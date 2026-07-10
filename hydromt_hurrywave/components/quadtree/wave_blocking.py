"""Wave blocking coefficients for HurryWave.

Directional wave-blocking coefficients are computed per quadtree cell by
projecting sub-grid-scale topographic obstacles onto directional bins.
The algorithm is ported from ``cht_hurrywave.WaveBlocking``.
"""

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import numba
import numpy as np
import pandas as pd
import shapely
import xarray as xr

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from hydromt import hydromt_step
from hydromt.model.components import ModelComponent

from hydromt_hurrywave.utils import make_regular_grid
from hydromt_hurrywave.workflows.merge import merge_multi_dataarrays

if TYPE_CHECKING:
    import geopandas as gpd

    from hydromt_hurrywave import HurrywaveModel

logger = logging.getLogger(f"hydromt.{__name__}")
_MATPLOTLIB_MISSING = "matplotlib is required for plotting."


class HurrywaveWaveBlocking(ModelComponent):
    """Directional wave-blocking coefficients on the HurryWave quadtree grid.

    Coefficients are stored in a netCDF file (``wblfile``) as a
    :class:`xarray.Dataset` with variable ``blocking_coefficient`` of
    dimensions ``(directions, cells)``.

    The :meth:`create` method computes blocking from sub-grid bathymetry using
    either a *cht_bathymetry* database or a HydroMT data-catalog elevation
    list — the same dual approach used by
    :class:`~hydromt_sfincs.SfincsQuadtreeSubgridTable`.
    """

    def __init__(self, model: "HurrywaveModel"):
        self._data: Optional[xr.Dataset] = None
        self._coastline_gdf: Optional[object] = None   # stored after reprojection
        self._nr_subgrid_pixels: int = 20
        super().__init__(model=model)

    @property
    def data(self) -> Optional[xr.Dataset]:
        """The wave-blocking dataset (lazily read on first access in read mode)."""
        if self._data is None and self.root.is_reading_mode():
            self.read()
        return self._data

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def read(self, filename: Optional[str] = None) -> None:
        """Read the wave-blocking file (``wblfile``).

        Parameters
        ----------
        filename : str or Path, optional
            Override the filename from config.
        """
        abs_path = self.model.config.get_set_file_variable("wblfile", value=filename)
        if abs_path is None or not abs_path.exists():
            return

        logger.info(f"Reading wave-blocking coefficients from {abs_path}")
        self._data = xr.open_dataset(str(abs_path))

    def write(self, filename: Optional[str] = None) -> None:
        """Write wave-blocking coefficients to a netCDF file.

        Parameters
        ----------
        filename : str or Path, optional
            Override the filename from config.  Defaults to ``hurrywave.wbl``.
        """
        if self._data is None:
            return

        abs_path = self.model.config.get_set_file_variable(
            "wblfile", value=filename, default="hurrywave.wbl"
        )
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Writing wave-blocking coefficients to {abs_path}")
        self._data.to_netcdf(str(abs_path))

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    @hydromt_step
    def create(
        self,
        elevation_list: Optional[list] = None,
        bathymetry_database: Optional[object] = None,
        coastline_gdf: Optional["gpd.GeoDataFrame"] = None,
        nr_dirs: int = 36,
        nr_subgrid_pixels: int = 20,
        threshold_level: float = -5.0,
        nrmax: int = 2000,
        quiet: bool = False,
        progress_bar: Optional[object] = None,
    ) -> None:
        """Compute directional wave-blocking coefficients.

        Exactly one obstacle source must be supplied:

        * **bathymetry_database** — cht_bathymetry database object; sub-grid
          elevation is fetched via ``get_bathymetry_on_grid()``.
        * **elevation_list** — HydroMT data-catalog datasets merged at sub-grid
          resolution (same format as
          :meth:`~hydromt_hurrywave.HurrywaveQuadtreeElevation.create`).
        * **coastline_gdf** — :class:`geopandas.GeoDataFrame` of
          ``LineString`` / ``MultiLineString`` geometries.  Any sub-grid pixel
          intersected by a line is treated as a full obstacle.

        Parameters
        ----------
        elevation_list : list, optional
            *cht_bathymetry* path:
                List of dataset names / dicts understood by
                ``bathymetry_database.get_bathymetry_on_grid()``.
            *HydroMT data-catalog path* (when ``bathymetry_database`` is
            ``None``):
                List of dicts with at least an ``"elevation"`` key pointing to
                a data-catalog entry and optional merge arguments, e.g.::

                    [
                        {'elevation': 'gebco'},
                        {'elevation': 'merit_hydro', 'merge_method': 'first'},
                    ]

        bathymetry_database : object, optional
            cht_bathymetry database object.
        coastline_gdf : geopandas.GeoDataFrame, optional
            Line geometries representing coastlines / obstacles.  Reprojected
            to the model CRS automatically if the GeoDataFrame carries a CRS.
            Cannot be combined with ``bathymetry_database`` or
            ``elevation_list``.
        nr_dirs : int, optional
            Number of directional bins, by default 36.  Must be even.
        nr_subgrid_pixels : int, optional
            Sub-grid pixels per quadtree cell edge, by default 20.
        threshold_level : float, optional
            Elevation threshold [m] above which a sub-grid pixel is treated as
            an obstacle (ignored for the ``coastline_gdf`` path), by default
            -5.0.
        nrmax : int, optional
            Maximum number of quadtree cells per processing block, by default
            2000.
        quiet : bool, optional
            Suppress per-block progress messages, by default False.
        progress_bar : object, optional
            Optional progress-bar object (cht_utils / DelftDashboard style)
            with ``set_text``, ``set_minimum``, ``set_maximum``,
            ``set_value``, and ``was_canceled`` methods.
        """
        # ----------------------------------------------------------
        # Validate inputs
        # ----------------------------------------------------------
        n_sources = sum([
            bathymetry_database is not None,
            elevation_list is not None,
            coastline_gdf is not None,
        ])
        if n_sources == 0:
            raise ValueError(
                "Provide exactly one of: bathymetry_database, elevation_list, "
                "or coastline_gdf."
            )
        if n_sources > 1:
            raise ValueError(
                "bathymetry_database, elevation_list, and coastline_gdf are "
                "mutually exclusive — supply only one."
            )

        # ----------------------------------------------------------
        # Pre-process coastline GDF
        # ----------------------------------------------------------
        coastline_union = None
        if coastline_gdf is not None:
            if coastline_gdf.crs is not None and coastline_gdf.crs != self.model.crs:
                coastline_gdf = coastline_gdf.to_crs(self.model.crs)
            coastline_union = coastline_gdf.union_all()

        # Store for later plotting
        self._coastline_gdf = coastline_gdf
        self._nr_subgrid_pixels = nr_subgrid_pixels
        # ----------------------------------------------------------
        # Grid metadata
        # ----------------------------------------------------------
        grid = self.model.quadtree_grid.data
        nr_cells = grid.sizes["mesh2d_nFaces"]
        x0 = grid.attrs["x0"]
        y0 = grid.attrs["y0"]
        dx = grid.attrs["dx"]
        dy = grid.attrs["dy"]
        rotation = grid.attrs["rotation"]
        nr_ref_levs = grid.attrs["nr_levels"]
        cosrot = np.cos(np.radians(rotation))
        sinrot = np.sin(np.radians(rotation))

        level = grid["level"].values[:] - 1  # 0-based
        n = grid["n"].values[:] - 1          # 0-based
        m = grid["m"].values[:] - 1          # 0-based
        cell_mask = self.model.quadtree_grid.data["mask"].values[:]

        # Level boundaries
        ifirst = np.zeros(nr_ref_levs, dtype=int)
        ilast = np.zeros(nr_ref_levs, dtype=int)
        ireflast = -1
        for ic in range(nr_cells):
            if level[ic] > ireflast:
                ifirst[level[ic]] = ic
                ireflast = level[ic]
        for ilev in range(nr_ref_levs - 1):
            ilast[ilev] = ifirst[ilev + 1] - 1
        ilast[nr_ref_levs - 1] = nr_cells - 1

        # ----------------------------------------------------------
        # Base resolution for elevation datasets — parsed lazily per level
        # ----------------------------------------------------------
        _elev_res_base: Optional[float] = None
        if elevation_list is not None:
            _elev_res_base = dx
            if self.model.crs.is_geographic:
                _elev_res_base *= 111111.0

        # ----------------------------------------------------------
        # Directional vectors (half-circle, mirrored for full 360° — symmetric by design)
        # Angles are in compass convention (0° = north, clockwise); converted to
        # math convention (0° = east, CCW) for the direction vectors.
        # ----------------------------------------------------------
        nvec = nr_dirs // 2
        dtheta = 360.0 / nr_dirs
        angles_half = np.linspace(
            0.5 * dtheta, 180.0 + 0.5 * dtheta, nvec, endpoint=False
        )
        math_radians_half = np.deg2rad(90.0 - angles_half)  # compass → math
        vectors = np.array(
            [[np.cos(a), np.sin(a), 0] for a in math_radians_half]
        )
        cos_angles = np.ascontiguousarray(vectors[:, 0])
        sin_angles = np.ascontiguousarray(vectors[:, 1])

        # ----------------------------------------------------------
        # Blocking coefficient array: (nr_dirs, nr_cells)
        # ----------------------------------------------------------
        block_coefficient = np.zeros((nr_dirs, nr_cells), dtype=float)
        counter = 0

        # ----------------------------------------------------------
        # Loop over refinement levels
        # ----------------------------------------------------------
        for ilev in range(nr_ref_levs):
            i0 = ifirst[ilev]
            i1 = ilast[ilev]
            cell_indices_in_level = np.arange(i0, i1 + 1, dtype=int)
            nr_cells_in_level = len(cell_indices_in_level)
            if nr_cells_in_level == 0:
                continue

            dxi = dx / 2**ilev
            dyi = dy / 2**ilev
            refi = nr_subgrid_pixels
            dxp = dxi / refi
            dyp = dyi / refi

            n0 = int(n[i0:i1 + 1].min())
            n1 = int(n[i0:i1 + 1].max())
            m0 = int(m[i0:i1 + 1].min())
            m1 = int(m[i0:i1 + 1].max())

            nrcb = max(1, int(np.floor(nrmax / refi)))
            nrbn = int(np.ceil((n1 - n0 + 1) / nrcb))
            nrbm = int(np.ceil((m1 - m0 + 1) / nrcb))

            if progress_bar is not None:
                progress_bar.set_text(
                    f"Computing wave-blocking coefficients (level {ilev + 1}/{nr_ref_levs}) ..."
                )
                progress_bar.set_minimum(0)
                progress_bar.set_maximum(nrbm * nrbn)
                progress_bar.set_value(0)

            ib = 0

            if not quiet:
                logger.info(
                    f"Processing level {ilev + 1}/{nr_ref_levs} — {nr_cells_in_level} cells, {nrbm}x{nrbn} blocks"
                )

            # Opt 3: fetch elevation datasets for this level only now (lazy per-level)
            parsed_elev_list: Optional[list] = None
            if _elev_res_base is not None:
                res_subgrid = (_elev_res_base / 2**ilev) / nr_subgrid_pixels
                parsed_elev_list = self.model._parse_datasets_elevation(
                    elevation_list, res=res_subgrid
                )

            for ii in range(nrbm):
                for jj in range(nrbn):

                    if progress_bar is not None:
                        progress_bar.set_value(ib)
                        if progress_bar.was_canceled():
                            logger.warning("Wave-blocking computation cancelled.")
                            return
                        
                    if not quiet:
                        logger.info(
                            f"Processing block (ilev={ilev}, ii={ii}, jj={jj}) ..."
                        )


                    ib += 1

                    bn0 = n0 + jj * nrcb
                    bn1 = min(bn0 + nrcb - 1, n1) + 1
                    bm0 = m0 + ii * nrcb
                    bm1 = min(bm0 + nrcb - 1, m1) + 1

                    if bn1 <= bn0 or bm1 <= bm0:
                        continue

                    # Subgrid pixel coordinates in the grid's local (rotated) frame
                    x00 = 0.5 * dxp + bm0 * refi * dxp
                    x01 = x00 + (bm1 - bm0) * refi * dxp
                    y00 = 0.5 * dyp + bn0 * refi * dyp
                    y01 = y00 + (bn1 - bn0) * refi * dyp

                    x0v = np.arange(x00, x01, dxp)
                    y0v = np.arange(y00, y01, dyp)
                    xg0, yg0 = np.meshgrid(x0v, y0v)

                    # Rotate and translate to real-world coordinates
                    xg = x0 + cosrot * xg0 - sinrot * yg0
                    yg = y0 + sinrot * xg0 + cosrot * yg0

                    # --------------------------------------------------
                    # Fetch sub-grid obstacle data
                    # --------------------------------------------------
                    if bathymetry_database is not None:
                        try:
                            zg = bathymetry_database.get_bathymetry_on_grid(
                                xg, yg, self.model.crs, elevation_list
                            )
                        except Exception as exc:
                            logger.error(
                                f"Error fetching bathymetry for block "
                                f"(ilev={ilev}, ii={ii}, jj={jj}): {exc}"
                            )
                            continue
                        cell_threshold = threshold_level
                    elif elevation_list is not None:
                        try:
                            da_like = make_regular_grid(
                                x0=x0,
                                y0=y0,
                                dx=dxp,
                                dy=dyp,
                                mmax=bm1 * refi,
                                nmax=bn1 * refi,
                                rotation=rotation,
                                crs=self.model.crs,
                                mmin=bm0 * refi,
                                nmin=bn0 * refi,
                                make_ugrid=False,
                            )
                            da_dep = merge_multi_dataarrays(
                                da_list=parsed_elev_list,
                                da_like=da_like,
                                buffer_cells=0,
                                interp_method="linear",
                                logger=logger,
                            )
                            # da_dep.values has shape (nmax_sub, mmax_sub)
                            zg = da_dep.values
                        except Exception as exc:
                            logger.debug(
                                f"Skipping block (ilev={ilev}, ii={ii}, jj={jj}): {exc}"
                            )
                            continue
                        cell_threshold = threshold_level
                    else:
                        # Coastline lines path — rasterise onto the block grid
                        block_box = shapely.box(
                            float(xg.min()) - dxp,
                            float(yg.min()) - dyp,
                            float(xg.max()) + dxp,
                            float(yg.max()) + dyp,
                        )
                        block_lines = coastline_union.intersection(block_box)
                        zg = _rasterize_lines_on_block(block_lines, xg, yg, dxp, dyp)
                        cell_threshold = 0.5  # binary raster: values are 0 or 1

                    # --------------------------------------------------
                    # Find cells in this block (vectorised)
                    # --------------------------------------------------
                    ni = n[cell_indices_in_level]
                    mi = m[cell_indices_in_level]
                    in_block = (
                        (ni >= bn0) & (ni < bn1) & (mi >= bm0) & (mi < bm1)
                    )
                    cells_in_block = cell_indices_in_level[in_block]

                    if cells_in_block.size == 0:
                        continue

                    # --------------------------------------------------
                    # Opt 5: skip block if no pixel exceeds threshold
                    # --------------------------------------------------
                    if not np.any(zg > cell_threshold):
                        continue

                    # --------------------------------------------------
                    # Opt 2: collect patches for all active cells at once
                    # --------------------------------------------------
                    active_indices = [
                        idx for idx in cells_in_block if cell_mask[idx] == 1
                    ]
                    if not active_indices:
                        continue

                    n_active = len(active_indices)
                    patches = np.empty((n_active, refi, refi), dtype=np.float64)
                    for k, idx in enumerate(active_indices):
                        nn = (n[idx] - bn0) * refi
                        mm = (m[idx] - bm0) * refi
                        patches[k] = zg[nn: nn + refi, mm: mm + refi]

                    # --------------------------------------------------
                    # Opt 1 & 2: JIT-compiled batch blocking computation
                    # --------------------------------------------------
                    ratios = _blocking_kernel(
                        patches, cell_threshold, cos_angles, sin_angles, 100
                    )  # (n_active, nvec)

                    for k, idx in enumerate(active_indices):
                        r = ratios[k]
                        if r.any():
                            block_coefficient[:nvec, idx] = r
                            block_coefficient[nvec:, idx] = r
                            counter += 1


        if not quiet:
            logger.info(
                f"Wave-blocking: {counter} cells with obstacles above threshold."
            )

        # ----------------------------------------------------------
        # Build xarray Dataset
        # ----------------------------------------------------------
        all_angles = np.concatenate((angles_half, angles_half + 180.0))
        da = xr.DataArray(
            block_coefficient,
            dims=("directions", "cells"),
            coords={
                "directions": all_angles,
                "cells": np.arange(nr_cells),
            },
            name="blocking_coefficient",
            attrs={"description": "Directional wave-blocking coefficient per cell [-]"},
        )
        self._data = xr.Dataset({"blocking_coefficient": da})
        self._data.attrs.update(
            {
                "title": "HurryWave wave-blocking file",
                "institution": "Deltares",
                "history": f"Created {pd.Timestamp.now()}",
            }
        )

    def plot(self, cell_idx: int = 0, ax=None):
        """Plot blocking coefficients as a polar bar chart for one cell.

        Parameters
        ----------
        cell_idx : int
            Cell index to plot, by default 0.
        ax : matplotlib.axes.Axes, optional
            Polar axes to draw on.  A new figure is created when ``None``.

        Returns
        -------
        matplotlib.axes.Axes
        """
        if not HAS_MATPLOTLIB:
            raise ImportError(_MATPLOTLIB_MISSING)
        if self.data is None:
            raise ValueError("No blocking data available — run create() or read() first.")

        bc = self.data["blocking_coefficient"].values[:, cell_idx]
        angles_deg = self.data["directions"].values  # compass convention (0°=north, CW)
        compass_rad = np.deg2rad(angles_deg)
        width = np.deg2rad(360.0 / len(angles_deg))

        if ax is None:
            _, ax = plt.subplots(subplot_kw={"projection": "polar"})

        ax.bar(compass_rad, bc, width=width, bottom=0.0, align="center", alpha=0.7)

        ax.set_title(f"Wave-blocking coefficients — cell {cell_idx}")
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_ylim(0, 1)
        return ax

    # ------------------------------------------------------------------
    # Spatial helpers
    # ------------------------------------------------------------------

    def _cell_subgrid_coords(self, cell_idx: int):
        """Return real-world pixel-centre coordinates for one cell's sub-grid.

        Returns
        -------
        xg, yg : ndarray of shape (refi, refi)
            Real-world pixel-centre coordinates.
        dxp, dyp : float
            Sub-grid pixel size.
        """
        grid = self.model.quadtree_grid.data
        x0 = grid.attrs["x0"]
        y0 = grid.attrs["y0"]
        dx = grid.attrs["dx"]
        dy = grid.attrs["dy"]
        rotation = grid.attrs["rotation"]

        level = int(grid["level"].values[cell_idx]) - 1
        n_cell = int(grid["n"].values[cell_idx]) - 1
        m_cell = int(grid["m"].values[cell_idx]) - 1

        dxi = dx / 2**level
        dyi = dy / 2**level
        refi = self._nr_subgrid_pixels
        dxp = dxi / refi
        dyp = dyi / refi

        cosrot = np.cos(np.radians(rotation))
        sinrot = np.sin(np.radians(rotation))

        x0v = 0.5 * dxp + m_cell * dxi + np.arange(refi) * dxp
        y0v = 0.5 * dyp + n_cell * dyi + np.arange(refi) * dyp
        xg0, yg0 = np.meshgrid(x0v, y0v)

        xg = x0 + cosrot * xg0 - sinrot * yg0
        yg = y0 + sinrot * xg0 + cosrot * yg0
        return xg, yg, dxp, dyp

    def _draw_lines_on_ax(self, ax, clipped_gdf, cell_idx: int, dxp: float, dyp: float):
        """Overlay clipped line geometries on *ax* in pixel-index coordinates."""
        grid = self.model.quadtree_grid.data
        x0 = grid.attrs["x0"]
        y0 = grid.attrs["y0"]
        dx = grid.attrs["dx"]
        rotation = grid.attrs["rotation"]
        cosrot = np.cos(np.radians(rotation))
        sinrot = np.sin(np.radians(rotation))
        level = int(grid["level"].values[cell_idx]) - 1
        dy = grid.attrs["dy"]
        m_cell = int(grid["m"].values[cell_idx]) - 1
        n_cell = int(grid["n"].values[cell_idx]) - 1
        dxi = dx / 2**level
        dyi = dy / 2**level

        for geom in clipped_gdf.geometry:
            if geom is None or geom.is_empty:
                continue
            parts = geom.geoms if hasattr(geom, "geoms") else [geom]
            for line in parts:
                coords = np.array(line.coords)
                rel_x = coords[:, 0] - x0
                rel_y = coords[:, 1] - y0
                lx = cosrot * rel_x + sinrot * rel_y
                ly = -sinrot * rel_x + cosrot * rel_y
                px = (lx - m_cell * dxi) / dxp - 0.5
                py = (ly - n_cell * dyi) / dyp - 0.5
                ax.plot(px, py, color="red", linewidth=1)

    def plot_map(self, cell_idx: int = 0, ax=None):
        """Plot the sub-grid obstacle map for one cell.

        Shows the rasterised coastline lines (or a blank grid when no coastline
        was supplied) and, if a coastline GeoDataFrame is stored, overlays the
        actual line geometry in pixel-index coordinates.

        Parameters
        ----------
        cell_idx : int
            Cell index to plot, by default 0.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on.  A new figure is created when ``None``.

        Returns
        -------
        matplotlib.axes.Axes
        """
        if not HAS_MATPLOTLIB:
            raise ImportError(_MATPLOTLIB_MISSING)

        xg, yg, dxp, dyp = self._cell_subgrid_coords(cell_idx)
        refi = self._nr_subgrid_pixels

        if ax is None:
            _, ax = plt.subplots()

        if self._coastline_gdf is not None:
            # Rasterise lines onto this cell's sub-grid
            cell_box = shapely.box(
                float(xg.min()) - dxp, float(yg.min()) - dyp,
                float(xg.max()) + dxp, float(yg.max()) + dyp,
            )
            clipped = self._coastline_gdf.clip(cell_box)

            zgc = _rasterize_lines_on_block(
                clipped.union_all() if not clipped.is_empty.all() else None,
                xg, yg, dxp, dyp,
            )

            ax.imshow(
                zgc, origin="lower", aspect="equal",
                cmap="Greys", vmin=0, vmax=1,
                extent=[0, refi, 0, refi],
            )

            # Overlay actual lines in pixel-index space
            self._draw_lines_on_ax(ax, clipped, cell_idx, dxp, dyp)
        else:
            # No obstacle data stored — show empty grid with message
            ax.imshow(
                np.zeros((refi, refi)), origin="lower", aspect="equal",
                cmap="Greys", vmin=0, vmax=1,
                extent=[0, refi, 0, refi],
            )
            ax.text(
                0.5, 0.5, "No coastline data stored\n(elevation path used)",
                transform=ax.transAxes, ha="center", va="center", fontsize=9,
            )

        ax.set_xlim(0, refi)
        ax.set_ylim(0, refi)
        ax.set_xlabel("m pixel")
        ax.set_ylabel("n pixel")
        ax.set_title(f"Sub-grid obstacle map — cell {cell_idx}")
        return ax

    def set(self, data: xr.Dataset) -> None:
        """Set blocking coefficients from an existing :class:`xarray.Dataset`.

        Parameters
        ----------
        data : xr.Dataset
            Must contain at minimum ``blocking_coefficient`` with dimensions
            ``(directions, cells)``.
        """
        self._data = data

    def clear(self) -> None:
        """Remove all wave-blocking data from memory."""
        self._data = None


# ---------------------------------------------------------------------------
# Coastline rasterisation helper
# ---------------------------------------------------------------------------

def _bounds_overlap(a: tuple, b: tuple) -> bool:
    """Return True if bounding boxes (xmin, ymin, xmax, ymax) overlap."""
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def _rasterize_lines_on_block(
    lines_geom, xg: np.ndarray, yg: np.ndarray, dxp: float, dyp: float
) -> np.ndarray:
    """Rasterise line geometries onto a sub-grid block.

    Parameters
    ----------
    lines_geom : shapely geometry or None
        Merged line geometry clipped to (approximately) the block extent.
    xg, yg : ndarray of shape (nrows, ncols)
        Real-world coordinates of sub-grid pixel centres.
    dxp, dyp : float
        Sub-grid pixel size in x and y.

    Returns
    -------
    ndarray of shape (nrows, ncols), dtype float32
        1.0 where a line crosses the pixel, 0.0 elsewhere.
    """
    if lines_geom is None or lines_geom.is_empty:
        return np.zeros(xg.shape, dtype=np.float32)

    hdx, hdy = dxp / 2.0, dyp / 2.0
    pixel_boxes = shapely.box(xg - hdx, yg - hdy, xg + hdx, yg + hdy)
    return shapely.intersects(pixel_boxes, lines_geom).astype(np.float32)


# ---------------------------------------------------------------------------
# Numba JIT kernel — batch blocking computation
# ---------------------------------------------------------------------------

@numba.njit(cache=True)
def _blocking_kernel(
    patches: np.ndarray,
    threshold: float,
    cos_angles: np.ndarray,
    sin_angles: np.ndarray,
    nrp: int,
) -> np.ndarray:
    """Compute directional blocking fractions for a batch of sub-grid patches.

    Parameters
    ----------
    patches : float64 array (n_cells, H, W)
        Sub-grid elevation patches, one per active cell.
    threshold : float
        Elevation threshold — pixels above this are obstacles.
    cos_angles, sin_angles : float64 array (ndirs,)
        Direction vector components for the half-circle bins.
    nrp : int
        Number of discretisation bins for the union-of-intervals calculation.

    Returns
    -------
    float64 array (n_cells, ndirs)
        Blocking ratios in [0, 1].
    """
    n_cells = patches.shape[0]
    H = patches.shape[1]
    W = patches.shape[2]
    ndirs = cos_angles.shape[0]

    W_f = float(W)
    H_f = float(H)
    cx_mid = W_f / 2.0
    cy_mid = H_f / 2.0
    r = math.sqrt(W_f * W_f + H_f * H_f) / 2.0

    result = np.zeros((n_cells, ndirs))

    # Pre-compute clipped screen geometry once — identical for every cell
    # (all cells in a block share the same patch dimensions refi × refi)
    x0c_arr = np.empty(ndirs)
    y0c_arr = np.empty(ndirs)
    vx_arr = np.empty(ndirs)
    vy_arr = np.empty(ndirs)
    cdenom_arr = np.empty(ndirs)

    for d in range(ndirs):
        cd = cos_angles[d]
        sd = sin_angles[d]
        angle = math.atan2(sd, cd)
        pmx = cx_mid + r * math.cos(angle + math.pi)
        pmy = cy_mid + r * math.sin(angle + math.pi)
        ox = -sd
        oy = cd
        x0 = pmx - W_f * ox
        y0 = pmy - H_f * oy
        x1 = pmx + W_f * ox
        y1 = pmy + H_f * oy
        dx = x1 - x0
        dy = y1 - y0
        denom = dx * dx + dy * dy
        if denom == 0.0:
            x0c_arr[d] = x0
            y0c_arr[d] = y0
            vx_arr[d] = 0.0
            vy_arr[d] = 0.0
            cdenom_arr[d] = 0.0
            continue
        # Clip screen to cell corners: (0,0), (W,0), (W,H), (0,H)
        min_c = 1.0e18
        max_c = -1.0e18
        for k in range(4):
            if k == 0:
                ck_x = 0.0; ck_y = 0.0
            elif k == 1:
                ck_x = W_f; ck_y = 0.0
            elif k == 2:
                ck_x = W_f; ck_y = H_f
            else:
                ck_x = 0.0; ck_y = H_f
            proj = ((ck_x - x0) * dx + (ck_y - y0) * dy) / denom
            if proj < min_c:
                min_c = proj
            if proj > max_c:
                max_c = proj
        x0c = x0 + min_c * dx
        y0c = y0 + min_c * dy
        x1c = x0 + max_c * dx
        y1c = y0 + max_c * dy
        cvx = x1c - x0c
        cvy = y1c - y0c
        cdenom = cvx * cvx + cvy * cvy
        x0c_arr[d] = x0c
        y0c_arr[d] = y0c
        vx_arr[d] = cvx
        vy_arr[d] = cvy
        cdenom_arr[d] = cdenom

    diff = np.zeros((ndirs, nrp + 2), dtype=np.int32)

    for ic in range(n_cells):
        # Reset difference array
        for d in range(ndirs):
            for k in range(nrp + 2):
                diff[d, k] = 0

        # Accumulate shadow intervals for each obstacle pixel
        for ni in range(H):
            for mi in range(W):
                v = patches[ic, ni, mi]
                if math.isnan(v) or v <= threshold:
                    continue

                # Pixel corners in index space
                ocx0 = float(mi)
                ocx1 = float(mi + 1)
                ocy0 = float(ni)
                ocy1 = float(ni + 1)

                for d in range(ndirs):
                    if cdenom_arr[d] == 0.0:
                        continue
                    x0c = x0c_arr[d]
                    y0c = y0c_arr[d]
                    cvx = vx_arr[d]
                    cvy = vy_arr[d]
                    cdenom = cdenom_arr[d]

                    # Project the 4 pixel corners onto the clipped screen
                    pmin = 2.0
                    pmax = -1.0
                    for k in range(4):
                        if k == 0:
                            px_k = ocx0; py_k = ocy0
                        elif k == 1:
                            px_k = ocx1; py_k = ocy0
                        elif k == 2:
                            px_k = ocx1; py_k = ocy1
                        else:
                            px_k = ocx0; py_k = ocy1
                        proj = ((px_k - x0c) * cvx + (py_k - y0c) * cvy) / cdenom
                        if proj < pmin:
                            pmin = proj
                        if proj > pmax:
                            pmax = proj

                    if pmin < 0.0:
                        pmin = 0.0
                    if pmin > 1.0:
                        pmin = 1.0
                    if pmax < 0.0:
                        pmax = 0.0
                    if pmax > 1.0:
                        pmax = 1.0

                    i0 = int(pmin * nrp)
                    i1 = int(pmax * nrp)
                    if i0 <= nrp:
                        diff[d, i0] += 1
                    if i1 <= nrp:
                        diff[d, i1] -= 1

        # Union-of-intervals: cumsum of diff → covered fraction
        for d in range(ndirs):
            covered = 0
            running = 0
            for k in range(nrp):
                running += diff[d, k]
                if running > 0:
                    covered += 1
            result[ic, d] = float(covered) / float(nrp)

    return result


# ---------------------------------------------------------------------------
# Sub-grid cell helper (ported from cht_hurrywave.Cell2)
# ---------------------------------------------------------------------------

class _Cell:
    """Compute directional blocking for a single sub-grid cell patch.

    Parameters
    ----------
    elevation_map : ndarray of shape (refi, refi)
        Sub-grid bathymetry/elevation values [m].
    threshold_level : float
        Elevation above which a pixel is treated as an obstacle [m].
    """

    def __init__(self, elevation_map: np.ndarray, threshold_level: float = 0.0):
        self.height, self.width = elevation_map.shape
        self.dx = 1.0
        self.dy = 1.0
        self.elevation_map = elevation_map
        self.threshold_level = threshold_level
        self._extract_obstacles(elevation_map, threshold_level)
        self.midpoint, self.circle_radius = self._circle_around_cell(
            self.width, self.height
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _circle_around_cell(w, h):
        diagonal = math.sqrt(w**2 + h**2)
        return (w / 2, h / 2), diagonal / 2

    def _extract_obstacles(self, elevation_map: np.ndarray, threshold: float):
        n_idx, m_idx = np.where(elevation_map > threshold)
        nobs = len(m_idx)
        self.obscor_x = np.zeros((nobs, 4))
        self.obscor_y = np.zeros((nobs, 4))
        self.obscor_x[:, 0] = m_idx
        self.obscor_x[:, 1] = m_idx + 1
        self.obscor_x[:, 2] = m_idx + 1
        self.obscor_x[:, 3] = m_idx
        self.obscor_y[:, 0] = n_idx
        self.obscor_y[:, 1] = n_idx
        self.obscor_y[:, 2] = n_idx + 1
        self.obscor_y[:, 3] = n_idx + 1

    @staticmethod
    def _project_points_on_line(
        point_x: np.ndarray, point_y: np.ndarray, line: np.ndarray
    ) -> np.ndarray:
        """Vectorised projection of obstacle corners onto a direction line.

        Parameters
        ----------
        point_x, point_y : ndarray of shape (nobs, 4)
            Corner coordinates.
        line : ndarray of shape (2, 2)
            ``[[x0, y0], [x1, y1]]`` defining the projection line.

        Returns
        -------
        ndarray of shape (nobs, 4)
            Normalised projection lengths along the line.
        """
        nobs = point_x.shape[0]
        px = point_x.ravel()
        py = point_y.ravel()

        line_vec = line[1] - line[0]
        point_vec = np.column_stack([px, py]) - line[0]
        denom = np.dot(line_vec, line_vec)
        if denom == 0.0:
            return np.zeros((nobs, 4))

        proj = np.dot(point_vec, line_vec) / denom
        return proj.reshape(nobs, 4)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def plot(self, ax=None):
        """Plot the sub-grid elevation map (zgc) with the obstacle mask overlaid.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on.  A new figure is created when ``None``.

        Returns
        -------
        matplotlib.axes.Axes
        """
        if not HAS_MATPLOTLIB:
            raise ImportError(_MATPLOTLIB_MISSING)

        if ax is None:
            _, ax = plt.subplots()

        im = ax.imshow(
            self.elevation_map, origin="lower", aspect="equal", cmap="terrain",
            vmin = -100, vmax = 0
        )
        obstacle_mask = self.elevation_map > self.threshold_level
        ax.contour(obstacle_mask.astype(float), levels=[0.5], colors="red", linewidths=1)
        plt.colorbar(im, ax=ax, label="elevation [m]")
        ax.set_title(f"Sub-grid elevation (zgc) — threshold {self.threshold_level} m")
        ax.set_xlabel("m index")
        ax.set_ylabel("n index")
        return ax

    def project_on_planes(self, directions: np.ndarray, nrp: int = 100) -> np.ndarray:
        """Compute blocking fractions for all direction bins in one vectorised pass.

        Parameters
        ----------
        directions : ndarray of shape (ndirs, 3)
            Unit vectors ``[cos θ, sin θ, 0]`` for each direction bin.
        nrp : int, optional
            Number of discretisation bins for the union-of-intervals calculation,
            by default 100.

        Returns
        -------
        ndarray of shape (ndirs,)
            Blocking ratios in [0, 1].
        """
        ndirs = directions.shape[0]
        nobs = self.obscor_x.shape[0]
        if nobs == 0:
            return np.zeros(ndirs)

        cx, cy = self.midpoint
        r = self.circle_radius
        W, H = float(self.width), float(self.height)

        # --- Screen geometry for all directions simultaneously --- (ndirs,)
        angle_rad = np.arctan2(directions[:, 1], directions[:, 0])
        proj_mid_x = cx + r * np.cos(angle_rad + np.pi)
        proj_mid_y = cy + r * np.sin(angle_rad + np.pi)
        ortho_x = -directions[:, 1]
        ortho_y =  directions[:, 0]

        x0 = proj_mid_x - W * ortho_x
        y0 = proj_mid_y - H * ortho_y
        x1 = proj_mid_x + W * ortho_x
        y1 = proj_mid_y + H * ortho_y

        # line_vec: (ndirs, 2),  denom: (ndirs,)
        line_vec = np.stack([x1 - x0, y1 - y0], axis=1)
        denom = np.einsum("di,di->d", line_vec, line_vec)

        # --- Clip screen to actual cell extent ---
        corners = np.array([[0., 0.], [W, 0.], [W, H], [0., H]])  # (4, 2)
        # point_vec from screen start to each corner: (ndirs, 4, 2)
        pvc = corners[np.newaxis] - np.stack([x0, y0], axis=1)[:, np.newaxis]
        # proj_corners: (ndirs, 4)
        proj_corners = np.einsum("nci,ni->nc", pvc, line_vec) / denom[:, np.newaxis]
        min_c = proj_corners.min(axis=1)  # (ndirs,)
        max_c = proj_corners.max(axis=1)

        dx = x1 - x0
        dy = y1 - y0
        x0c = x0 + min_c * dx
        y0c = y0 + min_c * dy
        x1c = x0 + max_c * dx
        y1c = y0 + max_c * dy

        # clipped_vec: (ndirs, 2),  clipped_denom: (ndirs,)
        clipped_vec = np.stack([x1c - x0c, y1c - y0c], axis=1)
        clipped_denom = np.einsum("di,di->d", clipped_vec, clipped_vec)

        # --- Project all obstacle corners onto every clipped screen ---
        # obs_pts: (nobs*4, 2)
        obs_pts = np.column_stack([self.obscor_x.ravel(), self.obscor_y.ravel()])
        # point_vec from clipped screen start to each obstacle corner: (ndirs, nobs*4, 2)
        pvo = obs_pts[np.newaxis] - np.stack([x0c, y0c], axis=1)[:, np.newaxis]
        # proj_obs_flat: (ndirs, nobs*4)  →  (ndirs, nobs, 4)
        proj_obs = (
            np.einsum("npi,ni->np", pvo, clipped_vec) / clipped_denom[:, np.newaxis]
        ).reshape(ndirs, nobs, 4)

        # min/max shadow interval per obstacle per direction: (ndirs, nobs)
        min_obs = np.clip(proj_obs.min(axis=2), 0.0, 1.0)
        max_obs = np.clip(proj_obs.max(axis=2), 0.0, 1.0)

        # --- Union of intervals via difference-array trick ---
        # diff[d, k] += 1 at interval start, -= 1 at interval end;
        # cumsum gives covered/not covered at each bin.
        i0_all = (min_obs * nrp).astype(int)  # (ndirs, nobs)
        i1_all = (max_obs * nrp).astype(int)
        diff = np.zeros((ndirs, nrp + 1), dtype=np.int16)
        d_idx = np.repeat(np.arange(ndirs), nobs)
        np.add.at(diff, (d_idx, i0_all.ravel()), 1)
        np.add.at(diff, (d_idx, i1_all.ravel()), -1)
        covered = np.cumsum(diff, axis=1)[:, :nrp] > 0

        return covered.sum(axis=1) / nrp

    def project_on_plane(self, incoming_direction: np.ndarray) -> float:
        """Compute the blocking fraction for a single direction.

        Thin wrapper around :meth:`project_on_planes` for convenience.

        Parameters
        ----------
        incoming_direction : array-like of length 3
            Unit vector ``[cos θ, sin θ, 0]``.

        Returns
        -------
        float
        """
        return float(self.project_on_planes(np.atleast_2d(incoming_direction))[0])

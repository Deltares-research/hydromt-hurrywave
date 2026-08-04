"""Wave blocking coefficients for HurryWave.

Directional wave-blocking coefficients are computed per quadtree cell by
projecting sub-grid-scale topographic obstacles onto directional bins.
The algorithm is ported from ``cht_hurrywave.WaveBlocking``.
"""

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import numpy as np
import pandas as pd
import xarray as xr

from hydromt import hydromt_step
from hydromt.model.components import ModelComponent

from hydromt_hurrywave.utils import make_regular_grid
from hydromt_hurrywave.workflows.merge import merge_multi_dataarrays

if TYPE_CHECKING:
    from hydromt_hurrywave import HurrywaveModel

logger = logging.getLogger(f"hydromt.{__name__}")


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
        elevation_list: list,
        bathymetry_database: Optional[object] = None,
        nr_dirs: int = 36,
        nr_subgrid_pixels: int = 20,
        threshold_level: float = -5.0,
        nrmax: int = 2000,
        quiet: bool = False,
        progress_bar: Optional[object] = None,
    ) -> None:
        """Compute directional wave-blocking coefficients.

        Sub-grid bathymetry is obtained either via a *cht_bathymetry* database
        (``bathymetry_database`` is not ``None``) or by merging HydroMT data-
        catalog datasets (same ``elevation_list`` format as
        :meth:`~hydromt_hurrywave.HurrywaveQuadtreeElevation.create`).

        Parameters
        ----------
        elevation_list : list
            With a *cht_bathymetry* database: list of dataset names / dicts
            understood by ``bathymetry_database.get_bathymetry_on_grid()``.
            With the HydroMT data-catalog path (``bathymetry_database`` is
            ``None``): list of dicts with at least an ``"elevation"`` key
            pointing to a data-catalog entry and optional merge arguments,
            e.g.::

                [
                    {'elevation': 'gebco'},
                    {'elevation': 'merit_hydro', 'merge_method': 'first'},
                ]

        bathymetry_database : object, optional
            cht_bathymetry database object.  When provided, sub-grid
            bathymetry is fetched via its ``get_bathymetry_on_grid()`` method.
            When ``None`` (default), the HydroMT data-catalog path is used.
        nr_dirs : int, optional
            Number of directional bins, by default 36.  Must be even.
        nr_subgrid_pixels : int, optional
            Sub-grid pixels per quadtree cell edge, by default 20.
        threshold_level : float, optional
            Elevation threshold [m] above which a sub-grid pixel is treated as
            an obstacle, by default -5.0.
        nrmax : int, optional
            Maximum number of quadtree cells per processing block, by default 2000.
        quiet : bool, optional
            Suppress per-block progress messages, by default False.
        progress_bar : object, optional
            Optional progress-bar object (cht_utils / DelftDashboard style)
            with ``set_text``, ``set_minimum``, ``set_maximum``,
            ``set_value``, and ``was_canceled`` methods.
        """
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
        # Pre-parse HydroMT elevation datasets (one set per level)
        # ----------------------------------------------------------
        elevation_list_per_level: Optional[list] = None
        if bathymetry_database is None:
            res = dx
            if self.model.crs.is_geographic:
                res *= 111111.0
            elevation_list_per_level = []
            for ilev in range(nr_ref_levs):
                res_subgrid = (res / 2**ilev) / nr_subgrid_pixels
                elevation_list_per_level.append(
                    self.model._parse_datasets_elevation(
                        elevation_list, res=res_subgrid
                    )
                )

        # ----------------------------------------------------------
        # Directional vectors (half-circle, mirrored for full 360°)
        # ----------------------------------------------------------
        nvec = nr_dirs // 2
        dtheta = 360.0 / nr_dirs
        angles_half = np.linspace(
            0.5 * dtheta, 180.0 + 0.5 * dtheta, nvec, endpoint=False
        )
        radians_half = np.deg2rad(angles_half)
        vectors = np.array(
            [[np.cos(a), np.sin(a), 0] for a in radians_half]
        )

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
            for ii in range(nrbm):
                for jj in range(nrbn):

                    if progress_bar is not None:
                        progress_bar.set_value(ib)
                        if progress_bar.was_canceled():
                            logger.warning("Wave-blocking computation cancelled.")
                            return

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

                    zg = np.full(xg.shape, np.nan)

                    # --------------------------------------------------
                    # Fetch sub-grid bathymetry
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
                    else:
                        # HydroMT data-catalog path
                        try:
                            da_like = make_regular_grid(
                                x0=x0,
                                y0=y0,
                                dx=dxp,
                                dy=dyp,
                                mmax=(bm1 - bm0) * refi,
                                nmax=(bn1 - bn0) * refi,
                                rotation=rotation,
                                crs=self.model.crs,
                                mmin=bm0 * refi,
                                nmin=bn0 * refi,
                                make_ugrid=False,
                            )
                            da_dep = merge_multi_dataarrays(
                                da_list=elevation_list_per_level[ilev],
                                da_like=da_like,
                                buffer_cells=0,
                                interp_method="linear",
                                logger=logger,
                            )
                            # da_dep.values has shape (nmax_sub, mmax_sub)
                            zg = da_dep.values
                        except Exception as exc:
                            logger.error(
                                f"Error merging elevation datasets for block "
                                f"(ilev={ilev}, ii={ii}, jj={jj}): {exc}"
                            )
                            continue

                    # --------------------------------------------------
                    # Find cells in this block
                    # --------------------------------------------------
                    cells_in_block = []
                    for ic in range(nr_cells_in_level):
                        idx = cell_indices_in_level[ic]
                        if (
                            n[idx] >= bn0 and n[idx] < bn1
                            and m[idx] >= bm0 and m[idx] < bm1
                        ):
                            cells_in_block.append(idx)

                    if not cells_in_block:
                        continue

                    # --------------------------------------------------
                    # Compute blocking coefficients per cell
                    # --------------------------------------------------
                    for idx in cells_in_block:
                        nn = (n[idx] - bn0) * refi
                        mm = (m[idx] - bm0) * refi
                        zgc = zg[nn: nn + refi, mm: mm + refi]
                        # BUG, somehow zbc can be empty, causing np.nanmax to raise an error.  Skip such cells.
                        if np.nanmax(zgc) < threshold_level:
                            # No obstacle above threshold → no blocking
                            continue

                        counter += 1
                        cell = _Cell(elevation_map=zgc, threshold_level=threshold_level)

                        for idx_dir in range(nvec):
                            ratio = cell.project_on_plane(vectors[idx_dir])
                            block_coefficient[idx_dir, idx] = ratio
                            block_coefficient[idx_dir + nvec, idx] = ratio

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
        px = point_x.reshape(nobs * 4, 1)
        py = point_y.reshape(nobs * 4, 1)

        line_vec = line[1] - line[0]
        point_vec = np.squeeze(np.array([px, py])).T - line[0]
        denom = np.dot(line_vec, line_vec)
        if denom == 0.0:
            return np.zeros((nobs, 4))

        proj = np.dot(point_vec, line_vec) / denom
        return proj.reshape(nobs, 4)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def project_on_plane(self, incoming_direction: np.ndarray) -> float:
        """Compute the fraction of the projection plane blocked by obstacles.

        Parameters
        ----------
        incoming_direction : array-like of length 3
            Unit vector ``[cos θ, sin θ, 0]`` for the wave direction.

        Returns
        -------
        float
            Blocking ratio in [0, 1].
        """
        if self.obscor_x.shape[0] == 0:
            return 0.0

        angle_rad = np.arctan2(incoming_direction[1], incoming_direction[0])
        cx, cy = self.midpoint
        r = self.circle_radius

        proj_mid_x = cx + r * np.cos(angle_rad + np.pi)
        proj_mid_y = cy - r * np.sin(angle_rad + np.pi)

        ortho = (-incoming_direction[1], incoming_direction[0])
        x0 = proj_mid_x - self.width * ortho[0]
        y0 = proj_mid_y - self.height * ortho[1]
        x1 = proj_mid_x + self.width * ortho[0]
        y1 = proj_mid_y + self.height * ortho[1]

        # Project cell corners to find the clipped line extent
        corners_x = np.array([[0.0, float(self.width), float(self.width), 0.0]])
        corners_y = np.array([[0.0, 0.0, float(self.height), float(self.height)]])
        line_corners = np.array([[x0, y0], [x1, y1]])
        proj_corners = self._project_points_on_line(
            corners_x, corners_y, line_corners
        )
        min_c = float(proj_corners.min())
        max_c = float(proj_corners.max())

        # Clip line
        x0c = x0 + min_c * (x1 - x0)
        y0c = y0 + min_c * (y1 - y0)
        x1c = x0 + max_c * (x1 - x0)
        y1c = y0 + max_c * (y1 - y0)
        line_clipped = np.array([[x0c, y0c], [x1c, y1c]])

        # Project obstacle corners
        proj_obs = self._project_points_on_line(
            self.obscor_x, self.obscor_y, line_clipped
        )
        min_obs = np.clip(proj_obs.min(axis=1), 0.0, 1.0)
        max_obs = np.clip(proj_obs.max(axis=1), 0.0, 1.0)

        # Discretise into bins and accumulate blocked fraction
        nrp = 100
        pnts = np.zeros(nrp, dtype=np.int8)
        for i0, i1 in zip(
            (min_obs * nrp).astype(int), (max_obs * nrp).astype(int)
        ):
            pnts[i0:i1] = 1

        return float(np.sum(pnts)) / nrp

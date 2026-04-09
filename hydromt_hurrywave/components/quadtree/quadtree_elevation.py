import logging
import os
from typing import TYPE_CHECKING, List, Optional

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.interpolate import RegularGridInterpolator
import xarray as xr
import xugrid as xu

from hydromt import hydromt_step
from hydromt.model.components import MeshComponent

from hydromt_sfincs.utils import make_regular_grid
from hydromt_sfincs.workflows.merge import merge_multi_dataarrays

try:
    import datashader as ds
    import datashader.transfer_functions as tf
    from datashader.utils import export_image
    HAS_DATASHADER = True
except ImportError:
    HAS_DATASHADER = False

if TYPE_CHECKING:
    from hydromt_hurrywave import HurrywaveModel

logger = logging.getLogger(f"hydromt.{__name__}")


class HurrywaveQuadtreeElevation(MeshComponent):
    def __init__(
        self,
        model: "HurrywaveModel",
    ):
        # Elevation is stored inside model.quadtree_grid.data["z"]
        super().__init__(model=model)

    @property
    def data(self):
        """Delegate to the quadtree grid dataset."""
        return self.model.quadtree_grid.data

    @property
    def mask(self):
        """Return the mask from the quadtree grid."""
        return self.model.quadtree_mask.data["mask"]

    def read(self):
        # Elevation is read as part of the quadtree grid nc file.
        pass

    def write(self):
        # Elevation is written as part of the quadtree grid nc file.
        pass

    @hydromt_step
    def create(
        self,
        elevation_list: List[dict],
        nrmax: int = 2000,
        buffer_cells: int = 0,
        interp_method: str = "linear",
        zmin: float = -1.0e9,
        zmax: float = 1.0e9,
        bathymetry_database: Optional[object] = None,
        mean_wet: bool = False,
        nr_subgrid_pixels: int = 20,
        threshold_level: float = 0.0,
    ):
        """Interpolate topobathy (z) data to the quadtree grid.

        Adds variable ``z`` [m+ref] to the quadtree grid dataset.

        By default, each cell receives the elevation at its centre point
        (standard bilinear interpolation).  When ``mean_wet=True``, a
        second pass is performed after the initial interpolation: each cell
        is assigned the *mean elevation of the wet sub-grid pixels*
        (i.e. pixels below ``threshold_level``).  This is equivalent to the
        ``set_bathymetry_mean_wet`` method in ``cht_hurrywave``.

        Parameters
        ----------
        elevation_list : list of dict
            Bathymetry datasets with optional merge arguments, e.g.::

                [
                    {'elevation': 'gebco', 'zmin': -100},
                    {'elevation': 'merit_hydro', 'merge_method': 'first'},
                ]

            When ``bathymetry_database`` is provided, this list is passed
            directly to the database (cht_bathymetry format).

        nrmax : int, optional
            Maximum number of grid cells per processing chunk, by default 2000.
        buffer_cells : int, optional
            Buffer cells between datasets for smooth transitions, by default 0.
        interp_method : str, optional
            Interpolation method for buffer cells, by default ``"linear"``.
        zmin, zmax : float, optional
            Clip final elevation to this range.
        bathymetry_database : object, optional
            cht_bathymetry database.  When provided, bathymetry is fetched
            via its ``get_bathymetry_on_points`` (standard interpolation)
            and ``get_bathymetry_on_grid`` (mean-wet pass) methods.
            When ``None``, the HydroMT data-catalog path is used.
        mean_wet : bool, optional
            Replace each cell's bed level with the mean elevation of its
            sub-grid pixels that are below ``threshold_level``, by default
            False.  Requires a second bathymetry fetch at sub-grid resolution.
        nr_subgrid_pixels : int, optional
            Sub-grid pixels per cell edge used in the mean-wet pass,
            by default 20.  Ignored when ``mean_wet=False``.
        threshold_level : float, optional
            Elevation threshold [m] separating wet from dry sub-grid pixels
            in the mean-wet pass, by default 0.0.  Ignored when
            ``mean_wet=False``.
        """
        # ----------------------------------------------------------
        # Standard cell-centre interpolation
        # ----------------------------------------------------------
        nlev = self.data.attrs["nr_levels"]
        xy = self.data.grid.face_coordinates
        nr_cells = len(xy)
        zz = np.full(nr_cells, np.nan)
        dx = self.data.attrs["dx"]
        dy = self.data.attrs["dy"]
        res = min(dx, dy)

        if self.model.crs.is_geographic:
            res *= 111111.0  # convert degrees → metres

        level = self.data["level"].values - 1  # 0-based
        level_indices = [np.where(level == ilev)[0] for ilev in range(nlev)]

        try:
            elevation_list_per_level = [
                self.model._parse_datasets_elevation(elevation_list, res=res / (2**ilev))
                for ilev in range(nlev)
            ]
        except Exception:
            logger.info("Using bathymetry database for cell-centre interpolation.")
            elevation_list_per_level = [None] * nlev

        n = self.data["n"] - 1  # 0-based
        m = self.data["m"] - 1

        def process_level(ilev):
            idx = level_indices[ilev]
            xz, yz = xy[idx, 0], xy[idx, 1]
            n_level, m_level = n[idx], m[idx]
            dxmin, dymin = dx / 2**ilev, dy / 2**ilev

            logger.info(f"Processing bathymetry level {ilev + 1} of {nlev} ...")

            x_min, x_max = xz.min() - dxmin, xz.max() + dxmin
            y_min, y_max = yz.min() - dymin, yz.max() + dymin
            x_chunks = np.arange(x_min, x_max, nrmax * dxmin)
            y_chunks = np.arange(y_min, y_max, nrmax * dymin)

            zgl = np.full(len(idx), np.nan)

            def process_chunk(ix, iy):
                x0c = x_chunks[ix]
                x1c = x_chunks[ix + 1] if ix < len(x_chunks) - 1 else x_max
                y0c = y_chunks[iy]
                y1c = y_chunks[iy + 1] if iy < len(y_chunks) - 1 else y_max

                in_chunk = np.where(
                    (xz >= x0c) & (xz < x1c) & (yz >= y0c) & (yz < y1c)
                )[0]
                if len(in_chunk) == 0:
                    return

                if bathymetry_database is not None:
                    zgl[in_chunk] = bathymetry_database.get_bathymetry_on_points(
                        xz[in_chunk],
                        yz[in_chunk],
                        min(dxmin, dymin),
                        self.model.crs,
                        elevation_list,
                    )
                else:
                    da_like = make_regular_grid(
                        x0=self.data.attrs["x0"],
                        y0=self.data.attrs["y0"],
                        dx=dxmin,
                        dy=dymin,
                        mmax=m_level[in_chunk].max().values + 1,
                        nmax=n_level[in_chunk].max().values + 1,
                        rotation=self.data.attrs["rotation"],
                        crs=self.model.crs,
                        mmin=m_level[in_chunk].min().values,
                        nmin=n_level[in_chunk].min().values,
                        make_ugrid=False,
                    )
                    da_dep = merge_multi_dataarrays(
                        da_list=elevation_list_per_level[ilev],
                        da_like=da_like,
                        buffer_cells=buffer_cells,
                        interp_method=interp_method,
                        logger=logger,
                    )
                    idx_y = np.searchsorted(da_dep.n.values, n_level[in_chunk].values)
                    idx_x = np.searchsorted(da_dep.m.values, m_level[in_chunk].values)
                    zgl[in_chunk] = da_dep.values[idx_y, idx_x]

            if len(x_chunks) > 1 or len(y_chunks) > 1:
                logger.info(
                    f"Processing in {len(x_chunks)} x {len(y_chunks)} chunks ..."
                )
                for ix in range(len(x_chunks)):
                    for iy in range(len(y_chunks)):
                        process_chunk(ix, iy)
            else:
                process_chunk(0, 0)

            return idx, np.clip(zgl, zmin, zmax)

        for ilev in range(nlev):
            idx, zgl_level = process_level(ilev)
            zz[idx] = zgl_level

        self.data["z"] = xu.UgridDataArray(
            xr.DataArray(data=zz, dims=[self.data.grid.face_dimension]),
            self.data.grid,
        )

        # ----------------------------------------------------------
        # Optional mean-wet sub-grid pass
        # ----------------------------------------------------------
        if mean_wet:
            self._create_mean_wet(
                elevation_list=elevation_list,
                bathymetry_database=bathymetry_database,
                nr_subgrid_pixels=nr_subgrid_pixels,
                threshold_level=threshold_level,
                nrmax=nrmax,
                zmin=zmin,
                zmax=zmax,
            )

    @hydromt_step
    def create_uniform(self, zb: float):
        """Set a uniform bed level across the entire grid."""
        self.data["z"][:] = zb

    def interpolate_bathymetry(self, x, y, z, method="linear"):
        """Interpolate scattered (x, y, z) points onto the quadtree grid."""
        xy = self.data.grid.face_coordinates
        xz, yz = xy[:, 0], xy[:, 1]
        zz = _interp2(x, y, z, xz, yz, method=method)
        ugrid2d = self.data.grid
        self.data["z"] = xu.UgridDataArray(
            xr.DataArray(data=zz, dims=[ugrid2d.face_dimension]), ugrid2d
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _create_mean_wet(
        self,
        elevation_list: list,
        bathymetry_database: Optional[object],
        nr_subgrid_pixels: int,
        threshold_level: float,
        nrmax: int,
        zmin: float,
        zmax: float,
    ) -> None:
        """Replace each cell's bed level with the mean of its wet sub-grid pixels.

        A sub-grid pixel is considered *wet* when its elevation is at or below
        ``threshold_level``.  If no wet pixels exist in a cell, the mean of
        *all* sub-grid pixels is used instead (matching cht_hurrywave behaviour).

        This method updates ``self.data["z"]`` in place.  It is called
        automatically from :meth:`create` when ``mean_wet=True``.

        Parameters
        ----------
        elevation_list : list
            Same elevation list passed to :meth:`create`.
        bathymetry_database : object or None
            cht_bathymetry database (or ``None`` for the HydroMT path).
        nr_subgrid_pixels : int
            Sub-grid pixels per cell edge.
        threshold_level : float
            Wet/dry threshold [m].
        nrmax : int
            Maximum number of cells per block.
        zmin, zmax : float
            Clip range applied to the computed mean-wet values.
        """
        logger.info("Computing mean-wet sub-grid elevation ...")

        grid = self.data
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
        n = grid["n"].values[:] - 1
        m = grid["m"].values[:] - 1

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

        # Pre-parse HydroMT elevation datasets at sub-grid resolution
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

        # Work on a mutable copy of the current z values
        zz = grid["z"].values.copy()

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

            n0 = int(n[i0: i1 + 1].min())
            n1 = int(n[i0: i1 + 1].max())
            m0 = int(m[i0: i1 + 1].min())
            m1 = int(m[i0: i1 + 1].max())

            nrcb = max(1, int(np.floor(nrmax / refi)))
            nrbn = int(np.ceil((n1 - n0 + 1) / nrcb))
            nrbm = int(np.ceil((m1 - m0 + 1) / nrcb))

            logger.info(
                f"Mean-wet: level {ilev + 1}/{nr_ref_levs}, "
                f"{nrbm * nrbn} block(s) ..."
            )

            for ii in range(nrbm):
                for jj in range(nrbn):
                    bn0 = n0 + jj * nrcb
                    bn1 = min(bn0 + nrcb - 1, n1) + 1
                    bm0 = m0 + ii * nrcb
                    bm1 = min(bm0 + nrcb - 1, m1) + 1

                    if bn1 <= bn0 or bm1 <= bm0:
                        continue

                    # Sub-grid pixel coordinates in the rotated local frame
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

                    # Fetch sub-grid bathymetry
                    if bathymetry_database is not None:
                        try:
                            zg = bathymetry_database.get_bathymetry_on_grid(
                                xg, yg, self.model.crs, elevation_list
                            )
                        except Exception as exc:
                            logger.error(
                                f"Mean-wet: error fetching bathymetry "
                                f"(ilev={ilev}, ii={ii}, jj={jj}): {exc}"
                            )
                            continue
                    else:
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
                            zg = da_dep.values  # shape (nmax_sub, mmax_sub)
                        except Exception as exc:
                            logger.error(
                                f"Mean-wet: error merging datasets "
                                f"(ilev={ilev}, ii={ii}, jj={jj}): {exc}"
                            )
                            continue

                    # Find cells in this block
                    cells_in_block = [
                        cell_indices_in_level[ic]
                        for ic in range(nr_cells_in_level)
                        if (
                            n[cell_indices_in_level[ic]] >= bn0
                            and n[cell_indices_in_level[ic]] < bn1
                            and m[cell_indices_in_level[ic]] >= bm0
                            and m[cell_indices_in_level[ic]] < bm1
                        )
                    ]

                    for idx in cells_in_block:
                        nn = (n[idx] - bn0) * refi
                        mm = (m[idx] - bm0) * refi
                        zgc = zg[nn: nn + refi, mm: mm + refi]

                        wet = zgc[zgc <= threshold_level]
                        if wet.size > 0:
                            zbmean = float(np.nanmean(wet))
                        else:
                            # No wet pixels — fall back to mean of all pixels
                            zbmean = float(np.nanmean(zgc))

                        zz[idx] = float(np.clip(zbmean, zmin, zmax))

        # Write updated values back to the grid dataset
        self.data["z"].values[:] = zz
        logger.info("Mean-wet sub-grid elevation done.")


    # ------------------------------------------------------------------
    # Datashader overlay
    # ------------------------------------------------------------------

    def get_datashader_dataframe(self):
        """Build a trimesh DataFrame for datashader rendering.

        Creates vertices (4 corners per cell) and simplices (2 triangles
        per cell) in EPSG:3857 for use with ``datashader.Canvas.trimesh()``.
        The vertex DataFrame has columns ``x, y, z`` and the simplices
        DataFrame has columns ``v0, v1, v2``.

        Stores results in ``self.datashader_vertices`` and
        ``self.datashader_simplices``.
        """
        if self.data is None or "z" not in self.data:
            self.datashader_vertices = pd.DataFrame()
            self.datashader_simplices = pd.DataFrame()
            return

        grid = self.data
        xy = grid.grid.face_coordinates
        z = grid["z"].values[:]
        level = grid["level"].values[:] - 1  # 0-based
        dx0 = grid.attrs["dx"]
        dy0 = grid.attrs["dy"]
        rotation = grid.attrs["rotation"]
        cosrot = np.cos(np.radians(rotation))
        sinrot = np.sin(np.radians(rotation))

        # Filter out NaN cells
        valid = np.isfinite(z)
        n_valid = np.sum(valid)
        if n_valid == 0:
            self.datashader_vertices = pd.DataFrame()
            self.datashader_simplices = pd.DataFrame()
            return

        cx = xy[valid, 0]
        cy = xy[valid, 1]
        cz = z[valid]
        clevel = level[valid]

        # Half-widths per cell
        hdx = (dx0 / 2.0 ** clevel) / 2.0
        hdy = (dy0 / 2.0 ** clevel) / 2.0

        # 4 corners per cell in local (rotated) frame:
        #   0=(-hdx,-hdy), 1=(+hdx,-hdy), 2=(+hdx,+hdy), 3=(-hdx,+hdy)
        dx_offsets = np.array([-1, 1, 1, -1], dtype=np.float64)
        dy_offsets = np.array([-1, -1, 1, 1], dtype=np.float64)

        # Broadcast: (n_valid, 4)
        local_x = hdx[:, None] * dx_offsets[None, :]
        local_y = hdy[:, None] * dy_offsets[None, :]

        # Rotate and translate to world coordinates
        vx = cx[:, None] + cosrot * local_x - sinrot * local_y
        vy = cy[:, None] + sinrot * local_x + cosrot * local_y

        # Flatten to vertex arrays
        vx_flat = vx.ravel()
        vy_flat = vy.ravel()
        vz_flat = np.repeat(cz, 4)  # same z for all 4 corners

        # Transform to EPSG:3857
        transformer = Transformer.from_crs(self.model.crs, 3857, always_xy=True)
        vx_3857, vy_3857 = transformer.transform(vx_flat, vy_flat)

        # Handle dateline crossing
        if self.model.crs.is_geographic and np.max(xy[:, 0]) > 180.0:
            vx_3857[vx_3857 < 0] += 40075016.68557849

        # Build vertex DataFrame
        self.datashader_vertices = pd.DataFrame({
            "x": vx_3857,
            "y": vy_3857,
            "z": vz_flat,
        })

        # Build simplices: 2 triangles per cell
        # Cell i has vertices at indices 4*i, 4*i+1, 4*i+2, 4*i+3
        cell_idx = np.arange(n_valid, dtype=np.int32)
        v_base = cell_idx * 4
        # Triangle 1: corners 0, 1, 2
        # Triangle 2: corners 0, 2, 3
        t0 = np.column_stack([v_base, v_base + 1, v_base + 2])
        t1 = np.column_stack([v_base, v_base + 2, v_base + 3])
        tris = np.vstack([t0, t1])

        self.datashader_simplices = pd.DataFrame({
            "v0": tris[:, 0],
            "v1": tris[:, 1],
            "v2": tris[:, 2],
        })

    def clear_datashader_dataframe(self):
        """Clear cached datashader data."""
        self.datashader_vertices = pd.DataFrame()
        self.datashader_simplices = pd.DataFrame()

    def map_overlay(
        self,
        file_name,
        xlim=None,
        ylim=None,
        cmap="gist_earth",
        cmin=None,
        cmax=None,
        width=800,
        **kwargs,
    ):
        """Create a map overlay image of the bathymetry using datashader.

        Parameters
        ----------
        file_name : str
            Output image file name (without extension — .png is appended).
        xlim : list
            [lon_min, lon_max] in WGS84 degrees.
        ylim : list
            [lat_min, lat_max] in WGS84 degrees.
        cmap : str or list
            Matplotlib colormap name or list of hex colors.
        cmin, cmax : float, optional
            Color scale range. If None, auto-scaled from data.
        width : int
            Output image width in pixels.

        Returns
        -------
        bool
            True if the image was created successfully.
        """
        if not HAS_DATASHADER:
            logger.warning("datashader is not installed.")
            return False

        if self.data is None or "z" not in self.data:
            return False

        try:
            # Build trimesh if not cached
            if not hasattr(self, "datashader_vertices") or self.datashader_vertices.empty:
                self.get_datashader_dataframe()

            if self.datashader_vertices.empty:
                return False

            # Transform view bounds to EPSG:3857
            transformer = Transformer.from_crs(4326, 3857, always_xy=True)
            xl0, yl0 = transformer.transform(xlim[0], ylim[0])
            xl1, yl1 = transformer.transform(xlim[1], ylim[1])
            if xl0 > xl1:
                xl1 += 40075016.68557849
            x_range = (xl0, xl1)
            y_range = (yl0, yl1)

            ratio = (y_range[1] - y_range[0]) / (x_range[1] - x_range[0])
            height = int(width * ratio)

            cvs = ds.Canvas(
                x_range=x_range, y_range=y_range,
                plot_width=width, plot_height=height,
            )

            # Pre-compute the mesh for datashader
            mesh = ds.utils.mesh(
                self.datashader_vertices,
                self.datashader_simplices,
            )

            # Render trimesh with flat shading (interp=False)
            agg = cvs.trimesh(
                self.datashader_vertices,
                self.datashader_simplices,
                mesh=mesh,
                agg=ds.mean("z"),
                interp=False,
            )

            # Resolve colormap: datashader needs a list of hex colors or a
            # matplotlib Colormap object, not a plain string name.
            if isinstance(cmap, str):
                from matplotlib import colormaps
                cmap = colormaps[cmap]

            span = None
            if cmin is not None and cmax is not None:
                span = (cmin, cmax)

            img = tf.shade(agg, cmap=cmap, span=span, how="linear")

            # Export
            path = os.path.dirname(file_name)
            if not path:
                path = os.getcwd()
            name = os.path.splitext(os.path.basename(file_name))[0]
            export_image(img, name, export_path=path)
            return True

        except Exception as e:
            logger.warning(f"map_overlay failed: {e}")
            return False


def _interp2(x0, y0, z0, x1, y1, method="linear"):
    f = RegularGridInterpolator(
        (y0, x0), z0, bounds_error=False, fill_value=np.nan, method=method
    )
    if x1.ndim > 1:
        sz = x1.shape
        x1 = x1.reshape(sz[0] * sz[1])
        y1 = y1.reshape(sz[0] * sz[1])
        return f((y1, x1)).reshape(sz)
    return f((y1, x1))

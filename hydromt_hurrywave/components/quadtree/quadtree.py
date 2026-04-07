import logging
import os
from os.path import isfile
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
import shapely

import xarray as xr
import xugrid as xu

from hydromt import hydromt_step
from hydromt.model.components import MeshComponent
from hydromt.model.processes.grid import create_grid_from_region

from hydromt_sfincs.utils import make_regular_grid
from .quadtree_builder import build_quadtree_xugrid, cut_inactive_cells

# optional dependency
try:
    import datashader.transfer_functions as tf
    from datashader import Canvas
    from datashader.utils import export_image

    HAS_DATASHADER = True
except ImportError:
    HAS_DATASHADER = False

if TYPE_CHECKING:
    from hydromt_hurrywave import HurrywaveModel

logger = logging.getLogger(f"hydromt.{__name__}")

# Variables stored in separate nc files (outside the main qtr nc).
# For HurryWave these are wave-blocking coefficients handled by their own component.
_QT_MAPS: List[str] = []


class HurrywaveQuadtreeGrid(MeshComponent):
    def __init__(
        self,
        model: "HurrywaveModel",
    ):
        self._filename: str = "hurrywave.nc"
        self._data: xu.UgridDataset = None
        self.version = 0
        self.datashader_dataframe = pd.DataFrame()

        super().__init__(
            model=model,
        )

    # NOTE @data and @initialize are inherited from MeshComponent

    @property
    def crs(self) -> CRS:
        """Return the CRS of the quadtree grid."""
        if self.data.grid.crs is not None:
            return self.data.grid.crs
        raise ValueError("No CRS defined for the quadtree grid.")

    @property
    def face_coordinates(self):
        if self.data is None:
            return None
        xy = self.data.grid.face_coordinates
        return xy[:, 0], xy[:, 1]

    @property
    def exterior(self):
        if self.data is None:
            return gpd.GeoDataFrame()
        indx = self.data.grid.edge_node_connectivity[self.data.grid.exterior_edges, :]
        x = self.data.grid.node_x[indx]
        y = self.data.grid.node_y[indx]

        linestrings = [
            shapely.LineString(np.column_stack((x[i], y[i]))) for i in range(len(x))
        ]
        merged = shapely.ops.linemerge(linestrings)
        polygons = shapely.ops.polygonize(merged)
        return gpd.GeoDataFrame(geometry=list(polygons), crs=self.crs)

    @property
    def empty_mask(self):
        if self.data is None:
            return None
        da0 = xr.DataArray(
            data=np.zeros(shape=len(self.data.grid.face_coordinates)),
            dims=self.data.grid.face_dimension,
        )
        return xu.UgridDataArray(da0, self.data.grid)

    @property
    def mask(self) -> xu.UgridDataArray:
        """Return the mask stored in the quadtree dataset."""
        if "mask" in self.data:
            return self.data["mask"]
        return self.empty_mask

    def read(
        self, filename: Union[str, Path] = "hurrywave.nc", data_vars: List[str] = None
    ):
        """Read the HurryWave quadtree netCDF file.

        Parameters
        ----------
        filename : str or Path, optional
            Filename to read, by default ``"hurrywave.nc"``.
        data_vars : list of str, optional
            Additional variables to load from separate nc files.
        """
        self.root._assert_read_mode()

        abs_file_path = self.model.config.get_set_file_variable(
            "qtrfile", value=filename
        )
        if abs_file_path is None:
            return

        if not abs_file_path.exists():
            raise FileNotFoundError(f"Quadtree grid file not found: {abs_file_path}")

        with xu.load_dataset(abs_file_path) as ds:
            ds.grid.set_crs(CRS.from_wkt(ds["crs"].crs_wkt))
            # backwards-compat rename
            ds = ds.rename({"msk": "mask"}) if "msk" in ds else ds

            self.nr_cells = ds.sizes["mesh2d_nFaces"]
            for key, value in ds.attrs.items():
                setattr(self, key, value)
            self._data = ds

        # keep crs_epsg in config in sync
        self.model.config.set("crs_epsg", self.model.crs.to_epsg())

        # load any extra variables stored in separate nc files
        if data_vars is None:
            data_vars = _QT_MAPS
        elif isinstance(data_vars, str):
            data_vars = [data_vars]

        for var in data_vars:
            fn_var = self.model.config.get(
                f"{var}file", fallback=f"{var}.nc", abs_path=True
            )
            if isfile(fn_var):
                try:
                    with xu.load_dataset(fn_var) as ds_var:
                        self._data[var] = ds_var[var]
                except Exception as e:
                    logger.error(f"Error reading variable {var}: {e}")

    def write(
        self, filename: Union[str, Path] = "hurrywave.nc", data_vars: List[str] = None
    ):
        """Write the HurryWave quadtree netCDF file.

        Parameters
        ----------
        filename : str or Path, optional
            Output filename, by default ``"hurrywave.nc"``.
        data_vars : list of str, optional
            Variables to write to separate nc files.
        """
        attrs = self.data.attrs
        ds = self.data.ugrid.to_dataset()
        ds["crs"] = self.crs.to_epsg()
        ds["crs"].attrs = self.crs.to_cf()

        if data_vars is None:
            data_vars = _QT_MAPS
        elif isinstance(data_vars, str):
            data_vars = [data_vars]

        for var in data_vars:
            fn_var = self.model.config.get(f"{var}file", abs_path=True)
            if fn_var is not None:
                fn_var.parent.mkdir(parents=True, exist_ok=True)
                try:
                    ds_var = self.data[
                        [var, "mesh2d_node_x", "mesh2d_node_y"]
                    ].ugrid.to_dataset()
                    ds_var.to_netcdf(fn_var)
                    ds = ds.drop_vars(var)
                except Exception as e:
                    logger.error(f"Error writing variable {var}: {e}")

        abs_file_path = self.model.config.get_set_file_variable(
            "qtrfile", value=filename, default="hurrywave.nc"
        )
        abs_file_path.parent.mkdir(parents=True, exist_ok=True)

        self.model.config.set("crs_epsg", self.model.crs.to_epsg())

        ds.attrs = attrs
        ds.to_netcdf(abs_file_path)
        ds.close()

    @hydromt_step
    def create(
        self,
        x0: float,
        y0: float,
        nmax: int,
        mmax: int,
        dx: float,
        dy: float,
        rotation: float,
        epsg: int,
        refinement_polygons: Optional[gpd.GeoDataFrame] = None,
        elevation_list: List[List[dict]] = None,
        bathymetry_database: Optional[object] = None,
    ):
        """Build the HurryWave quadtree grid.

        Parameters
        ----------
        x0, y0 : float
            Lower-left corner coordinates.
        nmax, mmax : int
            Grid dimensions at the coarsest level.
        dx, dy : float
            Cell size (positive, metres or degrees).
        rotation : float
            Rotation angle in degrees.
        epsg : int
            EPSG code of the CRS.
        refinement_polygons : gpd.GeoDataFrame, optional
            Polygons with a ``refinement_level`` column.
        elevation_list : list of list of dict, optional
            Bathymetry datasets per refinement level.
        bathymetry_database : object, optional
            cht_bathymetry database object.
        """
        self.clear_datashader_dataframe()
        self.model.quadtree_mask.clear_datashader_dataframe()

        self.model.grid_type = "quadtree"
        crs = CRS.from_epsg(epsg)

        elevation_list_per_level = []
        if elevation_list is not None and bathymetry_database is None:
            self._data = make_regular_grid(
                x0, y0, dx, dy, mmax, nmax, rotation=rotation, crs=crs, make_ugrid=True
            )
            res = dx
            if refinement_polygons is not None:
                nlev = max(refinement_polygons["refinement_level"].unique())
            else:
                nlev = 1
            if crs.is_geographic:
                res = res * 111111.0
            for lev in range(nlev):
                res_level = res / (2**lev)
                elevation_list_per_level.append(
                    self.model._parse_datasets_elevation(elevation_list, res=res_level)
                )
            elevation_list = elevation_list_per_level

        self._data = build_quadtree_xugrid(
            x0, y0, nmax, mmax, dx, dy, rotation, crs,
            refinement_polygons=refinement_polygons,
            elevation_list=elevation_list,
            bathymetry_database=bathymetry_database,
        )

        self.model.config.set("crs_epsg", self.model.crs.to_epsg())

    @hydromt_step
    def create_from_region(
        self,
        region: dict,
        res: float = 100,
        crs: Union[str, int] = "utm",
        rotated: bool = False,
        hydrography_fn: str = None,
        basin_index_fn: str = None,
        align: bool = True,
        dec_origin: int = 0,
        dec_rotation: int = 3,
        refinement_polygons: Optional[gpd.GeoDataFrame] = None,
        elevation_list: List[List[dict]] = None,
    ):
        """Setup a quadtree grid from a region description.

        Parameters
        ----------
        region : dict
            Region description, e.g. ``{'bbox': [xmin, ymin, xmax, ymax]}``
            (WGS84 coordinates).
        res : float, optional
            Grid resolution in metres, by default 100.
        crs : str or int, optional
            ``"utm"`` auto-selects best UTM zone, by default ``"utm"``.
        rotated : bool, optional
            Fit minimum rotated rectangle, by default False.
        refinement_polygons : gpd.GeoDataFrame, optional
            Refinement polygons.
        elevation_list : list of list of dict, optional
            Bathymetry datasets per level.
        """
        ds = create_grid_from_region(
            region=region,
            data_catalog=self.model.data_catalog,
            res=res,
            crs=crs,
            region_crs=4326,
            rotated=rotated,
            hydrography_path=hydrography_fn,
            basin_index_path=basin_index_fn,
            add_mask=False,
            align=align,
            dec_origin=dec_origin,
            dec_rotation=dec_rotation,
        )

        if ds.raster.res[1] < 0:
            ds = ds.raster.flipud()

        nmax, mmax = ds.raster.shape
        dx, dy = ds.raster.res
        x0, y0 = ds.raster.origin
        rotation = ds.raster.rotation
        epsg = ds.raster.crs.to_epsg()

        self.create(
            x0=x0, y0=y0, nmax=nmax, mmax=mmax,
            dx=dx, dy=dy, rotation=rotation, epsg=epsg,
            refinement_polygons=refinement_polygons,
            elevation_list=elevation_list,
        )

    def cut_inactive_cells(self):
        """Remove inactive cells (mask == 0) from the grid dataset."""
        self.clear_datashader_dataframe()
        self.model.quadtree_mask.clear_datashader_dataframe()
        self._data = cut_inactive_cells(self.data)

    def snap_to_grid(self, polyline):
        """Snap a polyline GeoDataFrame to the nearest grid edges."""
        if len(polyline) == 0:
            return gpd.GeoDataFrame()
        max_snap_distance = 1.0e-6 if self.model.crs.is_geographic else 0.1
        geom_list = [
            line["geometry"]
            for _, line in polyline.iterrows()
            if line["geometry"].geom_type == "LineString"
        ]
        gdf = gpd.GeoDataFrame({"geometry": geom_list})
        _, snapped_gdf = xu.snap_to_grid(
            gdf, self.data.grid, max_snap_distance=max_snap_distance
        )
        return snapped_gdf.set_crs(self.crs)

    def map_overlay(self, file_name, xlim=None, ylim=None, color="black", width=800):
        """Create a PNG map overlay of the grid using datashader.

        Returns ``True`` on success, ``False`` if datashader is unavailable or
        the grid is empty.
        """
        if not HAS_DATASHADER:
            logger.warning("Datashader is not available. Please install datashader.")
            return False

        if self.data is None:
            return False

        try:
            if self.datashader_dataframe.empty:
                self.get_datashader_dataframe()

            transformer = Transformer.from_crs(4326, 3857, always_xy=True)
            xl0, yl0 = transformer.transform(xlim[0], ylim[0])
            xl1, yl1 = transformer.transform(xlim[1], ylim[1])
            if xl0 > xl1:
                xl1 += 40075016.68557849
            xlim = [xl0, xl1]
            ylim = [yl0, yl1]
            ratio = (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])
            height = int(width * ratio)
            cvs = Canvas(
                x_range=xlim, y_range=ylim, plot_height=height, plot_width=width
            )
            agg = cvs.line(
                self.datashader_dataframe, x=["x1", "x2"], y=["y1", "y2"], axis=1
            )
            img = tf.shade(agg, cmap=color)
            path_dir = os.path.dirname(file_name) or os.getcwd()
            name = os.path.splitext(os.path.basename(file_name))[0]
            export_image(img, name, export_path=path_dir)
            return True
        except Exception:
            return False

    def get_indices_at_points(self, x, y):
        """Return 2-D index array of quadtree cell indices at the given points."""
        if np.ndim(x) == 0:
            x = np.array([[x]])
        if np.ndim(y) == 0:
            y = np.array([[y]])

        x0 = self.data.attrs["x0"]
        y0 = self.data.attrs["y0"]
        dx = self.data.attrs["dx"]
        dy = self.data.attrs["dy"]
        rotation = self.data.attrs["rotation"]
        nr_refinement_levels = self.data.attrs["nr_levels"]
        nr_cells = len(self.data["level"])

        cosrot = np.cos(-rotation * np.pi / 180)
        sinrot = np.sin(-rotation * np.pi / 180)
        x00 = x - x0
        y00 = y - y0
        xg = x00 * cosrot - y00 * sinrot
        yg = x00 * sinrot + y00 * cosrot

        if not hasattr(self, "ifirst"):
            ifirst = np.zeros(nr_refinement_levels, dtype=int)
            for ilev in range(nr_refinement_levels):
                ifirst[ilev] = np.where(
                    self.data["level"].to_numpy()[:] == ilev + 1
                )[0][0]
            self.ifirst = ifirst

        ifirst = self.ifirst
        i0_lev, i1_lev, nmax_lev, mmax_lev, nm_lev = [], [], [], [], []

        for level in range(nr_refinement_levels):
            i0 = ifirst[level]
            i1 = ifirst[level + 1] if level < nr_refinement_levels - 1 else nr_cells
            i0_lev.append(i0)
            i1_lev.append(i1)
            nmax_lev.append(np.amax(self.data["n"].to_numpy()[i0:i1]) + 1)
            mmax_lev.append(np.amax(self.data["m"].to_numpy()[i0:i1]) + 1)
            nn = self.data["n"].to_numpy()[i0:i1] - 1
            mm = self.data["m"].to_numpy()[i0:i1] - 1
            nm_lev.append(mm * nmax_lev[level] + nn)

        indx = np.full(np.shape(x), -999, dtype=np.int32)

        for ilev in range(nr_refinement_levels):
            nmax_l = nmax_lev[ilev]
            mmax_l = mmax_lev[ilev]
            i0 = i0_lev[ilev]
            dxr = dx / 2**ilev
            dyr = dy / 2**ilev
            iind = np.floor(xg / dxr).astype(int)
            jind = np.floor(yg / dyr).astype(int)
            ind = iind * nmax_l + jind
            ind[iind < 0] = -999
            ind[jind < 0] = -999
            ind[iind >= mmax_l] = -999
            ind[jind >= nmax_l] = -999

            ingrid = np.isin(ind, nm_lev[ilev], assume_unique=False)
            incell = np.where(ingrid)
            if incell[0].size > 0:
                try:
                    cell_indices = (
                        _binary_search(nm_lev[ilev], ind[incell[0], incell[1]])
                        + i0
                    )
                    indx[incell[0], incell[1]] = cell_indices
                except Exception as e:
                    logger.warning(f"Error in binary search: {e}")

        return indx

    # ------------------------------------------------------------------ internal
    def get_datashader_dataframe(self):
        """Build an edge-coordinate dataframe for datashader rendering."""
        x1 = self.data.grid.edge_node_coordinates[:, 0, 0]
        x2 = self.data.grid.edge_node_coordinates[:, 1, 0]
        y1 = self.data.grid.edge_node_coordinates[:, 0, 1]
        y2 = self.data.grid.edge_node_coordinates[:, 1, 1]

        cross_dateline = False
        if self.model.crs.is_geographic:
            if np.max(x1) > 180.0 or np.max(x2) > 180.0:
                cross_dateline = True

        transformer = Transformer.from_crs(self.model.crs, 3857, always_xy=True)
        x1, y1 = transformer.transform(x1, y1)
        x2, y2 = transformer.transform(x2, y2)
        if cross_dateline:
            x1[x1 < 0] += 40075016.68557849
            x2[x2 < 0] += 40075016.68557849
        self.datashader_dataframe = pd.DataFrame(dict(x1=x1, y1=y1, x2=x2, y2=y2))

    def clear_datashader_dataframe(self):
        self.datashader_dataframe = pd.DataFrame()


def _binary_search(val_array, vals):
    """Binary search returning cell indices; -1 for out-of-bounds."""
    indx = np.searchsorted(val_array, vals)
    not_ok = np.where(indx == len(val_array))[0]
    indx[not_ok] = 0
    is_ok = np.where(val_array[indx] == vals)[0]
    indices = np.zeros(len(vals), dtype=int) - 1
    indices[is_ok] = indx[is_ok]
    indices[not_ok] = -1
    return indices

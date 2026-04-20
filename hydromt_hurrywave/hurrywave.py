"""HurrywaveModel – HydroMT plugin for the HurryWave spectral wave model."""

from __future__ import annotations

import logging
from typing import List, Union

import geopandas as gpd
from pyproj import CRS

from hydromt.model import Model

from hydromt_hurrywave.components.config import HurrywaveConfig
from hydromt_hurrywave.components.quadtree import (
    HurrywaveQuadtreeGrid,
    HurrywaveQuadtreeElevation,
    HurrywaveQuadtreeMask,
    HurrywaveWaveBlocking,
)
from hydromt_hurrywave.components.forcing import (
    HurrywaveBoundaryConditions,
    HurrywaveWind,
)
from hydromt_hurrywave.components.geometries import (
    HurrywaveObservationPoints,
    HurrywaveObservationPointsSpectra,
)

__all__ = ["HurrywaveModel"]
__hydromt_eps__ = ["HurrywaveModel"]
logger = logging.getLogger(f"hydromt.{__name__}")


class HurrywaveModel(Model):
    """HydroMT model class for HurryWave.

    HurryWave is a spectral wave model that uses an adaptive quadtree mesh.
    This class follows the same component-based pattern as
    :class:`hydromt_sfincs.SfincsModel`.

    HurryWave is quadtree-only, so ``grid_type`` is always ``"quadtree"``.

    Components
    ----------
    config
        ``hurrywave.inp`` via :class:`~hydromt_hurrywave.components.config.HurrywaveConfig`.
    quadtree_grid
        Quadtree mesh + mask + elevation stored as a single :class:`xugrid.UgridDataset`.
    quadtree_elevation
        Bed elevation delegate (data lives in ``quadtree_grid``).
    quadtree_mask
        Cell-activity mask delegate (data lives in ``quadtree_grid``):
        0 = inactive, 1 = active, 2 = open boundary.
    wave_blocking
        Directional wave-blocking coefficients (separate netCDF).
    boundary_conditions
        Wave boundary conditions: *timeseries* (Hs/Tp/Wd/Ds) or *spectra* (2-D).
    wind
        Wind forcing: uniform time series or gridded u/v netCDF.
    observation_points
        Regular (bulk-parameter) output locations.
    observation_points_spectra
        Spectral output locations.
    """

    name: str = "hurrywave"

    # HurryWave is always quadtree – no regular grid branch
    grid_type: str = "quadtree"

    _CONFIG_COMPONENTS = {"config": HurrywaveConfig}

    _QUADTREE_COMPONENTS = {
        "quadtree_grid": HurrywaveQuadtreeGrid,
        "quadtree_elevation": HurrywaveQuadtreeElevation,
        "quadtree_mask": HurrywaveQuadtreeMask,
        "wave_blocking": HurrywaveWaveBlocking,
    }

    _GEOMETRY_COMPONENTS = {
        "observation_points": HurrywaveObservationPoints,
        "observation_points_spectra": HurrywaveObservationPointsSpectra,
    }

    _FORCING_COMPONENTS = {
        "boundary_conditions": HurrywaveBoundaryConditions,
        "wind": HurrywaveWind,
    }

    _ALL_COMPONENTS = {
        **_CONFIG_COMPONENTS,
        **_QUADTREE_COMPONENTS,
        **_GEOMETRY_COMPONENTS,
        **_FORCING_COMPONENTS,
    }

    _QUADTREE_GRID_NAMES = set(_QUADTREE_COMPONENTS.keys())

    def __init__(
        self,
        root: str = None,
        mode: str = "w",
        write_gis: bool = True,
        data_libs: Union[List[str], str] = None,
        **catalog_keys,
    ):
        """Initialise a HurrywaveModel.

        Parameters
        ----------
        root : str or Path, optional
            Path to the model directory.
        mode : {'w', 'r+', 'r'}
            Open in write, append, or read mode.  Default ``'w'``.
        data_libs : list of str or str, optional
            HydroMT data catalog YAML files.
        **catalog_keys
            Additional keyword arguments forwarded to the DataCatalog.
        """
        # HurryWave only supports quadtree grids
        self.grid_type = "quadtree"

        super().__init__(
            root=root,
            mode=mode,
            data_libs=data_libs,
            **catalog_keys,
        )

        for name, cls in self._ALL_COMPONENTS.items():
            self.add_component(name, cls(self))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def crs(self) -> CRS | None:
        """CRS of the model (from the quadtree grid)."""
        try:
            return self.quadtree_grid.crs
        except (AttributeError, ValueError):
            return None

    @property
    def region(self) -> gpd.GeoDataFrame:
        """Exterior boundary polygon of the active grid cells."""
        return self.quadtree_grid.exterior

    @property
    def bounds(self):
        """Total bounds of the grid: (xmin, ymin, xmax, ymax)."""
        if self.quadtree_grid.data is None:
            return None
        return self.quadtree_grid.empty_mask.ugrid.total_bounds

    @property
    def bbox(self):
        """Bounds reprojected to WGS 84."""
        if self.quadtree_grid.data is None:
            return None
        return self.quadtree_grid.empty_mask.ugrid.to_crs(4326).ugrid.total_bounds

    # Convenience aliases matching cht_hurrywave attribute names
    @property
    def grid(self):
        """Alias for ``quadtree_grid``."""
        return self.quadtree_grid

    @property
    def mask(self):
        """Alias for ``quadtree_mask``."""
        return self.quadtree_mask

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def read(self) -> None:
        """Read the full HurryWave model from *root*.

        Config is always read first (it provides file paths), then all
        other components are read in registration order.
        """
        self.config.read()

        for name, comp in self.components.items():
            if name == "config":
                continue
            try:
                comp.read()
            except Exception as exc:
                logger.warning(f"Could not read component '{name}': {exc}")

    def write(self) -> None:
        """Write the full HurryWave model to *root*.

        Individual component ``write()`` calls may register file names in the
        config, so ``config`` is always written last.
        """
        for name, comp in self.components.items():
            if name == "config":
                continue
            try:
                comp.write()
            except Exception as exc:
                logger.warning(f"Could not write component '{name}': {exc}")

        self.config.write()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_model_time(self):
        """Return ``(tstart, tstop)`` as :class:`pandas.Timestamp` objects."""
        import pandas as pd

        tstart = pd.Timestamp(self.config.get("tstart"))
        tstop = pd.Timestamp(self.config.get("tstop"))
        return tstart, tstop
    
    def _parse_datasets_elevation(self, elevation_list, res):
        """Parse filenames or paths of Datasets in list of dictionaries elevation_list
        into xr.DataArray and gdf.GeoDataFrames:

        * "elevation" is parsed into da (xr.DataArray)
        * "offset" is parsed into da_offset (xr.DataArray)
        * "mask" is parsed into gdf (gpd.GeoDataFrame)

        Parameters
        ----------
        elevation_list : List[dict]
            List of dictionaries with topography and bathymetry data, each containing a dataset name or
            Path (dep) and optional merge arguments.
        res : float
            Resolution of the model grid in meters. Used to obtain the correct zoom
            level of the depth datasets.
        """
        parse_keys = ["elevation", "offset", "mask", "da"]
        copy_keys = ["zmin", "zmax", "reproj_method", "merge_method", "offset"]

        datasets_out = []
        for dataset in elevation_list:
            dd = {}
            # read in depth datasets; replace dep (source name; filename or xr.DataArray)
            if "elevation" in dataset or "da" in dataset:
                try:
                    data_name = dataset.get("elevation", dataset.get("da"))
                    da_elv = self.data_catalog.get_rasterdataset(
                        data_name,
                        bbox=self.bbox,
                        buffer=10,
                        variables=["elevtn"],  # NOTE this is still hydromt convention
                        zoom=(res, "meter"),
                    )
                    # rename elevtn to elevation if present
                    da_elv.name = "elevation"
                    actual_res = da_elv.raster.res
                    logger.debug(
                        f"Dataset '{data_name}': requested zoom {res:.1f} m, "
                        f"actual resolution {actual_res[0]:.6g} x {actual_res[1]:.6g} "
                        f"(CRS units), shape {da_elv.shape}"
                    )
                # TODO remove ValueError after fix in hydromt core
                except (IndexError, ValueError):
                    data_name = dataset.get("elevation")
                    logger.warning(f"No data in domain for {data_name}, skipped.")
                    continue
                dd.update({"da": da_elv})
            else:
                raise ValueError(
                    "No 'elevation' (topobathy) dataset provided in elevation_list."
                )

            # read offset filenames
            # NOTE offsets can be xr.DataArrays and floats
            if "offset" in dataset and not isinstance(dataset["offset"], (float, int)):
                da_offset = self.data_catalog.get_rasterdataset(
                    dataset.get("offset"),
                    bbox=self.bbox,
                    buffer=10,
                )
                dd.update({"offset": da_offset})

            # read geodataframes describing valid areas
            if "mask" in dataset:
                gdf_valid = self.data_catalog.get_geodataframe(
                    dataset.get("mask"),
                    bbox=self.bbox,
                )
                dd.update({"gdf_valid": gdf_valid})

            # copy remaining keys
            for key, value in dataset.items():
                if key in copy_keys and key not in dd:
                    dd.update({key: value})
                elif key not in copy_keys + parse_keys:
                    logger.warning(f"Unknown key {key} in elevation_list. Ignoring.")
            datasets_out.append(dd)

        return datasets_out

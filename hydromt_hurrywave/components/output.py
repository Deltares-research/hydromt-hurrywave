"""HurryWave model output component.

Reads ``hurrywave_map.nc`` (quadtree map) and ``hurrywave_his.nc``
(station history) into a dict of xarray / xugrid datasets, mirroring
the :class:`hydromt_sfincs.components.output.SfincsOutput` API.
"""

import logging
from os.path import isabs, isfile
from typing import TYPE_CHECKING, Dict, Optional, Union

import xarray as xr
import xugrid as xu
from pyproj import CRS

from hydromt.model.components import ModelComponent

if TYPE_CHECKING:
    from hydromt_hurrywave import HurrywaveModel

logger = logging.getLogger(f"hydromt.{__name__}")


class HurrywaveOutput(ModelComponent):
    """HurryWave model output component.

    Handles reading of model results from ``hurrywave_map.nc`` (quadtree
    map) and ``hurrywave_his.nc`` (station time-series). Results are
    exposed as a dict of :class:`xarray.DataArray` (or
    :class:`xugrid.UgridDataArray`) via the :attr:`data` property.
    """

    def __init__(self, model: "HurrywaveModel") -> None:
        self._data: Optional[dict] = None
        super().__init__(model=model)

    @property
    def data(
        self,
    ) -> Dict[str, Union[xr.Dataset, xr.DataArray, xu.UgridDataArray, xu.UgridDataset]]:
        """Model results. Returns a dict of xarray / xugrid DataArrays."""
        if self._data is None:
            self._initialize()
        return self._data

    def _initialize(self, skip_read: bool = False) -> None:
        """Initialise the results store, reading from disk in read-mode."""
        if self._data is None:
            self._data = {}
            if self.root.is_reading_mode() and not skip_read:
                self.read()

    def read(
        self,
        chunksize: int = 100,
        drop=("crs", "hurrywavegrid"),
        fn_map: str = "hurrywave_map.nc",
        fn_his: str = "hurrywave_his.nc",
        **kwargs,
    ) -> None:
        """Read ``hurrywave_map.nc`` and ``hurrywave_his.nc`` into :attr:`data`.

        Parameters
        ----------
        chunksize : int, optional
            Chunk size along the time dimension, by default 100.
        drop : iterable of str, optional
            Variables to drop on read.
        fn_map, fn_his : str, optional
            Filenames relative to the model root (absolute paths are also
            accepted), by default ``"hurrywave_map.nc"`` and
            ``"hurrywave_his.nc"``.
        """
        self.root._assert_read_mode()
        # Make sure config is available (for CRS etc.)
        self.model.config.read()

        drop = list(drop)

        if not isabs(fn_map):
            fn_map = self.model.root.path / fn_map
        if isfile(fn_map):
            self.read_map_file(fn_map=fn_map, drop=drop, chunksize=chunksize, **kwargs)
        else:
            logger.warning(f"File {fn_map} not found.")

        if not isabs(fn_his):
            fn_his = self.model.root.path / fn_his
        if isfile(fn_his):
            ds_his = self.read_his_file(fn_his=fn_his, drop=drop, chunksize=chunksize)
            self.set(ds_his, split_dataset=True)
        else:
            logger.warning(f"File {fn_his} not found.")

    def write(self) -> None:
        """No-op — the HurryWave kernel writes its own output files."""

    def read_map_file(
        self,
        fn_map: str = "hurrywave_map.nc",
        drop=("crs", "hurrywavegrid"),
        chunksize: int = 100,
        **kwargs,
    ) -> None:
        """Read ``hurrywave_map.nc`` (quadtree map) and store the variables.

        HurryWave is quadtree-only so the file is always a UGRID2D
        dataset; use :mod:`xugrid` to load it.
        """
        drop = list(drop)
        with xu.load_dataset(fn_map, chunks={"time": chunksize}, **kwargs) as ds:
            # Set node-coord coordinates so xugrid recognises them.
            node_coords = [c for c in ("mesh2d_node_x", "mesh2d_node_y") if c in ds]
            if node_coords:
                ds = ds.set_coords(node_coords)
            # Extract and apply the CRS if the file carries one.
            if "crs" in ds:
                try:
                    ds.grid.set_crs(CRS.from_user_input(ds["crs"].values))
                except Exception:
                    pass
            # Drop housekeeping variables that users don't care about.
            drop_vars = [v for v in drop if v in ds.data_vars]
            if drop_vars:
                ds = ds.drop_vars(drop_vars)
            self.set(ds, split_dataset=True)

    def read_his_file(
        self,
        fn_his: str = "hurrywave_his.nc",
        drop=("crs", "hurrywavegrid"),
        chunksize: int = 100,
    ) -> xr.Dataset:
        """Read ``hurrywave_his.nc`` (station time series) into a :class:`xarray.Dataset`.

        Attempts to set the station x/y/index as a hydromt ``vector``
        dimension so downstream code can treat it as a GeoDataset. Falls
        back to a plain Dataset if the expected station coordinates
        aren't present.
        """
        drop = list(drop)
        with xr.open_dataset(fn_his, chunks={"time": chunksize}) as ds:
            # Promote station descriptors to coordinates (name/id/x/y).
            cvars = ("id", "name", "x", "y", "station_x", "station_y")
            ds = ds.set_coords(
                [v for v in ds.data_vars if v.split("_")[-1] in ("id", "name", "x", "y")]
            )
            if "station_x" in ds and "station_y" in ds and "stations" in ds.dims:
                try:
                    ds.vector.set_spatial_dims(
                        x_name="station_x", y_name="station_y", index_dim="stations"
                    )
                    if self.model.crs is not None:
                        ds.vector.set_crs(self.model.crs)
                except Exception:
                    pass
            drop_vars = [v for v in drop if v in ds.data_vars]
            if drop_vars:
                ds = ds.drop_vars(drop_vars)
            # Avoid shadowing variables already loaded from the map file.
            dup = [v for v in ds.data_vars if v in self.data]
            if dup:
                ds = ds.drop_vars(dup)
            return ds

    def set(
        self,
        data: Union[xr.DataArray, xr.Dataset, xu.UgridDataArray, xu.UgridDataset],
        name: Optional[str] = None,
        split_dataset: bool = False,
    ) -> None:
        """Add data to :attr:`data`.

        Mirrors :meth:`hydromt_sfincs.SfincsOutput.set`.
        """
        self._initialize()
        data_dict = _check_data(data, name, split_dataset)
        for key, value in data_dict.items():
            if key in self._data:
                logger.warning(f"Replacing result: {key}")
            self._data[key] = value


def _check_data(
    data: Union[xr.DataArray, xr.Dataset, xu.UgridDataArray, xu.UgridDataset],
    name: Optional[str] = None,
    split_dataset: bool = True,
) -> Dict:
    if isinstance(data, (xr.DataArray, xu.UgridDataArray)):
        if data.name is None and name is not None:
            data.name = name
        elif name is None and data.name is not None:
            name = data.name
        elif data.name is None and name is None:
            raise ValueError("Name required for DataArray.")
        return {name: data}
    if isinstance(data, (xr.Dataset, xu.UgridDataset)):
        if split_dataset:
            return {n: data[n] for n in data.data_vars}
        if name is None:
            raise ValueError("Name required for Dataset.")
        return {name: data}
    raise ValueError(f'Data type "{type(data).__name__}" not recognized')

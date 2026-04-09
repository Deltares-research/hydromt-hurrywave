"""Observation point components for HurryWave.

HurryWave supports two types of output points:

* :class:`HurrywaveObservationPoints` – regular time-series output
  (Hm0, Tp, wave direction, directional spreading).  Written to ``obsfile``.

* :class:`HurrywaveObservationPointsSpectra` – 2-D spectral output at
  selected locations.  Written to ``ospfile``.

Both classes follow the same pattern as ``SfincsObservationPoints`` in
hydromt_sfincs: geometry stored in a GeoDataFrame, serialised as a simple
ASCII ``x y name`` file.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import xarray as xr
from hydromt import hydromt_step
from hydromt.model.components import ModelComponent

if TYPE_CHECKING:
    from hydromt_hurrywave import HurrywaveModel

logger = logging.getLogger(f"hydromt.{__name__}")


class _ObservationPointsBase(ModelComponent):
    """Shared logic for both observation point types."""

    # Subclasses must override these
    _config_key: str = None       # e.g. "obsfile"
    _default_filename: str = None  # e.g. "hurrywave.obs"

    def __init__(self, model: "HurrywaveModel"):
        self._data: Optional[gpd.GeoDataFrame] = None
        super().__init__(model=model)

    @property
    def data(self) -> gpd.GeoDataFrame:
        if self._data is None:
            self._initialize()
        return self._data

    def _initialize(self, skip_read: bool = False) -> None:
        if self._data is None:
            self._data = gpd.GeoDataFrame()
            if self.root.is_reading_mode() and not skip_read:
                self.read()

    @property
    def nr_points(self) -> int:
        return len(self._data) if self._data is not None else 0

    @property
    def list_names(self) -> List[str]:
        if self.nr_points == 0:
            return []
        return list(self._data["name"].values)

    @property
    def gdf(self) -> gpd.GeoDataFrame:
        return self.data

    def read(self, filename: Optional[str] = None) -> None:
        """Read an observation points file (``x y name`` per line)."""
        abs_path = self.model.config.get_set_file_variable(self._config_key, value=filename)
        if abs_path is None or not abs_path.exists():
            return

        logger.info(f"Reading observation points from {abs_path}")
        rows = []
        with open(str(abs_path)) as fid:
            for line in fid:
                parts = line.strip().split()
                if not parts:
                    continue
                x, y = float(parts[0]), float(parts[1])
                name = parts[2] if len(parts) > 2 else str(len(rows) + 1).zfill(4)
                rows.append({"name": name, "geometry": shapely.geometry.Point(x, y)})
        self._data = gpd.GeoDataFrame(rows, crs=self.model.crs)

    def write(self, filename: Optional[str] = None) -> None:
        """Write observation points to file."""
        if self.nr_points == 0:
            return

        abs_path = self.model.config.get_set_file_variable(
            self._config_key, value=filename, default=self._default_filename
        )
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Writing observation points to {abs_path}")
        with open(str(abs_path), "w") as fid:
            for _, row in self._data.iterrows():
                x, y = row.geometry.x, row.geometry.y
                name = row.get("name", "")
                if self.model.crs.is_geographic:
                    fid.write(f'{x:12.6f}  {y:12.6f}  "{name}"\n')
                else:
                    fid.write(f'{x:12.1f}  {y:12.1f}  "{name}"\n')

    @hydromt_step
    def add_point(self, x: float, y: float, name: str = None) -> None:
        """Add a single observation point.

        Parameters
        ----------
        x, y : float
            Coordinates in the model CRS.
        name : str, optional
            Point name (auto-generated if omitted).
        """
        if name is None:
            name = str(self.nr_points + 1).zfill(4)
        new_row = gpd.GeoDataFrame(
            [{"name": name, "geometry": shapely.geometry.Point(x, y)}],
            crs=self.model.crs,
        )
        if self.nr_points == 0:
            self._data = new_row
        else:
            self._data = pd.concat([self._data, new_row], ignore_index=True)

    @hydromt_step
    def add_points(self, gdf: gpd.GeoDataFrame) -> None:
        """Add multiple observation points from a GeoDataFrame.

        Parameters
        ----------
        gdf : GeoDataFrame
            Points with a ``name`` column and Point geometry.
        """
        if gdf.crs != self.model.crs:
            gdf = gdf.to_crs(self.model.crs)
        if self.nr_points == 0:
            self._data = gdf.reset_index(drop=True)
        else:
            self._data = pd.concat([self._data, gdf], ignore_index=True)

    def delete(self, index: Union[int, List[int]]) -> None:
        """Delete observation point(s) by index.

        Parameters
        ----------
        index : int or list of int
        """
        if self.nr_points == 0:
            return
        if not isinstance(index, list):
            index = [index]
        self._data = self._data.drop(index=index).reset_index(drop=True)

    def clear(self) -> None:
        self._data = gpd.GeoDataFrame()


class HurrywaveObservationPoints(_ObservationPointsBase):
    """Regular (bulk-parameter) observation points.

    Output variables: Hm0, Tp, wave direction, directional spreading.
    Config key: ``obsfile`` (default file: ``hurrywave.obs``).
    """

    _config_key = "obsfile"
    _default_filename = "hurrywave.obs"


class HurrywaveObservationPointsSpectra(_ObservationPointsBase):
    """Spectral output observation points.

    2-D energy spectra are written at these locations.
    Config key: ``ospfile`` (default file: ``hurrywave.osp``).
    """

    _config_key = "ospfile"
    _default_filename = "hurrywave.osp"

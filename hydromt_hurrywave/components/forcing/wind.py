"""Wind forcing for HurryWave.

Two modes:

* **uniform** – a single time series of wind speed and direction written to a
  text file (``wndfile``).  Columns: seconds-since-tref, wind-speed [m/s],
  wind-direction [°N].

* **gridded** – spatially varying wind on a regular lat/lon grid stored in
  netCDF files (``amufile`` for the u-component and ``amvfile`` for the
  v-component), following the Delft3D FLOW / SFINCS convention.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr
from hydromt import hydromt_step
from hydromt.model.components import ModelComponent

if TYPE_CHECKING:
    from hydromt_hurrywave import HurrywaveModel

logger = logging.getLogger(f"hydromt.{__name__}")


class HurrywaveWind(ModelComponent):
    """Wind forcing component.

    ``data`` is an ``xr.Dataset`` that contains either:

    * ``wind_speed`` and ``wind_dir`` (uniform mode), or
    * ``wind10_u`` and ``wind10_v`` (gridded mode).
    """

    def __init__(self, model: "HurrywaveModel"):
        self._data: Optional[xr.Dataset] = None
        super().__init__(model=model)

    @property
    def data(self) -> Optional[xr.Dataset]:
        if self._data is None and self.root.is_reading_mode():
            self.read()
        return self._data

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def read(self) -> None:
        """Read wind forcing from whichever files are referenced in config."""
        wndfile = self.model.config.get("wndfile")
        amufile = self.model.config.get("amufile")
        amvfile = self.model.config.get("amvfile")

        if wndfile is not None:
            self._read_uniform()
        elif amufile is not None or amvfile is not None:
            self._read_gridded()

    def _read_uniform(self) -> None:
        """Read uniform wind speed / direction time series."""
        abs_path = self.model.config.get_set_file_variable("wndfile")
        if abs_path is None or not abs_path.exists():
            return

        logger.info(f"Reading uniform wind from {abs_path}")
        tref = self.model.config.get("tref")
        df = pd.read_csv(
            str(abs_path), index_col=0, header=None, sep=r"\s+",
            names=["speed", "direction"],
        )
        df.index = tref + pd.to_timedelta(df.index, unit="s")
        self._data = xr.Dataset(
            {
                "wind_speed": xr.DataArray(df["speed"].values, dims="time",
                                           coords={"time": df.index}),
                "wind_dir": xr.DataArray(df["direction"].values, dims="time",
                                         coords={"time": df.index}),
            }
        )

    def _read_gridded(self) -> None:
        """Read gridded wind u/v components from netCDF."""
        datasets = []
        for key, var in (("amufile", "wind10_u"), ("amvfile", "wind10_v")):
            abs_path = self.model.config.get_set_file_variable(key)
            if abs_path is None or not abs_path.exists():
                continue
            logger.info(f"Reading gridded wind ({var}) from {abs_path}")
            ds = xr.open_dataset(str(abs_path))
            # rename to canonical variable name if necessary
            first_var = list(ds.data_vars)[0]
            if first_var != var:
                ds = ds.rename({first_var: var})
            datasets.append(ds)
        if datasets:
            self._data = xr.merge(datasets)

    def write(self) -> None:
        """Write wind forcing to files."""
        if self._data is None:
            return
        if "wind_speed" in self._data:
            self._write_uniform()
        elif "wind10_u" in self._data or "wind10_v" in self._data:
            self._write_gridded()

    def _write_uniform(self) -> None:
        """Write uniform wind to a text file."""
        abs_path = self.model.config.get_set_file_variable(
            "wndfile", default="hurrywave.wnd"
        )
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        tref = self.model.config.get("tref")
        if tref is None:
            raise ValueError("tref must be set in config before writing wind forcing.")

        logger.info(f"Writing uniform wind to {abs_path}")
        times = self._data["wind_speed"].time.values
        seconds = (times.astype("datetime64[s]").astype(float) -
                   np.datetime64(tref, "s").astype(float))
        speeds = self._data["wind_speed"].values
        dirs = self._data["wind_dir"].values

        with open(str(abs_path), "w") as fid:
            for t, spd, dr in zip(seconds, speeds, dirs):
                fid.write(f"{t:12.1f}  {spd:.3f}  {dr:.3f}\n")

    def _write_gridded(self) -> None:
        """Write gridded wind components to netCDF."""
        for key, var in (("amufile", "wind10_u"), ("amvfile", "wind10_v")):
            if var not in self._data:
                continue
            abs_path = self.model.config.get_set_file_variable(
                key, default=f"hurrywave.{key[:-4]}"
            )
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Writing gridded wind ({var}) to {abs_path}")
            self._data[[var]].to_netcdf(str(abs_path))

    # ------------------------------------------------------------------
    # Setters
    # ------------------------------------------------------------------

    @hydromt_step
    def set_uniform(self, df: pd.DataFrame) -> None:
        """Set uniform wind from a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DatetimeIndex, columns ``["speed", "direction"]`` (or ``["u", "v"]``
            which will be renamed).
        """
        if set(df.columns) >= {"speed", "direction"}:
            self._data = xr.Dataset(
                {
                    "wind_speed": xr.DataArray(df["speed"].values, dims="time",
                                               coords={"time": df.index}),
                    "wind_dir": xr.DataArray(df["direction"].values, dims="time",
                                             coords={"time": df.index}),
                }
            )
        elif set(df.columns) >= {"u", "v"}:
            spd = np.sqrt(df["u"] ** 2 + df["v"] ** 2)
            dr = (270.0 - np.degrees(np.arctan2(df["v"], df["u"]))) % 360.0
            self._data = xr.Dataset(
                {
                    "wind_speed": xr.DataArray(spd.values, dims="time",
                                               coords={"time": df.index}),
                    "wind_dir": xr.DataArray(dr.values, dims="time",
                                             coords={"time": df.index}),
                }
            )
        else:
            raise ValueError("DataFrame must have columns 'speed'+'direction' or 'u'+'v'.")

    @hydromt_step
    def set_gridded(self, ds: xr.Dataset) -> None:
        """Set gridded wind from an xarray Dataset with ``wind10_u`` and/or ``wind10_v``."""
        self._data = ds

    def clear(self) -> None:
        self._data = None

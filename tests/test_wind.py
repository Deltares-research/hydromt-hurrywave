"""Tests for HurrywaveWind component."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from constants import TSTART, TSTOP, make_time_index


class TestWindUniform:
    """Uniform wind forcing."""

    def test_set_uniform_speed_direction(self, model_with_mask):
        wind = model_with_mask.wind
        times = make_time_index(12)
        df = pd.DataFrame(
            {"speed": [10.0] * 12, "direction": [270.0] * 12},
            index=times,
        )
        wind.set_uniform(df)
        assert wind.data is not None

    def test_set_uniform_uv(self, model_with_mask):
        wind = model_with_mask.wind
        times = make_time_index(12)
        df = pd.DataFrame(
            {"u": [-10.0] * 12, "v": [0.0] * 12},
            index=times,
        )
        wind.set_uniform(df)
        assert wind.data is not None

    def test_clear(self, model_with_mask):
        wind = model_with_mask.wind
        times = make_time_index(12)
        df = pd.DataFrame(
            {"speed": [10.0] * 12, "direction": [270.0] * 12},
            index=times,
        )
        wind.set_uniform(df)
        wind.clear()


class TestWindIO:
    """Wind read/write round-trip."""

    def test_write_read_uniform(self, model_with_mask, tmp_path):
        mod = model_with_mask
        wind = mod.wind
        times = make_time_index(12)
        df = pd.DataFrame(
            {"speed": [10.0] * 12, "direction": [270.0] * 12},
            index=times,
        )
        wind.set_uniform(df)

        root = tmp_path / "wind_rt"
        root.mkdir()
        mod.root.set(str(root), mode="w")
        wind.write()
        mod.config.write()  # config must be written AFTER wind so wndfile key is set

        from hydromt_hurrywave import HurrywaveModel

        mod2 = HurrywaveModel(root=str(root), mode="r")
        mod2.config.read()
        mod2.wind.read()
        assert mod2.wind.data is not None

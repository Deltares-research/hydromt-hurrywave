"""Tests for HurrywaveConfig component."""

from datetime import datetime

import pytest

from constants import TREF, TSTART, TSTOP


class TestConfigGetSet:
    """Basic get / set / update operations."""

    def test_set_and_get(self, model_write):
        config = model_write.config
        config.set("mmax", 20)
        assert config.get("mmax") == 20

    def test_get_default_returns_none(self, model_write):
        assert model_write.config.get("mmax") is None

    def test_get_fallback(self, model_write):
        assert model_write.config.get("nonexistent", fallback=42) == 42

    def test_set_datetime(self, model_write):
        config = model_write.config
        config.set("tref", TREF)
        assert config.get("tref") == TREF

    def test_update_dict(self, model_write):
        config = model_write.config
        config.update({"mmax": 5, "nmax": 10})
        assert config.get("mmax") == 5
        assert config.get("nmax") == 10

    def test_update_kwargs(self, model_write):
        config = model_write.config
        config.update(dx=500.0, dy=500.0)
        assert config.get("dx") == 500.0
        assert config.get("dy") == 500.0


class TestConfigIO:
    """Read/write round-trip for hurrywave.inp."""

    def test_write_read_roundtrip(self, model_with_grid, tmp_path):
        mod = model_with_grid
        root = tmp_path / "config_rt"
        root.mkdir()
        mod.root.set(str(root), mode="w")
        mod.config.write()

        # Verify the file exists
        assert (root / "hurrywave.inp").exists()

        # Read back
        from hydromt_hurrywave import HurrywaveModel

        mod2 = HurrywaveModel(root=str(root), mode="r")
        mod2.config.read()

        assert mod2.config.get("tref") == TREF
        assert mod2.config.get("tstart") == TSTART
        assert mod2.config.get("tstop") == TSTOP

    def test_time_values_preserved(self, model_with_grid):
        config = model_with_grid.config
        assert config.get("tref") == TREF
        assert config.get("tstart") == TSTART
        assert config.get("tstop") == TSTOP

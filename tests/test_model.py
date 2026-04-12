"""Tests for the HurrywaveModel class (integration tests)."""

import pytest

from constants import EPSG, TREF, TSTART, TSTOP


class TestModelInit:
    """Model creation and basic properties."""

    def test_create_write_mode(self, model_write):
        assert model_write is not None
        assert model_write.name == "hurrywave"

    def test_model_with_grid_has_crs(self, model_with_grid):
        from pyproj import CRS

        assert model_with_grid.crs == CRS.from_epsg(EPSG)

    def test_model_with_grid_has_region(self, model_with_grid):
        region = model_with_grid.region
        assert region is not None
        assert len(region) > 0

    def test_model_with_grid_has_bounds(self, model_with_grid):
        bounds = model_with_grid.bounds
        assert bounds is not None
        assert len(bounds) == 4

    def test_get_model_time(self, model_with_grid):
        tstart, tstop = model_with_grid.get_model_time()
        assert tstart == TSTART
        assert tstop == TSTOP


class TestModelIO:
    """Full model write / read round-trip."""

    def test_write_creates_files(self, model_with_mask, tmp_path):
        root = tmp_path / "model_io"
        root.mkdir()
        mod = model_with_mask
        mod.root.set(str(root), mode="w")
        mod.write()

        assert (root / "hurrywave.inp").exists()
        assert (root / "hurrywave.nc").exists()

    def test_roundtrip_preserves_grid(self, model_roundtrip, model_with_mask):
        assert model_roundtrip.quadtree_grid.data.sizes["mesh2d_nFaces"] == model_with_mask.quadtree_grid.data.sizes["mesh2d_nFaces"]

    def test_roundtrip_preserves_config(self, model_roundtrip):
        assert model_roundtrip.config.get("tref") == TREF
        assert model_roundtrip.config.get("tstart") == TSTART
        assert model_roundtrip.config.get("tstop") == TSTOP

    def test_roundtrip_preserves_mask(self, model_roundtrip, model_with_mask):
        mask_orig = model_with_mask.quadtree_grid.data["mask"].values
        mask_read = model_roundtrip.quadtree_grid.data["mask"].values
        assert (mask_orig == mask_read).all()


class TestModelWithForcing:
    """Model with boundary conditions and observation points."""

    def test_full_model_with_obs(self, model_with_mask, tmp_path):
        mod = model_with_mask
        mod.observation_points.add_point(x=5000.0, y=5000.0, name="obs_01")
        mod.observation_points_spectra.add_point(x=5000.0, y=5000.0, name="osp_01")

        root = tmp_path / "model_full"
        root.mkdir()
        mod.root.set(str(root), mode="w")
        mod.write()

        from hydromt_hurrywave import HurrywaveModel

        mod2 = HurrywaveModel(root=str(root), mode="r")
        mod2.read()
        assert mod2.observation_points.nr_points == 1
        assert mod2.observation_points_spectra.nr_points == 1

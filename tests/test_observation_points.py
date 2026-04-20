"""Tests for observation point components."""

import geopandas as gpd
import pytest
from shapely.geometry import Point

from constants import EPSG


class TestObservationPoints:
    """HurrywaveObservationPoints component."""

    def test_initial_empty(self, model_with_mask):
        obs = model_with_mask.observation_points
        assert obs.nr_points == 0

    def test_add_point(self, model_with_mask):
        obs = model_with_mask.observation_points
        obs.add_point(x=5000.0, y=5000.0, name="obs_01")
        assert obs.nr_points == 1
        assert "obs_01" in obs.list_names

    def test_add_multiple_points(self, model_with_mask):
        obs = model_with_mask.observation_points
        obs.add_point(x=1000.0, y=1000.0, name="obs_01")
        obs.add_point(x=2000.0, y=2000.0, name="obs_02")
        obs.add_point(x=3000.0, y=3000.0, name="obs_03")
        assert obs.nr_points == 3

    def test_add_points_gdf(self, model_with_mask):
        obs = model_with_mask.observation_points
        gdf = gpd.GeoDataFrame(
            {"name": ["a", "b"]},
            geometry=[Point(1000, 1000), Point(2000, 2000)],
            crs=EPSG,
        )
        obs.add_points(gdf)
        assert obs.nr_points == 2

    def test_delete_point(self, model_with_mask):
        obs = model_with_mask.observation_points
        obs.add_point(x=1000.0, y=1000.0, name="obs_01")
        obs.add_point(x=2000.0, y=2000.0, name="obs_02")
        obs.delete(0)
        assert obs.nr_points == 1

    def test_clear(self, model_with_mask):
        obs = model_with_mask.observation_points
        obs.add_point(x=1000.0, y=1000.0, name="obs_01")
        obs.clear()
        assert obs.nr_points == 0

    def test_gdf_property(self, model_with_mask):
        obs = model_with_mask.observation_points
        obs.add_point(x=1000.0, y=1000.0, name="obs_01")
        gdf = obs.gdf
        assert len(gdf) == 1
        assert "name" in gdf.columns


class TestObservationPointsIO:
    """Read/write round-trip."""

    def test_write_read(self, model_with_mask, tmp_path):
        mod = model_with_mask
        obs = mod.observation_points
        obs.add_point(x=1000.0, y=1000.0, name="obs_01")
        obs.add_point(x=5000.0, y=5000.0, name="obs_02")

        root = tmp_path / "obs_rt"
        root.mkdir()
        mod.root.set(str(root), mode="w")
        mod.write()  # write full model so grid file is available for read-back

        from hydromt_hurrywave import HurrywaveModel

        mod2 = HurrywaveModel(root=str(root), mode="r")
        mod2.config.read()
        mod2.observation_points.read()
        assert mod2.observation_points.nr_points == 2
        assert "obs_01" in mod2.observation_points.list_names
        assert "obs_02" in mod2.observation_points.list_names


class TestObservationPointsSpectra:
    """HurrywaveObservationPointsSpectra component."""

    def test_add_point(self, model_with_mask):
        osp = model_with_mask.observation_points_spectra
        osp.add_point(x=5000.0, y=5000.0, name="osp_01")
        assert osp.nr_points == 1

    def test_write_read(self, model_with_mask, tmp_path):
        mod = model_with_mask
        osp = mod.observation_points_spectra
        osp.add_point(x=5000.0, y=5000.0, name="osp_01")

        root = tmp_path / "osp_rt"
        root.mkdir()
        mod.root.set(str(root), mode="w")
        mod.write()  # write full model so grid file is available for read-back

        from hydromt_hurrywave import HurrywaveModel

        mod2 = HurrywaveModel(root=str(root), mode="r")
        mod2.config.read()
        mod2.observation_points_spectra.read()
        assert mod2.observation_points_spectra.nr_points == 1

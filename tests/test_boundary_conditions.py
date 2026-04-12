"""Tests for HurrywaveBoundaryConditions component."""

import numpy as np
import pandas as pd
import pytest

from constants import N_BND_POINTS, N_TIMES, make_boundary_gdf, make_time_index, make_wave_df


class TestBoundaryTimeseries:
    """Timeseries-mode boundary conditions."""

    def test_initial_state_empty(self, model_with_mask):
        bc = model_with_mask.boundary_conditions
        assert bc.nr_points == 0
        assert bc.forcing == "timeseries"

    def test_set_timeseries_hs(self, model_with_mask, wave_df, boundary_gdf):
        bc = model_with_mask.boundary_conditions
        bc.set_timeseries(wave_df, gdf=boundary_gdf, varname="hs")
        assert bc.nr_points == N_BND_POINTS
        assert "hs" in bc.data

    def test_set_all_variables(self, model_with_mask, boundary_gdf):
        bc = model_with_mask.boundary_conditions
        for var, val in [("hs", 1.5), ("tp", 8.0), ("wd", 270.0), ("ds", 30.0)]:
            df = make_wave_df(hs=val, tp=val, wd=val, ds=val)
            bc.set_timeseries(df, gdf=boundary_gdf, varname=var)
        assert "hs" in bc.data
        assert "tp" in bc.data
        assert "wd" in bc.data
        assert "ds" in bc.data

    def test_set_uniform(self, model_with_mask, boundary_gdf):
        bc = model_with_mask.boundary_conditions
        bc.set_uniform(hs=2.0, tp=10.0, wd=180.0, ds=20.0, gdf=boundary_gdf)
        assert bc.nr_points == N_BND_POINTS

    def test_gdf_property(self, model_with_mask, wave_df, boundary_gdf):
        bc = model_with_mask.boundary_conditions
        bc.set_timeseries(wave_df, gdf=boundary_gdf, varname="hs")
        gdf = bc.gdf
        assert len(gdf) == N_BND_POINTS

    def test_add_point(self, model_with_mask):
        bc = model_with_mask.boundary_conditions
        bc.add_point(x=100.0, y=200.0, name="test_pt")
        assert bc.nr_points == 1

    def test_delete_point(self, model_with_mask):
        bc = model_with_mask.boundary_conditions
        bc.add_point(x=100.0, y=200.0, name="pt1")
        bc.add_point(x=300.0, y=400.0, name="pt2")
        assert bc.nr_points == 2
        bc.delete(0)
        assert bc.nr_points == 1


class TestBoundaryIO:
    """Read/write round-trip for boundary conditions."""

    @pytest.mark.xfail(reason="mod.write() triggers lazy read on BC component, causing dimension conflict")
    def test_write_read_timeseries(self, model_with_mask, boundary_gdf, tmp_path):
        mod = model_with_mask
        bc = mod.boundary_conditions

        # Set all 4 wave variables (only pass gdf on first call)
        for i, (var, val) in enumerate([("hs", 1.5), ("tp", 8.0), ("wd", 270.0), ("ds", 30.0)]):
            df = make_wave_df(hs=val, tp=val, wd=val, ds=val)
            bc.set_timeseries(df, gdf=boundary_gdf if i == 0 else None, varname=var)

        # Write full model so grid file is available for read-back
        root = tmp_path / "bc_rt"
        root.mkdir()
        mod.root.set(str(root), mode="w")
        mod.write()

        # Read back
        from hydromt_hurrywave import HurrywaveModel

        mod2 = HurrywaveModel(root=str(root), mode="r")
        mod2.config.read()
        mod2.boundary_conditions.read()
        assert mod2.boundary_conditions.nr_points == N_BND_POINTS

    def test_set_all_variables_no_gdf_reuse(self, model_with_mask, boundary_gdf):
        """Ensure passing gdf only once works for all 4 variables."""
        bc = model_with_mask.boundary_conditions
        for i, (var, val) in enumerate([("hs", 1.5), ("tp", 8.0), ("wd", 270.0), ("ds", 30.0)]):
            df = make_wave_df(hs=val, tp=val, wd=val, ds=val)
            bc.set_timeseries(df, gdf=boundary_gdf if i == 0 else None, varname=var)
        for var in ["hs", "tp", "wd", "ds"]:
            assert var in bc.data

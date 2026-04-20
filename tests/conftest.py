"""Shared fixtures for hydromt_hurrywave tests."""

import pytest

from hydromt_hurrywave import HurrywaveModel

from constants import (
    DX, DY, EPSG, MMAX, NMAX, TREF, TSTART, TSTOP, X0, Y0,
    make_boundary_gdf, make_wave_df,
)


@pytest.fixture
def model_write(tmp_path):
    """A HurrywaveModel in write mode with no data."""
    root = tmp_path / "model_w"
    root.mkdir()
    return HurrywaveModel(root=str(root), mode="w")


@pytest.fixture
def model_with_grid(tmp_path):
    """A HurrywaveModel with a 10x10 quadtree grid at 1 km resolution."""
    root = tmp_path / "model_grid"
    root.mkdir()
    mod = HurrywaveModel(root=str(root), mode="w")

    mod.quadtree_grid.create(
        x0=X0, y0=Y0, nmax=NMAX, mmax=MMAX,
        dx=DX, dy=DY, rotation=0.0, epsg=EPSG,
    )
    mod.quadtree_elevation.create_uniform(zb=-10.0)
    mod.config.set("tref", TREF)
    mod.config.set("tstart", TSTART)
    mod.config.set("tstop", TSTOP)

    return mod


@pytest.fixture
def model_with_mask(model_with_grid):
    """A model with grid + active mask."""
    model_with_grid.quadtree_mask.create(zmin=-99999.0)
    return model_with_grid


@pytest.fixture
def model_roundtrip(model_with_mask, tmp_path):
    """Write a model to disk and read it back."""
    root = tmp_path / "model_rt"
    root.mkdir()
    model_with_mask.root.set(str(root), mode="w")
    model_with_mask.write()

    mod2 = HurrywaveModel(root=str(root), mode="r")
    mod2.read()
    return mod2


@pytest.fixture
def wave_df():
    return make_wave_df()


@pytest.fixture
def boundary_gdf():
    return make_boundary_gdf()

"""Tests for HurrywaveQuadtreeGrid, Elevation, and Mask components."""

import numpy as np
import pytest
from pyproj import CRS

from constants import DX, DY, EPSG, MMAX, NMAX, X0, Y0


class TestQuadtreeGrid:
    """Grid creation and properties."""

    def test_create_grid(self, model_with_grid):
        grid = model_with_grid.quadtree_grid
        assert grid.data is not None
        assert grid.data.sizes["mesh2d_nFaces"] > 0

    def test_grid_crs(self, model_with_grid):
        crs = model_with_grid.quadtree_grid.crs
        assert crs is not None
        assert crs == CRS.from_epsg(EPSG)

    def test_grid_exterior(self, model_with_grid):
        ext = model_with_grid.quadtree_grid.exterior
        assert ext is not None
        assert len(ext) > 0

    def test_grid_face_coordinates(self, model_with_grid):
        xc, yc = model_with_grid.quadtree_grid.face_coordinates
        n = model_with_grid.quadtree_grid.data.sizes["mesh2d_nFaces"]
        assert len(xc) == n
        assert len(yc) == n


class TestQuadtreeCreateIndexTiles:
    """Index-tile generation for the quadtree grid."""

    def test_create_index_tiles_png_and_html(self, model_with_mask, tmp_path):
        # Small grid (10 x 10 @ 1 km) — need enough zoom to resolve pixels.
        zoom_range = [12, 13]
        root = tmp_path / "tiles_png"
        model_with_mask.quadtree_grid.create_index_tiles(
            root=root, zoom_range=zoom_range
        )
        png_files = list((root / "indices").rglob("*.png"))
        assert len(png_files) > 0
        # <indices>/<zoom>/<x>/<y>.png
        zoom_levels = {int(p.parts[-3]) for p in png_files}
        assert zoom_levels.issubset(set(range(zoom_range[0], zoom_range[1] + 1)))
        # HTML viewer is written by default alongside PNG tiles
        html_file = root / "indices" / "index.html"
        assert html_file.is_file()
        assert "{z}/{x}/{y}.png" in html_file.read_text()

    def test_create_index_tiles_bin(self, model_with_mask, tmp_path):
        zoom_range = [12, 13]
        root = tmp_path / "tiles_bin"
        model_with_mask.quadtree_grid.create_index_tiles(
            root=root, zoom_range=zoom_range, fmt="bin"
        )
        dat_files = list((root / "indices").rglob("*.dat"))
        assert len(dat_files) > 0
        # Each .dat tile should be 256*256 int32 = 262144 bytes
        assert dat_files[0].stat().st_size == 256 * 256 * 4
        # No HTML viewer for bin format
        assert not (root / "indices" / "index.html").exists()


class TestQuadtreeGridIO:
    """Grid read/write round-trip."""

    def test_write_read_roundtrip(self, model_with_grid, tmp_path):
        mod = model_with_grid
        root = tmp_path / "grid_rt"
        root.mkdir()
        mod.root.set(str(root), mode="w")
        mod.quadtree_grid.write()
        mod.config.write()  # config must be written AFTER grid so qtrfile key is set

        from hydromt_hurrywave import HurrywaveModel

        mod2 = HurrywaveModel(root=str(root), mode="r")
        mod2.config.read()
        mod2.quadtree_grid.read()

        assert (
            mod2.quadtree_grid.data.sizes["mesh2d_nFaces"]
            == mod.quadtree_grid.data.sizes["mesh2d_nFaces"]
        )
        assert mod2.quadtree_grid.crs == mod.quadtree_grid.crs


class TestQuadtreeElevation:
    """Elevation creation and properties."""

    def test_uniform_elevation(self, model_with_grid):
        z = model_with_grid.quadtree_grid.data["z"]
        assert z is not None
        assert np.allclose(z.values, -10.0)

    def test_create_uniform_overwrites(self, model_with_grid):
        mod = model_with_grid
        mod.quadtree_elevation.create_uniform(zb=-20.0)
        z = mod.quadtree_grid.data["z"]
        assert np.allclose(z.values, -20.0)


class TestQuadtreeMask:
    """Mask creation and properties."""

    def test_create_mask(self, model_with_mask):
        mask = model_with_mask.quadtree_grid.data["mask"]
        assert mask is not None
        # All cells should be active (1) or boundary (2)
        assert (mask.values >= 0).all()

    def test_mask_has_active_cells(self, model_with_mask):
        mask = model_with_mask.quadtree_grid.data["mask"]
        assert (mask.values == 1).any()

    def test_to_gdf(self, model_with_mask):
        gdf = model_with_mask.quadtree_mask.to_gdf(option="active")
        assert len(gdf) > 0

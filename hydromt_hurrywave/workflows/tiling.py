"""Tiling functions for fast visualization of the HurryWave model in- and output data."""

import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path
from typing import List, Optional, Union

import geopandas as gpd
import numpy as np
import xarray as xr
from affine import Affine
from PIL import Image
from pyproj import Transformer

from .merge import merge_multi_dataarrays

# numba is an optional runtime accelerator for the index-tile kernel.
try:
    from numba import njit, prange

    HAS_NUMBA = True
except ImportError:  # pragma: no cover - exercised only on installs without numba
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        """Fallback no-op decorator used when numba is not installed."""
        if args and callable(args[0]) and not kwargs:
            return args[0]

        def decorator(func):
            return func

        return decorator

    prange = range

__all__ = [
    "make_index_tiles",
    "write_html",
]

logger = logging.getLogger(__name__)


def write_html(
    file_name: Union[str, Path],
    title: str = "hydromt-hurrywave tiles",
    legend_title: str = "Legend",
    max_native_zoom: int = 19,
) -> None:
    """Write a standalone Leaflet HTML viewer for a tiled map layer.

    The generated page loads OpenStreetMap and Esri World Imagery base
    layers and overlays tiles from the URL pattern ``{z}/{x}/{y}.png``
    relative to the HTML file. Drop it alongside a ``{z}/{x}/{y}.png``
    tile tree and open it in a browser to preview the tiles.

    Parameters
    ----------
    file_name : str or Path
        Output HTML file path.
    title : str, optional
        Page heading shown above the map, by default
        ``"hydromt-hurrywave tiles"``.
    legend_title : str, optional
        Text displayed inside the legend box, by default ``"Legend"``.
    max_native_zoom : int, optional
        Maximum native zoom level of the tile layer, by default ``19``.
    """
    with open(file_name, "w") as f:
        f.write("<!DOCTYPE html>\r\n")
        f.write("<head>\r\n")
        f.write(
            "  <meta http-equiv='content-type' content='text/html; charset=UTF-8' />\r\n"
        )
        f.write("  <script>\r\n")
        f.write("    L_NO_TOUCH = false;\r\n")
        f.write("    L_DISABLE_3D = false;\r\n")
        f.write("  </script>\r\n")
        f.write(
            "  <style>html, body {width: 100%;height: 100%;margin: 0;padding: 0;}</style>\r\n"
        )
        f.write(
            "  <script src='https://cdn.jsdelivr.net/npm/leaflet@1.6.0/dist/leaflet.js'></script>\r\n"
        )
        f.write(
            "  <script src='https://code.jquery.com/jquery-1.12.4.min.js'></script>\r\n"
        )
        f.write(
            "  <script src='https://maxcdn.bootstrapcdn.com/bootstrap/3.2.0/js/bootstrap.min.js'></script>\r\n"
        )
        f.write(
            "  <script src='https://cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.js'></script>\r\n"
        )
        f.write(
            "  <link rel='stylesheet' href='https://cdn.jsdelivr.net/npm/leaflet@1.6.0/dist/leaflet.css'/>\r\n"
        )
        f.write(
            "  <link rel='stylesheet' href='https://maxcdn.bootstrapcdn.com/bootstrap/3.2.0/css/bootstrap.min.css'/>\r\n"
        )
        f.write(
            "  <link rel='stylesheet' href='https://maxcdn.bootstrapcdn.com/bootstrap/3.2.0/css/bootstrap-theme.min.css'/>\r\n"
        )
        f.write(
            "  <link rel='stylesheet' href='https://maxcdn.bootstrapcdn.com/font-awesome/4.6.3/css/font-awesome.min.css'/>\r\n"
        )
        f.write(
            "  <link rel='stylesheet' href='https://cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.css'/>\r\n"
        )
        f.write(
            "  <link rel='stylesheet' href='https://cdn.jsdelivr.net/gh/python-visualization/folium/folium/templates/leaflet.awesome.rotate.min.css'/>\r\n"
        )
        f.write(
            "  <meta name='viewport' content='width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no' />\r\n"
        )
        f.write("\r\n")
        f.write("  <style>\r\n")
        f.write("      #map { width: 800px; height: 500px; }\r\n")
        f.write(
            "      .info { padding: 6px 8px; font: 17px/19px Arial, Helvetica, sans-serif; background: white; background: rgba(255,255,255,0.8); box-shadow: 0 0 15px rgba(0,0,0,0.2); border-radius: 5px; } .info h4 { margin: 0 0 5px; color: #777; }\r\n"
        )
        f.write(
            "      .legend     { text-align: center; line-height: 18px; color: #555; } .legend i     { width: 20px; height: 15px; float: left; margin-right: 8px; opacity: 0.7; border-style: solid; border-width: 1px;}\r\n"
        )
        f.write("  </style>\r\n")
        f.write("\r\n")
        f.write("</head>\r\n")
        f.write("<body>\r\n")
        f.write(f"  <h3> {title}</h3>\r\n")
        f.write("  <div id='map' style='width: 100%; height: 90%;'></div>\r\n")
        f.write("</body>\r\n")
        f.write("<script>\r\n")
        f.write("\r\n")
        f.write("// Base layers\r\n")
        f.write(
            "var tile_layer_osm = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',\r\n"
        )
        f.write(
            "    {'attribution': 'Data by http://openstreetmap.org href=http://www.openstreetmap.org',\r\n"
        )
        f.write("     'detectRetina': false,\r\n")
        f.write("     'maxZoom': 19,\r\n")
        f.write("     'minZoom': 0,\r\n")
        f.write("     'noWrap': false,\r\n")
        f.write("     'opacity': 1,\r\n")
        f.write("     'maxNativeZoom': 13,\r\n")
        f.write("     'subdomains': 'abc',\r\n")
        f.write("     'tms': false});\r\n")
        f.write(
            "var Esri_WorldImagery = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {\r\n"
        )
        f.write(
            "	attribution: 'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community'\r\n"
        )
        f.write("});\r\n")
        f.write("\r\n")
        f.write("// Data layer\r\n")
        f.write("var tile_layer = L.tileLayer(\r\n")
        f.write("    '{z}/{x}/{y}.png',\r\n")
        f.write("    {'attribution': 'hydromt-hurrywave',\r\n")
        f.write("     'detectRetina': false,\r\n")
        f.write("     'opacity': 0.7,\r\n")
        f.write(f"     'maxNativeZoom': {max_native_zoom},\r\n")
        f.write("     'maxZoom': 19,\r\n")
        f.write("     'minZoom': 0,\r\n")
        f.write("     'noWrap': false,\r\n")
        f.write("     'subdomains': 'abc',\r\n")
        f.write("     'zIndex':10,\r\n")
        f.write("     'tms': false}\r\n")
        f.write(");\r\n")
        f.write("\r\n")
        f.write("var legend = L.control({position: 'bottomright'});\r\n")
        f.write("legend.onAdd = function (map) {\r\n")
        f.write("        var div = L.DomUtil.create('div', 'info legend')\r\n")
        f.write(f"        div.innerHTML += '{legend_title}<br>'\r\n")
        f.write("        return div;\r\n")
        f.write("};\r\n")
        f.write("\r\n")
        f.write("// Map\r\n")
        f.write("var map = L.map('map',{\r\n")
        f.write("    center: [0, 0],\r\n")
        f.write("    crs: L.CRS.EPSG3857,\r\n")
        f.write("    zoom: 2,\r\n")
        f.write("    zoomControl: true,\r\n")
        f.write("    preferCanvas: false,\r\n")
        f.write("    layers: [tile_layer_osm, tile_layer]\r\n")
        f.write("    }\r\n")
        f.write(");\r\n")
        f.write("\r\n")
        f.write("legend.addTo(map);\r\n")
        f.write("\r\n")
        f.write("// Layer control\r\n")
        f.write("var baseMaps = {\r\n")
        f.write("    'Open Street Map': tile_layer_osm,\r\n")
        f.write("    'Satellite': Esri_WorldImagery\r\n")
        f.write("};\r\n")
        f.write("\r\n")
        f.write("var overlayMaps = {};\r\n")
        f.write("\r\n")
        f.write("L.control.layers(baseMaps, overlayMaps).addTo(map);\r\n")
        f.write("\r\n")
        f.write("</script>\r\n")


@njit(cache=True, nogil=True)
def _resolve_pixel_indices(
    xm: np.ndarray,
    ym: np.ndarray,
    x0: float,
    y0: float,
    dx: float,
    dy: float,
    cosrot: float,
    sinrot: float,
    nr_levels: int,
    nm_all: np.ndarray,
    nm_start: np.ndarray,
    nm_len: np.ndarray,
    i0_lev: np.ndarray,
    nmax_lev: np.ndarray,
    mmax_lev: np.ndarray,
) -> np.ndarray:
    """Resolve (xm, ym) pixel coordinates to quadtree cell indices in parallel.

    Mirrors the multi-level index lookup in
    :py:meth:`HurrywaveQuadtreeGrid.get_indices_at_points`, but compiled
    with numba and parallelised across pixel rows via ``prange``. Pixels
    that fall outside every refinement level are returned as ``-999``.

    The per-level ``nm`` arrays (one sorted array of ``(m-1)*nmax + (n-1)``
    per level) are concatenated into ``nm_all`` and located via
    ``nm_start`` / ``nm_len`` to stay numba-friendly.
    """
    ny, nx = xm.shape
    indx = np.full((ny, nx), -999, dtype=np.int32)
    for i in range(ny):
        for j in range(nx):
            x = xm[i, j] - x0
            y = ym[i, j] - y0
            xg = x * cosrot - y * sinrot
            yg = x * sinrot + y * cosrot
            for ilev in range(nr_levels):
                dxr = dx / (2.0**ilev)
                dyr = dy / (2.0**ilev)
                xi_f = xg / dxr
                yi_f = yg / dyr
                if xi_f < 0.0 or yi_f < 0.0:
                    continue
                iind = int(xi_f)
                jind = int(yi_f)
                nmax_l = nmax_lev[ilev]
                mmax_l = mmax_lev[ilev]
                if iind >= mmax_l or jind >= nmax_l:
                    continue
                target = iind * nmax_l + jind
                start = nm_start[ilev]
                end = start + nm_len[ilev]
                lo = start
                hi = end
                while lo < hi:
                    mid = (lo + hi) // 2
                    if nm_all[mid] < target:
                        lo = mid + 1
                    else:
                        hi = mid
                if lo < end and nm_all[lo] == target:
                    indx[i, j] = i0_lev[ilev] + (lo - start)
    return indx


def _precompute_quadtree_lookup(quadtree_grid) -> dict:
    """Extract and flatten quadtree arrays needed by ``_resolve_pixel_indices``.

    Returns a dict of scalars / numpy arrays that together describe the
    per-refinement-level cell layout: ``x0``, ``y0``, ``dx``, ``dy``,
    ``cosrot``, ``sinrot``, ``nr_levels``, ``nm_all``, ``nm_start``,
    ``nm_len``, ``i0_lev``, ``nmax_lev``, ``mmax_lev``.
    """
    attrs = quadtree_grid.data.attrs
    x0 = float(attrs["x0"])
    y0 = float(attrs["y0"])
    dx = float(attrs["dx"])
    dy = float(attrs["dy"])
    rotation = float(attrs["rotation"])
    nr_levels = int(attrs["nr_levels"])

    cosrot = math.cos(-rotation * math.pi / 180.0)
    sinrot = math.sin(-rotation * math.pi / 180.0)

    level = quadtree_grid.data["level"].to_numpy()
    n = quadtree_grid.data["n"].to_numpy().astype(np.int64)
    m = quadtree_grid.data["m"].to_numpy().astype(np.int64)

    i0_lev = np.zeros(nr_levels, dtype=np.int64)
    nmax_lev = np.zeros(nr_levels, dtype=np.int64)
    mmax_lev = np.zeros(nr_levels, dtype=np.int64)
    nm_list = []
    for ilev in range(nr_levels):
        idx = np.where(level == ilev + 1)[0]
        if idx.size == 0:
            # Empty level — still need a placeholder so nm_start/nm_len line up.
            nm_list.append(np.empty(0, dtype=np.int64))
            continue
        i0_lev[ilev] = idx[0]
        mm = m[idx] - 1
        nn = n[idx] - 1
        nmax_l = int(nn.max()) + 1
        mmax_l = int(mm.max()) + 1
        nmax_lev[ilev] = nmax_l
        mmax_lev[ilev] = mmax_l
        # Cells within a level are stored in sorted (m, n) order in the
        # quadtree netcdf, so nm is naturally sorted and searchable.
        nm_list.append(mm * nmax_l + nn)

    nm_len = np.array([len(a) for a in nm_list], dtype=np.int64)
    nm_start = np.zeros(nr_levels, dtype=np.int64)
    if nr_levels > 1:
        nm_start[1:] = np.cumsum(nm_len[:-1])
    nm_all = (
        np.concatenate(nm_list).astype(np.int64)
        if nm_list
        else np.empty(0, dtype=np.int64)
    )

    return dict(
        x0=x0,
        y0=y0,
        dx=dx,
        dy=dy,
        cosrot=cosrot,
        sinrot=sinrot,
        nr_levels=nr_levels,
        nm_all=nm_all,
        nm_start=nm_start,
        nm_len=nm_len,
        i0_lev=i0_lev,
        nmax_lev=nmax_lev,
        mmax_lev=mmax_lev,
    )


def _resolve_elevation_list_from_catalog(
    data_catalog, res: Optional[float] = None, bbox=None
) -> List[dict]:
    """Build an elevation_list by resolving every source in a data catalog.

    Used as a fallback when no explicit ``elevation_list`` is supplied to
    the tiling workflows. Assumes all sources in ``data_catalog`` are
    raster elevation datasets (true in DelftDashboard, where the model's
    catalog is populated only with topobathy sources).
    """
    resolved: List[dict] = []
    for name in list(data_catalog.sources):
        try:
            zoom = (res, "meter") if res is not None else None
            da = data_catalog.get_rasterdataset(name, bbox=bbox, buffer=10, zoom=zoom)
            da.name = "elevation"
        except Exception:
            logger.warning(f"No data in domain for {name}, skipped.")
            continue
        resolved.append({"da": da})
    return resolved


def make_index_tiles(
    quadtree_grid,
    root: Union[str, Path],
    region: Optional[gpd.GeoDataFrame] = None,
    elevation_list: Optional[List[dict]] = None,
    z_range: Optional[List[float]] = None,
    zoom_range: Union[int, List[int]] = [0, 13],
    fmt: str = "png",
    write_html_viewer: bool = True,
    max_workers: Optional[int] = None,
    logger: logging.Logger = logger,
) -> None:
    """Create webmercator index tiles for a HurryWave quadtree grid.

    Index tiles map each pixel of a webmercator XYZ tile to the index of
    the corresponding quadtree cell (``-999`` where there is no match).
    Tiles are written to ``<root>/indices/<zoom>/<x>/<y>.<ext>``.

    When both ``elevation_list`` and ``z_range`` are supplied, pixel
    elevations are resampled from ``elevation_list`` (using the same
    merge logic as topobathy tiling) and any pixel whose elevation falls
    outside ``z_range`` is masked out (set to ``-999``). This is how you
    restrict the tiles to, e.g., wet cells only.

    Parameters
    ----------
    quadtree_grid : HurrywaveQuadtreeGrid
        Grid component providing ``get_indices_at_points``, ``exterior``,
        and ``model.crs``.
    root : Union[str, Path]
        Parent directory; tiles land in ``<root>/indices``.
    region : gpd.GeoDataFrame, optional
        GeoDataFrame defining the area for which tiles are generated.
        Defaults to ``quadtree_grid.exterior``.
    elevation_list : list of dict, optional
        Topobathy datasets (same format as
        :py:func:`hydromt_sfincs.workflows.tiling.create_topobathy_tiles`).
        Required when ``z_range`` is set. If ``None`` and ``z_range`` is
        set, falls back to ``quadtree_grid.model.data_catalog`` — every
        source in the catalog is treated as an elevation dataset.
    z_range : list of float, optional
        ``[zmin, zmax]`` pixel-elevation window. Pixels outside this
        window are masked out. Requires ``elevation_list``.
    zoom_range : Union[int, List[int]], optional
        Range of zoom levels, by default ``[0, 13]``. A single int is
        treated as ``[0, zoom_range]``.
    fmt : str, optional
        Output format: ``"png"`` (default) or ``"bin"`` (raw int32).
    write_html_viewer : bool, optional
        If True (default) and ``fmt == "png"``, write a Leaflet
        ``index.html`` alongside the tiles.
    max_workers : int, optional
        Number of worker threads used to render tiles concurrently.
        Defaults to ``os.cpu_count()``. Pass ``1`` to disable
        parallelism. The numba kernel is GIL-free so threads scale
        well for pure-index tiling; threads also help overlap file I/O
        when many small tiles are written.
    """
    if region is None:
        region = quadtree_grid.exterior

    # If no elevation_list is given but a z_range is active, fall back to the
    # model's data catalog (if populated). Explicit elevation_list always wins.
    if elevation_list is None and z_range is not None:
        data_catalog = getattr(quadtree_grid.model, "data_catalog", None)
        if data_catalog is not None and len(list(data_catalog.sources)) > 0:
            if isinstance(zoom_range, int):
                zr_max = zoom_range
            else:
                zr_max = zoom_range[1]
            res = 40075016.686 / 256 / 2**zr_max
            elevation_list = _resolve_elevation_list_from_catalog(
                data_catalog, res=res, bbox=quadtree_grid.model.bbox
            )

    if z_range is not None and (elevation_list is None or len(elevation_list) == 0):
        raise ValueError(
            "`z_range` requires `elevation_list` or a populated model data catalog."
        )

    index_path = os.path.join(root, "indices")
    npix = 256

    if fmt == "bin":
        extension = "dat"
    else:
        extension = fmt

    if isinstance(zoom_range, int):
        zoom_range = [0, zoom_range]

    # webmercator bounds of the region
    minx, miny, maxx, maxy = region.total_bounds
    transformer = Transformer.from_crs(region.crs.to_epsg(), 3857)
    if region.crs.is_geographic:
        minx, miny = map(
            max, zip(transformer.transform(miny, minx), [-20037508.34] * 2)
        )
        maxx, maxy = map(min, zip(transformer.transform(maxy, maxx), [20037508.34] * 2))
    else:
        minx, miny = map(
            max, zip(transformer.transform(minx, miny), [-20037508.34] * 2)
        )
        maxx, maxy = map(min, zip(transformer.transform(maxx, maxy), [20037508.34] * 2))

    # transformer from webmercator back to the model CRS for index lookup
    transformer_inv = Transformer.from_crs(
        3857, quadtree_grid.model.crs, always_xy=True
    )

    # Precompute the grid-lookup tables once; the kernel is called per tile.
    lut = _precompute_quadtree_lookup(quadtree_grid)

    def _process_tile(izoom, transform, col, row):
        """Render a single tile. Safe to call concurrently from worker threads."""
        zoom_path = os.path.join(index_path, str(izoom))
        file_name = os.path.join(zoom_path, str(col), f"{row}.{extension}")

        # pixel centres in webmercator
        x = np.arange(0, npix) + 0.5
        y = np.arange(0, npix) + 0.5
        x3857, y3857 = transform * (x, y)
        x3857_2d, y3857_2d = np.meshgrid(x3857, y3857)

        # resolve cell indices in model CRS via the nogil numba kernel
        xm, ym = transformer_inv.transform(x3857_2d, y3857_2d)
        ind = _resolve_pixel_indices(
            np.ascontiguousarray(xm, dtype=np.float64),
            np.ascontiguousarray(ym, dtype=np.float64),
            lut["x0"],
            lut["y0"],
            lut["dx"],
            lut["dy"],
            lut["cosrot"],
            lut["sinrot"],
            lut["nr_levels"],
            lut["nm_all"],
            lut["nm_start"],
            lut["nm_len"],
            lut["i0_lev"],
            lut["nmax_lev"],
            lut["mmax_lev"],
        )

        # optionally mask pixels by elevation sampled from elevation_list
        if elevation_list is not None and z_range is not None:
            da_dep = xr.DataArray(
                np.full([npix, npix], np.nan, dtype=np.float32),
                coords={"y": y3857, "x": x3857},
                dims=["y", "x"],
            )
            da_dep.raster.set_crs(3857)
            # Tell rioxarray which dims are spatial so it can derive the
            # transform from the coordinate arrays (avoids
            # NotGeoreferencedWarning during reprojection).
            da_dep.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
            da_dep = merge_multi_dataarrays(
                da_list=elevation_list,
                da_like=da_dep,
            )
            z = da_dep.values
            out_of_range = np.isnan(z) | (z < z_range[0]) | (z > z_range[1])
            ind[out_of_range] = -999

        if np.any(ind >= 0):
            os.makedirs(os.path.join(zoom_path, str(col)), exist_ok=True)
            if fmt == "bin":
                with open(file_name, "wb") as fid:
                    fid.write(ind.astype(np.int32))
            elif fmt == "png":
                ind = ind.copy()
                ind[ind == -999] = 0
                int2png(ind, file_name)

    # Fan the tiles out across a thread pool. The numba kernel releases the
    # GIL (nogil=True) and the per-tile file I/O is independent, so threads
    # scale well even though Python code between kernel calls still takes
    # the GIL.
    max_workers = max_workers or os.cpu_count() or 1
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for izoom in range(zoom_range[0], zoom_range[1] + 1):
            logger.debug("Processing zoom level " + str(izoom))
            for transform, col, row in tile_window(izoom, minx, miny, maxx, maxy):
                futures.append(
                    executor.submit(_process_tile, izoom, transform, col, row)
                )
        # .result() propagates any exception raised in a worker thread.
        for future in futures:
            future.result()

    if write_html_viewer and fmt == "png":
        os.makedirs(index_path, exist_ok=True)
        write_html(
            os.path.join(index_path, "index.html"),
            title="Index tiles",
            legend_title="Cell indices",
            max_native_zoom=zoom_range[1],
        )


def deg2num(lat_deg, lon_deg, zoom):
    """Convert lat/lon to webmercator tile number"""
    lat_rad = math.radians(lat_deg)
    n = 2**zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(-lat_rad)) / math.pi) / 2.0 * n)
    return (xtile, ytile)


def num2deg(xtile, ytile, zoom):
    """Convert webmercator tile number to lat/lon"""
    n = 2**zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(-lat_rad)
    return (lat_deg, lon_deg)


def rgba2int(rgba):
    """Convert rgba tuple to int"""
    r, g, b, a = rgba
    return (r * 256**3) + (g * 256**2) + (b * 256) + a


def int2rgba(int_val):
    """Convert int to rgba tuple"""
    r = (int_val // 256**3) % 256
    g = (int_val // 256**2) % 256
    b = (int_val // 256) % 256
    a = int_val % 256
    return (r, g, b, a)


def elevation2rgb(val):
    """Convert elevation to rgb tuple"""
    val += 32768
    r = np.floor(val / 256)
    g = np.floor(val % 256)
    b = np.floor((val - np.floor(val)) * 256)

    return (r, g, b)


def rgb2elevation(r, g, b):
    """Convert rgb tuple to elevation"""
    val = (r * 256 + g + b / 256) - 32768
    return val


def png2int(png_file):
    """Convert png to int array"""
    # Open the PNG image
    image = Image.open(png_file)

    # Convert the image to RGBA mode if it's not already in RGBN mode
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    # Get the pixel data from the image
    pixel_data = list(image.getdata())

    # Convert RGBA values to unique integers
    val = []
    for rgba in pixel_data:
        val.append(rgba2int(rgba))

    return val


def int2png(val, png_file):
    """Convert int array to png"""
    # Convert index integers to RGBA values
    rgba = np.zeros((256 * 256, 4), "uint8")
    r, g, b, a = int2rgba(val)

    rgba[:, 0] = r.flatten()
    rgba[:, 1] = g.flatten()
    rgba[:, 2] = b.flatten()
    rgba[:, 3] = a.flatten()

    rgba = rgba.reshape([256, 256, 4])

    # Create PIL Image from RGB values and save as PNG
    img = Image.fromarray(rgba)
    img.save(png_file)


def png2elevation(png_file):
    """Convert png to elevation array based on terrarium interpretation"""
    img = Image.open(png_file)
    arr = np.array(img.convert("RGB"))
    # Convert RGB values to elevation values
    elevations = np.apply_along_axis(rgb2elevation, 2, arr)
    return elevations


def elevation2png(val, png_file):
    """Convert elevation array to png using terrarium interpretation"""
    rgb = np.zeros((256 * 256, 3), "uint8")
    r, g, b = elevation2rgb(val)

    rgb[:, 0] = r.values.flatten()
    rgb[:, 1] = g.values.flatten()
    rgb[:, 2] = b.values.flatten()

    rgb = rgb.reshape([256, 256, 3])

    # Create PIL Image from RGB values and save as PNG
    img = Image.fromarray(rgb)
    img.save(png_file)


def tile_window(zl, minx, miny, maxx, maxy):
    """Window generator for a given zoom level and bounding box"""
    dxy = (20037508.34 * 2) / (2**zl)
    # Origin displacement
    odx = np.floor(abs(-20037508.34 - minx) / dxy)
    ody = np.floor(abs(20037508.34 - maxy) / dxy)

    # Set the new origin
    minx = -20037508.34 + odx * dxy
    maxy = 20037508.34 - ody * dxy

    # Create window generator
    lu = product(np.arange(minx, maxx, dxy), np.arange(maxy, miny, -dxy))
    for l, u in lu:
        col = round(odx + (l - minx) / dxy)
        row = round(ody + (maxy - u) / dxy)
        yield Affine(dxy / 256, 0, l, 0, -dxy / 256, u), col, row

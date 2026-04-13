"""Datashader-based map-overlay rendering for HurryWave quadtree grids."""

import logging
import os
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer

__all__ = ["make_edge_dataframe", "make_map_overlay"]

logger = logging.getLogger(__name__)

# optional dependency
try:
    import datashader.transfer_functions as tf
    from datashader import Canvas
    from datashader.utils import export_image

    HAS_DATASHADER = True
except ImportError:
    HAS_DATASHADER = False


def make_edge_dataframe(ugrid, source_crs: CRS) -> pd.DataFrame:
    """Build a datashader-ready edge-coordinate DataFrame in webmercator.

    Produces one row per mesh edge with its two endpoints reprojected
    from ``source_crs`` to EPSG:3857. If the source CRS is geographic and
    any longitude exceeds 180, negative webmercator x values are shifted
    by one world width so the line doesn't wrap back across the dateline.

    Parameters
    ----------
    ugrid : xugrid.Ugrid2d
        Source mesh exposing ``edge_node_coordinates`` (shape
        ``(n_edges, 2, 2)``).
    source_crs : pyproj.CRS
        CRS of the mesh coordinates.

    Returns
    -------
    pd.DataFrame
        Columns ``x1, y1, x2, y2`` in EPSG:3857.
    """
    x1 = ugrid.edge_node_coordinates[:, 0, 0]
    x2 = ugrid.edge_node_coordinates[:, 1, 0]
    y1 = ugrid.edge_node_coordinates[:, 0, 1]
    y2 = ugrid.edge_node_coordinates[:, 1, 1]

    cross_dateline = False
    if source_crs.is_geographic and (np.max(x1) > 180.0 or np.max(x2) > 180.0):
        cross_dateline = True

    transformer = Transformer.from_crs(source_crs, 3857, always_xy=True)
    x1, y1 = transformer.transform(x1, y1)
    x2, y2 = transformer.transform(x2, y2)
    if cross_dateline:
        x1[x1 < 0] += 40075016.68557849
        x2[x2 < 0] += 40075016.68557849

    return pd.DataFrame(dict(x1=x1, y1=y1, x2=x2, y2=y2))


def make_map_overlay(
    dataframe: pd.DataFrame,
    file_name: Union[str, Path],
    xlim: Optional[List[float]] = None,
    ylim: Optional[List[float]] = None,
    color: str = "black",
    width: int = 800,
) -> bool:
    """Render a PNG map overlay of mesh edges using datashader.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Edge DataFrame as produced by :py:func:`make_edge_dataframe`
        (columns ``x1, y1, x2, y2`` in EPSG:3857).
    file_name : str or Path
        Output image path. The extension is stripped; datashader appends
        ``.png``.
    xlim : list of float, optional
        Longitude limits ``[xmin, xmax]`` in EPSG:4326.
    ylim : list of float, optional
        Latitude limits ``[ymin, ymax]`` in EPSG:4326.
    color : str, optional
        Line colour, by default ``"black"``.
    width : int, optional
        Output image width in pixels; height is derived from the aspect
        ratio of ``xlim`` / ``ylim``. Defaults to ``800``.

    Returns
    -------
    bool
        ``True`` if the overlay was written, ``False`` if datashader is
        unavailable, the dataframe is empty, or rendering raised an
        exception.
    """
    if not HAS_DATASHADER:
        logger.warning("Datashader is not available. Please install datashader.")
        return False

    if dataframe is None or dataframe.empty:
        return False

    try:
        transformer = Transformer.from_crs(4326, 3857, always_xy=True)
        xl0, yl0 = transformer.transform(xlim[0], ylim[0])
        xl1, yl1 = transformer.transform(xlim[1], ylim[1])
        if xl0 > xl1:
            xl1 += 40075016.68557849
        xlim = [xl0, xl1]
        ylim = [yl0, yl1]
        ratio = (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])
        height = int(width * ratio)

        cvs = Canvas(x_range=xlim, y_range=ylim, plot_height=height, plot_width=width)
        agg = cvs.line(dataframe, x=["x1", "x2"], y=["y1", "y2"], axis=1)
        img = tf.shade(agg, cmap=color)

        path_dir = os.path.dirname(file_name) or os.getcwd()
        name = os.path.splitext(os.path.basename(file_name))[0]
        export_image(img, name, export_path=path_dir)
        return True
    except Exception:
        return False

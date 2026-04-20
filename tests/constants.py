"""Shared constants and helpers for hydromt_hurrywave tests."""

from datetime import datetime

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

EPSG = 32631
DX = DY = 1000.0
MMAX = NMAX = 10
X0 = Y0 = 0.0
TREF = datetime(2020, 1, 1)
TSTART = datetime(2020, 1, 1)
TSTOP = datetime(2020, 1, 2)
N_TIMES = 12
N_BND_POINTS = 3


def make_time_index(n=N_TIMES):
    return pd.date_range(TSTART, TSTOP, periods=n)


def make_wave_df(n_times=N_TIMES, n_points=N_BND_POINTS, hs=1.5, tp=8.0, wd=270.0, ds=30.0):
    times = make_time_index(n_times)
    return pd.DataFrame(
        {i: [hs] * n_times for i in range(n_points)},
        index=times,
    )


def make_boundary_gdf(n_points=N_BND_POINTS, crs=EPSG):
    points = [Point(X0, Y0 + i * DY * 2) for i in range(n_points)]
    names = [f"bnd_{i:04d}" for i in range(n_points)]
    return gpd.GeoDataFrame({"name": names}, geometry=points, crs=crs)

#%%

import os
from pprint import pprint
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

import hydromt
from hydromt_hurrywave import HurrywaveModel


#%% Initialize model
# Specify root_folder and logger_name
root_folder  = Path(r'p:\11212696-fhics-bes\1_models\HurryWave\01_modelsetup\hurrywave_caribbean_v4')

# initialize model
hw = HurrywaveModel(
    root=root_folder,
    mode="w+",
    data_libs = ['deltares_data'],
)

region_gdf = gpd.read_file(r"p:\11212696-fhics-bes\1_models\HurryWave\00_data\caribbean_inlcude.geojson")
refinement_polygons = gpd.read_file(r"p:\11212696-fhics-bes\1_models\HurryWave\00_data\refinement_1_v2.geojson").explode()

gdf_1 = gpd.GeoDataFrame(geometry=refinement_polygons.buffer(0.04))
gdf_1["refinement_level"] = 1   

gdf_2 = refinement_polygons
gdf_2["refinement_level"] = 2  

gdf_all = gpd.GeoDataFrame(pd.concat([gdf_1, gdf_2], ignore_index=True), crs=gdf_1.crs)
# #%%

hw.quadtree_grid.create_from_region(
    region = {"geom": region_gdf},
    res = 0.2,
    refinement_polygons = gdf_all,
    crs = 4326
)

#%%

hw.quadtree_grid.data.mask.ugrid.to_geodataframe().to_file(root_folder / "quadtree_grid.gpkg", driver="GPKG")

# %%
elevation_list = [{"elevation": r"p:\11212696-fhics-bes\0_data\destination_earth\caribbean_2022_mean.tif"},
                  {"elevation": "gebco"}]

hw.quadtree_elevation.create(
        elevation_list = elevation_list,
        )

#%%
hw.quadtree_mask.create(zmax = 0.0)

#%% Add observation points
gdf_obs = gpd.read_file(r"p:\11212696-fhics-bes\1_models\HurryWave\00_data\observation_points.geojson")
hw.observation_points.add_points(gdf_obs)
#%%
hw.config.update(
    {
        "tref": "20170917 000000",
        "tstart": "20170917 000000",
        "tstop": "20170921 000000",
        "amufile": "hurrywave.amu",
        "amvfile": "hurrywave.amv",
        "outputformat": "net"
    }
)
#%% Add wave blocking
hw.wave_blocking.create(
    elevation_list = elevation_list,
)

#%%
hw.write()

# %%

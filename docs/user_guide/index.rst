User guide
==========

Model components
----------------

The HurryWave model is built from several components, each responsible for a
specific part of the model setup:

**Configuration** (``config``)
   Reads and writes ``hurrywave.inp``, the main HurryWave settings file.
   See :class:`~hydromt_hurrywave.components.config.HurrywaveConfig`.

**Quadtree grid** (``quadtree_grid``)
   Generates an adaptive quadtree mesh with refinement controlled by bathymetry
   gradients or user-defined polygons.
   See :class:`~hydromt_hurrywave.components.quadtree.HurrywaveQuadtreeGrid`.

**Elevation** (``quadtree_elevation``)
   Interpolates bed elevation onto the quadtree mesh from one or more
   bathymetry datasets.
   See :class:`~hydromt_hurrywave.components.quadtree.HurrywaveQuadtreeElevation`.

**Mask** (``quadtree_mask``)
   Defines cell activity: 0 = inactive, 1 = active, 2 = open boundary.
   See :class:`~hydromt_hurrywave.components.quadtree.HurrywaveQuadtreeMask`.

**Wave blocking** (``wave_blocking``)
   Directional wave-blocking coefficients stored as a separate netCDF file.
   See :class:`~hydromt_hurrywave.components.quadtree.HurrywaveWaveBlocking`.

**Boundary conditions** (``boundary_conditions``)
   Wave boundary conditions: timeseries (Hs, Tp, Wd, Ds) or 2-D spectra.
   See :class:`~hydromt_hurrywave.components.forcing.HurrywaveBoundaryConditions`.

**Wind forcing** (``wind``)
   Uniform wind time series or gridded u/v netCDF fields.
   See :class:`~hydromt_hurrywave.components.forcing.HurrywaveWind`.

**Observation points** (``observation_points``)
   Bulk-parameter output locations.
   See :class:`~hydromt_hurrywave.components.geometries.HurrywaveObservationPoints`.

**Spectral observation points** (``observation_points_spectra``)
   Spectral output locations.
   See :class:`~hydromt_hurrywave.components.geometries.HurrywaveObservationPointsSpectra`.

Grid setup
----------

HurryWave uses a quadtree grid exclusively. The grid is built by specifying a
bounding box and a base resolution, then optionally refining cells based on
bathymetry gradients or user-supplied polygons.

Boundary conditions
-------------------

Boundary conditions can be supplied as:

- **Timeseries**: uniform or spatially varying Hs, Tp, wind direction, and
  directional spreading at each boundary point.
- **Spectra**: full 2-D energy-density spectra at each boundary point.

Forcing
-------

Wind forcing supports two modes:

- **Uniform**: a single time series of wind speed and direction applied everywhere.
- **Gridded**: spatially varying u- and v-component fields on a regular grid,
  typically from reanalysis or forecast products (e.g. ERA5).

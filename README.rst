hydromt_hurrywave
=================

HydroMT plugin for the `HurryWave <https://github.com/deltares-research/cht_hurrywave>`_
spectral wave model.

HurryWave is an adaptive quadtree spectral wave model developed at Deltares.
This plugin wraps its I/O and parameterisation in the
`HydroMT <https://deltares.github.io/hydromt>`_ framework, following the same
component-based pattern used by
`hydromt_sfincs <https://github.com/deltares-research/hydromt_sfincs>`_.

Installation
------------

.. code-block:: bash

    pip install -e .

Quick start
-----------

.. code-block:: python

    from hydromt_hurrywave import HurrywaveModel

    # Read an existing model
    mod = HurrywaveModel(root="path/to/model", mode="r")
    mod.read()

    # Inspect the grid
    print(mod.quadtree_grid.crs)
    print(mod.quadtree_mask.has_open_boundaries)

    # Access boundary conditions
    bc = mod.boundary_conditions
    print(bc.forcing)   # "timeseries" or "spectra"
    print(bc.gdf)       # boundary point locations

    # Write a modified model
    mod2 = HurrywaveModel(root="path/to/new_model", mode="w")
    # ... set components ...
    mod2.write()

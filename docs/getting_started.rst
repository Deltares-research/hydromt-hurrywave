Getting started
===============

Installation
------------

Install from source:

.. code-block:: bash

   pip install git+https://github.com/Deltares/hydromt_hurrywave.git

For development:

.. code-block:: bash

   git clone https://github.com/Deltares/hydromt_hurrywave.git
   cd hydromt_hurrywave
   pip install -e ".[dev]"

Requirements
~~~~~~~~~~~~

- Python >= 3.10
- `HydroMT <https://deltares.github.io/hydromt/>`_
- xugrid, geopandas, xarray, numpy

Quick start
-----------

Build a HurryWave model from a configuration file and a data catalog:

.. code-block:: python

   from hydromt_hurrywave import HurrywaveModel

   model = HurrywaveModel(root="my_model", mode="w")
   model.build(
       region={"bbox": [5.0, 52.0, 6.0, 53.0]},
       opt={"setup_config": {}, "setup_quadtree": {"resolution": 500}},
   )

Or use the HydroMT CLI:

.. code-block:: bash

   hydromt build hurrywave my_model -r "{'bbox': [5.0, 52.0, 6.0, 53.0]}" -i config.yml -d data_catalog.yml

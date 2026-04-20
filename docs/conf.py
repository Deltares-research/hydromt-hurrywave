"""Sphinx configuration for hydromt_hurrywave."""

project = "hydromt_hurrywave"
copyright = "2024, Deltares"
author = "Maarten van Ormondt"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build"]

html_theme = "sphinx_rtd_theme"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "hydromt": ("https://deltares.github.io/hydromt/latest/", None),
}

napoleon_numpy_docstring = True
autodoc_member_order = "bysource"

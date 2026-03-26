"""HydroMT plugin for the HurryWave wave model."""

from os.path import abspath, dirname, join

from .hurrywave import *

DATADIR = join(dirname(abspath(__file__)), "data")

__version__ = "0.1.0"

"""Spatiotemporal modeling of spatial transcriptomics
"""

from .get_version import get_version

__version__ = get_version(__file__)
del get_version

from . import cs, dd, io, pl, pp, tdr, tl, sample_data
from .configuration import config
from .data_io import *

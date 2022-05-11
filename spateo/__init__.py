"""A complete solution of spatialtemporal dynamics analyses toolkit of single
cell spatial transcriptomics.
"""

from .get_version import get_version

__version__ = get_version(__file__)
del get_version

from . import io, pl, pp, sg, tl
from .configuration import config
from .data_io import *
from .sample_data import *

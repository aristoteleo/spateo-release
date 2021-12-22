"""A complete solution of spatialtemporal dynamics analyses toolkit of single cell spatial transcriptomics
"""

from .get_version import get_version

__version__ = get_version(__file__)
del get_version

from .io import *
from . import pp
from . import tl
from . import pl

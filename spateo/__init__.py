"""A complete solution of spatialtemporal dynamics analyses toolkit of single
cell spatial transcriptomics.

Todo:
    * Fix warnings during import
"""

from .get_version import get_version

__version__ = get_version(__file__)
del get_version

from . import io, pl, pp, tl

"""Spatiotemporal modeling of spatial transcriptomics
"""

from .get_version import get_version

__version__ = get_version(__file__)
del get_version

# Use lazy loading to speed up imports
from ._lazy_loader import LazyLoader

# Lazy load submodules
align = LazyLoader("spateo.align", globals())
cs = LazyLoader("spateo.cs", globals())
dd = LazyLoader("spateo.dd", globals())
io = LazyLoader("spateo.io", globals())
pl = LazyLoader("spateo.pl", globals())
pp = LazyLoader("spateo.pp", globals())
sample_data = LazyLoader("spateo.sample_data", globals())
svg = LazyLoader("spateo.svg", globals())
tdr = LazyLoader("spateo.tdr", globals())
tl = LazyLoader("spateo.tl", globals())

# Lazy load config
from ._lazy_loader import LazyAttribute
config = LazyAttribute("spateo.configuration", "config")

# These are simple re-exports from anndata, keep them as direct imports
from .data_io import *

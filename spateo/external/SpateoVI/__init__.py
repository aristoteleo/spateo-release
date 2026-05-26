"""SpateoVI — spatially-aware single-cell VAE for MERFISH / Xenium data.

Public API:

* :class:`SpateoVI`         — gene-only spatial VAE (GATv2 + adversarial batch
                               removal + spatial smoothness).
* :class:`SpateoVIProtein`  — SpateoVI plus an optional protein (antibody)
                               modality with TotalVI-style mosaic mask.

Backwards-compatibility aliases (the older spateo ``MERFISHVI`` module pointed
at the same kind of model; keep code that used those names working):

* ``MERFISHVI``  is an alias for :class:`SpateoVI`.
* ``SpatialVI``  is an alias for :class:`SpateoVI`.
* ``MERFISHVI3`` is an alias for :class:`SpateoVIProtein`.
"""
from ._model import SpateoVI, SpatialVI
from ._model_protein import SpateoVIProtein

# Backwards-compatibility aliases.
MERFISHVI = SpateoVI
MERFISHVI3 = SpateoVIProtein

__all__ = [
    "SpateoVI",
    "SpateoVIProtein",
    # legacy aliases
    "MERFISHVI",
    "SpatialVI",
    "MERFISHVI3",
]

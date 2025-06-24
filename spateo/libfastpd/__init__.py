try:
    from .fastpd import *

    # If you want to keep a reference: fastpd = __import__('spateo.libfastpd.fastpd', fromlist=['fastpd']).fastpd
    __all__ = ["fastpd"]  # List the class or function names you exposed in init_fastpd.cpp here
except ImportError as e:
    import warnings

    warnings.warn(f"Cannot import fastpd extension: {e}")
    __all__ = []

# spateo/libfastpd/fastpd.py

"""
This file serves as a "wrapper" that exposes symbols from the compiled C++ libfastpd extension module under the name fastpd.
"""

# The following line loads the libfastpd extension from the same directory:
from .libfastpd import *

# If you want "from spateo.libfastpd.fastpd import FastPD" to work directly, you can list it in __all__
__all__ = [name for name in dir() if not name.startswith("_")]

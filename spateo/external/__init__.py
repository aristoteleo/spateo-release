from . import lack


# Lazy loading for MERFISHVI to avoid importing scvi unless actually needed
class _MERFISHVILazyLoader:
    """Lazy loader for MERFISHVI module to make scvi-tools optional."""

    def __init__(self):
        self._module = None
        self._import_attempted = False
        self._import_error = None

    def _try_import(self):
        """Attempt to import MERFISHVI module."""
        if self._import_attempted:
            if self._import_error:
                raise self._import_error
            return self._module

        self._import_attempted = True
        try:
            import importlib
            self._module = importlib.import_module(".MERFISHVI", package="spateo.external")
            return self._module
        except ImportError as e:
            # Check if the error is related to scvi or other required dependencies
            if "scvi" in str(e) or "torch_geometric" in str(e) or "torch" in str(e) or "anndata.io" in str(e):
                self._import_error = ImportError(
                    "MERFISHVI requires additional dependencies that are not installed. "
                    "Please install them with: pip install scvi-tools torch torch-geometric\n"
                    f"Original error: {str(e)}"
                )
                raise self._import_error
            else:
                # Re-raise if it's a different import error
                raise

    def __getattr__(self, name):
        """Get attribute from the lazily loaded MERFISHVI module."""
        module = self._try_import()
        return getattr(module, name)

    def __dir__(self):
        """Return available attributes in the MERFISHVI module."""
        try:
            module = self._try_import()
            return dir(module)
        except ImportError:
            return []

    def __repr__(self):
        if self._module is not None:
            return f"<MERFISHVI module (loaded)>"
        elif self._import_error is not None:
            return f"<MERFISHVI module (unavailable: missing dependencies)>"
        else:
            return f"<MERFISHVI module (not loaded)>"


# Create lazy loader instance
MERFISHVI = _MERFISHVILazyLoader()

# Check availability by attempting import without raising error
_MERFISHVI_AVAILABLE = None  # Unknown until first access


def is_merfishvi_available():
    """
    Check if MERFISHVI and its dependencies are available.

    Returns
    -------
    bool
        True if MERFISHVI can be imported successfully, False otherwise.
    """
    global _MERFISHVI_AVAILABLE
    if _MERFISHVI_AVAILABLE is None:
        # Check availability by attempting import
        try:
            MERFISHVI._try_import()
            _MERFISHVI_AVAILABLE = True
        except ImportError:
            _MERFISHVI_AVAILABLE = False
    return _MERFISHVI_AVAILABLE


def get_merfishvi_requirements():
    """
    Get the list of required packages for MERFISHVI functionality.

    Returns
    -------
    list
        List of required package names.
    """
    return ["scvi-tools", "torch", "torch-geometric", "scipy", "sklearn"]

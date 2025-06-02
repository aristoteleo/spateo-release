from . import lack

# Try to import MERFISHVI only if scvi dependencies are available
try:
    from . import MERFISHVI

    _MERFISHVI_AVAILABLE = True
except ImportError as e:
    # Check if the error is related to scvi or other required dependencies
    if "scvi" in str(e) or "torch_geometric" in str(e) or "torch" in str(e):
        import warnings

        warnings.warn(
            "MERFISHVI could not be imported due to missing dependencies. "
            "To use MERFISHVI functionality, please install the required packages: "
            "pip install scvi-tools torch torch-geometric",
            ImportWarning,
            stacklevel=2,
        )
        _MERFISHVI_AVAILABLE = False
        # Create a dummy MERFISHVI module that raises an informative error when accessed
        class _MERFISHVIPlaceholder:
            def __getattr__(self, name):
                raise ImportError(
                    f"Cannot access {name} from MERFISHVI. "
                    "MERFISHVI requires additional dependencies. "
                    "Please install them with: pip install scvi-tools torch torch-geometric"
                )

        MERFISHVI = _MERFISHVIPlaceholder()
    else:
        # Re-raise if it's a different import error
        raise


def is_merfishvi_available():
    """
    Check if MERFISHVI and its dependencies are available.

    Returns
    -------
    bool
        True if MERFISHVI can be imported successfully, False otherwise.
    """
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

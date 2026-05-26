from . import lack


# Lazy loading for SpateoVI to avoid importing scvi unless actually needed.
# scvi-tools is heavyweight + brings torch, lightning, pyro, etc., so we only
# pull it in when the user actually touches ``spateo.external.SpateoVI``.
class _SpateoVILazyLoader:
    """Lazy loader for the SpateoVI module to keep scvi-tools optional."""

    def __init__(self, dotted_path: str = ".SpateoVI"):
        self._dotted_path = dotted_path
        self._module = None
        self._import_attempted = False
        self._import_error = None

    def _try_import(self):
        if self._import_attempted:
            if self._import_error:
                raise self._import_error
            return self._module

        self._import_attempted = True
        try:
            import importlib

            self._module = importlib.import_module(self._dotted_path, package="spateo.external")
            return self._module
        except ImportError as e:
            if any(dep in str(e) for dep in ("scvi", "torch_geometric", "torch", "anndata.io")):
                self._import_error = ImportError(
                    "SpateoVI requires additional dependencies that are not installed. "
                    "Please install them with: pip install scvi-tools torch torch-geometric\n"
                    f"Original error: {str(e)}"
                )
                raise self._import_error
            raise

    def __getattr__(self, name):
        return getattr(self._try_import(), name)

    def __dir__(self):
        try:
            return dir(self._try_import())
        except ImportError:
            return []

    def __repr__(self):
        if self._module is not None:
            return "<SpateoVI module (loaded)>"
        if self._import_error is not None:
            return "<SpateoVI module (unavailable: missing dependencies)>"
        return "<SpateoVI module (not loaded)>"


# Public name.
SpateoVI = _SpateoVILazyLoader(".SpateoVI")
# Backwards-compatibility alias for the old spateo MERFISHVI namespace.
MERFISHVI = SpateoVI


_SPATEOVI_AVAILABLE = None


def is_spateovi_available() -> bool:
    """Return True iff SpateoVI and its dependencies can be imported."""
    global _SPATEOVI_AVAILABLE
    if _SPATEOVI_AVAILABLE is None:
        try:
            SpateoVI._try_import()
            _SPATEOVI_AVAILABLE = True
        except ImportError:
            _SPATEOVI_AVAILABLE = False
    return _SPATEOVI_AVAILABLE


def get_spateovi_requirements() -> list[str]:
    """Required packages for SpateoVI functionality."""
    return ["scvi-tools", "torch", "torch-geometric", "scipy", "sklearn"]


# Legacy function-name aliases (so `is_merfishvi_available()` keeps working).
is_merfishvi_available = is_spateovi_available
get_merfishvi_requirements = get_spateovi_requirements

import pyvista as pv
from pyvista import Plotter

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


def _interactive_plotter() -> Plotter:
    """Create an interactive window for using widgets."""

    plotter = pv.Plotter(
        off_screen=False,
        window_size=(1024, 768),
        notebook=False,
        lighting="light_kit",
    )

    plotter.background_color = "white"
    plotter.add_camera_orientation_widget()

    return plotter

import pyvista as pv
from pyvista import Plotter

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


def _interactive_plotter(message=True) -> Plotter:
    """Create an interactive window for using widgets."""

    plotter = pv.Plotter(
        off_screen=False,
        window_size=(1024, 768),
        notebook=False,
        lighting="light_kit",
    )

    plotter.camera_position = "iso"
    plotter.background_color = "white"
    plotter.add_camera_orientation_widget()
    if message is True:
        plotter.add_text(
            "Please double-click the camera orientation widget in the upper right corner first.",
            font_size=15,
            color="black",
            font="arial",
        )

    return plotter

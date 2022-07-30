"""
This file implements the LiveWire segmentation algorithm. The code is ported from:
 1. https://github.com/pdyban/livewire: LiveWireSegmentation and compute_shortest_path functions/class.
 2. https://github.com/Usama3627/live-wire: include the _compute_graph and compute_shortest_path functions.
"""

import math
from itertools import cycle
from typing import List, Optional, Tuple

import numpy as np

from ..logging import logger_manager as lm


class LiveWireSegmentation(object):
    def __init__(self, image: Optional = None, smooth_image: bool = False, threshold_gradient_image: bool = False):
        super(LiveWireSegmentation, self).__init__()

        # init internal containers

        # container for input image
        self._image = None

        # container for the gradient image
        self.edges = None

        # stores the image as an undirected graph for shortest path search
        self.G = None

        # init parameters

        # should smooth the original image using bilateral smoothing filter
        self.smooth_image = smooth_image

        # should use the thresholded gradient image for shortest path computation
        self.threshold_gradient_image = threshold_gradient_image

        # init image

        # store image and compute the gradient image
        self.image = image

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, value):
        self._image = value

        if self._image is not None:
            if self.smooth_image:
                self._smooth_image()

            self._compute_gradient_image()

            if self.threshold_gradient_image:
                self._threshold_gradient_image()

            self._compute_graph()

        else:
            self.edges = None
            self.G = None

    def _smooth_image(self):
        from skimage import restoration

        self._image = restoration.denoise_bilateral(self.image)

    def _compute_gradient_image(self):
        from skimage import filters

        self.edges = filters.scharr(self._image)

    def _threshold_gradient_image(self):
        from skimage.filters import threshold_otsu

        threshold = threshold_otsu(self.edges)
        self.edges = self.edges > threshold
        self.edges = self.edges.astype(float)

    def _compute_graph(self):

        try:
            from dijkstar import Graph
        except ImportError:
            raise ImportError(
                "You need to install the package `dijkstar`." "\nInstall dijkstar via `pip install --upgrade dijkstar`"
            )

        vertex = self.edges
        h, w = self.edges.shape[1::-1]

        graph = Graph(undirected=True)

        # Iterating over an image and avoiding boundaries
        for i in range(1, w - 1):
            for j in range(1, h - 1):
                G_x = float(vertex[i, j]) - float(vertex[i, j + 1])  # Center - right
                G_y = float(vertex[i, j]) - float(vertex[i + 1, j])  # Center - bottom
                G = np.sqrt((G_x) ** 2 + (G_y) ** 2)
                if G_x > 0 or G_x < 0:
                    theeta = math.atan(G_y / G_x)
                else:
                    theeta = 0
                # Theeta is rotated in clockwise direction (90 degrees) to align with edge
                theeta_a = theeta + math.pi / 2
                G_x_a = abs(G * math.cos(theeta_a)) + 0.00001
                G_y_a = abs(G * math.sin(theeta_a)) + 0.00001

                # Strongest Edge will have lowest weights
                W_x = 1 / G_x_a
                W_y = 1 / G_y_a

                # Assigning weights
                graph.add_edge((i, j), (i, j + 1), W_x)  # W_x is given to right of current vertex
                graph.add_edge((i, j), (i + 1, j), W_y)  # W_y is given to bottom of current vertex

        self.G = graph

    def compute_shortest_path(self, startPt, endPt):
        try:
            from dijkstar import find_path
        except ImportError:
            raise ImportError(
                "You need to install the package `dijkstar`." "\nInstall dijkstar via `pip install --upgrade dijkstar`"
            )

        if self.image is None:
            raise AttributeError("Load an image first!")

        path = find_path(self.G, startPt, endPt)[0]

        return path


def compute_shortest_path(image: np.ndarray, startPt: Tuple[float, float], endPt: Tuple[float, float]) -> List:
    """Inline function for easier computation of shortest_path in an image.
    This function will create a new instance of LiveWireSegmentation class every time it is called, calling for a
    recomputation of the gradient image and the shortest path graph. If you need to compute the shortest path in one
    image more than once, use the class-form initialization instead.

    Args:
        image: image on which the shortest path should be computed
        startPt: starting point for path computation
        endPt: target point for path computation

    Returns:
        path: shortest path as a list of tuples (x, y), including startPt and endPt
    """

    lm.main_info("Build LiveWireSegmentation object")
    algorithm = LiveWireSegmentation(image)

    lm.main_info("run compute_shortest_path to identify the shortest path")
    path = algorithm.compute_shortest_path(startPt, endPt)
    lm.main_finish_progress("compute_shortest_path")

    return path


def live_wire(
    image: np.ndarray,
    smooth_image: bool = False,
    threshold_gradient_image: bool = False,
    interactive: bool = True,
) -> List[np.ndarray]:
    """Use LiveWire segmentation algorithm for image segmentation aka intelligent scissors. The general idea of the
    algorithm is to use image information for segmentation and avoid crossing object boundaries. A gradient image
    highlights the boundaries, and Dijkstra’s shortest path algorithm computes a path using gradient differences as
    segment costs. Thus the line avoids strong gradients in the gradient image, which corresponds to following object
    boundaries in the original image.

    Now let's display the image using matplotlib front end. A click on the image starts livewire segmentation.
    The suggestion for the best segmentation will appear as you will be moving mouse across the image. To submit a
    suggestion, click on the image for the second time. To finish the segmentation, press Escape key.

    Args:
        image: image on which the shortest path should be computed.
        smooth_image: Whether to smooth the original image using bilateral smoothing filter.
        threshold_gradient_image: Wheter to use otsu method generate a thresholded gradient image for shortest path
            computation.
        interactive: Wether to generate the path interactively.

    Returns:
        A list of paths that are generated when running this algorithm. Paths can be used to segment a particular
        spatial domain of interests.
    """

    import matplotlib.pyplot as plt

    algorithm = LiveWireSegmentation(
        image, smooth_image=smooth_image, threshold_gradient_image=threshold_gradient_image
    )

    plt.gray()

    COLORS = cycle("rgbyc")  # use separate colors for consecutive segmentations

    start_point = None
    current_color = next(COLORS)
    current_path = None
    global path_list

    def button_pressed(event):
        global start_point
        if start_point is None:
            start_point = (int(event.ydata), int(event.xdata))
        else:
            end_point = (int(event.ydata), int(event.xdata))

            # the line below is calling the segmentation algorithm
            path = algorithm.compute_shortest_path(start_point, end_point)

            path = np.array(path)
            plt.plot(path[:, 1], path[:, 0], c=current_color)
            start_point = end_point

            path_list.append(path)

    def mouse_moved(event):
        if start_point is None:
            return

        end_point = (int(event.ydata), int(event.xdata))

        # the line below is calling the segmentation algorithm
        path = algorithm.compute_shortest_path(start_point, end_point)

        global current_path
        if current_path is not None:
            current_path.pop(0).remove()

        path = np.array(path)
        current_path = plt.plot(path[:, 1], path[:, 0], c=current_color)

        path_list.append(path)

        plt.show()

    def key_pressed(event):
        if event.key == "escape":
            global start_point, current_color
            start_point = None
            current_color = next(COLORS)

            global current_path
            if current_path is not None:
                current_path.pop(0).remove()
                current_path = None
                plt.draw()

        plt.show()

    plt.connect("button_release_event", button_pressed)
    if interactive:
        plt.connect("motion_notify_event", mouse_moved)
    plt.connect("key_press_event", key_pressed)

    plt.imshow(image)
    plt.autoscale(False)
    plt.title("Livewire")
    plt.show()

    return path_list

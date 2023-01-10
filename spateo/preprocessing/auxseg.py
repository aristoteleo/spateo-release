import math
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


class LiveWireSegmentation(object):
    def __init__(self, image=None, smooth_image: bool = False, threshold_gradient_image: bool = False):
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

        self.current_point = None
        self.path = None
        self.current_path_plot = None
        self.point_list = []
        self.point_plot_list = []
        self.path_list = np.empty(shape=[0, 2], dtype="int")
        self.path_plot_list = []
        self.rst = None

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

        return np.array(path)

    @staticmethod
    def LineDDA(start, end):
        start_x = start[0]
        start_y = start[1]
        end_x = end[0]
        end_y = end[1]
        delta_x = end_x - start_x
        delta_y = end_y - start_y

        if abs(delta_x) > abs(delta_y):
            steps = abs(delta_x)
        else:
            steps = abs(delta_y)

        x_step = delta_x / (steps + 10**-9)
        y_step = delta_y / (steps + 10**-9)

        x = start_x
        y = start_y
        points = []
        while steps >= 0:
            points.append([round(x), round(y)])
            x += x_step
            y += y_step
            steps -= 1
        return np.array(points)

    @staticmethod
    def fill_contours(arr):
        img = np.zeros(shape=[np.max(arr[:, 0]) + 1, np.max(arr[:, 1]) + 1], dtype="uint8")
        for line in arr:
            img[line[0], line[1]] = 1
        img_full = np.maximum.accumulate(img, 1) & np.maximum.accumulate(img[:, ::-1], 1)[:, ::-1]
        return np.array(np.where(img_full == 1)).T

    def connect(self):
        plt.connect("button_release_event", self.button_pressed)
        plt.connect("motion_notify_event", self.mouse_moved)
        plt.connect("key_press_event", self.key_pressed)

    def button_pressed(self, event):
        self.current_point = (int(event.ydata), int(event.xdata))
        self.point_list.append(self.current_point)
        self.point_plot_list.extend(plt.plot([event.xdata], [event.ydata], marker="o", color="k"))
        if len(self.point_list) > 1:
            self.path_list = np.row_stack((self.path_list, self.path))
            self.path_plot_list.extend(plt.plot(self.path[:, 1], self.path[:, 0]))
            first_point = self.point_list[0]
            if np.sum((np.array(self.current_point) - np.array(first_point)) ** 2) ** 0.5 <= 2:
                path_final = self.compute_shortest_path(self.current_point, first_point)
                path_rst = np.row_stack((self.path_list, path_final))
                path_full = self.fill_contours(path_rst)
                # np.savetxt('full.txt', path_full, fmt='%d')
                self.rst = path_full
                plt.close()
        plt.draw()

    def mouse_moved(self, event):
        if self.current_point is not None:
            mouse_point = (int(event.ydata), int(event.xdata))
            if event.key == "s":
                self.path = self.LineDDA(self.current_point, mouse_point)
            else:
                self.path = self.compute_shortest_path(self.current_point, mouse_point)
            if self.current_path_plot is not None:
                self.current_path_plot.pop(0).remove()
            self.current_path_plot = plt.plot(self.path[:, 1], self.path[:, 0])
            plt.draw()

    def key_pressed(self, event):
        if event.key == "ctrl+z":
            self.point_plot_list.pop(-1).remove()
            self.path_plot_list.pop(-1).remove()
            self.point_list.pop(-1)
            self.current_point = self.point_list[-1]

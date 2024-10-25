"""Spatiotemporal modeling of spatial transcriptomics
"""
from .agg import box_qc_regions, imshow, qc_regions
from .align import (
    optimization_animation,
    overlay_slices_2d,
    plot_deformation_grid,
    slices_2d,
)
from .bbs import delaunay, polygon
from .colorlabel import color_label
from .contour import spatial_domains
from .dotplot import dotplot
from .geo import geo
from .glm import glm_fit, glm_heatmap
from .interactions import ligrec, plot_connections
from .lisa import lisa, lisa_quantiles
from .polarity import *
from .scatters import scatters
from .space import space
from .three_d_plot import (
    acceleration,
    backbone,
    curl,
    curvature,
    deformation,
    divergence,
    jacobian,
    merge_animations,
    multi_models,
    pairwise_iteration,
    pairwise_iteration_panel,
    pairwise_mapping,
    pi_heatmap,
    three_d_animate,
    three_d_multi_plot,
    three_d_plot,
    torsion,
)

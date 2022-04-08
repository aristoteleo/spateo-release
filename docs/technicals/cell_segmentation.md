# Cell segmentation

This section describes the technical details behind Spateo's cell segmentation pipeline.

## Alignment of stain and RNA coordinates

* {py:func}`spateo.preprocessing.segmentation.refine_alignment`

Correct alignment between the stain and RNA coordinates is imperative to obtain correct cell segmentation. Slight misalignments can cause inaccuracies when aggregating UMIs in a patch of pixels into a single cell. We've found that for the most part, the alignments produced by the spatial assays themselves are quite good, but there are sometimes slight misalignments. Therefore, Spateo includes a couple of strategies to further refine this alignment to achieve the best possible cell segmentation results. In both cases, the stain image is considered as the "source" image (a.k.a. the image that will be transfomed), and the RNA coordinates represent the "target" image. This convention was chosen because the original RNA coordinates should be maintained as much as possible.

### Rigid alignment
The goal is to find the [affine transformation](https://en.wikipedia.org/wiki/Affine_transformation) of the stain such that the normalized cross-correlation (NCC) between the stain and RNA is minimized. Mathematically, we wish to find matrix $T$ such that

```{math}
:nowrap: true

$$\begin{bmatrix}
    x_{target}\\y_{target}\\1
\end{bmatrix} = T\begin{bmatrix}
    x_{source}\\y_{source}\\1
\end{bmatrix}$$
```

The matrix $T$ is optimized with [PyTorch](https://pytorch.org/)'s automatic differentiation capabilities. Internally, $T$ is represented as a $2 \times 3$ matrix, following PyTorch convention, and the [`affine_grid`](https://pytorch.org/docs/stable/generated/torch.nn.functional.affine_grid.html) and [`grid_sample`](https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html) functions are used.


### Non-rigid alignment
The goal is to find a transformation from the source coordinates to the target coordinates (also minimizing the NCC as with [Rigid alignment](#rigid-alignment)) where the transformation is defined using a set of reference ("control") points on the source image arranged in a grid (or a mesh) and displacements in each dimension (X and Y) from these points to a set of target points on the target image. Then, to obtain the transformation from the source to the target, the displacement of *every* source coordinate is computed by [thin-plate-spline](https://en.wikipedia.org/wiki/Thin_plate_spline) (TPS) interpolation. Here is an illustration of an image warped using control points and TPS interpolation [^ref1].

![Thin Plate Spline](../_static/technicals/cell_segmentation/thin_plate_spline.png)

The displacements for each of the control points are also optimized using PyTorch. Internally, the [`thin_plate_spline`](https://kornia.readthedocs.io/en/latest/_modules/kornia/geometry/transform/thin_plate_spline.html) module in the [Kornia](https://kornia.github.io/) library is used for TPS calculations.

## Watershed-based segmentation

* {py:func}`spateo.preprocessing.segmentation.mask_nuclei_from_stain`
* {py:func}`spateo.preprocessing.segmentation.watershed_markers`
* {py:func}`spateo.preprocessing.segmentation.watershed`

[Watershed](https://en.wikipedia.org/wiki/Watershed_(image_processing))-based nuclei segmentation works in three broad steps. First, a boolean mask, indicating where the nuclei are, is constructed. Second, Watershed markers are obtained by iteratively eroding the mask. Finally, the Watershed algorithm is used to label each nuclei. We will go over each of these steps in more detail.

First, the boolean mask is obtained by a combination of global and local thresholding. A [Multi-Otsu threshold](https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_multiotsu.html) is obtained and the first class is classified as the "background" pixels. Then, [adaptive thresholding](https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html) (with a gaussian-weighted sum) is applied to identify nuclei. The downside of the adaptive ("local") thresholding approach is that regions with sparse nuclei tend to be very noisy. Therefore, the background mask is used to remove noise in background regions by taking the pixels that are `False` in the background mask (indicating foreground) and `True` in the local mask. The final mask is obtained by applying the [morphological](https://docs.opencv.org/3.4/d9/d61/tutorial_py_morphological_ops.html) closing and opening operations to fill in holes and remove noise.

With the nuclei mask in hand, we need to identify initial markers in order to run the Watershed algorithm. These markers are identified by iteratively [eroding](https://docs.opencv.org/3.4/d9/d61/tutorial_py_morphological_ops.html) the mask until all components are less than a certain area. Then, each connected component is used as a marker for a single label. This method approximates the [distance transform](https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_watershed.html) that is commonly used to identify Watershed markers, with the additional benefit that it is much faster and efficient.

Finally, the Watershed algorithm is applied on a blurred stain image, limiting the labels to within the mask obtained in the first step, using the initial markers from the previous step.

## Deep-learning-based segmentation

Spateo provides a variety of existing deep-learning-based segmentation models. These include:

* StarDist [^ref2] ({py:func}`spateo.preprocessing.segmentation.stardist`)
* Cellpose [^ref3] ({py:func}`spateo.preprocessing.segmentation.cellpose`)
* Deepcell [^ref4] ({py:func}`spateo.preprocessing.segmentation.deepcell`)

In our experiments, StarDist performed the most consistently and is the recommended method to try first.

## Integrating both segmentation approaches

* {py:func}`spateo.preprocessing.segmentation.augment_labels`

The Watershed-based and deep-learning-based segmentation approaches have their own advantages. The former tends to perform reasonably well on most staining images and does not require any kind of training. However, the resulting segmentation tends to be "choppy" and sometimes breaks convexity. The latter performs very well on images similar to those they were trained on and results in more evenly-shaped segmentations, but does poorly on outlier regions. In particular, we've found that the deep-learning-based approaches have difficulty identifying nuclei in dense regions, often resulting in segmentations spanning multiple nuclei, or missing them entirely. The latter problem is mitigated to a certain degree by [adaptive histogram equalization](https://en.wikipedia.org/wiki/Adaptive_histogram_equalization#Contrast_Limited_AHE).

Therefore, we implemented a simple method of integrating the Watershed-based segmentation into the deep-learning-based segmentation. For every label in the Watershed segmentation that does not overlap with any label in the deep-learning-based segmentation, the label is copied to the deep-learning-based segmentation. This effectively results in "filling the gaps" ("augmenting") in the deep-learning segmentation using the Watershed segmentation.

## RNA-only segmentation

### Density binning

### Negative binomial mixture model

### Belief propagation

[^ref1]: Yin-Chiao Tsai, Hong-Dun Lin, Yu-Chang Hu, Chin-Lung Yu, Kang-Ping Lin (2006),
    *Thin-plate spline technique for medical image deformation*,
    [Journal of medical and biological engineering](https://www.airitilibrary.com/Publication/alDetailedMesh?docid=16090985-200012-20-4-203-210-a).
[^ref2]: Uwe Schmidt, Martin Weigert, Coleman Broaddus, Gene Myers (2018),
    *Cell Detection with Star-convex Polygons*,
    [International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI)](https://arxiv.org/abs/1806.03535).
[^ref3]: Carsen Stringer, Tim Wang, Michalis Michaelos, Marius Pachitariu (2020),
    *Cellpose: a generalist algorithm for cellular segmentation*,
    [Nature Methods](https://doi.org/10.1038/s41592-020-01018-x).
[^ref4]: Noah F. Greenwald, Geneva Miller, Erick Moen, Alex Kong, Adam Kagel, Thomas Dougherty, Christine Camacho Fullaway, Brianna J. McIntosh, Ke Xuan Leow, Morgan Sarah Schwartz, Cole Pavelchek, Sunny Cui, Isabella Camplisson, Omer Bar-Tal, Jaiveer Singh, Mara Fong, Gautam Chaudhry, Zion Abraham, Jackson Moseley, Shiri Warshawsky, Erin Soon, Shirley Greenbaum, Tyler Risom, Travis Hollmann, Sean C. Bendall, Leeat Keren, William Graf, Michael Angelo, David Van Valen (2021),
    *Whole-cell segmentation of tissue images with human-level performance using large-scale data annotation and deep learning*,
    [Nature Biotechnology](https://doi.org/10.1038/s41587-021-01094-0).

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

The goal is to find a transformation from the source coordinates to the target coordinates (also minimizing the NCC as with [](#rigid-alignment)) where the transformation is defined using a set of reference ("control") points on the source image arranged in a grid (or a mesh) and displacements in each dimension (X and Y) from these points to a set of target points on the target image. Then, to obtain the transformation from the source to the target, the displacement of *every* source coordinate is computed by [thin-plate-spline](https://en.wikipedia.org/wiki/Thin_plate_spline) (TPS) interpolation. Here is an illustration of an image warped using control points and TPS interpolation [^ref1].

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


## Identifying cytoplasm

* {py:func}`spateo.preprocessing.segmentation.mask_cells_from_stain`

Taking advantage of the fact that most nuclei labeling strategies also very weakly label the cytoplasm, cytoplasmic regions can be identified by using a very lenient threshold. Then, cells can be labeled by iteratively expanding the nuclei labels by a certain distance.


## RNA-only segmentation

Spateo features a novel method for cell segmentation using only RNA signal. As with any approximation/estimation method, some level of dropouts and noise is expected. A key requirement for RNA-only segmentation is that the unit of measure ("pixel") is much smaller than the expected size of a cell. Otherwise, there is no real benefit of attempting cell segmentation because each pixel most likely contains more than one cell.


### Density binning

* {py:func}`spateo.preprocessing.segmentation.segment_densities`
* {py:func}`spateo.preprocessing.segmentation.merge_densities`

We've found in testing that the RNA-only segmentation method, without separating out regions with different RNA (UMI) densities, tends to be too lenient in RNA-dense regions and too strict in RNA-sparse regions. Therefore, unless the UMI counts are sufficiently sparse (such as the case when only certain genes are being used), we recommend to first separate out the pixels into different "bins" according to their RNA density. Then, RNA- only segmentation can be performed on each bin separately, resulting in a better-calibrated algorithm.

One important detail is that density binning, as with any kind of clustering, is highly subjective. We suggest testing various parameters to find one that qualitatively "makes sense" to you.

Spateo employs a spatially-constrained hierarchical Ward clustering approach to segment the tissue into different RNA densities. Each pixel (or bin, if binning is used) is considered as an observation of a single feature: the number of UMIs observed for that pixel. Spatial constraints are imposed by providing a connectivity matrix with an edge between each neighboring pixel (so each pixel has four neighbors).

There are a few additional considerations in this approach.

* As briefly noted in the previous paragraph, pixels can be grouped in the square bins (where the UMI count of each bin is simply the sum of UMIs of its constituent pixels). We highly recommend binning as it drastically reduces runtime with minimal downsides.
* The pixel UMI counts (or equivalently, the binned UMI counts) are Gaussian-blurred prior to clustering to reduce noise. The size of the kernel is controlled by the `k` parameter.
* After the pixels (or bins) have been clustered, each bin is "[dilated](https://docs.opencv.org/3.4/db/df6/tutorial_erosion_dilatation.html)" one at a time, in ascending order of mean UMI counts. The size of the kernel is controlled by the `dk` parameter. This is done to reduce sharp transitions between bins in downstream RNA-based cell segmentation that we've observed in our testing.


### Negative binomial mixture model

* {py:func}`spateo.preprocessing.segmentation.score_and_mask_pixels`

The [negative binomial distribution](https://en.wikipedia.org/wiki/Negative_binomial_distribution) is widely used to model count data. It has been applied to many different kinds of biological data (such as sequencing data [^ref5]), and this is also how Spateo models the number of UMIs detected per pixel. For the purpose of cell segmentation, any pixel is either occupied or unoccupied by a cell. This is modeled as a two-component negative binomial mixture model, with one component for the UMIs detected in pixels occupied by a cell and the other for those unoccupied ("background"). Mathematically,

```{math}
:nowrap: true

$$P(X|p,r,r',\theta,\theta')=p \cdot NB(X|r,\theta)+(1-p) \cdot NB(X|r',\theta')$$
```
where
```{math}
:nowrap: true
\begin{align}
    X &\text{: number of observed UMIs}\\
    p &\text{: proportion of occupied pixels}\\
    NB(\cdot|r,\theta) &\text{: negative binomial PDF with parameters } r, \theta\\
    r,\theta &\text{: parameters to the negative binomial for occupied pixels}\\
    r',\theta' &\text{: parameters to the negative binomial for unoccupied pixels}
\end{align}
```

Ultimately, we wish to obtain estimates for $r,r',\theta,\theta'$. Spateo offers two strategies: expectation-maximization (EM) [^ref6] and a custom variational inference (VI) model. The latter is implemented with [Pyro](https://pyro.ai/), which is a Bayesian modeling and estimation framework built on top of PyTorch.

Once the desired parameter estimates are obtained, likelihoods of obtaining the number of observed UMIs $X$, for each pixel $(x,y)$, conditional on the pixel being occupied and unoccupied, are calculated.
```{math}
:nowrap: true

\begin{align}
    P_{(x,y)}(X|r,\theta) &\triangleq P_{(x,y)}(X|occupied)\\
    P_{(x,y)}(X|r',\theta') &\triangleq P_{(x,y)}(X|unoccupied)
\end{align}
```
These probabilities can be used directly to classify each pixel as occupied or unoccupied.


### Belief propagation

* {py:func}`spateo.preprocessing.segmentation.score_and_mask_pixels`

One important caveat of the [](#negative-binomial-mixture-model) is that it does *not* yield the marginal probabilities $P_{(x,y)}(occupied),P_{(x,y)}(unoccupied)$. In order to obtain these probabilities directly, Spateo can apply an efficient [belief propagation](https://en.wikipedia.org/wiki/Belief_propagation) algorithm. An undirected graphical model is constructed by considering each pixel as a node, and edges ("potentials") between neighboring pixels (a.k.a. a grid [Markov random field](https://en.wikipedia.org/wiki/Markov_random_field)).

![Markov Random Field](../_static/technicals/cell_segmentation/markov_random_field.jpg) [^ref7]

Each pixel has two possible states: occupied or unoccupied. The conditional probabilities obtained with the [](#negative-binomial-mixture-model) are used as the node potentials, and the edge potentials are defined in such a way that it is more probable for two connected nodes have the same state. This encodes the expectation that if a pixel is occupied, then its neighbors are also likely to also be occupied (and vice-versa). Loopy belief propagation is run on this graph until convergence, which yields estimates for the desired marginal probabilities.

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
[^ref5]: Valentine Svensson (2020),
    *Droplet scRNA-seq is not zero-inflated*,
    [Nature Biotechnology](https://doi.org/10.1038/s41587-019-0379-5).
[^ref6]: Chunmao Huang, Xingwang Liu, Tianyuan Yao, Xiaoqiang Wang (2019),
    *An efficient EM algorithm for the mixture of negative binomial models*,
    [Journal of Physics: Conference Series](https://doi.org/10.1088/1742-6596/1324/1/012093).
[^ref7]: Peter Orchard,
    *Markov Random Field Optimisation*,
    [https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/AV0809/ORCHARD/](https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/AV0809/ORCHARD/).

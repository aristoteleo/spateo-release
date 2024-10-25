# Spatial transcriptomics alignment

This section describes the technical details behind Spateo's spatial transcriptomics alignment pipeline. 

## Background

The sequential slicing and subsequent spatial transcriptomic profiling at the whole embryo level offer us an unprecedented opportunity to reconstruct the molecular hologram of the entire 3D embryo structure. However, conventional sectioning and downstream library preparation can rotate, transform, deform, and introduce missing regions in each profiled tissue section. In addition, with advancements in technology, spatial transcriptomics techniques with single-cell and even subcellular resolution are gradually emerging, and a single slice often contains hundreds of thousands of cells. Therefore, it is in general necessary to develop scalable and robust algorithms to reconstruct 3D structures to recover the relative spatial locations of single cells across different slices while allowing local distortion within the same slice. 

In general, aligning spatial transcriptomics data faces the following challenges:

1. Handling both rigid and non-rigid deformations simultaneously;
2. Managing partial alignments and outliers;
3. Addressing error propagation when aligning multiple consecutive slices;
4. Ensuring efficiency to handle large-scale datasets;
5. Maintaining flexibility to accommodate different technologies and data types.

Next, we will dive into the technical details of Spateo.


## Methodology
Consider a series of spatially-resolved transcriptomics samples, such as consecutive tissue sections from the same embryo, denote as $\mathcal{D} = \{\mathcal{S}^i\}_{i=1}^k,\ k\geq 2$, where $\mathcal{S}^i=\{\mathbf{X}^i, \mathbf{Z}^i\}$ is the $i$-th section, $\mathbf{X}^i\in\mathbb{R}^{N_i\times D}$ denotes the -dimensional spatial coordinates of $N_i$ spots or cells in section $i$ where $D$ can be either 2 or 3, and $\mathbf{Z}^i\in\mathbb{R}^{N_i\times G}$ corresponds to $G$ features, *e.g.*, gene expression of the measured readout, label information, or PCA and deep feature at those spots. Our goal is to align the two samples such that corresponding spots between samples have similar readout while the spatial distributions of spots are also preserved across samples. In the following, we first describe the alignment between two slices, i.e., pairwise alignment, and then introduce how to consider multiple slices in the optimization process. Thus, we assume we have slice A and slice B, and suppose slice B is the reference and that slice A will be aligned to the coordinate 
system of slice B by a transformation $\mathcal{T}$.  In what follows, we present a Bayesian generative model for aligning ST data that is robust, efficient, and capable of performing flexible partial alignment, non-rigid deformations. This model can additionally be used to jointly align multiple sections all at once via a global refinement mode to alleviate error accumulation of sequential 
alignment of many slices.

### Generative process

### Transformation model

### Define prior distributions

### Variational Bayesian Inference

## Function Design



# Spatial transcriptomics alignment

This section describes the technical details behind Spateo's spatial transcriptomics alignment pipeline. 

## Background

The sequential slicing and subsequent spatial transcriptomic profiling at the whole embryo level offer us an unprecedented opportunity to reconstruct the molecular hologram of the entire 3D embryo structure. However, conventional sectioning and downstream library preparation can rotate, transform, deform, and introduce missing regions in each profiled tissue section. In addition, with advancements in technology, spatial transcriptomics techniques with single-cell and even subcellular resolution are gradually emerging, and a single slice often contains hundreds of thousands of cells. Therefore, it is in general necessary to develop scalable and robust algorithms to reconstruct 3D structures to recover the relative spatial locations of single cells across different slices while allowing local distortion within the same slice. 


## Methodology
Consider a series of spatially-resolved transcriptomics samples, such as consecutive tissue sections from the same embryo, denote as $\mathcal{D} = \{\mathcal{S}^i\}_{i=1}^k$, where $\mathcal{S}^i=$ is the $i$-th section


### Problem formulation

### Generative process

### Transformation model

### Define prior distributions

### Variational Bayesian Inference

## Function Design



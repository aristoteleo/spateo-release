<p align="center">
  <img height="150" src="https://github.com/aristoteleo/spateo-release/blob/main/docs/_static/logo.png" />
</p

[![documentation](https://readthedocs.org/projects/spateo-release/badge/?version=latest)](https://spateo-release.readthedocs.io/en/latest/)


[Installation](https://spateo-release.readthedocs.io/en/latest/installation.html) - [Tutorials](https://spateo-release.readthedocs.io/en/latest/tutorials/index.html) - [API](https://spateo-release.readthedocs.io/en/latest/autoapi/spateo/index.html) - [Citation](https://linkinghub.elsevier.com/retrieve/pii/S0092-8674(22)00399-3) - [Technical](https://spateo-release.readthedocs.io/en/latest/technicals/index.html)
  
# spateo-release
Spateo: multidimensional spatiotemporal modeling of single-cell spatial transcriptomics 

<p align="justify">
Cells do not live in a vacuum, but in a milieu defined by cell–cell communication that can be measured via emerging high-resolution spatial transcriptomics approaches. However, analytical tools that fully leverage such data for kinetic modeling remain lacking. Here we present Spateo (aristoteleo/spateo-release), a general framework for quantitative spatiotemporal modeling of single-cell resolution spatial transcriptomics. Spateo delivers novel methods for digitizing spatial layers/columns to identify spatially-polar genes, and develops a comprehensive framework of cell-cell interaction to reveal spatial effects of niche factors and cell type-specific ligand-receptor interactions. Furthermore, Spateo reconstructs 3D models of whole embryos, and performs 3D morphometric analyses. Lastly, Spateo introduces the concept of “morphometric vector field” of cell migrations, and integrates spatial differential geometry to unveil regulatory programs underlying various organogenesis patterns of Drosophila. Thus, Spateo enables the study of the ecology of organs at a molecular level in 3D space, beyond isolated single cells. 
</p

![Spateo](https://user-images.githubusercontent.com/7456281/206298806-eb7df755-5fc4-46b6-80cc-baab86f0611f.png)

## Highlights of Spateo:

*  <p align="justify"> Spateo introduces a sophisticated approach, Starro, to segment single cells based purely on RNA signal, unsupervisedly identifies continuous tissue domains via spatially-constrained clustering, and dissect the intricate spatial cell type distribution and tissue composition;

* <p align="justify"> Spateo identifies spatial polarity/gradient genes (e.g. neuronal layer specific genes) by solving a partial differential equation to digitize layers and columns of a spatial domain. </p

* <p align="justify"> Spateo implements a full suite of spatially-aware modules for differential expression inference, including novel parametric models for spatially-informed prediction of cell-cell interactions and interpretable estimation of downstream effects. </p

* <p align="justify"> Spateo enables reconstruction of 3D whole-organ models from 2D slices, identifying different “organogenesis modes” (patterns of cell migration during organogenesis) for each organ and quantifying morphometric properties (such as organ surface area, volume, length and cell density) over time. </p

* <p align="justify"> Spateo brings in the concept of the “morphometric vector field” that predicts migration paths for each cell within an organ in a 3D fashion and reveals principles of cell migration by exploring various differential geometry quantities. </p

## Spateo Development Process
- Follow feature-staging-main review process
    - create a specific branch for new feature
    - implement and test on your branch; add unit tests
    - create pull request
    - discuss with lab members and merge into the main branch once all checks pass
- Follow python [Google code style](https://google.github.io/styleguide/pyguide.html)

## Code quality
- File and function docstrings should be written in [Google style](https://google.github.io/styleguide/pyguide.html)
- We use `black` to automatically format code in a standardized format. To ensure that any code changes are up to standard, use `pre-commit` as such.
```
# Run the following two lines ONCE.
pip install pre-commit
pre-commit install
```
Then, all future commits will call `black` automatically to format the code. Any code that does not follow the standard will cause a check to fail.

## Unit testing
Unit-tests should be written for most functions. To run unit tests, simply run the following.
```
# Install ONCE.
pip install -r dev-requirements.txt

# Run test
make test
```
Any failing tests will cause a check to fail.

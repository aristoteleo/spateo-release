[![documentation](https://readthedocs.org/projects/spateo-release/badge/?version=latest)](https://spateo-release.readthedocs.io/en/latest/)

# spateo-release
Spatiotemporal modeling of spatial transcriptomics 

Cells do not live in a vacuum, but in a milieu defined by cell–cell 
communication that can be quantified via recent advances in spatial 
transcriptomics. Here we present spateo, a open source framework that 
welcomes community contributions for quantitative spatiotemporal 
modeling of spatial transcriptomics. Leveraging the ultra-high 
spatial-resolution, large field of view and high RNA capture sensitivity 
of stereo-seq, spateo enables single cell resolution spatial 
transcriptomics via nuclei-staining and RNA signal based cell 
segmentation. Spateo also delivers novel methods for spatially 
constrained clustering to identify continuous tissue domains, spatial 
aware differential analyses to reveal spatial gene expression hotspots 
and modules, as well as the intricate ligand-receptor interactions. 
Importantly, spateo is equipped with sophisticated methods for building 
whole-body 3D models of embryogenesis by leveraging serial profilings of 
drosophila embryos across different stages. Spateo thus enables us to 
evolve from the reductionism of single cells to the holisticism of 
tissues and organs, heralding a paradigm shift in moving toward studying 
the ecology of tissue and organ while still offering us the opportunity 
to reveal associated molecular mechanisms.

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

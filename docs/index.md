# Welcome to Spateo documentation

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
Importantly, spateo is also equipped with sophisticated methods for
building whole-body 3D models of embryogenesis by leveraging serial
profilings of drosophila embryos across different stages. Spateo thus
enables us to evolve from reductionism of single cells to the
holisticism of tissues and organs, heralding a paradigm shift in moving
toward studying the ecology of tissue and organ while still offering us
the opportunity to reveal associated molecular mechanisms.

```{eval-rst}
.. card:: Installation :octicon:`plug;1em;`
    :link: installation
    :link-type: doc

    Click here to view a brief *spateo* installation guide.
```

```{eval-rst}
.. card:: Tutorials :octicon:`play;1em;`
    :link: tutorials/index
    :link-type: doc

    End-to-end tutorials showcasing key features in the package.
```

```{eval-rst}
.. card:: Technicals :octicon:`info;1em;`
    :link: technicals/index
    :link-type: doc

    Technical information on algorithms and tools provided by *spateo*.
```

```{eval-rst}
.. card:: API reference :octicon:`book;1em;`
    :link: autoapi/spateo/index
    :link-type: doc

    Detailed descriptions of *spateo* API and internals.
```

```{eval-rst}
.. card:: GitHub :octicon:`mark-github;1em;`
    :link: https://github.com/aristoteleo/spateo-release

    Ask questions, report bugs, and contribute to *spateo* at our GitHub repository.
```

*This documentation was heavily inspired and adapted from the [scvi-tools documentation](https://docs.scvi-tools.org/en/stable/). Go check them out!*

```{toctree}
:hidden: true
:maxdepth: 3
:titlesonly: true

Documentation home <self>
installation
tutorials/index
Technicals <technicals/index>
API <autoapi/spateo/index>
references
contributing/index
GitHub <https://github.com/aristoteleo/spateo-release>
```

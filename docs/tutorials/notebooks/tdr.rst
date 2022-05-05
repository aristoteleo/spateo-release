Three dimensional reconstruction and morphometric analyses
===========================================================
Spateo is equipped with sophisticated methods for building 3D spatiotemporal models by aligning serial sections of
tissue, organ or embryos across different stages that can be measured by stereo-seq or other high definition spatial
transcriptomics technologies. The ability of Spateo to 3D spatiotemporal models enables us to evolve from reductionism
of single cells to the holisticism of tissues and organs, heralding a paradigm shift in moving toward studying the
ecology of tissue and organ while still offering us the opportunity to reveal associated molecular mechanisms.

Here we will present a series of notebooks that showcase:

#. how we can leverage Spateo to build whole body 3D point-cloud, surface and volume models of drosophila embryo at the E8-10 stage.
#. how we can leverage Spateo to build 3D surface and volume models of each tissue type.
#. how we can leverage Spateo to perform novel morphometric analyses.
#. how we can leverage Spateo to learn continuous expression pattern in the 3D volume model by kernel methods.
#. how we can leverage Spateo to learn continuous expression pattern in the 3D volume model by a new deep learning methods.

.. toctree::
    :maxdepth: 0

    ./tdr_1_reconstruct_whole_body.ipynb
    ./tdr_2_reconstruct_tissues.ipynb
    ./tdr_3_morphometric_analysis.ipynb
    ./tdr_4_kernel_interpolation.ipynb
    ./tdr_5_deep_interpolation.ipynb

import argparse
import random
import sys
from collections.abc import Iterable

import numpy as np
from mpi4py import MPI

# For testing:
# For now, add Spateo working directory to sys path so compiler doesn't look in the installed packages:
sys.path.insert(0, "/mnt/c/Users/danie/Desktop/Github/Github/spateo-release-main")

from spateo.plotting.static.space import plot_cell_signaling
from spateo.tools.ST_modeling.MuSIC import MuSIC, VMuSIC
from spateo.tools.ST_modeling.MuSIC_downstream import MuSIC_Interpreter
from spateo.tools.ST_modeling.MuSIC_upstream import MuSIC_target_selector

np.random.seed(888)
random.seed(888)


def define_spateo_argparse(**kwargs):
    """Defines and returns MPI and argparse objects for model fitting and interpretation.

    Args:
        kwargs: Keyword arguments for any of the argparse arguments defined below.

    Parser arguments:
        run_upstream: Flag to run the upstream target selection step. If True, will run the target selection step
        adata_path: Path to AnnData object containing gene expression data. This or 'csv_path' must be given to run.
        csv_path: Path to .csv file containing gene expression data. This or 'adata_path' must be given to run.
        subsample: Flag to subsample the data. Recommended for large datasets (>5000 samples).
        multiscale: Flag to create multiscale models. Currently, it is recommended to only create multiscale models
            for Gaussian data.
        multiscale_params_only: Flag to return additional metrics along with the coefficients for multiscale models (
            specifying this argument sets Flag to True)
        mod_type: The type of model that will be employed- this dictates how the data will be processed and
            prepared. Options:
                - "niche": Spatially-aware, uses categorical cell type labels as independent variables.
                - "lr": Spatially-aware, essentially uses the combination of receptor expression in the "target" cell
                    and spatially lagged ligand expression in the neighboring cells as independent variables.
                - "ligand": Spatially-aware, essentially uses ligand expression in the neighboring cells as
                    independent variables.
                - "receptor": Uses receptor expression in the "target" cell as independent variables.
        cci_dir: Path to directory containing cell-cell interaction databases
        species: Selects the cell-cell communication database the relevant ligands will be drawn from. Options:
                "human", "mouse".
        output_path: Full path name for the .csv file in which results will be saved. Make sure the parent directory
            is empty- any existing files will be deleted. It is recommended to create a new folder to serve as the
            output directory. This should be supplied of the form '/path/to/file.csv', where file.csv will store
            coefficients. The name of the target will be appended at runtime.


        custom_lig_path: Path to .txt file containing a custom list of ligands. Each ligand should have its own line
            in the .txt file.
        ligand: Alternative to the custom ligand path, can be used to provide a single ligand or a list of ligands (
            separated by whitespace in the command line).
        custom_rec_path: Path to .txt file containing a custom list of receptors. Each receptor should have its own
            line in the .txt file.
        receptor: Alternative to the custom receptor path, can be used to provide a single receptor or a list of
            receptors (separated by whitespace in the command line).
        custom_pathways_path: Path to .txt file containing a custom list of pathways. Each pathway should have its own
            line in the .txt file.
        pathway: Alternative to the custom pathway path, can be used to provide a single pathway or a list of pathways (
            separated by whitespace in the command line).
        targets_path: Path to .txt file containing a custom list of targets. Each target should have its own line in
            the .txt file.
        target: Alternative to the custom target path, can be used to provide a single target or a list of targets (
            separated by whitespace in the command line).


        init_betas_path: Optional path to a .json file or .csv file containing initial coefficient values for the model
            for each target variable. If encoded in .json, keys should be target gene names, values should be numpy
            arrays containing coefficients. If encoded in .csv, columns should be target gene names. Initial
            coefficients should have shape [n_features, ].


        normalize: Flag to perform library size normalization, to set total counts in each cell to the same
            number (adjust for cell size). Will be set to True if provided.
        smooth: Flag to correct for dropout effects by leveraging gene expression neighborhoods to smooth
            expression. It is advisable not to do this if performing Poisson or negative binomial regression. Will
            be set to True if provided.
        log_transform: Flag for whether log-transformation should be applied to expression. It is advisable not to do
            this if performing Poisson or negative binomial regression. Will be set to True if provided.
        normalize_signaling: Flag to minmax scale the final ligand expression array (for :attr `mod_type` =
            "ligand"), or the final ligand-receptor array (for :attr `mod_type` = "lr"). This is recommended to
            associate downstream expression with rarer/less prevalent signaling mechanisms.
        target_expr_threshold: Only used when automatically selecting targets- finds the L:R-downstream TFs and their
            targets and searches for expression above a threshold proportion of cells to filter to a subset of
            candidate target genes. This argument sets that proportion, and defaults to 0.2.
        multicollinear_threshold: Variance inflation factor threshold used to filter out multicollinear features. A
            value of 5 or 10 is recommended.


        coords_key: Entry in :attr:`adata` .obsm that contains spatial coordinates. Defaults to "spatial".
        group_key: Entry in :attr:`adata` .obs that contains cell type labels. Defaults to "cell_type".
        group_subset: Subset of cell types to include in the model (provided as a whitespace-separated list in
            command line). If given, will consider only cells of these types in modeling. Defaults to all cell types.
        covariate_keys: Entries in :attr:`adata` .obs or :attr:`adata` .var that contain covariates to include
            in the model. Can be provided as a whitespace-separated list in the command line.


        bw: Bandwidth for kernel density estimation. Consists of either a distance value or N for the number of
            nearest neighbors, depending on :attr:`bw_fixed`
        minbw: For use in automated bandwidth selection- the lower-bound bandwidth to test.
        maxbw: For use in automated bandwidth selection- the upper-bound bandwidth to test.
        bw_fixed: Flag to use a fixed bandwidth (True) or to automatically select a bandwidth (False). This should be
            True if the input to/values to test for :attr:`bw` are distance values, and False if they are numbers of
            neighbors.
        exclude_self: Flag to exclude the target cell from the neighborhood when computing spatial weights. Note that
            if True and :attr:`bw` is defined by the number of neighbors, your desired bw should be 1 + the number of
            neighbors you want to include.


        kernel: Type of kernel function used to weight observations; one of "bisquare", "exponential", "gaussian",
            "quadratic", "triangular" or "uniform".
        n_neighbors: For :attr:`mod_type` "ligand" or "lr"- ligand expression will be taken from the neighboring
            cells- this defines the number of cells to use. A value of 10 should typically capture the region where
            the majority of the signaling is sourced from.
        distr: Distribution family for the dependent variable; one of "gaussian", "poisson", "nb"


        fit_intercept: Flag to fit an intercept term in the model. Will be set to True if provided.
        no_hurdle: By default, a hurdle model will be used to account for the zero-inflated nature of single-cell
            data. To skip this step, set this flag.


        tolerance: Convergence tolerance for IWLS
        max_iter: Maximum number of iterations for IWLS
        patience: When checking various values for the bandwidth, this is the number of iterations to wait for
            without the score changing before stopping. Defaults to 5.
        ridge_lambda: Sets the strength of the regularization, between 0 and 1. The higher values typically will
            result in more features removed.


        search_bw: For downstream analysis; specifies the bandwidth to search for senders/receivers. Recommended to
            set equal to the bandwidth of a fitted model.
        top_k_receivers: For downstream analysis, specifically when constructing vector fields of signaling effects.
            Specifies the number of nearest neighbors to consider when computing signaling effect vectors.
        filter_targets: For downstream analysis, specifically :func `infer_effect_direction`; if True, will subset to
            only the targets that were predicted well by the model.
        filter_target_threshold: For downstream analysis, specifically :func `infer_effect_direction`; specifies the
            threshold Pearson coefficient for target subsetting. Only used if `filter_targets` is True.
        diff_sending_or_receiving: For downstream analyses, specifically :func
            `sender_receiver_effect_deg_detection`; specifies whether to compute differential expression of genes in
            cells with high or low sending effect potential ('sending cells') or high or low receiving effect
            potential ('receiving cells').
        target_for_downstream: A string or a list (provided as a whitespace-separated list in the command line) of
            target genes for :func `get_effect_potential`, :func `get_pathway_potential` and :func
             `calc_and_group_sender_receiver_effect_degs` (provide only one target), as well as :func
             `compute_cell_type_coupling` (can provide multiple targets).
        ligand_for_downstream: For downstream analyses; used for :func `get_effect_potential` and :func
            `calc_and_group_sender_receiver_effect_degs`, used to specify the ligand gene to consider with respect
            to the target.
        receptor_for_downstream: For downstream analyses; used for :func `get_effect_potential` and :func
            `calc_and_group_sender_receiver_effect_degs`, used to specify the receptor gene to consider with respect
            to the target.
        pathway_for_downstream: For downstream analyses; used for :func `get_pathway_potential` and :func
            `calc_and_group_sender_receiver_effect_degs`, used to specify the pathway to consider with respect to
            the target.
        sender_ct_for_downstream: For downstream analyses; used for :func `get_effect_potential` and :func
            `calc_and_group_sender_receiver_effect_degs`, used to specify the cell type to consider as a sender.
        receiver_ct_for_downstream: For downstream analyses; used for :func `get_effect_potential` and :func
            `calc_and_group_sender_receiver_effect_degs`, used to specify the cell type to consider as a receiver.
        no_cell_type_markers: For downstream analyses; used for :func `calc_and_group_sender_receiver_effect_degs`;
            if True, will exclude cell type markers from the set of genes for which to compare to sent/received signal.
        compute_pathway_effect: For downstream analyses; used for :func `inferred_effect_direction`; if True,
            will summarize the effects of all ligands/ligand-receptor interactions in a pathway.
        n_GAM_points: For downstream analyses; used for :func `calc_and_group_sender_receiver_effect_degs`;
            specifies the number of points to sample along the spline function when evaluating a fitted GAM model.
        num_leiden_pcs: For downstream analyses; used for :func `calc_and_group_sender_receiver_effect_degs`;
            specifies the number of principal components to use when computing partitioning for the discovered
            differentially expressed genes.
        num_leiden_neighbors: For downstream analyses; used for :func `calc_and_group_sender_receiver_effect_degs`;
            specifies the number of neighbors to use when computing partitioning for the discovered differentially
            expressed genes.
        leiden_resolution: For downstream analyses; used for :func `calc_and_group_sender_receiver_effect_degs`;
            specifies the resolution parameter to use for Leiden partitioning.
        top_n_DE_genes: For downstream analyses; used for :func `calc_and_group_sender_receiver_effect_degs`; if
            given, will restrict the number of genes that are clustered and displayed on the final plot. Note that
            if there are more than 200 differentially expressed genes, the program will automatically use 200 for
            this parameter.

    Returns:
        comm: MPI comm object for parallel processing
        parser: Argparse object defining important arguments for model fitting and interpretation
    """
    # Initialize MPI
    comm = MPI.COMM_WORLD

    arg_dict = {
        "-run_upstream": {
            "type": bool,
            "action": "store_true",
            "default": False,
        },
        "-adata_path": {
            "type": str,
        },
        "-csv_path": {
            "type": str,
            "help": "Can be used to provide a .csv file, containing gene expression data or any other kind of data. "
            "Assumes the first three columns contain x- and y-coordinates and then dependent variable "
            "values, in that order.",
        },
        "-subsample": {
            "type": bool,
            "action": "store_true",
            "default": False,
            "help": "Recommended for large datasets (>5000 samples), otherwise model fitting is quite slow.",
        },
        "-multiscale": {
            "type": bool,
            "action": "store_true",
            "default": False,
            "help": "Currently, it is recommended to only create multiscale models for Gaussian regression models.",
        },
        "-multiscale_params_only": {
            "type": bool,
            "default": False,
            "action": "store_true",
        },
        "-mod_type": {
            "type": str,
            "default": "niche",
        },
        "-cci_dir": {"type": str},
        "-species": {"type": str, "default": "human"},
        "-output_path": {
            "type": str,
            "default": "./output/stgwr_results.csv",
            "help": "Path to output file. Make sure the parent directory is empty- "
            "any existing files will be deleted. It is recommended to create "
            "a new folder to serve as the output directory. This should be "
            "supplied of the form '/path/to/file.csv', where file.csv will "
            "store coefficients. The name of the target will be appended at runtime.",
        },
        "-custom_lig_path": {"type": str},
        "-ligand": {
            "nargs": "+",
            "type": str,
            "help": "Alternative to the custom ligand path, can be used to provide a custom list of ligands.",
        },
        "-custom_rec_path": {"type": str},
        "-receptor": {
            "nargs": "+",
            "type": str,
            "help": "Alternative to the custom receptor path, can be used to provide a custom list of receptors.",
        },
        "-custom_pathways_path": {"type": str},
        "-pathway": {
            "nargs": "+",
            "type": str,
            "help": "Alternative to the custom pathway path, can be used to provide a custom list of pathways.",
        },
        "-targets_path": {"type": str},
        "-target": {
            "nargs": "+",
            "type": str,
            "help": "Alternative to the custom target path, can be used to provide a custom list of target molecules.",
        },
        "-init_betas_path": {"type": str},
        "-normalize": {
            "type": bool,
            "action": "store_true",
            "default": False,
        },
        "-smooth": {
            "type": bool,
            "action": "store_true",
            "default": False,
        },
        "-log_transform": {
            "type": bool,
            "action": "store_true",
            "default": False,
        },
        "-normalize_signaling": {
            "type": bool,
            "action": "store_true",
            "default": False,
            "help": "For ligand, receptor or L:R models, normalize computed signaling values. This should be used to "
            "find signaling effects that may be mediated by rarer signals.",
        },
        "-target_expr_threshold": {
            "default": 0.05,
            "type": float,
            "help": "For automated selection, the threshold proportion of cells for which transcript "
            "needs to be expressed in to be selected as a target of interest. Not used if 'targets_path' is not None.",
        },
        "-multicollinear_threshold": {
            "type": float,
            "help": "Used only if `mod_type` is 'slice'. If this argument is provided, independent variables that are "
            "highly correlated will be filtered out based on variance inflation factor threshold. A value "
            "of 5 or 10 is recommended. This can be useful in reducing computation time.",
        },
        "-coords_key": {"default": "spatial", "type": str},
        "-group_key": {
            "default": "cell_type",
            "type": str,
            "help": "Key to entry in .obs containing cell type "
            "or other category labels. Required if "
            "'mod_type' is 'niche' or 'slice'.",
        },
        "-group_subset": {
            "nargs": "+",
            "type": str,
            "help": "If provided, only cells with labels that correspond to these group(s) will be used as prediction "
            "targets. Will search in key corresponding to the input to arg 'cell_type' if given.",
        },
        "-covariate_keys": {
            "nargs": "+",
            "type": str,
            "help": "Any number of keys to entry in .obs or .var_names of an "
            "AnnData object. Values here will be added to"
            "the model as covariates.",
        },
        "-bw": {
            "type": float,
        },
        "-minbw": {
            "type": float,
        },
        "-maxbw": {
            "type": float,
        },
        "-bw_fixed": {
            "type": bool,
            "action": "store_true",
            "default": False,
            "help": "If this argument is provided, the bandwidth will be interpreted as a distance during kernel "
            "operations. If not, it will be interpreted as the number of nearest neighbors.",
        },
        "-exclude_self": {
            "type": bool,
            "action": "store_true",
            "help": "When computing spatial weights, do not count the cell itself as a neighbor. Recommended to set to "
            "True for the CCI models because the independent variable array is also spatially-dependent.",
        },
        "-kernel": {"default": "bisquare", "type": str},
        "-n_neighbors": {
            "default": 10,
            "type": int,
            "help": "Only used if `mod_type` is 'niche', to define the number of neighbors to consider for each "
            "cell when defining the independent variable array.",
        },
        "-distr": {"default": "gaussian", "type": str},
        "-fit_intercept": {"type": bool, "action": "store_true", "default": False},
        "-no_hurdle": {
            "type": bool,
            "action": "store_true",
            "default": False,
            "help": "If True, do not implement spatially-weighted hurdle model- will only perform generalized linear "
            "modeling.",
        },
        "-tolerance": {"default": 1e-3, "type": float},
        "-max_iter": {"default": 500, "type": int},
        "-patience": {"default": 5, "type": int},
        "-ridge_lambda": {"type": float},
        "-chunks": {
            "default": 1,
            "type": int,
            "help": "For use if `multiscale` is True- increase the number of parallel processes. Can be used to help "
            "prevent memory from running out, otherwise keep as low as possible.",
        },
        # Downstream arguments:
        "-search_bw": {
            "type": float,
            "help": "Used for downstream analyses; specifies the bandwidth to search for "
            "senders/receivers. Recommended to set equal to the bandwidth of a fitted "
            "model.",
        },
        "-top_k_receivers": {
            "default": 10,
            "type": int,
            "help": "Used for downstream analyses, specifically for :func `define_effect_vf`; specifies the number of "
            "nearest neighbors to consider when computing signaling effect vectors.",
        },
        "-filter_targets": {
            "type": bool,
            "action": "store_true",
            "default": False,
            "help": "Used for downstream analyses, specifically :func `infer_effect_direction`; if True, will subset to only "
            "the targets that were predicted well by the model.",
        },
        "-filter_targets_threshold": {
            "default": 0.65,
            "type": float,
            "help": "Used for downstream analyses, specifically :func `infer_effect_direction`; specifies the threshold "
            "Pearson coefficient for target subsetting. Only used if `filter_targets` is True.",
        },
        "-diff_sending_or_receiving": {
            "default": "sending",
            "type": str,
            "help": "Used for downstream analyses, specifically :func `sender_receiver_effect_deg_detection`; specifies "
            "whether to compute differential expression of genes in cells with high or low sending effect potential "
            "('sending cells') or high or low receiving effect potential ('receiving cells').",
        },
        "-target_for_downstream": {
            "nargs": "+",
            "type": str,
            "help": "Used for :func `get_effect_potential`, :func `get_pathway_potential` and :func "
            "`calc_and_group_sender_receiver_effect_degs` (provide only one target), as well as :func "
            "`compute_cell_type_coupling` (can provide multiple targets). Used to specify the target "
            "gene(s) to analyze with these functions.",
        },
        "-ligand_for_downstream": {
            "type": str,
            "help": "Used for :func `get_effect_potential` and :func `calc_and_group_sender_receiver_effect_degs`, "
            "used to specify the ligand gene to consider with respect to the target.",
        },
        "-receptor_for_downstream": {
            "type": str,
            "help": "Used for :func `get_effect_potential` and :func `calc_and_group_sender_receiver_effect_degs`, "
            "used to specify the receptor gene to consider with respect to the target.",
        },
        "-pathway_for_downstream": {
            "type": str,
            "help": "Used for :func `get_pathway_potential` and :func `calc_and_group_sender_receiver_effect_degs`, "
            "used to specify the pathway to consider with respect to the target.",
        },
        "-sender_ct_for_downstream": {
            "type": str,
            "help": "Used for :func `get_effect_potential` and :func `calc_and_group_sender_receiver_effect_degs`, "
            "used to specify the cell type to consider as a sender.",
        },
        "-receiver_ct_for_downstream": {
            "type": str,
            "help": "Used for :func `get_effect_potential` and :func `calc_and_group_sender_receiver_effect_degs`, "
            "used to specify the cell type to consider as a receiver.",
        },
        "-no_cell_type_markers": {
            "type": bool,
            "action": "store_true",
            "default": False,
            "help": "Used for :func `calc_and_group_sender_receiver_effect_degs`; if True, will exclude cell type "
            "markers from the set of genes for which to compare to sent/received signal.",
        },
        "-compute_pathway_effect": {
            "type": bool,
            "action": "store_true",
            "default": False,
            "help": "Used for :func `inferred_effect_direction`; if True, will summarize the effects of all "
            "ligands/ligand-receptor interactions in a pathway.",
        },
        "-n_GAM_points": {
            "default": 50,
            "type": int,
            "help": "Used for :func `calc_and_group_sender_receiver_effect_degs`; specifies the number of points to "
            "sample along the spline function when evaluating a fitted GAM model.",
        },
        "-num_leiden_pcs": {
            "default": 10,
            "type": int,
            "help": "Used for :func `calc_and_group_sender_receiver_effect_degs`; specifies the number of principal "
            "components to use when computing partitioning for the discovered differentially expressed genes.",
        },
        "-num_leiden_neighbors": {
            "default": 5,
            "type": int,
            "help": "Used for :func `calc_and_group_sender_receiver_effect_degs`; specifies the number of neighbors to "
            "use when computing partitioning for the discovered differentially expressed genes.",
        },
        "-leiden_resolution": {
            "default": 0.5,
            "type": float,
            "help": "Used for :func `calc_and_group_sender_receiver_effect_degs`; specifies the resolution parameter to"
            "use for Leiden partitioning.",
        },
        "-top_n_DE_genes": {
            "type": int,
            "help": "Used for :func `calc_and_group_sender_receiver_effect_degs`; if given, will restrict the number "
            "of genes that are clustered and displayed on the final plot. Note that if there are more than 200 "
            "differentially expressed genes, the program will automatically use 200 for this parameter.",
        },
    }

    # Update kwargs to match the argparse signature:
    kwargs = {f"-{key}": value for key, value in kwargs.items()}

    # Iterate over the keyword items to validate the form of the given arguments:
    for key, value in kwargs.items():
        if key not in arg_dict.keys():
            raise ValueError(f"Argument {key} not recognized. Please check the documentation for valid arguments.")

        arg_info = arg_dict[key]
        arg_type = arg_info.get("type")
        if arg_type is not None:
            if not isinstance(value, arg_type):
                raise TypeError(f"Argument {key} must be of type {arg_type}.")

        if arg_info.get("action") == "store_true" and not isinstance(value, bool):
            raise TypeError(f"Argument {key} must be of type bool.")

        # Check for iterable to allow input to be set, list, tuple, etc. Also a single string is fine.
        if arg_info.get("nargs") is not None:
            if not isinstance(value, Iterable):
                raise TypeError(f"Argument {key} must be an iterable.")

            element_type = type(next(iter(value)))
            if not all(isinstance(element, element_type) for element in value):
                raise TypeError(f"Argument {key} must be an iterable containing values of type {element_type}.")

    # Update the argument dictionary with arguments from kwargs:
    arg_dict.update(kwargs)

    # Initialize parser:
    parser = argparse.ArgumentParser(description="MuSIC arguments")

    # Use arg_dict to population parser:
    for arg, arg_info in arg_dict.items():
        parser.add_argument(arg, **arg_info)

    return comm, parser


if __name__ == "__main__":
    # From the command line, run spatial GWR
    comm, parser = define_spateo_argparse()

    rank = comm.Get_rank()
    size = comm.Get_size()

    t1 = MPI.Wtime()

    # Testing time! Uncomment this (and comment anything below) to test the downstream functions:
    test_downstream = MuSIC_Interpreter(comm, parser)
    test_downstream.compute_coeff_significance()
    # test_downstream.get_sig_potential()
    test_downstream.compute_cell_type_coupling()

    # # Testing time! Uncomment this (and then comment anything above and below) to test the upstream functions:
    # # test = MuSIC_target_selector(parser)
    # test_gene = self.adata[:, "MMP1"].X
    # # test.parse_predictors(data=test_gene)
    #
    # Check if GRN model is specified:
    # if parser.parse_args().grn:
    #     "filler"

    # else:
    # For use only with VMuSIC:
    # n_multiscale_chunks = parser.parse_args().chunks
    #
    # if parser.parse_args().run_upstream:
    #     swr_selector = MuSIC_target_selector(parser)
    #     swr_selector.select_features()
    #
    # if parser.parse_args().multiscale:
    #     print(
    #         "Multiscale algorithm may be computationally intensive for large number of features- if this is the "
    #         "case, it is advisable to reduce the number of parameters."
    #     )
    #     multiscale_model = VMuSIC(comm, parser)
    #     multiscale_model.multiscale_backfitting()
    #     multiscale_model.multiscale_compute_metrics(n_chunks=int(n_multiscale_chunks))
    #     multiscale_model.predict_and_save()
    #
    # else:
    #     swr_model = MuSIC(comm, parser)
    #     swr_model._set_up_model()
    #     swr_model.fit()
    #     swr_model.predict_and_save()
    #
    # t_last = MPI.Wtime()
    #
    # wt = comm.gather(t_last - t1, root=0)
    # if rank == 0:
    #     print("Total Time Elapsed:", np.round(max(wt), 2), "seconds")
    #     print("-" * 60)

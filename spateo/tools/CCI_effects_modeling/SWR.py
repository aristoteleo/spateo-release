import argparse
import random
import time
from collections.abc import Iterable
from typing import List, Set, Tuple

import numpy as np

from ...logging import logger_manager as lm
from .MuSIC import MuSIC

np.random.seed(888)
random.seed(888)


# To allow for running by function definition (for more flexible option)
def define_spateo_argparse(**kwargs):
    """Defines and returns MPI and argparse objects for model fitting and interpretation.

    Args:
        kwargs: Keyword arguments for any of the argparse arguments defined below.

    Parser arguments:
        run_upstream: Flag to run the upstream target selection step. If True, will run the target selection step
        adata_path: Path to AnnData object containing gene expression data. This or 'csv_path' must be given to run.
        csv_path: Path to .csv file containing gene expression data. This or 'adata_path' must be given to run.
        n_spatial_dim_csv: Number of spatial dimensions to the data provided to 'csv_path'. Defaults to 2.
        spatial_subsample: Flag to subsample the data- at a big picture level, this will be done by dividing the tissue
            into regions and subsampling from each of these regions. Recommended for large datasets (>5000 samples).
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
                - "downstream": For the purposes of downstream analysis, used to model ligand expression as a
                    function of upstream regulators
        include_unpaired_lr: Only if :attr:`mod_type` is "lr"- if True, will include individual ligands/complexes and
            individual receptors in the design matrix if their cognate interacting partners cannot also be found.
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
            candidate target genes. This argument sets that proportion, and defaults to 0.05.
        multicollinear_threshold: Variance inflation factor threshold used to filter out multicollinear features. A
            value of 5 or 10 is recommended.


        coords_key: Entry in :attr:`adata` .obsm that contains spatial coordinates. Defaults to "spatial".
        group_key: Entry in :attr:`adata` .obs that contains cell type labels. Required for 'mod_type' = "niche".
        group_subset: Subset of cell types to include in the model (provided as a whitespace-separated list in
            command line). If given, will consider only cells of these types in modeling. Defaults to all cell types.
        covariate_keys: Entries in :attr:`adata` .obs or :attr:`adata` .var that contain covariates to include
            in the model. Can be provided as a whitespace-separated list in the command line. Numerical covariates
            should be minmax scaled between 0 and 1.
        total_counts_key: Entry in :attr:`adata` .obs that contains total counts for each cell. Required if subsetting
            by total counts. Defaults to "total_counts".
        total_counts_threshold: Threshold for total counts to subset cells by- cells with total counts greater than
            this threshold will be retained.


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


        kernel: Type of kernel function used to weight observations when computing spatial weights and fitting the
            model; one of "bisquare", "exponential", "gaussian", "quadratic", "triangular" or "uniform".
        distance_membrane_bound: In model setup, distance threshold to consider cells as neighbors for membrane-bound
            ligands. If provided, will take priority over :attr 'n_neighbors_membrane_bound'.
        distance_secreted: In model setup, distance threshold to consider cells as neighbors for secreted or ECM
            ligands. If provided, will take priority over :attr 'n_neighbors_secreted'.
        n_neighbors_membrane_bound: For :attr:`mod_type` "ligand" or "lr"- ligand expression will be taken from the
            neighboring cells- this defines the number of cells to use for membrane-bound ligands. Defaults to 8.
        n_neighbors_secreted: For :attr:`mod_type` "ligand" or "lr"- ligand expression will be taken from the
            neighboring cells- this defines the number of cells to use for secreted or ECM ligands.
        distr: Distribution family for the dependent variable; one of "gaussian", "poisson", "nb"
        fit_intercept: Flag to fit an intercept term in the model. Will be set to True if provided.


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
        n_components: Used for :func `CCI_sender_deg_detection` and :func `CCI_receiver_deg_detection`;
            determines the dimensionality of the space to embed into using UMAP.
        cci_degs_model_interactions: Used for :func `CCI_sender_deg_detection`; if True, will consider transcription
            factor interactions with cofactors and other transcription factors, with these interactions combined into
            features. If False, will use each cofactor independently in the prediction.
        no_cell_type_markers: Used for :func `CCI_receiver_deg_detection`; if True, will exclude cell type markers
            from the set of genes for which to compare to sent/received signal.
        compute_pathway_effect: Used for :func `inferred_effect_direction`; if True, will summarize the effects of all
            ligands/ligand-receptor interactions in a pathway.

    Returns:
        parser: Argparse object defining important arguments for model fitting and interpretation
        args_list: If argparse object is returned from a function, the parser must read in arguments in the form of a
            list- this return contains that processed list.
    """

    logger = lm.get_main_logger()

    arg_dict = {
        "-run_upstream": {
            "action": "store_true",
        },
        "-adata_path": {
            "type": str,
        },
        "-csv_path": {
            "type": str,
            "help": "Can be used to provide a .csv file, containing gene expression data or any other kind of data. "
            "Assumes the first few columns contain spatial coordinates (depending on input to 'n_spatial_dim_csv') and "
            "then "
            "dependent variable values, in that order.",
        },
        "-n_spatial_dim_csv": {
            "type": int,
            "default": 2,
            "help": "If using a .csv file, specifies the number of spatial dimensions. Default is 2.",
        },
        "-spatial_subsample": {
            "action": "store_true",
            "help": "Recommended for large datasets (>5000 samples), otherwise model fitting is quite slow.",
        },
        "-mod_type": {
            "type": str,
            "default": "niche",
        },
        "-include_unpaired_lr": {
            "action": "store_true",
            "help": "Only if 'mod_type' is 'lr'- if True, will include individual ligands/complexes and individual "
            "receptors/complexes together with L:R pairs if their cognate interacting partners cannot also be "
            "found.",
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
            "action": "store_true",
        },
        "-smooth": {
            "action": "store_true",
        },
        "-log_transform": {
            "action": "store_true",
        },
        "-normalize_signaling": {
            "action": "store_true",
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
            "help": "Key to entry in .obs containing cell type or other category labels. Required if 'mod_type' is "
            "'niche'.",
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
        "-total_counts_key": {
            "default": "total_counts",
            "type": str,
            "help": "Key to entry in .obs containing total counts per cell. Will be used if 'total_counts_threshold' "
            "is provided.",
        },
        "-total_counts_threshold": {
            "default": 0.0,
            "type": float,
            "help": "If provided, cells with total counts below this threshold will be removed in preprocessing.",
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
            "action": "store_true",
            "help": "If this argument is provided, the bandwidth will be interpreted as a distance during kernel "
            "operations. If not, it will be interpreted as the number of nearest neighbors.",
        },
        "-exclude_self": {
            "action": "store_true",
            "help": "When computing spatial weights, do not count the cell itself as a neighbor. Recommended to set to "
            "True for the CCI models because the independent variable array is also spatially-dependent.",
        },
        "-kernel": {
            "default": "bisquare",
            "type": str,
            "help": "Kernel to use when computing spatial weights and when fitting the model.",
        },
        "-use_expression_neighbors": {
            "action": "store_true",
            "help": "The default for finding spatial neighborhoods for the modeling process is to use neighbors in "
            "physical space. If this argument is provided, expression will instead be used to find neighbors.",
        },
        "-distance_membrane_bound": {
            "type": float,
            "help": "In model setup, distance threshold to consider cells as neighbors for membrane-bound ligands. If"
            "provided, will take priority over 'n_neighbors_membrane_bound'.",
        },
        "-distance_secreted": {
            "type": float,
            "help": "In model setup, distance threshold to consider cells as neighbors for secreted or ECM ligands. If"
            "provided, will take priority over 'n_neighbors_secreted'.",
        },
        "-n_neighbors_membrane_bound": {
            "default": 6,
            "type": int,
            "help": "Only used if `mod_type` is 'niche', to define the number of neighbors to consider for each "
            "cell when defining the independent variable array for membrane-bound ligands. Will also be used to "
            "define the number of neighbors to consider for lagged ligand expression.",
        },
        "-n_neighbors_secreted": {
            "default": 25,
            "type": int,
            "help": "Only used if `mod_type` is 'niche', to define the number of neighbors to consider for each "
            "cell when defining the independent variable array for secreted or ECM ligands. Will also be used to "
            "define the number of neighbors to consider for lagged ligand expression.",
        },
        "-distr": {"default": "gaussian", "type": str},
        "-fit_intercept": {"action": "store_true"},
        "-tolerance": {"default": 1e-3, "type": float},
        "-max_iter": {"default": 500, "type": int},
        "-patience": {"default": 5, "type": int},
        "-ridge_lambda": {"default": 1.0, "type": float},
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
            "action": "store_true",
            "help": "Used for downstream analyses, specifically :func `inferred_effect_direction`; if True, "
            "will subset to only the targets that were predicted well by the model.",
        },
        "-filter_target_threshold": {
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
        "-n_components": {
            "default": 20,
            "type": int,
            "help": "Used for :func `CCI_sender_deg_detection` and :func `CCI_receiver_deg_detection`; determines the "
            "dimensionality of the space to embed into using UMAP.",
        },
        "-cci_degs_model_interactions": {
            "action": "store_true",
            "help": "Used for :func `CCI_sender_deg_detection`; if True, will consider transcription factor "
            "interactions with cofactors and other transcription factors, with these interactions combined into "
            "features. If False, will use each cofactor independently in the prediction.",
        },
        "-no_cell_type_markers": {
            "action": "store_true",
            "help": "Used for :func `CCI_receiver_deg_detection`; if True, will exclude cell type "
            "markers from the set of genes for which to compare to sent/received signal.",
        },
        "-compute_pathway_effect": {
            "action": "store_true",
            "help": "Used for :func `inferred_effect_direction`; if True, will summarize the effects of all "
            "ligands/ligand-receptor interactions in a pathway.",
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
            if isinstance(value, Iterable):
                check = next(iter(value))
            else:
                check = value
            if not isinstance(check, arg_type):
                raise TypeError(f"Argument {key} must be of type {arg_type}.")

        if arg_info.get("action") == "store_true" and not isinstance(value, bool):
            raise TypeError(f"Argument {key} must be of type bool.")

        # Check for iterable to allow input to be set, list, tuple, etc. Also a single string is fine. Note that
        # currently all arguments that allow this option take string inputs, so there is not an explicit need to
        # check for typing matches:
        if arg_info.get("nargs") is not None:
            if not isinstance(value, Iterable):
                raise TypeError(f"Argument {key} must be an iterable.")

            element_type = type(next(iter(value)))
            if not all(isinstance(element, element_type) for element in value):
                raise TypeError(f"Argument {key} must be an iterable containing values of type {element_type}.")

    # Initialize parser:
    parser = argparse.ArgumentParser(description="MuSIC arguments", allow_abbrev=False)

    # Use arg_dict to populate the parser:
    for arg, arg_info in arg_dict.items():
        parser.add_argument(arg, **arg_info)

    # Compile arguments list:
    all_args = []
    for key, value in kwargs.items():
        # Check if key is valid:
        if key not in arg_dict.keys():
            logger.info(
                f"Argument {key} not recognized and will be skipped and not included in the final processed " f"list."
            )
            continue
        all_args.append(key)
        # If Boolean, don't do anything- giving the argument will store it as True and so appending the key is enough:
        if isinstance(value, bool):
            continue
        # Check for arguments that allow multiple inputs:
        arg_info = arg_dict[key]
        if arg_info.get("nargs") is not None:
            if isinstance(value, (List, Tuple, Set)):
                all_args.extend(map(str, value))
        all_args.append(str(value))

    return parser, all_args


if __name__ == "__main__":
    # Run from the command line:
    parser = argparse.ArgumentParser(description="MuSIC arguments", allow_abbrev=False)

    parser.add_argument("-run_upstream", action="store_true")

    parser.add_argument("-adata_path", type=str)
    parser.add_argument(
        "-csv_path",
        type=str,
        help="Can be used to provide a .csv file, containing gene expression data or any other kind of data. "
        "Assumes the first three columns contain x- and y-coordinates and then dependent variable "
        "values, in that order.",
    )
    parser.add_argument(
        "-n_spatial_dim_csv",
        type=int,
        default=2,
        help="If using a .csv file, specifies the number of spatial dimensions. Default is 2.",
    )
    parser.add_argument(
        "-spatial_subsample",
        action="store_true",
        help="Recommended for large datasets (>5000 samples), otherwise model fitting is quite slow.",
    )
    parser.add_argument("-mod_type", type=str, default="niche")
    parser.add_argument("-include_unpaired_lr", action="store_true")
    parser.add_argument("-cci_dir", type=str)
    parser.add_argument("-species", type=str, default="human")
    parser.add_argument(
        "-output_path",
        default="./output/stgwr_results.csv",
        type=str,
        help="Path to output file. Make sure the parent directory is empty- "
        "any existing files will be deleted. It is recommended to create "
        "a new folder to serve as the output directory. This should be "
        "supplied of the form '/path/to/file.csv', where file.csv will "
        "store coefficients. The name of the target will be appended at runtime.",
    )
    parser.add_argument("-custom_lig_path", type=str)
    parser.add_argument(
        "-ligand",
        nargs="+",
        type=str,
        help="Alternative to the custom ligand path, can be used to provide a custom list of ligands.",
    )
    parser.add_argument("-custom_rec_path", type=str)
    parser.add_argument(
        "-receptor",
        nargs="+",
        type=str,
        help="Alternative to the custom receptor path, can be used to provide a custom list of receptors.",
    )
    parser.add_argument("-custom_pathways_path", type=str)
    parser.add_argument(
        "-pathway",
        nargs="+",
        type=str,
        help="Alternative to the custom pathway path, can be used to provide a custom list of pathways.",
    )
    parser.add_argument("-targets_path", type=str)
    parser.add_argument(
        "-target",
        nargs="+",
        type=str,
        help="Alternative to the custom target path, can be used to provide a custom list of target molecules.",
    )
    parser.add_argument("-init_betas_path", type=str)

    parser.add_argument("-normalize", action="store_true")
    parser.add_argument("-smooth", action="store_true")
    parser.add_argument("-log_transform", action="store_true")
    parser.add_argument(
        "-normalize_signaling",
        action="store_true",
        help="For ligand, receptor or L:R models, "
        "normalize computed signaling values. This should be used to find signaling effects that may be mediated by "
        "rarer signals.",
    )
    parser.add_argument(
        "-target_expr_threshold",
        default=0.05,
        type=float,
        help="For automated selection, the threshold proportion of cells for which transcript "
        "needs to be expressed in to be selected as a target of interest. Not used if 'targets_path' is not None.",
    )
    parser.add_argument(
        "-multicollinear_threshold",
        type=float,
        help="Used only if `mod_type` is 'slice'. If this argument is provided, independent variables that are highly "
        "correlated will be filtered out based on variance inflation factor threshold. A value of 5 or 10 is "
        "recommended. This can be useful in reducing computation time.",
    )

    parser.add_argument("-coords_key", default="spatial", type=str)
    parser.add_argument(
        "-group_key",
        default="cell_type",
        type=str,
        help="Key to entry in .obs containing cell type or other category labels.",
    )
    parser.add_argument(
        "-group_subset",
        nargs="+",
        type=str,
        help="If provided, only cells with labels that correspond to these group(s) will be used as prediction "
        "targets. Will search in key corresponding to the input to arg 'cell_type' if given.",
    )
    parser.add_argument(
        "-covariate_keys",
        nargs="+",
        type=str,
        help="Any number of keys to entry in .obs or .var_names of an "
        "AnnData object. Values here will be added to"
        "the model as covariates.",
    )
    parser.add_argument(
        "-total_counts_key",
        default="total_counts",
        type=str,
        help="Key to entry in .obs containing total counts per cell.",
    )
    parser.add_argument(
        "-total_counts_threshold",
        default=0.0,
        type=float,
        help="If provided, cells with total counts below this threshold will be removed.",
    )

    parser.add_argument("-bw")
    parser.add_argument("-minbw")
    parser.add_argument("-maxbw")
    parser.add_argument(
        "-bw_fixed",
        action="store_true",
        help="If this argument is provided, the bandwidth will be "
        "interpreted as a distance during kernel operations. If not, it will be interpreted "
        "as the number of nearest neighbors.",
    )
    parser.add_argument(
        "-exclude_self",
        action="store_true",
        help="When computing spatial weights, do not count the "
        "cell itself as a neighbor. Recommended to set to "
        "True for the CCI models because the independent "
        "variable array is also spatially-dependent.",
    )
    parser.add_argument(
        "-kernel",
        default="bisquare",
        type=str,
        help="Kernel to use when computing spatial weights and fitting the model.",
    )
    parser.add_argument(
        "-distance_membrane_bound",
        type=float,
        help="In model setup, distance threshold to consider cells as neighbors for membrane-bound ligands. If"
        "provided, will take priority over 'n_neighbors_membrane_bound'.",
    )
    parser.add_argument(
        "-distance_secreted",
        type=float,
        help="In model setup, distance threshold to consider cells as neighbors for secreted or ECM ligands. If"
        "provided, will take priority over 'n_neighbors_secreted'.",
    )
    parser.add_argument(
        "-n_neighbors_membrane_bound",
        default=8,
        type=int,
        help="Only used if `mod_type` is 'niche', to define the number of neighbors "
        "to consider for each cell when defining the independent variable array for "
        "membrane-bound ligands.",
    )
    parser.add_argument(
        "-n_neighbors_secreted",
        default=25,
        type=int,
        help="Only used if `mod_type` is 'niche', to define the number of neighbors "
        "to consider for each cell when defining the independent variable array for "
        "secreted or ECM ligands.",
    )
    parser.add_argument(
        "-use_expression_neighbors",
        action="store_true",
        help="The default for finding spatial neighborhoods for the modeling process is to use neighbors in "
        "physical space. If this argument is provided, expression will be used instead to find neighbors.",
    )

    parser.add_argument("-distr", default="gaussian", type=str)
    parser.add_argument("-fit_intercept", action="store_true")
    # parser.add_argument("-include_offset", action="store_true")
    parser.add_argument(
        "-no_hurdle",
        action="store_true",
        help="If True, do not implement spatially-weighted hurdle model- will only perform generalized linear "
        "modeling.",
    )
    parser.add_argument("-tolerance", default=1e-3, type=float)
    parser.add_argument("-max_iter", default=500, type=int)
    parser.add_argument("-patience", default=5, type=int)
    parser.add_argument("-ridge_lambda", default=0.3, type=float)

    parser.add_argument(
        "-chunks",
        default=1,
        type=int,
        help="For use if `multiscale` is True- increase the number of parallel processes. Can be used to help prevent"
        "memory from running out, otherwise keep as low as possible.",
    )

    # Options for downstream analysis:
    parser.add_argument(
        "-search_bw",
        help="Used for downstream analyses; specifies the bandwidth to search for "
        "senders/receivers. Recommended to set equal to the bandwidth of a fitted "
        "model.",
    )
    parser.add_argument(
        "-top_k_receivers",
        default=10,
        type=int,
        help="Used for downstream analyses, specifically for :func `define_effect_vf`; specifies the number of "
        "nearest neighbors to consider when computing signaling effect vectors.",
    )
    parser.add_argument(
        "-filter_targets",
        action="store_true",
        help="Used for downstream analyses, specifically :func `infer_effect_direction`; if True, will subset to only "
        "the targets that were predicted well by the model.",
    )
    parser.add_argument(
        "-filter_target_threshold",
        default=0.65,
        type=float,
        help="Used for downstream analyses, specifically :func `infer_effect_direction`; specifies the threshold "
        "Pearson coefficient for target subsetting. Only used if `filter_targets` is True.",
    )
    parser.add_argument(
        "-diff_sending_or_receiving",
        default="sending",
        type=str,
        help="Used for downstream analyses, specifically :func `sender_receiver_effect_deg_detection`; specifies "
        "whether to compute differential expression of genes in cells with high or low sending effect potential "
        "('sending cells') or high or low receiving effect potential ('receiving cells').",
    )
    parser.add_argument(
        "-target_for_downstream",
        nargs="+",
        type=str,
        help="Used for :func `get_effect_potential`, :func `get_pathway_potential` and :func "
        "`calc_and_group_sender_receiver_effect_degs` (provide only one target), as well as :func "
        "`compute_cell_type_coupling` (can provide multiple targets). Used to specify the target "
        "gene(s) to analyze with these functions.",
    )
    parser.add_argument(
        "-ligand_for_downstream",
        type=str,
        help="Used for :func `get_effect_potential` and :func `calc_and_group_sender_receiver_effect_degs`, "
        "used to specify the ligand gene to consider with respect to the target.",
    )
    parser.add_argument(
        "-receptor_for_downstream",
        type=str,
        help="Used for :func `get_effect_potential` and :func `calc_and_group_sender_receiver_effect_degs`, "
        "used to specify the receptor gene to consider with respect to the target.",
    )
    parser.add_argument(
        "-pathway_for_downstream",
        type=str,
        help="Used for :func `get_pathway_potential` and :func `calc_and_group_sender_receiver_effect_degs`, "
        "used to specify the pathway to consider with respect to the target.",
    )
    parser.add_argument(
        "-sender_ct_for_downstream",
        type=str,
        help="Used for :func `get_effect_potential` and :func `calc_and_group_sender_receiver_effect_degs`, "
        "used to specify the cell type to consider as a sender.",
    )
    parser.add_argument(
        "-receiver_ct_for_downstream",
        type=str,
        help="Used for :func `get_effect_potential` and :func `calc_and_group_sender_receiver_effect_degs`, "
        "used to specify the cell type to consider as a receiver.",
    )
    parser.add_argument(
        "-n_components",
        type=int,
        help="Used for :func `CCI_sender_deg_detection` and :func `CCI_receiver_deg_detection`; determines the "
        "dimensionality of the space to embed into using UMAP.",
    )
    parser.add_argument(
        "-cci_degs_model_interactions",
        type=bool,
        help="Used for :func `CCI_sender_deg_detection`; if True, will consider transcription factor "
        "interactions with cofactors and other transcription factors. If False, will use only "
        "transcription factor expression for prediction.",
    )
    parser.add_argument(
        "-no_cell_type_markers",
        action="store_true",
        help="Used for :func `CCI_receiver_deg_detection`; if True, will exclude cell type markers "
        "from the set of genes for which to compare to sent/received signal.",
    )
    parser.add_argument(
        "-compute_pathway_effect",
        action="store_true",
        help="Used for :func `inferred_effect_direction`; if True, will summarize the effects of all "
        "ligands/ligand-receptor interactions in a pathway.",
    )

    t1 = time.time()

    swr_model = MuSIC(parser)
    swr_model._set_up_model()
    swr_model.fit()
    swr_model.predict_and_save()

    t_last = time.time()

    print("Total Time Elapsed:", np.round(t_last - t1, 2), "seconds")
    print("-" * 60)

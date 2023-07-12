"""
Enables STGWR to be run using the "run" command rather than needing to navigate to and call the main file
(SWR_mpi.py).
"""
import os
import sys

import click

# For now, add Spateo working directory to sys path so compiler doesn't look in the installed packages:
# sys.path.insert(0, "/mnt/c/Users/danie/Desktop/Github/Github/spateo-release-main")
from ...tools import ST_modeling as fast_swr


@click.group()
@click.version_option("0.3.2")
def main():
    pass


@main.command()
@click.option(
    "-np",
    default=2,
    help="Number of processes to use. Note the max number of processes is " "determined by the number of CPUs.",
    required=True,
)
@click.option("-adata_path")
@click.option("-coords_key", default="spatial")
@click.option(
    "-group_key",
    default="cell_type",
    help="Key to entry in .obs containing cell type "
    "or other category labels. Required if "
    "'mod_type' is 'niche' or 'slice'.",
)
@click.option(
    "-group_subset",
    required=False,
    multiple=True,
    help="If provided, only cells with labels that correspond to these group(s) will be used as prediction targets.",
)
@click.option(
    "-csv_path",
    required=False,
    help="Can be used to provide a .csv file, containing gene expression data or any other kind of data. "
    "Assumes the first three columns contain x- and y-coordinates and then dependent variable "
    "values, in that order.",
)
@click.option(
    "-n_spatial_dim_csv",
    default=2,
    help="If csv_path is provided, this argument specifies the number of spatial dimensions (e.g. 2 for X-Y, "
    "3 for X-Y-Z) in the data.",
)
@click.option(
    "-subsample",
    default=False,
    is_flag=True,
    help="Recommended for large datasets (>5000 samples), " "otherwise model fitting is quite slow.",
)
@click.option("-multiscale", default=False, is_flag=True)
@click.option(
    "-mod_type",
    default="niche",
    help="If adata_path is provided, one of the STGWR models " "will be used. Options: 'niche', 'lr', 'slice'.",
)
@click.option("-grn", default=False, is_flag=True)
@click.option("-cci_dir", required=True)
@click.option("-species", default="human")
@click.option(
    "-output_path",
    default="./output/stgwr_results.csv",
    help="Path to output file. Make sure the parent " "directory is empty- any existing files will " "be deleted.",
)
@click.option("-custom_lig_path", required=False)
@click.option("-ligand", required=False, multiple=True)
@click.option("-custom_rec_path", required=False)
@click.option("-receptor", required=False, multiple=True)
@click.option("-custom_pathways_path", required=False)
@click.option("-pathway", required=False, multiple=True)
@click.option("-targets_path", required=False)
@click.option("-target", required=False, multiple=True)
@click.option(
    "-target_expr_threshold",
    default=0.2,
    help="For automated selection, the threshold "
    "proportion of cells for which transcript "
    "needs to be expressed in to be selected as a target of interest. "
    "Not used if 'targets_path' is not None.",
)
@click.option(
    "-multicollinear_threshold",
    required=False,
    help="Used only if `mod_type` is 'slice'. If this argument is provided, independent variables that are highly "
    "correlated will be filtered out based on variance inflation factor threshold. A value of 5 or 10 is "
    "recommended. This can be useful in reducing computation time.",
)
@click.option("-init_betas_path", required=False)
@click.option("-normalize", default=False, is_flag=True)
@click.option("-smooth", default=False, is_flag=True)
@click.option("-log_transform", default=False, is_flag=True)
@click.option(
    "-normalize_signaling",
    default=False,
    is_flag=True,
    help="For ligand, receptor or L:R models, "
    "normalize computed signaling values. This "
    "should be used to find signaling effects that"
    "may be mediated by rarer signals.",
)
@click.option(
    "-covariate_keys",
    required=False,
    multiple=True,
    help="Any number of keys to entry in .obs or "
    ".var_names of an "
    "AnnData object. Values here will be added to"
    "the model as covariates.",
)
@click.option("-bw", required=False)
@click.option("-minbw", required=False)
@click.option("-maxbw", required=False)
@click.option(
    "-bw_fixed",
    default=False,
    is_flag=True,
    help="If this argument is provided, the bandwidth will be "
    "interpreted as a distance during kernel operations. If not, it will be interpreted "
    "as the number of nearest neighbors.",
)
@click.option(
    "-exclude_self",
    default=False,
    is_flag=True,
    help="When computing spatial weights, do not count the "
    "cell itself as a neighbor. Recommended to set to "
    "True for the CCI models because the independent "
    "variable array is also spatially-dependent.",
)
@click.option("-kernel", default="bisquare")
@click.option("-distr", default="gaussian")
@click.option("-n_neighbors_membrane_bound", default=8)
@click.option("-n_neighbors_secreted", default=25)
@click.option(
    "-use_expression_neighbors_only",
    default=False,
    is_flag=True,
    help="The default for finding spatial neighborhoods for the modeling process is to use neighbors in "
    "physical space, and turn to expression space if there is not enough signal in the physical "
    "neighborhood. If this argument is provided, only expression will be used to find neighbors.",
)
@click.option("-fit_intercept", default=False, is_flag=True)
@click.option(
    "-include_offset",
    default=False,
    is_flag=True,
    help="Set True to include offset to account for "
    "differences in library size in predictions. If "
    "True, will compute scaling factor using trimmed "
    "mean of M-value with singleton pairing (TMMswp).",
)
@click.option("-no_hurdle", default=False, is_flag=True)
@click.option("-tolerance", default=1e-3)
@click.option("-max_iter", default=1000)
@click.option(
    "-patience",
    default=5,
    help="Number of iterations to wait before stopping if parameters have "
    "stabilized. Only used if `multiscale` is True.",
)
@click.option("-ridge_lambda", required=False)

# Downstream analysis arguments
@click.option("-search_bw", default=10)
@click.option(
    "-top_k_receivers",
    default=10,
    help="Used for :func `infer_effect_direction`; specifies the number of top "
    "senders/receivers to consider for each cell.",
)
@click.option(
    "-filter_targets",
    default=False,
    is_flag=True,
    help="Used for :func `infer_effect_direction`; if True, will subset to only "
    "the targets that were predicted well by the model.",
)
@click.option(
    "-filter_target_threshold",
    default=0.65,
    help="Used for :func `infer_effect_direction`; specifies the "
    "threshold "
    "Pearson coefficient for target subsetting. Only used if `filter_targets` is True.",
)
@click.option(
    "-diff_sending_or_receiving",
    default="sending",
    help="Used for :func `sender_receiver_effect_deg_detection`; specifies "
    "whether to compute differential expression of genes in cells with high or low sending effect potential "
    "('sending cells') or high or low receiving effect potential ('receiving cells').",
)
@click.option(
    "-target_for_downstream",
    required=False,
    multiple=True,
    help="Used for :func `get_effect_potential`, :func `get_pathway_potential` and :func "
    "`calc_and_group_sender_receiver_effect_degs` (provide only one target), as well as :func "
    "`compute_cell_type_coupling` (can provide multiple targets). Used to specify the target "
    "gene(s) to analyze with these functions.",
)
@click.option(
    "-ligand_for_downstream",
    required=False,
    help="Used for :func `get_effect_potential` and :func `calc_and_group_sender_receiver_effect_degs`, "
    "used to specify the ligand gene to consider with respect to the target.",
)
@click.option(
    "-receptor_for_downstream",
    required=False,
    help="Used for :func `get_effect_potential` and :func `calc_and_group_sender_receiver_effect_degs`, "
    "used to specify the receptor gene to consider with respect to the target.",
)
@click.option(
    "-pathway_for_downstream",
    required=False,
    help="Used for :func `get_pathway_potential` and :func `calc_and_group_sender_receiver_effect_degs`, "
    "used to specify the pathway to consider with respect to the target.",
)
@click.option(
    "-sender_ct_for_downstream",
    required=False,
    help="Used for :func `get_effect_potential` and :func `calc_and_group_sender_receiver_effect_degs`, "
    "used to specify the cell type to consider as a sender.",
)
@click.option(
    "-receiver_ct_for_downstream",
    required=False,
    help="Used for :func `get_effect_potential` and :func `calc_and_group_sender_receiver_effect_degs`, "
    "used to specify the cell type to consider as a receiver.",
)
@click.option(
    "-cci_degs_model_interactions",
    default=False,
    is_flag=True,
    help="Used for :func `CCI_sender_deg_detection`; if True, will consider transcription factor "
    "interactions with cofactors and other transcription factors. If False, will use only "
    "transcription factor expression for prediction.",
)
@click.option(
    "-no_cell_type_markers",
    default=False,
    is_flag=True,
    help="Used for :func `CCI_receiver_deg_detection`; if True, will exclude cell type markers from the set of "
    "genes for which to compare to sent/received signal.",
)
@click.option(
    "-compute_pathway_effect",
    default=False,
    is_flag=True,
    help="Used for :func `inferred_effect_direction`; if True, will summarize the effects of all "
    "ligands/ligand-receptor interactions in a pathway.",
)
def run(
    np,
    adata_path,
    coords_key,
    group_key,
    group_subset,
    csv_path,
    n_spatial_dim_csv,
    multiscale,
    multiscale_params_only,
    mod_type,
    grn,
    cci_dir,
    species,
    output_path,
    custom_lig_path,
    ligand,
    custom_rec_path,
    receptor,
    custom_pathways_path,
    pathway,
    targets_path,
    target,
    target_expr_threshold,
    init_betas_path,
    normalize,
    smooth,
    log_transform,
    normalize_signaling,
    covariate_keys,
    bw,
    minbw,
    maxbw,
    bw_fixed,
    exclude_self,
    kernel,
    n_neighbors_membrane_bound,
    n_neighbors_secreted,
    use_expression_neighbors_only,
    distr,
    fit_intercept,
    include_offset,
    no_hurdle,
    tolerance,
    max_iter,
    patience,
    ridge_lambda,
    search_bw,
    top_k_receivers,
    filter_targets,
    filter_target_threshold,
    diff_sending_or_receiving,
    target_for_downstream,
    ligand_for_downstream,
    receptor_for_downstream,
    pathway_for_downstream,
    sender_ct_for_downstream,
    receiver_ct_for_downstream,
    cci_degs_model_interactions,
    no_cell_type_markers,
    compute_pathway_effect,
    chunks,
):
    """Command line shortcut to run any STGWR models.

    Args:
        n_processes: Number of processes to use. Note the max number of processes is determined by the number of CPUs.
        adata_path: Path to AnnData object containing gene expression data
        coords_key: Key to entry in .obs containing x- and y-coordinates
        group_key: Key to entry in .obs containing cell type or other category labels. Required if 'mod_type' is
            'niche' or 'slice'.
        csv_path: Can be used to provide a .csv file, containing gene expression data or any other kind of data.
            Assumes the first three columns contain x- and y-coordinates and then dependent variable values,
            in that order.
        n_spatial_dim_csv: If csv_path is provided, this argument specifies the number of spatial dimensions (e.g. 2
            for X-Y, 3 for X-Y-Z) in the data
        multiscale: If True, the MGWR model will be used
        multiscale_params_only: If True, will only fit parameters for MGWR model and no other metrics. Otherwise,
            the effective number of parameters and leverages will be returned.
        mod_type: If adata_path is provided, one of the SWR models will be used. Options: 'niche', 'lr', 'ligand',
            'receptor'.


        cci_dir: Path to directory containing CCI files
        species: Species for which CCI files were generated. Options: 'human', 'mouse'.
        output_path: Path to output file
        custom_lig_path: Path to file containing a list of ligands to be used in the GRN model
        ligand: Can be used as an alternative to `custom_lig_path`. Can be used to provide a custom list of ligands.
        custom_rec_path: Path to file containing a list of receptors to be used in the GRN model
        receptor: Can be used as an alternative to `custom_rec_path`. Can be used to provide a custom list of
            receptors.
        custom_pathways_path: Rather than providing a list of receptors, can provide a list of signaling pathways-
            all receptors with annotations in this pathway will be included in the model. Only used if :attr `mod_type`
            is "lr".
        targets_path: Path to file containing a list of targets to be used in the GRN model
        target: Can be used as an alternative to `targets_path`. Can be used to provide a custom list of
            targets.
        target_expr_threshold: For automated selection, the threshold proportion of cells for which transcript needs
            to be expressed in to be selected as a target of interest.


        init_betas_path: Path to file containing initial values for beta coefficients
        normalize: If True, the data will be normalized
        smooth: If True, the data will be smoothed
        log_transform: If True, the data will be log-transformed
        normalize_signaling: If True, will minmax scale the signaling matrix- meant to find and characterize rarer
            signaling patterns
        covariate_keys: Any number of keys to entry in .obs or .var_names of an AnnData object. Values here will
            be added to the model as covariates.
        bw: Bandwidth to use for spatial weights
        minbw: Minimum bandwidth to use for spatial weights
        maxbw: Maximum bandwidth to use for spatial weights
        bw_fixed: If this argument is provided, the bandwidth will be interpreted as a distance during kernel
            operations. If not, it will be interpreted as the number of nearest neighbors.
        exclude_self: When computing spatial weights, do not count the cell itself as a neighbor. Recommended to
            set to True for the CCI models because the independent variable array is also spatially-dependent.
        kernel: Kernel to use for spatial weights. Options: 'bisquare', 'quadratic', 'gaussian', 'triangular',
            'uniform', 'exponential'.
        n_neighbors_membrane_bound: Only used if `mod_type` is 'niche', to define the number of neighbors  to consider
            for each cell when defining the independent variable array; used for membrane-bound ligands. Defaults to 8.
        n_neighbors_secreted: Only used if `mod_type` is 'niche', to define the number of neighbors  to consider
            for each cell when defining the independent variable array; used for secreted and ECM ligands. Defaults
            to 25.
        use_expression_neighbors_only: The default for finding spatial neighborhoods for the modeling process is to
            use neighbors in physical space, and turn to expression space if there is not enough signal in the physical
            neighborhood. If this argument is provided, only expression will be used to find neighbors.
        distr: Distribution to use for spatial weights. Options: 'gaussian', 'poisson', 'nb'.
        fit_intercept: If True, will include intercept in model
        include_offset: If True, include offset to account for differences in library size in predictions. If True,
            will compute scaling factor using trimmed mean of M-value with singleton pairing (TMMswp).
        no_hurdle: If True, will implement spatially-weighted hurdle model to attempt to account for biological zeros.
        tolerance: Tolerance for convergence of model
        max_iter: Maximum number of iterations for model
        patience: Number of iterations to wait before stopping if parameters have stabilized. Only used if
            `multiscale` is True.
        ridge_lambda: Ridge lambda value to use for L2 regularization
        chunks: Number of chunks for multiscale computation (default: 1). Increase the number if run out of memory
            but should keep it as low as possible.


        search_bw: Used for downstream analyses; specifies the bandwidth to search for senders/receivers.
            Recommended to set equal to the `n_neighbors_membrane_bound` argument given during model fitting.
        top_k_receivers: Used for downstream analyses, specifically :func `infer_effect_direction`; if True,
            will subset to only the targets that were predicted well by the model.
        filter_targets: Used for downstream analyses, specifically :func `infer_effect_direction`; if True,
            will subset to only the targets that were predicted well by the model.
        filter_targets_threshold: Used for downstream analyses, specifically :func `infer_effect_direction`;
            specifies the threshold Pearson coefficient for target subsetting. Only used if `filter_targets` is True.
        diff_sending_or_receiving: Used for downstream analyses, specifically :func
            `sender_receiver_effect_deg_detection`; specifies whether to compute differential expression of genes
            in cells with high or low sending effect potential ('sending cells') or high or low receiving effect
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
        cci_degs_model_interactions: Used for :func `CCI_sender_deg_detection`; if True, will consider
            transcription factor interactions with cofactors and other transcription factors. If False, will use only
            transcription factor expression for prediction.
        no_cell_type_markers: For downstream analyses; used for :func `calc_and_group_sender_receiver_effect_degs`;
            if True, will exclude cell type markers from the set of genes for which to compare to sent/received signal.
        compute_pathway_effect: For downstream analyses; used for :func `inferred_effect_direction`; if True,
            will summarize the effects of all ligands/ligand-receptor interactions in a pathway.
    """

    mpi_path = os.path.dirname(fast_swr.__file__) + "/SWR_mpi.py"

    command = (
        "mpiexec "
        + " -np "
        + str(np)
        + " python "
        + mpi_path
        + " -mod_type "
        + mod_type
        + " -species "
        + species
        + " -output_path "
        + output_path
        + " -target_expr_threshold "
        + str(target_expr_threshold)
        + " -coords_key "
        + coords_key
        + " -group_key "
        + group_key
        + " -kernel "
        + kernel
        + " -distr "
        + distr
        + " -tolerance "
        + str(tolerance)
        + " -max_iter "
        + str(max_iter)
        + " -patience "
        + str(patience)
    )

    if adata_path is not None:
        command += " -adata_path " + adata_path
    elif csv_path is not None:
        command += " -csv_path " + csv_path

    if n_spatial_dim_csv is not None:
        command += " -n_spatial_dim_csv " + str(n_spatial_dim_csv)

    if group_subset is not None:
        command += " -group_subset "
        for key in group_subset:
            command += key + " "

    if multiscale:
        command += " -multiscale "
    if multiscale_params_only:
        command += " -multiscale_params_only "
    if grn:
        command += " -grn "
    if cci_dir is not None:
        command += " -cci_dir " + cci_dir

    if custom_lig_path is not None:
        command += " -custom_lig_path " + custom_lig_path
    if ligand is not None:
        command += " -ligand "
        for lig in ligand:
            command += lig + " "

    if custom_rec_path is not None:
        command += " -custom_rec_path " + custom_rec_path
    if receptor is not None:
        command += " -receptor "
        for rec in receptor:
            command += rec + " "

    if custom_pathways_path is not None:
        command += " -custom_pathways_path " + custom_pathways_path
    if pathway is not None:
        command += " -pathway "
        for path in pathway:
            command += path + " "

    if targets_path is not None:
        command += " -targets_path " + targets_path
    if target is not None:
        command += " -target "
        for tar in target:
            command += tar + " "

    if init_betas_path is not None:
        command += " -init_betas_path " + init_betas_path
    if normalize:
        command += " -normalize "
    if smooth:
        command += " -smooth "
    if log_transform:
        command += " -log_transform "
    if normalize_signaling:
        command += " -normalize_signaling "
    if covariate_keys is not None:
        command += " -covariate_keys "
        for key in covariate_keys:
            command += key + " "
    if bw is not None:
        command += " -bw " + str(bw)
    if minbw is not None:
        command += " -minbw " + str(minbw)
    if maxbw is not None:
        command += " -maxbw " + str(maxbw)
    if bw_fixed:
        command += " -bw_fixed "
    if n_neighbors_membrane_bound is not None:
        command += "-n_neighbors_membrane_bound " + str(n_neighbors_membrane_bound)
    if n_neighbors_secreted is not None:
        command += "-n_neighbors_secreted " + str(n_neighbors_secreted)
    if use_expression_neighbors_only:
        command += " -use_expression_neighbors_only "
    if exclude_self:
        command += " -exclude_self "
    if fit_intercept:
        command += " -fit_intercept "
    if include_offset:
        command += " -include_offset "
    if no_hurdle:
        command += " -no_hurdle "
    if chunks is not None:
        command += " -chunks " + str(chunks)
    if ridge_lambda is not None:
        command += " -ridge_lambda " + str(ridge_lambda)

    if search_bw is not None:
        command += " -search_bw " + str(search_bw)
    if top_k_receivers is not None:
        command += " -top_k_receivers " + str(top_k_receivers)
    if filter_targets:
        command += " -filter_targets "
    if filter_target_threshold is not None:
        command += " -filter_target_threshold " + str(filter_target_threshold)
    if diff_sending_or_receiving is not None:
        command += " -diff_sending_or_receiving " + str(diff_sending_or_receiving)
    if target_for_downstream is not None:
        command += " -target "
        for tar in target_for_downstream:
            command += tar + " "
    if ligand_for_downstream is not None:
        command += " -ligand " + ligand_for_downstream
    if receptor_for_downstream is not None:
        command += " -receptor " + receptor_for_downstream
    if pathway_for_downstream is not None:
        command += " -pathway " + pathway_for_downstream
    if sender_ct_for_downstream is not None:
        command += " -sender_ct " + sender_ct_for_downstream
    if receiver_ct_for_downstream is not None:
        command += " -receiver_ct " + receiver_ct_for_downstream
    if cci_degs_model_interactions:
        command += " -cci_degs_model_interactions "
    if no_cell_type_markers:
        command += " -no_cell_type_markers "
    if compute_pathway_effect:
        command += " -compute_pathway_effect "

    os.system(command)
    pass


if __name__ == "__main__":
    main()

# ADD UPSTREAM PROCESSING COMMANDS BELOW:


# ADD INTERPRETATION COMMANDS BELOW:
# Note: for the interpretation command that clusters the sender-receiver effect DEGs, call a wrapper function that
# first computes sender_receiver_effect_deg_detection, returns GAM_adata and bs_obj.

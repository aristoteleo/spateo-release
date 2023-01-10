# -*- coding: utf-8 -*-
"""
@File    :   cci_two_cluster.py
@Time    :   2022/07/03 11:50:40
@Author  :   LuluZuo
@Version :   1.0
@Desc    :   spatial cell cell communication
"""

import random
from typing import Tuple

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import issparse
from tqdm import tqdm as tqdm

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from ..configuration import SKM
from ..logging import logger_manager as lm
from .cci_fdr import fdr_correct


@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE, "adata")
def find_cci_two_group(
    adata: AnnData,
    path: str,
    species: Literal["human", "mouse", "drosophila", "zebrafish", "axolotl"] = "human",
    layer: Tuple[None, str] = None,
    group: str = None,
    lr_pair: list = None,
    sender_group: str = None,
    receiver_group: str = None,
    filter_lr: Literal["outer", "inner"] = "outer",
    top: int = 20,
    spatial_neighbors: str = "spatial_neighbors",
    spatial_distances: str = "spatial_distances",
    min_cells_by_counts: int = 0,
    min_pairs: int = 5,
    min_pairs_ratio: float = 0.01,
    num: int = 1000,
    pvalue: float = 0.05,
) -> dict:
    """Performing cell-cell transformation on an anndata object, while also
       limiting the nearest neighbor per cell to n_neighbors. This function returns
       a dictionary, where the key is 'cell_pair' and 'lr_pair'.

    Args:
        adata: An Annodata object.
        path: Path to ligand_receptor network of NicheNet (prior lr_network).
        species: Which species is your adata generated from. Will be used to determine the proper ligand-receptor
            database.
        layer: the key to the layer. If it is None, adata.X will be used by default.
        group: The group name in adata.obs
        lr_pair: given a lr_pair list.
        sender_group: the cell group name of send ligands.
        receiver_group: the cell group name of receive receptors.
        spatial_neighbors:  spatial neighbor key {spatial_neighbors} in adata.uns.keys(),
        spatial_distances: spatial neighbor distance key {spatial_distances} in adata.obsp.keys().
        min_cells_by_counts: threshold for minimum number of cells expressing ligand/receptor to avoid being filtered
            out. Only used if 'lr_pair' is None.
        min_pairs: minimum number of cell pairs between cells from two groups.
        min_pairs_ratio: minimum ratio of cell pairs to theoretical cell pairs (n x M / 2) between cells
            from two groups.
        num: number of permutations. It is recommended that this number be at least 1000.
        pvalue: the p-value threshold that will be used to filter for significant ligand-receptor pairs.
        filter_lr: filter ligand and receptor based on specific expressed in sender groups
            and receiver groups. 'inner': specific both in sender groups and receiver groups;
            'outer': specific in sender groups or receiver groups.
        top: the number of top expressed fraction in given sender groups(receiver groups)
            for each gene(ligand or receptor).

    Returns:
        result_dict: a dictionary where the key is 'cell_pair' and 'lr_pair'.
    """

    logger = lm.get_main_logger()

    # prior lr_network
    if species == "human":
        lr_network = pd.read_csv(path + "lr_network_human.csv", index_col=0)
        lr_network["lr_pair"] = lr_network["from"].str.cat(lr_network["to"], sep="-")
    elif species == "mouse":
        lr_network = pd.read_csv(path + "lr_network_mouse.csv", index_col=0)
        lr_network["lr_pair"] = lr_network["from"].str.cat(lr_network["to"], sep="-")
    elif species == "drosophila":
        lr_network = pd.read_csv(path + "lr_network_drosophila.csv", index_col=0)
        lr_network["lr_pair"] = lr_network["from"].str.cat(lr_network["to"], sep="-")
    elif species == "zebrafish":
        lr_network = pd.read_csv(path + "lr_network_zebrafish.csv", index_col=0)
        lr_network["lr_pair"] = lr_network["from"].str.cat(lr_network["to"], sep="-")
    elif species == "axolotl":
        lr_network = pd.read_csv(path + "lr_network_axolotl.csv", index_col=0)
        lr_network["lr_pair"] = lr_network["human_ligand"].str.cat(lr_network["human_receptor"], sep="-")
    # layer
    if layer is None:
        adata.X = adata.X
    else:
        adata.X = adata.layers[layer]

    x_sparse = issparse(adata.X)

    # filter lr

    if lr_pair is None:
        # expressed lr_network in our data
        ligand = lr_network["from"].unique()
        expressed_ligand = list(set(ligand) & set(adata.var_names))
        if len(expressed_ligand) == 0:
            raise ValueError(f"No intersected ligand between your adata object and lr_network dataset.")
        lr_network = lr_network[lr_network["from"].isin(expressed_ligand)]
        receptor = lr_network["to"].unique()
        expressed_receptor = list(set(receptor) & set(adata.var_names))
        if len(expressed_receptor) == 0:
            raise ValueError(f"No intersected receptor between your adata object and lr_network dataset.")
        lr_network = lr_network[lr_network["to"].isin(expressed_receptor)]

        # ligand_sender_spec
        adata_l = adata[:, list(set(lr_network["from"]))]
        for g in adata.obs[group].unique():
            # Of all cells expressing particular ligand, what proportion are group g:
            frac = (adata_l[adata_l.obs[group] == g].X > 0).sum(axis=0) / (adata_l.X > 0).sum(axis=0)
            adata_l.var[g + "_frac"] = np.asarray(frac.A1) if x_sparse else np.asarray(frac)

        # Check if preprocessing has already been done:
        if "n_cells_by_counts" not in adata_l.var_keys():
            if issparse(adata_l.X):
                adata_l.var["n_cells_by_counts"] = adata_l.X.getnnz(axis=0)
            else:
                adata_l.var["n_cells_by_counts"] = np.count_nonzero(adata_l.X, axis=0)

        dfl = adata_l.var[adata_l.var[sender_group + "_frac"] > 0]
        dfl = dfl[dfl["n_cells_by_counts"] > min_cells_by_counts]

        ligand_sender_spec = dfl.sort_values(by=sender_group + "_frac", ascending=False)[:top].index
        logger.info(
            f"{top} ligands for cell type {sender_group} with highest fraction of prevalence: "
            f"{list(ligand_sender_spec)}. Testing interactions involving these genes."
        )
        lr_network_l = lr_network.loc[lr_network["from"].isin(ligand_sender_spec.tolist())]

        # receptor_receiver_spec
        adata_r = adata[:, list(set(lr_network["to"]))]
        for g in adata.obs[group].unique():
            # Of all cells expressing particular receptor, what proportion are group g:
            frac = (adata_r[adata_r.obs[group] == g].X > 0).sum(axis=0) / (adata_r.X > 0).sum(axis=0)
            adata_r.var[g + "_frac"] = np.asarray(frac.A1) if x_sparse else np.asarray(frac)

        # Check if preprocessing has already been done:
        if "n_cells_by_counts" not in adata_r.var_keys():
            if issparse(adata_r.X):
                adata_r.var["n_cells_by_counts"] = adata_r.X.getnnz(axis=0)
            else:
                adata_r.var["n_cells_by_counts"] = np.count_nonzero(adata_r.X, axis=0)

        dfr = adata_r.var[adata_r.var[receiver_group + "_frac"] > 0]
        dfr = dfr[dfr["n_cells_by_counts"] > min_cells_by_counts]

        receptor_receiver_spec = dfr.sort_values(by=receiver_group + "_frac", ascending=False)[:top].index
        logger.info(
            f"{top} receptors for cell type {receiver_group} with highest fraction of prevalence: "
            f"{list(set(receptor_receiver_spec))}. Testing interactions involving these genes."
        )
        lr_network_r = lr_network.loc[lr_network["to"].isin(receptor_receiver_spec.tolist())]

        if filter_lr == "inner":
            # inner merge
            lr_network_inner = lr_network_l.merge(lr_network_r, how="inner", on=["from", "to"])
            lr_network = lr_network.loc[
                lr_network["from"].isin(lr_network_inner["from"].tolist())
                & lr_network["to"].isin(lr_network_inner["to"].tolist())
            ]
        elif filter_lr == "outer":
            # outer merge
            lr_network = pd.concat([lr_network_l, lr_network_r], axis=0, join="outer")
            lr_network.drop_duplicates(keep="first", inplace=True)
    else:
        lr_network = lr_network.loc[lr_network["lr_pair"].isin(lr_pair)]

    # find cell_pair

    # cell_pair_all
    sender_id = adata[adata.obs[group].isin([sender_group])].obs.index
    receiver_id = adata[adata.obs[group].isin([receiver_group])].obs.index
    cell_pair_all = len(sender_id) * len(receiver_id) / 2

    # spatial constrain cell pair
    nw = {"neighbors": adata.uns[spatial_neighbors]["indices"], "weights": adata.obsp[spatial_distances]}
    k = adata.uns[spatial_neighbors]["params"]["n_neighbors"]

    # cell_pair:all cluster spatial constrain cell pair
    cell_pair = []
    for i, cell_id in enumerate(nw["neighbors"]):
        # - sometimes will be used in adata.obs_names, use >-<in stead
        cell_pair.append(str(adata.obs.index[i]) + ">-<" + adata.obs.index[cell_id])
    cell_pair = [i for j in cell_pair for i in j]
    cell_pair = pd.DataFrame({"cell_pair_name": cell_pair})
    cell_pair[["cell_sender", "cell_receiver"]] = cell_pair["cell_pair_name"].str.split(">-<", 2, expand=True)
    # cell_pair:sender_group
    cell_pair = cell_pair.loc[cell_pair["cell_sender"].isin(sender_id.tolist())]
    # cell_pair:receiver_group
    cell_pair = cell_pair.loc[cell_pair["cell_receiver"].isin(receiver_id.tolist())]
    # filter cell pairs
    if cell_pair.shape[0] < min_pairs:
        raise ValueError(f"cell pairs found between", sender_group, "and", receiver_group, "less than min_pairs")
    if cell_pair.shape[0] / cell_pair_all < min_pairs_ratio:
        raise ValueError(f"cell pairs found between", sender_group, "and", receiver_group, "less than min_pairs_ratio")

    # calculate score

    # real lr_cp_exp_score
    ligand_data = adata[cell_pair["cell_sender"], lr_network["from"]]
    receptor_data = adata[cell_pair["cell_receiver"], lr_network["to"]]
    lr_data = ligand_data.X.A * receptor_data.X.A if x_sparse else ligand_data.X * receptor_data.X
    lr_data = np.array(lr_data)
    if cell_pair.shape[0] == 0:
        lr_prod = np.zeros(lr_network.shape[0])
        lr_co_exp_ratio = np.zeros(lr_network.shape[0])
        lr_co_exp_num = np.zeros(lr_network.shape[0])
    else:
        lr_prod = np.apply_along_axis(lambda x: np.mean(x), 0, lr_data)
        lr_co_exp_ratio = np.apply_along_axis(lambda x: np.sum(x > 0) / x.size, 0, lr_data)
        lr_co_exp_num = np.apply_along_axis(lambda x: np.sum(x > 0), 0, lr_data)
    lr_network["lr_product"] = lr_prod
    lr_network["lr_co_exp_num"] = lr_co_exp_num
    lr_network["lr_co_exp_ratio"] = lr_co_exp_ratio

    # permutation test
    per_data = np.zeros((lr_network.shape[0], num))
    for i in tqdm(range(num)):
        random.seed(i)
        try:
            cell_id = random.sample(adata.obs.index.tolist(), k=cell_pair.shape[0] * 2)
            per_sender_id = cell_id[0 : cell_pair.shape[0]]
            per_receiver_id = cell_id[cell_pair.shape[0] : cell_pair.shape[0] * 2]
        except:
            # If cell_pair * 2 is too large a number:
            import itertools

            combinations = itertools.permutations(adata.obs.index.tolist(), r=2)
            pairs = random.sample(list(combinations), k=cell_pair.shape[0])
            per_sender_id = [pair[0] for pair in pairs]
            per_receiver_id = [pair[1] for pair in pairs]

        per_ligand_data = adata[per_sender_id, lr_network["from"]]
        per_receptor_data = adata[per_receiver_id, lr_network["to"]]
        per_lr_data = (
            per_ligand_data.X.A * per_receptor_data.X.A if x_sparse else per_ligand_data.X * per_receptor_data.X
        )
        per_lr_co_exp_ratio = np.apply_along_axis(lambda x: np.sum(x > 0) / x.size, 0, per_lr_data)
        if np.isnan(per_lr_co_exp_ratio).all():
            per_data[:, i] = np.zeros(lr_network.shape[0])
        else:
            per_data[:, i] = per_lr_co_exp_ratio

    per_data = pd.DataFrame(per_data)
    per_data.index = lr_network["from"]
    per_data["real"] = lr_network["lr_co_exp_ratio"].tolist()
    lr_network["lr_co_exp_ratio_pvalue"] = per_data.apply(lambda x: sum(x[:num] >= x["real"]) / num, axis=1).tolist()
    lr_network["is_significant"] = lr_network["lr_co_exp_ratio_pvalue"] < pvalue

    # Multiple hypothesis testing correction:
    qvalues = fdr_correct(pd.DataFrame(lr_network["lr_co_exp_ratio_pvalue"]), corr_method="fdr_bh")
    lr_network["lr_co_exp_ratio_qvalues"] = qvalues

    # After multiple testing correction:
    lr_network["is_significant_fdr"] = qvalues < pvalue
    # lr_network = lr_network.loc[lr_network["lr_co_exp_ratio_pvalue"] < pvalue]
    lr_network["sr_pair"] = sender_group + "-" + receiver_group

    res = {"cell_pair": cell_pair, "lr_pair": lr_network}
    return res


# Wrapper for preprocessing for plotting:
def prepare_cci_df(cci_df: pd.DataFrame, means_col: str, pval_col: str, lr_pair_col: str, sr_pair_col: str):
    """
    Given a dataframe generated from the output of :func `cci_two_cluster`, prepare for visualization by heatmap by
    splitting into two dataframes, corresponding to the mean cell type-cell type L:R product and probability values
    from the permutation test.

    Args:
        cci_df: CCI dataframe with columns for: ligand name, receptor name, L:R product, p value, and sender-receiver
            cell types
        means_col: Label for the column corresponding to the mean product of L:R expression between two cell types
        pval_col: Label for the column corresponding to the p-value of the interaction
        lr_pair_col: Label for the column corresponding to the ligand-receptor pair in format "{ligand}-{receptor}"
        sr_pair_col: Label for the column corresponding to the sending-receiving cell type pair in format "{
        sender}-{receiver}"

    Returns:
        dict: If 'adata' is None. Keys: 'means', 'pvalues', values: mean cell type-cell type L:R product, probability
            values, respectively

    Example:
        res = find_cci_two_group(adata, ...)
        # The df to save can be found under "lr_pair":
        res["lr_pair"].to_csv(...)

        adata, dict = prepare_cci_df(res["lr_pair"])
    """

    logger = lm.get_main_logger()

    # Dictionary to store mean and p-value dataframes:
    dict = {}

    # Split sender and receiver into separate columns:
    cci_df[["sender", "receiver"]] = cci_df[sr_pair_col].str.split("-", expand=True)
    all_lr_products, all_lr_pvals = {}, {}
    # Split dataframe based on ligand-receptor pair, set "sender" and "receiver" as multiindex, keep only the means
    # or p-values to get series for each LR interaction:
    cci_grouped = cci_df.groupby(lr_pair_col)
    for group in cci_grouped.groups.keys():
        lig, rec = group.split("-")
        df_group = cci_grouped.get_group(group)
        df_group.set_index(["sender", "receiver"])
        df_group = df_group.transpose()
        # Series to row dataframe for means and p-values:
        prod_df_group = df_group.loc[means_col].to_frame().transpose()
        prod_df_group[["source", "target"]] = [lig, rec]
        prod_df_group.set_index(["source", "target"])
        pval_df_group = df_group.loc[pval_col].to_frame().transpose()
        pval_df_group[["source", "target"]] = [lig, rec]
        pval_df_group.set_index(["source", "target"])

        all_lr_products[group] = prod_df_group
        all_lr_pvals[group] = pval_df_group

    means = pd.concat(all_lr_products.values())
    pvals = pd.concat(all_lr_pvals.values())

    dict["means"] = means
    dict["pvalues"] = pvals

    return dict


@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE, "adata")
def prepare_cci_cellpair_adata(
    adata: AnnData,
    sender_group: str = None,
    receiver_group: str = None,
    group: str = None,
    cci_dict: dict = None,
    all_cell_pair: bool = False,
) -> AnnData:
    """prepare for visualization cellpairs by func `st.tl.space`, plot all_cell_pair,
    or cell pairs which constrain by spatial distance(output of :func `cci_two_cluster`).
        Args:
            adata:An Annodata object.
            sender_group: the cell group name of send ligands.
            receiver_group: the cell group name of receive receptors.
            group:The group name in adata.obs, Unused unless 'all_cell_pair' is True.
            cci_dict: a dictionary result from :func `cci_two_cluster`, where the key is 'cell_pair' and 'lr_pair'.
                     Unused unless 'all_cell_pair' is False.
            all_cell_pair: show all cells of the sender and receiver cell group, spatial_key: Key in .obsm containing coordinates for each bucket. Defult `False`.
        Returns:
            adata: Updated AnnData object containing 'spec' in .obs.
    """
    logger = lm.get_main_logger()

    adata.obs["spec"] = "other"
    if all_cell_pair:
        adata.obs.loc[adata.obs[group] == sender_group, "spec"] = sender_group
        adata.obs.loc[adata.obs[group] == receiver_group, "spec"] = receiver_group
    else:
        adata.obs.loc[adata.obs.index.isin(cci_dict["cell_pair"]["cell_sender"].tolist()), "spec"] = sender_group
        adata.obs.loc[adata.obs.index.isin(cci_dict["cell_pair"]["cell_receiver"].tolist()), "spec"] = receiver_group
    return adata

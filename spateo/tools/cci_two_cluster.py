# -*- coding: utf-8 -*-
"""
@File    :   cci_new.py
@Time    :   2022/07/03 11:50:40
@Author  :   LuluZuo
@Version :   1.0
@Desc    :   spatial cell cell communication
"""

import random
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse
from scipy.sparse import issparse
from scipy.stats import gmean, pearsonr
from typing_extensions import Literal

from ..configuration import SKM


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
    min_pairs: int = 5,
    min_pairs_ratio: float = 0.01,
    num: int = 1000,
    pvalue: float = 0.05,
) -> AnnData:
    """Performing cell-cell transformation on an anndata object, while also
       limiting the nearest neighbor per cell to n_neighbors. This function returns
       another anndata object, in which the columns of the matrix are bucket
       -bucket pairs, while the rows ligand-receptor mechanisms. This resultant
       anndated object allows flexible downstream manipulations such as the
       dimensional reduction of the row or column of this object.

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
        min_pairs: minimal number of cell pairs between cells from two groups.
        min_pairs_ratio: minimal ratio of cell pairs to theoretical cell pairs (n x M / 2) between cells
            from two groups.
        num: number of permutations.
        pvalue: the p-value threshold that will be used to filter for significant ligan-receptor pairs.
        filter_lr: filter ligand and receptor based on specific expressed in sender groups
            and receiver groups. 'inner': specific both in sender groups and receiver groups;
            'outer': specific in sender groups or receiver groups.
        top: the number of top expressed fraction in given sender groups(receiver groups)
            for each gene(ligand or receptor).

    Returns:
        An anndata of Niches, which rows are mechanisms and columns are all
        possible cell x cell interactions.

    """
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
        adata_l = adata[:, lr_network["from"]]
        for g in adata.obs[group].unique():
            frac = (adata_l[adata_l.obs[group] == g].X > 0).sum(axis=0) / (adata_l.X > 0).sum(axis=0)
            adata_l.var[g + "_frac"] = frac.A1 if x_sparse else frac
        dfl = adata_l.var[adata_l.var[sender_group + "_frac"] > 0]
        ligand_sender_spec = dfl.sort_values(by=sender_group + "_frac", ascending=False)[:top].index
        lr_network_l = lr_network.loc[lr_network["from"].isin(ligand_sender_spec.tolist())]

        # receptor_receiver_spec
        adata_r = adata[:, lr_network["to"]]
        for g in adata.obs[group].unique():
            frac = (adata_r[adata_r.obs[group] == g].X > 0).sum(axis=0) / adata_r.X.sum(axis=0)
            adata_r.var[g + "_frac"] = frac.A1 if x_sparse else frac
        dfr = adata_r.var[adata_r.var[receiver_group + "_frac"] > 0]
        receptor_receiver_spec = dfr.sort_values(by=receiver_group + "_frac", ascending=False)[:top].index
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
        cell_pair.append(str(adata.obs.index[i]) + ">-<" + i for i in adata.obs.index[cell_id])
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
    lr_co_exp_ratio = np.apply_along_axis(lambda x: np.sum(x > 0) / x.size, 0, lr_data)
    lr_co_exp_num = np.apply_along_axis(lambda x: np.sum(x > 0), 0, lr_data)
    lr_network["lr_co_exp_num"] = lr_co_exp_num
    lr_network["lr_co_exp_ratio"] = lr_co_exp_ratio

    # permutation test
    per_data = np.zeros((lr_network.shape[0], num))
    for i in range(num):
        random.seed(i)
        cell_id = random.sample(adata.obs.index.tolist(), k=cell_pair.shape[0] * 2)
        per_sender_id = cell_id[0 : cell_pair.shape[0]]
        per_receiver_id = cell_id[cell_pair.shape[0] : cell_pair.shape[0] * 2]
        per_ligand_data = adata[per_sender_id, lr_network["from"]]
        per_receptor_data = adata[per_receiver_id, lr_network["to"]]
        per_lr_data = (
            per_ligand_data.X.A * per_receptor_data.X.A if x_sparse else per_ligand_data.X * per_receptor_data.X
        )
        per_lr_co_exp_ratio = np.apply_along_axis(lambda x: np.sum(x > 0) / x.size, 0, per_lr_data)
        per_data[:, i] = per_lr_co_exp_ratio

    per_data = pd.DataFrame(per_data)
    per_data.index = lr_network["from"]
    per_data["real"] = lr_network["lr_co_exp_ratio"].tolist()
    lr_network["lr_co_exp_ratio_pvalue"] = per_data.apply(lambda x: sum(x[:num] > x["real"]) / num, axis=1).tolist()
    lr_network = lr_network.loc[lr_network["lr_co_exp_ratio_pvalue"] < pvalue]
    lr_network["sr_pair"] = sender_group + "-" + receiver_group

    res = {"cell_pair": cell_pair, "lr_pair": lr_network}
    return res

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse
from scipy.sparse import issparse
from scipy.stats import gmean, pearsonr

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from ..configuration import SKM


# Niches
@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE)
def niches(
    adata: AnnData,
    path: str,
    layer: Tuple[None, str] = None,
    weighted: bool = False,
    spatial_neighbors: str = "spatial_neighbors",
    spatial_distances: str = "spatial_distances",
    species: Literal["human", "mouse", "drosophila", "zebrafish", "axolotl"] = "human",
    system: Literal["niches_c2c", "niches_n2c", "niches_c2n", "niches_n2n"] = "niches_n2n",
    method: Literal["gmean", "mean", "sum"] = "sum",
) -> AnnData:
    """Performing cell-cell transformation on an anndata object, while also
       limiting the nearest neighbor per cell to k. This function returns
       another anndata object, in which the columns of the matrix are bucket
       -bucket pairs, while the rows ligand-receptor mechanisms. This resultant
       anndated object allows flexible downstream manipulations such as the
       dimensional reduction of the row or column of this object.

        Our method is adapted from:
        Micha Sam Brickman Raredon, Junchen Yang, Neeharika Kothapalli,  Naftali Kaminski, Laura E. Niklason,
        Yuval Kluger. Comprehensive visualization of cell-cell interactions in single-cell and spatial transcriptomics
        with NICHES. doi: https://doi.org/10.1101/2022.01.23.477401



    Args:
        adata: An Annodata object.
        path: Path to ligand_receptor network of NicheNet (prior lr_network).
        layer: the key to the layer. If it is None, adata.X will be used by default.
        weighted: 'False' (defult)
            whether to supply the edge weights according to the actual spatial
            distance(just as weighted kNN). Defult is 'False', means all neighbor
            edge weights equal to 1, others is 0.
        spatial_neighbors : neighbor_key {spatial_neighbors} in adata.uns.keys(),
        spatial_distances : neighbor_key {spatial_distances} in adata.obsp.keys().
        system: 'niches_n2n'(defult)
            cell-cell signaling ('niches_c2c'), defined as the signals passed between
            cells, determined by the product of the ligand expression of the sending
            cell and the receptor expression of the receiving cell, and system-cell
            signaling ('niches_n2c'), defined as the signaling input to a cell,
            determined by taking the geometric mean of the ligand profiles of the
            surrounding cells and the receptor profile of the receiving cell.similarly,
            'niches_c2n','niches_n2n'.


    Returns:
        An anndata of Niches, which rows are mechanisms and columns are all
        possible cell x cell interactions.

    """

    # prior lr_network
    if species == "human":
        lr_network = pd.read_csv(path + "lr_network_human.csv", index_col=0)
        if system == "niches_n2c":
            lr_network[["from", "to"]] = lr_network[["to", "from"]]
    elif species == "mouse":
        lr_network = pd.read_csv(path + "lr_network_mouse.csv", index_col=0)
        if system == "niches_n2c":
            lr_network[["from", "to"]] = lr_network[["to", "from"]]
    elif species == "drosophila":
        lr_network = pd.read_csv(path + "lr_network_drosophila.csv", index_col=0)
        if system == "niches_n2c":
            lr_network[["from", "to"]] = lr_network[["to", "from"]]
    elif species == "zebrafish":
        lr_network = pd.read_csv(path + "lr_network_zebrafish.csv", index_col=0)
        if system == "niches_n2c":
            lr_network[["from", "to"]] = lr_network[["to", "from"]]
    elif species == "axolotl":
        lr_network = pd.read_csv(path + "lr_network_axolotl.csv", index_col=0)
        if system == "niches_n2c":
            lr_network[["from", "to"]] = lr_network[["to", "from"]]

    if layer is None:
        adata.X = adata.X
    else:
        adata.X = adata.layers[layer]

    x_sparse = issparse(adata.X)

    # expressed lr_network
    ligand = lr_network["from"].unique()
    expressed_ligand = list(set(ligand) & set(adata.var_names))
    if len(expressed_ligand) == 0:
        raise ValueError(f"No intersected ligand between your adata object" f" and lr_network dataset.")
    lr_network = lr_network[lr_network["from"].isin(expressed_ligand)]
    receptor = lr_network["to"].unique()
    expressed_receptor = list(set(receptor) & set(adata.var_names))
    if len(expressed_receptor) == 0:
        raise ValueError(f"No intersected receptor between your adata object" f" and lr_network dataset.")
    lr_network = lr_network[lr_network["to"].isin(expressed_receptor)]
    ligand_matrix = adata[:, lr_network["from"]].X.A.T if x_sparse else adata[:, lr_network["from"]].X.T

    # spatial neighbors
    if spatial_neighbors not in adata.uns.keys():
        raise ValueError(
            f"No spatial_key {spatial_neighbors} exists in adata,"
            f"using 'dyn.tl.neighbors' to calulate the spatial neighbors first."
        )
    if spatial_distances not in adata.obsp.keys():
        raise ValueError(
            f"No spatial_key {spatial_distances} exists in adata,"
            f"using 'dyn.tl.neighbors' to calulate the spatial diatances first."
        )

    nw = {"neighbors": adata.uns["spatial_neighbors"]["indices"], "weights": adata.obsp["spatial_distances"]}
    k = adata.uns["spatial_neighbors"]["params"]["n_neighbors"]

    # construct c2c matrix
    if system == "niches_c2c":
        X = np.zeros(shape=(ligand_matrix.shape[0], k * adata.n_obs))
        if weighted:
            # weighted matrix (weighted distance)
            row, col = np.diag_indices_from(nw["weights"])
            nw["weights"][row, col] = 1
            weight = np.zeros(shape=(adata.n_obs, k))
            for i, row in enumerate(nw["weights"].A):
                weight[i, :] = 1 / row[nw["neighbors"][i]]
            for i in range(ligand_matrix.shape[1]):
                receptor_matrix = adata[nw["neighbors"][i], lr_network["to"]].X.A.T * weight[i, :]
                X[:, i * k : (i + 1) * k] = receptor_matrix * ligand_matrix[:, i].reshape(-1, 1)
        else:
            for i in range(ligand_matrix.shape[1]):
                receptor_matrix = adata[nw["neighbors"][i], lr_network["to"]].X.A.T
                X[:, i * k : (i + 1) * k] = receptor_matrix * ligand_matrix[:, i].reshape(-1, 1)
        # bucket-bucket pair
        cell_pair = []
        for i, cell_id in enumerate(nw["neighbors"]):
            cell_pair.append(adata.obs.index[i] + "-" + adata.obs.index[cell_id])
        cell_pair = [i for j in cell_pair for i in j]
        cell_pair = pd.DataFrame({"cell_pair_name": cell_pair})
        cell_pair.set_index("cell_pair_name", inplace=True)

    # construct n2c matrix or construct c2n matrix
    if system == "niches_n2c" or system == "niches_c2n":
        X = np.zeros(shape=(ligand_matrix.shape[0], adata.n_obs))
        if weighted:
            # weighted matrix (weighted distance)
            row, col = np.diag_indices_from(nw["weights"])
            nw["weights"][row, col] = 1
            weight = np.zeros(shape=(adata.n_obs, k))
            for i, row in enumerate(nw["weights"].A):
                weight[i, :] = 1 / row[nw["neighbors"][i]]
            for i in range(ligand_matrix.shape[1]):
                if method == "gmean":
                    receptor_matrix = (
                        gmean((adata[nw["neighbors"][i], lr_network["to"]].X.A.T + 1) * weight[i, :], axis=1)
                        if x_sparse
                        else gmean((adata[nw["neighbors"][i], lr_network["to"]].X.T + 1) * weight[i, :], axis=1)
                    )
                elif method == "mean":
                    receptor_matrix = (
                        np.mean(adata[nw["neighbors"][i], lr_network["to"]].X.A.T * weight[i, :], axis=1)
                        if x_sparse
                        else np.mean(adata[nw["neighbors"][i], lr_network["to"]].X.T * weight[i, :], axis=1)
                    )
                else:
                    receptor_matrix = (
                        np.sum(adata[nw["neighbors"][i], lr_network["to"]].X.A.T * weight[i, :], axis=1)
                        if x_sparse
                        else np.sum(adata[nw["neighbors"][i], lr_network["to"]].X.T * weight[i, :], axis=1)
                    )
                X[:, i] = receptor_matrix * ligand_matrix[:, i]
        else:
            for i in range(ligand_matrix.shape[1]):
                if method == "gmean":
                    receptor_matrix = (
                        gmean((adata[nw["neighbors"][i], lr_network["to"]].X.A.T + 1), axis=1)
                        if x_sparse
                        else gmean((adata[nw["neighbors"][i], lr_network["to"]].X.T + 1), axis=1)
                    )
                elif method == "mean":
                    receptor_matrix = (
                        np.mean(adata[nw["neighbors"][i], lr_network["to"]].X.A.T, axis=1)
                        if x_sparse
                        else np.mean(adata[nw["neighbors"][i], lr_network["to"]].X.T, axis=1)
                    )
                else:
                    receptor_matrix = (
                        np.sum(adata[nw["neighbors"][i], lr_network["to"]].X.A.T, axis=1)
                        if x_sparse
                        else np.sum(adata[nw["neighbors"][i], lr_network["to"]].X.T, axis=1)
                    )
                X[:, i] = receptor_matrix * ligand_matrix[:, i]
        # bucket-bucket pair
        cell_pair = []
        for i, cell_id in enumerate(nw["neighbors"]):
            cell_pair.append(adata.obs.index[i] + "-" + adata.obs.index[cell_id])
        cell_pair = pd.DataFrame({"cell_pair_name": cell_pair})

    # construct n2n matrix
    if system == "niches_n2n":
        X = np.zeros(shape=(ligand_matrix.shape[0], adata.n_obs))
        if weighted:
            # weighted matrix (weighted distance)
            row, col = np.diag_indices_from(nw["weights"])
            nw["weights"][row, col] = 1
            weight = np.zeros(shape=(adata.n_obs, k))
            for i, row in enumerate(nw["weights"].A):
                weight[i, :] = 1 / row[nw["neighbors"][i]]
            for i in range(ligand_matrix.shape[1]):
                if method == "gmean":
                    receptor_matrix = (
                        gmean((adata[nw["neighbors"][i], lr_network["to"]].X.A.T + 1) * weight[i, :], axis=1)
                        if x_sparse
                        else gmean((adata[nw["neighbors"][i], lr_network["to"]].X.T + 1) * weight[i, :], axis=1)
                    )
                    ligand_matrix = (
                        gmean((adata[nw["neighbors"][i], lr_network["from"]].X.A.T + 1) * weight[i, :], axis=1)
                        if x_sparse
                        else gmean((adata[nw["neighbors"][i], lr_network["from"]].X.T + 1) * weight[i, :], axis=1)
                    )
                elif method == "mean":
                    receptor_matrix = (
                        np.mean(adata[nw["neighbors"][i], lr_network["to"]].X.A.T * weight[i, :], axis=1)
                        if x_sparse
                        else np.mean(adata[nw["neighbors"][i], lr_network["to"]].X.T * weight[i, :], axis=1)
                    )
                    ligand_matrix = (
                        np.mean(adata[nw["neighbors"][i], lr_network["from"]].X.A.T * weight[i, :], axis=1)
                        if x_sparse
                        else np.mean(adata[nw["neighbors"][i], lr_network["from"]].X.T * weight[i, :], axis=1)
                    )
                else:
                    receptor_matrix = (
                        np.sum(adata[nw["neighbors"][i], lr_network["to"]].X.A.T * weight[i, :], axis=1)
                        if x_sparse
                        else np.sum(adata[nw["neighbors"][i], lr_network["to"]].X.T * weight[i, :], axis=1)
                    )
                    ligand_matrix = (
                        np.sum(adata[nw["neighbors"][i], lr_network["from"]].X.A.T * weight[i, :], axis=1)
                        if x_sparse
                        else np.sum(adata[nw["neighbors"][i], lr_network["from"]].X.T * weight[i, :], axis=1)
                    )
                X[:, i] = np.array(receptor_matrix).reshape(receptor_matrix.shape[0]) * np.array(ligand_matrix).reshape(
                    ligand_matrix.shape[0]
                )
        else:
            for i in range(ligand_matrix.shape[1]):
                if method == "gmean":
                    receptor_matrix = (
                        gmean((adata[nw["neighbors"][i], lr_network["to"]].X.A.T + 1), axis=1)
                        if x_sparse
                        else gmean((adata[nw["neighbors"][i], lr_network["to"]].X.T + 1), axis=1)
                    )
                    ligand_matrix = (
                        gmean((adata[nw["neighbors"][i], lr_network["from"]].X.A.T + 1), axis=1)
                        if x_sparse
                        else gmean((adata[nw["neighbors"][i], lr_network["from"]].X.T + 1), axis=1)
                    )
                elif method == "mean":
                    receptor_matrix = (
                        np.mean(adata[nw["neighbors"][i], lr_network["to"]].X.A.T, axis=1)
                        if x_sparse
                        else np.mean(adata[nw["neighbors"][i], lr_network["to"]].X.T, axis=1)
                    )
                    ligand_matrix = (
                        np.mean(adata[nw["neighbors"][i], lr_network["from"]].X.A.T, axis=1)
                        if x_sparse
                        else np.mean(adata[nw["neighbors"][i], lr_network["from"]].X.T, axis=1)
                    )
                else:
                    receptor_matrix = (
                        np.sum(adata[nw["neighbors"][i], lr_network["to"]].X.A.T, axis=1)
                        if x_sparse
                        else np.sum(adata[nw["neighbors"][i], lr_network["to"]].X.T, axis=1)
                    )
                    ligand_matrix = (
                        np.sum(adata[nw["neighbors"][i], lr_network["from"]].X.A.T, axis=1)
                        if x_sparse
                        else np.sum(adata[nw["neighbors"][i], lr_network["from"]].X.T, axis=1)
                    )
                X[:, i] = np.array(receptor_matrix).reshape(receptor_matrix.shape[0]) * np.array(ligand_matrix).reshape(
                    ligand_matrix.shape[0]
                )
        # bucket-bucket pair
        cell_pair = []
        for i, cell_id in enumerate(nw["neighbors"]):
            cell_pair.append(adata.obs.index[i] + "-" + adata.obs.index[cell_id])
        cell_pair = pd.DataFrame({"cell_pair_name": cell_pair})

    # lr_pair
    lr_pair = lr_network["from"] + "-" + lr_network["to"]
    lr_pair = pd.DataFrame({"lr_pair_name": lr_pair})
    lr_pair.set_index("lr_pair_name", inplace=True)

    # csr_matrix
    X = sparse.csr_matrix(X.T)

    # adata_nichec2c
    adata_niche = AnnData(X=X, obs=cell_pair, var=lr_pair)
    return adata_niche


# NicheNet
@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE)
def predict_ligand_activities(
    adata: AnnData,
    path: str,
    sender_cells: Optional[List[str]] = None,
    receiver_cells: Optional[List[str]] = None,
    geneset: Optional[List[str]] = None,
    ratio_expr_thresh: float = 0.01,
    species: Literal["human", "mouse"] = "human",
) -> pd.DataFrame:
    """Function to predict the ligand activity.

        Our method is adapted from:
        Robin Browaeys, Wouter Saelens & Yvan Saeys. NicheNet: modeling intercellular communication by linking ligands
        to target genes. Nature Methods volume 17, pages159–162 (2020).

    Args:
        path: Path to ligand_target_matrix, lr_network (human and mouse).
        adata: An Annodata object.
        sender_cells: Ligand cells.
        receiver_cells: Receptor cells.
        geneset: The genes set of interest. This may be the differentially
            expressed genes in receiver cells (comparing cells in case and
            control group). Ligands activity prediction is based on this gene
            set. By default, all genes expressed in receiver cells is used.
        ratio_expr_thresh: The minimum percentage of buckets expressing the ligand
            (target) in sender(receiver) cells.
    Returns:
        A pandas DataFrame of the predicted activity ligands.

    """
    # load ligand_target_matrix
    if species == "human":
        ligand_target_matrix = pd.read_csv(path + "ligand_target_matrix_human_nichenet.csv", index_col=0)
        lr_network = pd.read_csv(path + "lr_network_human.csv", index_col=0)
    else:
        ligand_target_matrix = pd.read_csv(path + "ligand_target_matrix_mouse_nichenet.csv", index_col=0)
        lr_network = pd.read_csv(path + "lr_network_mouse.csv", index_col=0)
    ligand_target_matrix = ligand_target_matrix.T  # row:ligand,col:target gene
    lr_network = lr_network.loc[lr_network["from"].isin(ligand_target_matrix.index)]

    # Define expressed genes in sender and receiver cell populations(pct>0.1)
    expressed_genes_sender = np.array(adata.var_names)[
        np.count_nonzero(adata[sender_cells, :].X.A, axis=0) / len(sender_cells) > 0.01
    ].tolist()
    expressed_genes_receiver = np.array(adata.var_names)[
        np.count_nonzero(adata[receiver_cells, :].X.A, axis=0) / len(receiver_cells) > 0.01
    ].tolist()

    # Define a set of potential ligands
    ligands = lr_network["from"].unique()
    expressed_ligand = list(set(ligands) & set(expressed_genes_sender))
    if len(expressed_ligand) == 0:
        raise ValueError(f"No intersected ligand between your adata object and lr_network dataset.")
    receptor = lr_network["to"].unique()
    expressed_receptor = list(set(receptor) & set(expressed_genes_receiver))
    if len(expressed_receptor) == 0:
        raise ValueError(f"No intersected receptor between your adata object and lr_network dataset.")
    lr_network_expressed = lr_network[
        lr_network["from"].isin(expressed_ligand) & lr_network["to"].isin(expressed_receptor)
    ]
    potential_ligands = lr_network_expressed["from"].unique()

    if geneset is None:
        # Calculate the pearson coeffcient between potential score and actual the average expression of genes
        response_expressed_genes = list(set(expressed_genes_receiver) & set(ligand_target_matrix.index))
        response_expressed_genes_df = pd.DataFrame(response_expressed_genes)
        response_expressed_genes_df = response_expressed_genes_df.rename(columns={0: "gene"})
        response_expressed_genes_df["avg_expr"] = np.mean(adata[receiver_cells, response_expressed_genes].X.A, axis=0)
        lt_matrix = ligand_target_matrix[potential_ligands.tolist()].loc[response_expressed_genes_df["gene"].tolist()]
        de = []
        for ligand in lt_matrix:
            pear_coef = pearsonr(lt_matrix[ligand], response_expressed_genes_df["avg_expr"])[0]
            pear_pvalue = pearsonr(lt_matrix[ligand], response_expressed_genes_df["avg_expr"])[1]
            de.append(
                (
                    ligand,
                    pear_coef,
                    pear_pvalue,
                )
            )
    else:
        # Define the gene set of interest and a background of genes
        gene_io = list(set(geneset) & set(ligand_target_matrix.index))
        background_expressed_genes = list(set(expressed_genes_receiver) & set(ligand_target_matrix.index))
        # Perform NicheNet’s ligand activity analysis on the gene set of interest
        gene_io = pd.DataFrame(gene_io)
        gene_io["logical"] = 1
        gene_io = gene_io.rename(columns={0: "gene"})
        background_expressed_genes = pd.DataFrame(background_expressed_genes)
        background = background_expressed_genes[~(background_expressed_genes[0].isin(gene_io))]
        background = background.rename(columns={0: "gene"})
        background["logical"] = 0
        # response gene vector
        response = pd.concat([gene_io, background], axis=0, join="outer")
        # lt_matrix potential score
        lt_matrix = ligand_target_matrix[potential_ligands.tolist()].loc[response["gene"].tolist()]
        # predict ligand activity by pearson coefficient.
        de = []
        for ligand in lt_matrix:
            pear_coef = pearsonr(lt_matrix[ligand], response["logical"])[0]
            pear_pvalue = pearsonr(lt_matrix[ligand], response["logical"])[1]
            de.append(
                (
                    ligand,
                    pear_coef,
                    pear_pvalue,
                )
            )
    res = pd.DataFrame(
        de,
        columns=[
            "ligand",
            "pearson_coef",
            "pearson_pvalue",
        ],
    )
    return res


@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE, optional=True)
def predict_target_genes(
    adata: AnnData,
    path: str,
    sender_cells: Optional[List[str]] = None,
    receiver_cells: Optional[List[str]] = None,
    geneset: Optional[List[str]] = None,
    species: Literal["human", "mouse"] = "human",
    top_ligand: int = 20,
    top_target: int = 300,
) -> pd.DataFrame:
    """Function to predict the target genes.

    Args:
        lt_matrix_path: Path to ligand_target_matrix of NicheNet.
        adata: An Annodata object.
        sender_cells: Ligand cells.
        receiver_cells: Receptor cells.
        geneset: The genes set of interest. This may be the differentially
            expressed genes in receiver cells (comparing cells in case and
            control group). Ligands activity prediction is based on this gene
            set. By default, all genes expressed in receiver cells is used.
        top_ligand: `int` (default=20)
            select 20 top-ranked ligands for further biological interpretation.
        top_target: `int` (default=300)
            Infer target genes of top-ranked ligands, and choose the top targets
            according to the general prior model.
    Returns:
        A pandas DataFrame of the predict target genes.

    """
    if species == "human":
        ligand_target_matrix = pd.read_csv(path + "ligand_target_matrix.csv", index_col=0)
    else:
        ligand_target_matrix = pd.read_csv(path + "ligand_target_matrix_mouse.csv", index_col=0)
    predict_ligand = predict_ligand_activities(
        adata=adata,
        path=path,
        sender_cells=sender_cells,
        receiver_cells=receiver_cells,
        geneset=geneset,
        species=species,
    )
    predict_ligand.sort_values(by="pearson_coef", axis=0, ascending=False, inplace=True)
    predict_ligand_top = predict_ligand[:top_ligand]["ligand"]
    res = pd.DataFrame(columns=("ligand", "targets", "weights"))
    for ligand in ligand_target_matrix[predict_ligand_top]:
        top_n_score = ligand_target_matrix[ligand].sort_values(ascending=False)[:top_target]
        if geneset is None:
            expressed_genes_receiver = np.array(adata.var_names)[
                np.count_nonzero(adata[receiver_cells, :].X.A, axis=0) / len(receiver_cells) > 0.01
            ].tolist()
            response_expressed_genes = list(set(expressed_genes_receiver) & set(ligand_target_matrix.index))
            targets = list(set(top_n_score.index) & set(response_expressed_genes))
        else:
            gene_io = list(set(geneset) & set(ligand_target_matrix.index))
            targets = list(set(top_n_score.index) & set(gene_io))
        res = pd.concat(
            [
                res,
                pd.DataFrame(
                    {"ligand": ligand, "targets": targets, "weights": ligand_target_matrix.loc[targets, ligand]}
                ).reset_index(drop=True),
            ],
            axis=0,
            join="outer",
        )
    return res

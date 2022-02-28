import pyreadr
import numpy as np
from scipy.stats import pearsonr
import pandas as pd
from typing import List, Optional
from anndata import AnnData
from scipy import sparse
from pysal import explore, lib
from libpysal.weights import W, full

# Niches
def run_niches_cell_to_cell(
    lr_network_path: str,
    adata: AnnData,
    k: int = 5,
    weighted: bool = False,
    x: Optional[List[int]] = None,
    y: Optional[List[int]] = None,
    
) -> AnnData:
    """Performs cell-cell transformation on an anndata object, and constrain the 
       interactions among cells based on the spatial distance. Outputs another 
       anndata object, but where the columns of the matrix are buckets-buckets pairs,
       and the rows of the matrix are ligand-receptor mechanisms. This allows rapid 
       manipulation and dimensional reduction of cell-cell connectivity data.

    Args:
        lr_network_path: Path to ligand_receptor network of NicheNet(prior lr_network).
        adata: An Annodata object.
        k: 'int' (defult=5)
            Number of neighbors to use by default for kneighbors queries.
        weighted: 'False'(defult)
            whether to supply the edge weights according to the actual spatial distance.
            (just as Gussian kernel). Defult is 'False', means all neighbor edge weights 
            equal to 1, others is 0. 
        x: 'list' or None(default: `None`)
            x-coordinates of all buckets.
        y: 'list' or None(default: `None`)
            y-coordinates of all buckets.
        
    Returns:
        An anndata of Niches.

    """
    # prior lr_network
    url = "https://zenodo.org/record/3260758/files/lr_network.rds"
    path = lr_network_path + 'lr_network.rds'
    path_load = pyreadr.download_file(url, path)
    lr_network = pyreadr.read_r(path)[None]
    # expressed lr_network
    ligand = lr_network['from'].unique()
    expressed_ligand = list(set(ligand) & set(adata.var_names))
    lr_network = lr_network[lr_network['from'].isin(expressed_ligand)]
    receptor = lr_network['to'].unique()
    expressed_receptor = list(set(receptor) & set(adata.var_names))
    lr_network = lr_network[lr_network['to'].isin(expressed_receptor)]
    # ligand_matrix
    ligand_matrix = adata[:, lr_network['from']].X.A.T
    # spatial cell to cell(scc) matrix
    # spatial adjcency matrix
    if x is None:
        x = adata.obsm["spatial"][:, 0].tolist()
    if y is None:
        y = adata.obsm["spatial"][:, 1].tolist()
    xymap = pd.DataFrame({"x": x, "y": y})
    if weighted:
        # weighted matrix (kernel distance)
        nw = lib.weights.Kernel(xymap, k, function="gaussian",diagonal=True)
        W = lib.weights.W(nw.neighbors, nw.weights)
        adj = full(W)[0]
        res ={}
        receptor_matrix_0 = adata[nw.neighbors[0], lr_network['to']].X.A.T
        X = np.zeros(shape=(receptor_matrix_0.shape[0], receptor_matrix_0.shape[1]))
        for i in range(ligand_matrix.shape[1]):
            res[i] = []
            receptor_matrix = adata[nw.neighbors[i], lr_network['to']].X.A.T
            for col in receptor_matrix.T:
                res[i].append(list(map(lambda x, y: x * y, ligand_matrix[:, i], col)))
            res[i] = np.array(res[i]).T
            X = np.hstack((X, res[i]))
    else:
        # weighted matrix(in:1 out:0)
        kd = lib.cg.KDTree(np.array(xymap))
        nw = lib.weights.KNN(kd, k)
        W = lib.weights.W(nw.neighbors, nw.weights)
        adj = full(W)[0]
        row, col = np.diag_indices_from(adj)
        adj[row, col] = 1
        res = {}
        receptor_matrix_0 = adata[nw.neighbors[0], lr_network['to']].X.A.T
        X = np.zeros(shape=(receptor_matrix_0.shape[0], receptor_matrix_0.shape[1]))
        for i in range(ligand_matrix.shape[1]):
            res[i] = []
            receptor_matrix = adata[nw.neighbors[i], lr_network['to']].X.A.T * np.array(nw.weights[i])
            for col in receptor_matrix.T:
                res[i].append(list(map(lambda x, y: x * y, ligand_matrix[:,i], col)))
            res[i] = np.array(res[i]).T
            X = np.hstack((X, res[i]))
    X = X[:, receptor_matrix_0.shape[1]:]
    # bucket-bucket pair
    cell_pair = []
    for i, cell_id in nw.neighbors.items():
        cell_pair.append(adata.obs.index[i]+'-'+adata.obs.index[cell_id])
    cell_pair = [i for j in cell_pair for i in j]
    cell_pair = pd.DataFrame({"cell_pair_name": cell_pair})
    cell_pair.set_index("cell_pair_name", inplace=True)
    # lr_pair
    lr_pair = lr_network['from']+'-'+lr_network['to']
    lr_pair = pd.DataFrame({"lr_pair_name": lr_pair})
    lr_pair.set_index("lr_pair_name", inplace=True)
    # csr_matrix
    X = sparse.csr_matrix(X.T)
    # adara_niche
    adata_niche = AnnData(X=X, obs=cell_pair, var=lr_pair)
    return adata_niche


def run_niches_system_to_cell(
    lr_network_path: str,
    adata: AnnData,
    k: int = 5,
    weighted: bool = False,
    x: Optional[List[int]] = None,
    y: Optional[List[int]] = None,
) -> AnnData:
    """Performs system(every other cells)-cell transformation on an anndata object, 
       and constrain the interactions among cells based on the spatial distance. Outputs
       another anndata object, but where the columns of the matrix are buckets-buckets pairs,
       and the rows of the matrix are ligand-receptor mechanisms. This allows rapid 
       manipulation and dimensional reduction of cell-cell connectivity data.

    Args:
        lr_network_path: Path to ligand_receptor network of NicheNet(prior lr_network).
        adata: An Annodata object.
        k: 'int' (defult=5)
            Number of neighbors to use by default for kneighbors queries.
        weighted: 'False'(defult)
            whether to supply the edge weights according to the actual spatial distance.
            (just as Gussian kernel). Defult is 'False', means all neighbor edge weights 
            equal to 1, others is 0. 
        x: 'list' or None(default: `None`)
            x-coordinates of all buckets.
        y: 'list' or None(default: `None`)
            y-coordinates of all buckets.
        
    Returns:
        An anndata of Niches.

    """
    # prior lr_network
    #url = "https://zenodo.org/record/3260758/files/lr_network.rds"
    path = lr_network_path + 'lr_network.rds'
    #path_load = pyreadr.download_file(url, path)
    lr_network = pyreadr.read_r(path)[None]
    # expressed lr_network
    ligand = lr_network['from'].unique()
    expressed_ligand = list(set(ligand) & set(adata.var_names))
    lr_network = lr_network[lr_network['from'].isin(expressed_ligand)]
    receptor = lr_network['to'].unique()
    expressed_receptor = list(set(receptor) & set(adata.var_names))
    lr_network = lr_network[lr_network['to'].isin(expressed_receptor)]
    # ligand_matrix
    ligand_matrix = adata[:, lr_network['from']].X.A.T
    # spatial adjcency matrix
    if x is None:
        x = adata.obsm["spatial"][:, 0].tolist()
    if y is None:
        y = adata.obsm["spatial"][:, 1].tolist()
    xymap = pd.DataFrame({"x": x, "y": y})
    if weighted:
        # weighted matrix (kernel distance)
        nw = lib.weights.Kernel(xymap, k, function="gaussian",diagonal=True)
        W = lib.weights.W(nw.neighbors, nw.weights)
        adj = full(W)[0]
        res = {}
        receptor_matrix_0 = adata[nw.neighbors[0], lr_network['to']].X.A.T
        X = np.array([0] * receptor_matrix_0.shape[0])
        for i in range(ligand_matrix.shape[1]):
            receptor_matrix = adata[nw.neighbors[i],lr_network['to']].X.A.T * np.array(nw.weights[i])
            res[i] = ligand_matrix[:, i] * receptor_matrix.sum(axis=1)
            X = np.vstack((X, res[i]))
    else:
        # weighted matrix(in:1 out:0)
        kd = lib.cg.KDTree(np.array(xymap))
        nw = lib.weights.KNN(kd, k)
        W = lib.weights.W(nw.neighbors, nw.weights)
        adj = full(W)[0]
        row, col = np.diag_indices_from(adj)
        adj[row, col] = 1
        res = {}
        receptor_matrix_0 = adata[nw.neighbors[0], lr_network['to']].X.A.T
        X = np.array([0] * receptor_matrix_0.shape[0])
        for i in range(ligand_matrix.shape[1]):
            receptor_matrix = adata[nw.neighbors[i], lr_network['to']].X.A.T
            res[i] = ligand_matrix[:, i] * receptor_matrix.sum(axis=1)
            X = np.vstack((X, res[i]))
    X = X.T[:, 1:]
    # bucket-bucket pair
    cell_pair = []
    for i, cell_id in nw.neighbors.items():
        cell_pair.append(adata.obs.index[i]+'-'+adata.obs.index[cell_id])
    cell_pair = pd.DataFrame({"cell_pair_name": cell_pair})
    #cell_pair.set_index("cell_pair_name", inplace=True)
    # lr_pair
    lr_pair = lr_network['from']+'-'+lr_network['to']
    lr_pair = pd.DataFrame({"lr_pair_name": lr_pair})
    lr_pair.set_index("lr_pair_name", inplace=True)
    # csr_matrix
    X = sparse.csr_matrix(X.T)
    # adara_niche
    adata_niche = AnnData(X=X, obs=cell_pair, var=lr_pair)
    return adata_niche

# NicheNet
def predict_ligand_activities(
    lt_matrix_path: str,
    adata: AnnData,
    sender_cells: List[str],
    receiver_cells: List[str],
    lr_network_path: str,
    geneset: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Function to predict the ligand activity. 

    Args:
        lt_matrix_path: Path to ligand_target_matrix of NicheNet.
        adata: An Annodata object.
        sender_cells: Ligand cells.
        receiver_cells: Receptor cells.
        geneset: The interest genes set, such as differential expression genes
            (case and control) in receiver cells. predicting activity ligands
            based on this geneset, defult is all genes expressed in receiver cells.
        lr_network_path: Path to ligand_receptor network of NicheNet.
        
    Returns:
        A pandas DataFrame of the predicted activity ligands.

    """
    # load ligand_target_matrix
    url = "https://zenodo.org/record/3260758/files/ligand_target_matrix.rds"
    path = lt_matrix_path + 'ligand_target_matrix.rds'
    path_load = pyreadr.download_file(url, path)
    ligand_target_matrix = pyreadr.read_r(path)[None]
    # Define expressed genes in sender and receiver cell populations(pct>0.1)
    expressed_genes_sender = np.array(adata.var_names)[np.count_nonzero(adata[sender_cells, :].X.A,axis=0)/len(sender_cells)>0.1].tolist()
    expressed_genes_receiver = np.array(adata.var_names)[np.count_nonzero(adata[receiver_cells, :].X.A,axis=0)/len(receiver_cells)>0.1].tolist()
    # Define a set of potential ligands
    url = "https://zenodo.org/record/3260758/files/lr_network.rds"
    path = lr_network_path + 'lr_network.rds'
    path_load = pyreadr.download_file(url, path)
    lr_network = pyreadr.read_r(path)[None]
    ligands = lr_network['from'].unique()
    expressed_ligands = list(set(ligands) & set(expressed_genes_sender))
    receptor = lr_network['to'].unique()
    expressed_receptor = list(set(receptor) & set(expressed_genes_receiver))
    lr_network_expressed = lr_network[lr_network['from'].isin(expressed_ligands) & lr_network['to'].isin(expressed_receptor)]
    potential_ligands = lr_network_expressed['from'].unique()
    if geneset is None:
        # Calculate the pearson coeffcient between potential score and actual average expression of genes
        response_expressed_genes = list(set(expressed_genes_receiver) & set(ligand_target_matrix.index))
        response_expressed_genes_df = pd.DataFrame(response_expressed_genes)
        response_expressed_genes_df = response_expressed_genes_df.rename(columns={0: 'gene'})
        response_expressed_genes_df['avg_expr'] = np.mean(adata[receiver_cells, response_expressed_genes].X.A,axis=0)
        lt_matrix = ligand_target_matrix[potential_ligands.tolist()].loc[response_expressed_genes_df['gene'].tolist()]
        de =[]
        for ligand in lt_matrix:
            pear_coef = pearsonr(lt_matrix[ligand], response_expressed_genes_df['avg_expr'])[0]
            pear_pvalue = pearsonr(lt_matrix[ligand], response_expressed_genes_df['avg_expr'])[1]
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
        gene_io['logical'] = 1
        gene_io = gene_io.rename(columns={0: 'gene'})
        background_expressed_genes = pd.DataFrame(background_expressed_genes)
        background = background_expressed_genes[~(background_expressed_genes[0].isin(gene_io))]
        background = background.rename(columns={0: 'gene'})
        background['logical'] = 0
        # response gene vector
        response = pd.concat([gene_io, background], axis=0, join='outer')
        # lt_matrix potential score
        lt_matrix = ligand_target_matrix[potential_ligands.tolist()].loc[response['gene'].tolist()]
        # predict ligand activity by pearson coefficient.
        de =[]
        for ligand in lt_matrix:
            pear_coef = pearsonr(lt_matrix[ligand], response['logical'])[0]
            pear_pvalue = pearsonr(lt_matrix[ligand], response['logical'])[1]
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
            'ligand',
            'pearson_coef',
            'pearson_pvalue',
        ]
    )
    return res


def predict_target_genes(
    lt_matrix_path: str,
    adata: AnnData,
    sender_cells: List[str],
    receiver_cells: List[str],
    lr_network_path: str,
    geneset: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Function to predict the target genes.

    Args:
        lt_matrix_path: Path to ligand_target_matrix of NicheNet.
        adata: An Annodata object.
        sender_cells: Ligand cells.
        receiver_cells: Receptor cells.
        geneset: The interest genes set, such as differential expression genes
            (case and control) in receiver cells. predicting activity ligands
            based on this geneset, defult is all genes expressed in receiver cells.
        lr_network_path: Path to ligand_receptor network of NicheNet.
        
    Returns:
        A pandas DataFrame of the predict target genes.

    """
    path = lt_matrix_path + 'ligand_target_matrix.rds'
    ligand_target_matrix = pyreadr.read_r(path)[None]
    predict_ligand = predict_ligand_activities(
        lt_matrix_path=lt_matrix_path,
        adata=adata,
        sender_cells=sender_cells,
        receiver_cells=receiver_cells,
        lr_network_path=lr_network_path,
        geneset=None)
    predict_ligand.sort_values(by='pearson_coef', axis=0, ascending=False, inplace=True)
    predict_ligand_top = predict_ligand[:20]['ligand']
    res = pd.DataFrame(columns=('ligand', 'targets', 'weights'))
    for ligand in ligand_target_matrix[predict_ligand_top]:
        top_n_score = ligand_target_matrix[ligand].sort_values(ascending=False)[:250]
        if geneset is None:
            expressed_genes_receiver = np.array(adata.var_names)[np.count_nonzero(adata[receiver_cells, :].X.A, axis=0)/len(receiver_cells) > 0.1].tolist()
            response_expressed_genes = list(set(expressed_genes_receiver) & set(ligand_target_matrix.index))
            targets = list(set(top_n_score.index) & set(response_expressed_genes))
        else:
            gene_io = list(set(geneset) & set(ligand_target_matrix.index))
            targets = list(set(top_n_score.index) & set(gene_io))
        res = pd.concat([res, pd.DataFrame({'ligand': ligand, 'targets': targets, 'weights': ligand_target_matrix.loc[targets, ligand]}).reset_index(drop=True)], axis=0, join='outer')  
    return res


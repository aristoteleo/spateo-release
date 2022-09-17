"""
Suite of tools for spatially-aware as well as spatially-lagged linear regression

Also performs downstream characterization following spatially-informed regression to characterize niche impact on gene
expression

Developer note: may change from the manual version of OLS here to a version of things that runs through
Statsmodels- I think the summary functionality is useful!
Developer note: haven't been able to find a good network for Drosophila and Zebrafish yet, so spatially lagged models
are restricted to human, mouse and axolotl. Might also be making a lot of assumptions about the axolotl,
but the axolotl LR network has columns for human equivalents for all LR so I assumed there's enough homology there
Developer note: for the sender/receiver effect functions, each row in pvalues, fold_change, etc. is a receiver,
each column is a sender

Developer note: currently, processing of 'connections' is set to encode the presence of cell type proximities as
either a 0 or 1. Another option is to minmax scale across rows.

Developer note: still to add: incorporation of Lasso OLS, maybe cross-validation wrapper in addition to the metrics at
the bottom to return all metrics for each fold.
"""
import os
import time
from itertools import product
from random import sample
from typing import List, Literal, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from anndata import AnnData
from dynamo.tools.moments import calc_1nd_moment
from joblib import Parallel, delayed
from matplotlib import rcParams
from patsy import dmatrix
from pysal.lib import weights
from pysal.model import spreg
from tqdm import tqdm

from ...configuration import SKM, config_spateo_rcParams, set_pub_style
from ...logging import logger_manager as lm
from ...plotting.static.utils import save_return_show_fig_utils
from ...preprocessing.normalize import normalize_total
from ...preprocessing.transform import log1p
from ...tools.find_neighbors import construct_pairwise, transcriptomic_connectivity
from ...tools.utils import update_dict
from .regression_utils import compute_wald_test, get_fisher_inverse, ols_fit_predict
from .spatial_lr_tsls import LR_GM_lag


# ---------------------------------------------------------------------------------------------------
# Wrapper classes for model running
# ---------------------------------------------------------------------------------------------------
class BaseInterpreter:
    """
    Basis class for all spatially-aware and spatially-lagged regression models that can be implemented through this
    toolkit. Includes necessary methods for data loading and preparation, computation of spatial weights matrices,
    computation of evaluation metrics and more.

    Args:
        adata: object of class `anndata.AnnData`
        group_key: Key in .obs where group (e.g. cell type) information can be found
        spatial_key: Key in .obsm where x- and y-coordinates are stored
        genes: Subset to genes of interest: will be used as dependent variables in non-ligand-based regression analyses,
            will be independent variables in ligand-based regression analyses
        drop_dummy: Name of the category to be dropped (the "dummy variable") in the regression. The dummy category
            can aid in interpretation as all model coefficients can be taken to be in reference to the dummy
            category. If None, will randomly select a few samples to constitute the dummy group.
        layer: Entry in .layers to use instead of .X
        cci_dir: Full path to the directory containing cell-cell communication databases. Only used in the case of models
            that use ligands for prediction.
        smooth: To correct for dropout effects, leverage gene expression neighborhoods to smooth expression
        log_transform: Set True if log-transformation should be applied to expression (otherwise, will assume
            preprocessing/log-transform was computed beforehand)
        weights_mode: Options "knn", "kernel", "band"; sets whether to use K-nearest neighbors, a kernel-based
            method, or distance band to compute spatial weights, respectively.
        data_id: If given, will save pairwise distance arrays & nearest neighbor arrays to folder in the working
            directory, under './neighbors/{data_id}_distance.csv' and './neighbors/{data_id}_adj.csv'. Will also
            check for existing files under these names to avoid re-computing these arrays. If not given, will not save.
        kwargs : Provides additional spatial weight-finding arguments. Note that these must specifically match the name
            that the function will look for (case sensitive). For reference:
                - n_neighbors : int
                    Number of nearest neighbors for KNN
                - p : int
                    Minkowski p-norm for KNN and distance band methods
                - distance_metric : str
                    Pairwise distance metric for KNN
                - bandwidth : float or array-like of floats
                    Sets kernel width for kernel method
                - fixed : bool
                    Allow bandwidth to vary across observations for kernel method
                - n_neighbors_bandwidth : int
                    Number of nearest neighbors for determining bandwidth for kernel method
                - kernel_function : str
                    "triangular", "uniform", "quadratic", "quartic" or "gaussian". Rule for setting how spatial
                    weight decays with distance
                - threshold : float
                    Distance for which to consider spots "neighbors" for each spot in distance band method (typically
                    in units of pixels)
                - alpha : float
                    Should be less than 0; can be used to set weights to decay with distance for distance band method
    """

    def __init__(
        self,
        adata: AnnData,
        spatial_key: str = "spatial",
        group_key: Union[None, str] = None,
        genes: Union[None, List] = None,
        drop_dummy: Union[None, str] = None,
        layer: Union[None, str] = None,
        cci_dir: Union[None, str] = None,
        smooth: bool = False,
        log_transform: bool = False,
        weights_mode: str = "knn",
        data_id: Union[None, str] = None,
        **kwargs,
    ):
        self.logger = lm.get_main_logger()

        self.adata = adata
        self.cell_names = self.adata.obs_names
        # Sort cell type categories (to keep order consistent for downstream applications):
        self.celltype_names = sorted(list(set(adata.obs[group_key])))

        self.spatial_key = spatial_key
        self.group_key = group_key
        self.genes = genes
        self.logger.info(
            "Note: argument provided to 'genes' represents the dependent variables for non-ligand-based "
            "analysis, but are used as independent variables for ligand-based analysis."
        )
        self.drop_dummy = drop_dummy
        self.layer = layer
        self.cci_dir = cci_dir
        self.smooth = smooth
        self.log_transform = log_transform
        self.weights_mode = weights_mode
        self.data_id = data_id

        # Kwargs can be used to adjust how spatial weights/spatial neighbors are found.
        # Default values for all possible spatial weight/spatial neighbor parameters:
        self.sp_kwargs = {
            "n_neighbors": 10,
            "p": 2,
            "distance_metric": "euclidean",
            "bandwidth": None,
            "fixed": True,
            "n_neighbors_bandwidth": 6,
            "kernel_function": "triangular",
            "threshold": 50,
            "alpha": -1.0,
        }

        # Update using user input:
        self.sp_kwargs = update_dict(self.sp_kwargs, kwargs)

        # Define reconstruction metrics:
        self.metrics = [mae, mse, nll, r_squared]

    def prepare_data(
        self,
        mod_type: str = "category",
        lig: Union[None, List[str]] = None,
        rec: Union[None, List[str]] = None,
        niche_lr_r_lag: bool = True,
        use_ds: bool = True,
        rec_ds: Union[None, List[str]] = None,
        species: Literal["human", "mouse", "axolotl"] = "human",
    ):
        """
        Handles any necessary data preparation, starting from given source AnnData object

        Args:
            mod_type: The type of model that will be employed- this dictates how the data will be processed and
                prepared.
                Options:
                    - category: spatially-aware, for each sample, computes category prevalence within the spatial
                    neighborhood and uses these as independent variables
                    - niche: spatially-aware, uses spatial connections between samples as independent variables
                    - ligand_lag: spatially-lagged, from database uses select ligand genes to perform regression on
                        select receptor and/or receptor-downstream genes, and additionally considers neighbor
                        expression of the ligands
                    - niche_ligand_lag: spatially-lagged, uses a combination of categories, spatial connections,
                        ligand genes and ligand gene expression of the neighbors to perform regression on select
                        receptor and/or receptor-downstream genes
                    - niche_lr: spatially-aware, uses a coupling of spatial category connections, ligand expression
                        and receptor expression to perform regression on select receptor-downstream genes
            lig: Only used if 'mod_type' contains "ligand". Provides the list of ligands to use as predictors. If not
                given, will attempt to subset self.genes
            rec: Only used if 'mod_type' contains "ligand". Provides the list of receptors to investigate. If not
                given, will search through database for all genes that correspond to the provided genes from 'ligands'.
            niche_lr_r_lag: Only used if 'mod_type' is "niche_lr". Uses the spatial lag of the receptor as the
                dependent variable rather than each spot's unique receptor expression. Defaults to True.
            use_ds: If True, uses receptor-downstream genes in addition to ligands and receptors.
            rec_ds: Only used if 'mod_type' is "niche_lr" or "ligand_lag". Can be used to optionally manually define a
                list of genes shown to be (or thought to potentially be) downstream of one or more of the provided
                L:R pairs. If not given, will find receptor-downstream genes from database based on input to 'lig'
                and 'rec'.
            species: Selects the cell-cell communication database the relevant ligands will be drawn from. Options:
                "human", "mouse", "axolotl" (EVENTUALLY, WILL ALSO INCLUDE DROSOPHILA/ZEBRAFISH)
        """
        # Can provide either a single L:R pair or multiple of ligands and/or receptors:
        if lig is not None:
            if not isinstance(lig, list):
                lig = [lig]
        if rec is not None:
            if not isinstance(rec, list):
                rec = [rec]
        if rec_ds is not None:
            if not isinstance(rec_ds, list):
                rec_ds = [rec_ds]

        # Normalize to size factor:
        normalize_total(self.adata)

        # Smooth data if 'smooth' is True and log-transform data matrix if 'log_transform' is True:
        if self.smooth:
            self.logger.info("Smoothing gene expression...")
            # Compute connectivity matrix if not already existing:
            try:
                conn = self.adata.obsp["connectivities"]
            except:
                _, adata = transcriptomic_connectivity(self.adata, n_neighbors_method="ball_tree")
                conn = adata.obsp["connectivities"]
            adata_smooth_norm, _ = calc_1nd_moment(self.adata.X, conn, normalize_W=True)
            self.adata.layers["M_s"] = adata_smooth_norm

            # Use smoothed layer for downstream processing:
            self.adata.layers["raw"] = self.adata.X
            self.adata.X = self.adata.layers["M_s"]

        if self.log_transform:
            if self.layer is None:
                log1p(self.adata)
            else:
                log1p(self.adata.layers[self.layer])
        else:
            self.logger.warn(
                "Linear regression models are not well suited to the distributional characteristics of "
                "raw transcriptomic data- it is recommended to perform a log-transformation first."
            )

        # General preprocessing required by multiple model types (for the models that use cellular niches):
        # Convert groups/categories into one-hot array:
        group_name = self.adata.obs[self.group_key]
        db = pd.DataFrame({"group": group_name})
        categories = np.array(self.adata.obs[self.group_key].unique().tolist())
        db["group"] = pd.Categorical(db["group"], categories=categories)

        self.logger.info("Preparing data: converting categories to one-hot labels for all samples.")
        X = pd.get_dummies(data=db, drop_first=False)
        # Ensure columns are in order:
        X = X.reindex(sorted(X.columns), axis=1)

        # Compute adjacency matrix- use the KNN value in 'sp_kwargs' (which may have been passed as an
        # argument when initializing the interpreter):
        construct_pairwise(
            self.adata, spatial_key=self.spatial_key, n_neighbors=self.sp_kwargs["n_neighbors"], exclude_self=True
        )
        adj = self.adata.obsm["adj"]

        # Construct category adjacency matrix (n_samples x n_categories array that records how many neighbors of
        # each category are present within the neighborhood of each sample):
        dmat_neighbors = (adj > 0).astype("int").dot(X.values)

        # Construct the category interaction matrix (1D array w/ n_categories ** 2 elements, encodes the niche of
        # each sample by documenting the category-category spatial connections within the niche- specifically,
        # for each sample, records the category identity of its neighbors in space):
        data = {"categories": X, "dmat_neighbours": dmat_neighbors}
        connections = np.asarray(dmatrix("categories:dmat_neighbours-1", data))

        # Set all elements of 'connections' to be binary to represent the presence/absence of a connection:
        connections[connections > 1] = 1

        # Specific preprocessing for each model type:
        if "category" in mod_type:
            self.logger.info(f"Using {self.group_key} values to predict feature expression...")
            # First, convert groups/categories into one-hot array:
            group_num = self.adata.obs[self.group_key].value_counts()
            max_group, min_group, min_group_ncells = (
                group_num.index[0],
                group_num.index[-1],
                group_num.values[-1],
            )

            group_name = self.adata.obs[self.group_key]
            db = pd.DataFrame({"group": group_name})
            categories = np.array(self.adata.obs[self.group_key].unique().tolist() + ["others"])
            db["group"] = pd.Categorical(db["group"], categories=categories)

            # Solve the dummy variable trap by dropping dummy category (deleting rows to avoid issues with
            # multicollinearity):
            if self.drop_dummy is None:
                # Leave some samples from every group intact
                db.iloc[sample(np.arange(self.adata.n_obs).tolist(), min_group_ncells), :] = "others"
            elif self.drop_dummy in categories:
                group_inds = np.where(db["group"] == self.drop_dummy)[0]
                db.iloc[group_inds, :] = "others"
                db = db["group"].cat.remove_unused_categories()
            else:
                raise ValueError(
                    f"Dummy category ({self.drop_dummy}) provided is not in the " f"adata.obs[{self.group_key}]."
                )
            drop_columns = ["group_others"]

            self.logger.info("Preparing data: converting categories to one-hot labels for all samples.")
            X = pd.get_dummies(data=db, drop_first=False)
            # Ensure columns are in order:
            X = X.reindex(sorted(X.columns), axis=1)
            # Compute adjacency matrix- use the KNN value in 'sp_kwargs' (which may have been passed as an
            # argument when initializing the interpreter):
            if self.data_id is not None:
                self.logger.info(f"Checking for pre-computed adjacency matrix for dataset {self.data_id}...")
                try:
                    self.adata.obsm["adj"] = pd.read_csv(
                        os.path.join(os.getcwd(), f"neighbors/{self.data_id}_neighbors.csv"), index_col=0
                    ).values
                    self.logger.info(f"Adjacency matrix loaded from file.")
                except:
                    self.logger.info(f"Pre-computed adjacency matrix not found. Computing adjacency matrix.")
                    start = time.time()
                    construct_pairwise(
                        self.adata,
                        spatial_key=self.spatial_key,
                        n_neighbors=self.sp_kwargs["n_neighbors"],
                        exclude_self=True,
                    )
                    self.logger.info(f"Computed adjacency matrix, time elapsed: {time.time()-start}s.")
            adj = self.adata.obsm["adj"]

            # Construct category adjacency matrix (n_samples x n_categories array that records how many neighbors of
            # each category are present within the neighborhood of each sample):
            dmat_neighbors = (adj > 0).astype("int").dot(X.values)
            self.X = pd.DataFrame(dmat_neighbors, columns=X.columns)
            self.X = self.X.reindex(sorted(self.X.columns), axis=1)
            self.n_features = self.X.shape[1]

            # To index all but the dummy column when fitting model:
            self.variable_names = self.X.columns.difference(drop_columns).to_list()

            # Get the names of all remaining groups:
            self.param_labels, group_name = (
                set(group_name).difference([self.drop_dummy]),
                group_name.to_list(),
            )

            self.param_labels = list(np.sort(list(self.param_labels)))

        elif mod_type == "niche":
            # If mod_type is 'niche', use the connections matrix as independent variables in the regression:
            connections_cols = list(product(X.columns, X.columns))
            connections_cols.sort(key=lambda x: x[1])
            connections_cols = [f"{i[0]}-{i[1]}" for i in connections_cols]
            self.X = pd.DataFrame(connections, columns=connections_cols)
            """
            # Otherwise if 'niche', combine two arrays:
            # 'connections', encoding pairwise *spatial adjacencies* between categories for each sample, and
            # 'dmat_neighbors', encoding *identity* and *prevalence* of the niche components
            elif mod_type == "niche":
                connections_cols = list(product(X.columns, X.columns))
                connections_cols.sort(key=lambda x: x[1])
                connections_cols = [f"{i[0]}-{i[1]}" for i in connections_cols]
                connections_df = pd.DataFrame(connections, index=X.index, columns=connections_cols)
                self.X = pd.concat([X, connections_df], axis=1)"""

            # Set self.param_labels to reflect inclusion of the interactions:
            self.param_labels = self.variable_names = self.X.columns

        elif "ligand_lag" in mod_type:
            ligands = lig
            receiving_genes = rec

            # Load signaling network file and find appropriate subset (if 'species' is axolotl, use the human portion
            # of the network):
            signet = pd.read_csv(os.path.join(self.cci_dir, "human_mouse_signaling_network.csv"), index_col=0)
            if species not in ["human", "mouse", "axolotl"]:
                self.logger.error("Invalid input to 'species'. Options: 'human', 'mouse', 'axolotl'.")
            if species == "axolotl":
                species = "human"
                axolotl_lr = pd.read_csv(os.path.join(self.cci_dir, "lr_network_axolotl.csv"), index_col=0)
                axolotl_l = set(axolotl_lr["human_ligand"])
            sig_net = signet[signet["species"] == species.title()]
            lig_available = set(sig_net["src"])
            if "axolotl_l" in locals():
                lig_available = lig_available.union(axolotl_l)

            # Set predictors and target- for consistency with field conventions, set ligands and ligand-downstream
            # gene names to uppercase (the AnnData object is assumed to follow this convention as well):
            # Use the argument provided to 'ligands' to set the predictor block:
            if ligands is None:
                ligands = [g.upper() for g in self.genes if g in lig_available]
            else:
                # Filter provided ligands to those that can be found in the database:
                ligands = [l for l in ligands if l in lig_available]
                self.logger.info("Proceeding with analysis using ligands {}".format(",".join(ligands)))

            # Filter ligands to those that can be found in the database:
            ligands = [l for l in ligands if l in self.adata.var_names]
            if len(ligands) == 0:
                self.logger.error(
                    "None of the ligands could be found in AnnData variable names. "
                    "Check that AnnData index names match database entries."
                    "Also possible to have selected only ligands that can't be found in AnnData- "
                    "select different ligands."
                )
            self.n_ligands = len(ligands)

            ligands_expr = pd.DataFrame(
                self.adata[:, ligands].X.toarray() if scipy.sparse.issparse(self.adata.X) else self.adata[:, ligands].X,
                index=X.index,
                columns=ligands,
            )

            if mod_type == "niche_ligand_lag":
                # Combine ligand expression with niche information:
                # category_df = pd.DataFrame(categories, columns=X.columns)

                connections_cols = list(product(X.columns, X.columns))
                connections_cols.sort(key=lambda x: x[1])
                connections_cols = [f"{i[0]}-{i[1]}" for i in connections_cols]
                connections_df = pd.DataFrame(connections, index=X.index, columns=connections_cols)
                self.X = pd.concat([connections_df, ligands_expr], axis=1)
            else:
                self.X = pd.DataFrame(ligands_expr, columns=ligands)

            if receiving_genes is None:
                # Append all receptors (direct connections to ligands):
                # Note that the database contains repeat L:R pairs- each pair is listed more than once if it is part
                # of more than one pathway. Furthermore, if two ligands bind the same receptor, the receptor will be
                # listed twice. Since we're looking for just the names, take the set of receptors/downstream genes
                # to get only unique molecules:
                receptors = set(list(sig_net.loc[sig_net["src"].isin(ligands)]["dest"].values))
                receiving_genes = list(receptors)
                self.logger.info(
                    "List of receptors was not provided- found these receptors from the provided "
                    f"ligands: {(', ').join(receiving_genes)}"
                )

            if rec_ds is not None:
                # If specific list of downstream genes (indirect connections to ligands) is provided, append:
                receiving_genes.extend(rec_ds)
            elif use_ds:
                #  Optionally append all downstream genes from the database (direct connections to receptors,
                #  indirect connections to ligands):
                self.logger.info(
                    "Downstream genes were not manually provided with 'rec_ds'...automatically "
                    "searching for downstream genes associated with the discovered 'receptors'."
                )
                receiver_ds = list(set(list(sig_net.loc[sig_net["src"].isin(receiving_genes)]["dest"].values)))
                self.logger.info(
                    "List of receptor-downstream genes was not provided- found these genes from the "
                    f"current list of receivers: {(', ').join(receiver_ds)}"
                )
                receiving_genes.extend(receiver_ds)
                receiving_genes = list(set(receiving_genes))

            # Filter receiving genes for those that can be found in the dataset:
            receiving_genes = [r for r in receiving_genes if r in self.adata.var_names]

            self.genes = receiving_genes

            if mod_type == "ligand_lag":
                # All ligands will have associated parameters and be used as variables in the model
                self.param_labels = self.variable_names = ligands
            elif mod_type == "niche_ligand_lag":
                # All ligands and connections will have associated parameters, but only ligands will be lagged
                self.param_labels = self.X.columns
                self.variable_names = ligands

        elif mod_type == "niche_lr":
            # Load LR database based on input to 'species':
            if species == "human":
                lr_network = pd.read_csv(os.path.join(self.cci_dir, "lr_network_human.csv"), index_col=0)
            elif species == "mouse":
                lr_network = pd.read_csv(os.path.join(self.cci_dir, "lr_network_mouse.csv"), index_col=0)
            # elif species == "drosophila":
            #    lr_network = pd.read_csv(os.path.join(self.cci_dir, "lr_network_drosophila.csv"), index_col=0)
            # elif species == "zebrafish":
            #    lr_network = pd.read_csv(os.path.join(self.cci_dir, "lr_network_zebrafish.csv"), index_col=0)
            elif species == "axolotl":
                lr_network = pd.read_csv(os.path.join(self.cci_dir, "lr_network_axolotl.csv"), index_col=0)
            else:
                self.logger.error("Invalid input given to 'species'. Options: 'human', 'mouse', or 'axolotl'.")

            if lig is None:
                self.logger.error("For 'mod_type' = 'niche_lr', ligands must be provided.")
            lig = [l.upper() for l in lig]
            # If no receptors are given, search database for matches w/ the ligand:
            if rec is None:
                rec = set(list(lr_network.loc[lr_network["from"].isin(lig)]["to"].values))
                rec = [r.upper() for r in rec]
                self.logger.info(
                    "List of receptors was not provided- found these receptors from the provided "
                    f"ligands: {(', ').join(rec)}"
                )

            # Filter ligand and receptor lists to those that can be found in the data:
            lig = [l for l in lig if l in self.adata.var_names]
            if len(lig) == 0:
                self.logger.error(
                    "None of the ligands could be found in AnnData variable names. "
                    "Check that AnnData index names match database entries."
                    "Also possible to have selected only ligands that can't be found in AnnData- "
                    "select different ligands."
                )
            rec = [r for r in rec if r in self.adata.var_names]

            # Convert groups/categories into one-hot array:
            group_name = self.adata.obs[self.group_key]
            db = pd.DataFrame({"group": group_name})
            categories = np.array(self.adata.obs[self.group_key].unique().tolist())
            n_categories = len(categories)
            db["group"] = pd.Categorical(db["group"], categories=categories)

            self.logger.info("Preparing data: converting categories to one-hot labels for all samples.")
            X = pd.get_dummies(data=db, drop_first=False)
            # Ensure columns are in order:
            X = X.reindex(sorted(X.columns), axis=1)

            # Compute adjacency matrix- use the KNN value in 'sp_kwargs' (which may have been passed as an
            # argument when initializing the interpreter):
            construct_pairwise(
                self.adata, spatial_key=self.spatial_key, n_neighbors=self.sp_kwargs["n_neighbors"], exclude_self=True
            )
            adj = self.adata.obsm["adj"]

            # 'l' and 'r' must be matched, and so must be the same length, unless it is a case of one ligand that can
            # bind multiple receptors or vice versa:
            if len(lig) != len(rec):
                self.logger.warning(
                    "Length of the provided list of ligands (input to 'l') does not match the length "
                    "of the provided list of receptors (input to 'r'). This is fine, so long as all ligands and "
                    "all receptors have at least one match in the other list."
                )

            pairs = []
            # This analysis takes ligand and receptor expression to predict expression of downstream genes- make sure
            # (1) input to 'r' are listed as receptors in the appropriate database, and (2) for each input to 'r',
            # there is a matched input in 'l':
            for ligand in lig:
                lig_key = "from" if species != "axolotl" else "human_ligand"
                rec_key = "to" if species != "axolotl" else "human_receptor"
                possible_receptors = set(lr_network.loc[lr_network[lig_key] == ligand][rec_key])
                possible_receptors = [r.upper() for r in possible_receptors]
                if not any(receptor in possible_receptors for receptor in rec):
                    self.logger.error(
                        "No record of {} interaction with any of {}. Ensure provided lists contain "
                        "paired ligand-receptors.".format(ligand, (",".join(rec)))
                    )
                found_receptors = list(set(possible_receptors).intersection(set(rec)))
                lig_pairs = list(product(lig, found_receptors))
                pairs.extend(lig_pairs)
            self.n_pairs = len(pairs)
            print(f"Setting up Niche-L:R model using the following ligand-receptor pairs: {pairs}")

            self.logger.info(
                "Starting from {} ligands and {} receptors, found {} ligand-receptor "
                "pairs.".format(len(lig), len(rec), len(pairs))
            )

            # Since features are combinatorial, it is not recommended to specify more than too many ligand-receptor
            # pairs:
            if len(pairs) > 200 / n_categories**2:
                self.logger.warning(
                    "Regression model has many predictors- consider measuring fewer ligands and " "receptors."
                )

            # Each ligand-receptor pair will have an associated niche matrix:
            self.niche_mats = {}

            for lr_pair in pairs:
                lig, rec = lr_pair[0], lr_pair[1]
                lig_expr_values = (
                    self.adata[:, lig].X.toarray() if scipy.sparse.issparse(self.adata.X) else self.adata[:, lig].X
                )
                rec_expr_values = (
                    self.adata[:, rec].X.toarray() if scipy.sparse.issparse(self.adata.X) else self.adata[:, rec].X
                )
                # Optionally, compute the spatial lag of the receptor:
                if niche_lr_r_lag:
                    if not hasattr(self, "w"):
                        self.compute_spatial_weights()
                    rec_lag = spreg.utils.lag_spatial(self.w, rec_expr_values)
                    self.adata.obs[f"{rec}_lag"] = rec_lag
                # Multiply one-hot category array by the expression of select receptor within that cell:
                rec_vals = self.adata[:, rec].X if not niche_lr_r_lag else self.adata.obs[f"{rec}_lag"].values
                rec_expr = np.multiply(X.values, np.tile(rec_vals.reshape(-1, 1), X.shape[1]))

                # Separately multiply by the expression of select ligand such that an expression value only exists
                # for one cell type per row:
                lig_vals = lig_expr_values
                lig_expr = np.multiply(X.values, np.tile(lig_vals, X.shape[1]))
                # Multiply adjacency matrix by the cell-specific expression of select ligand:
                nbhd_lig_expr = (adj > 0).astype("int").dot(lig_expr)

                # Construct the category interaction matrix (1D array w/ n_categories ** 2 elements, encodes the
                # ligand-receptor niches of each sample by documenting the cell type-specific L:R enrichment within
                # the niche:
                data = {"category_rec_expr": rec_expr, "neighborhood_lig_expr": nbhd_lig_expr}
                lr_connections = np.asarray(dmatrix("category_rec_expr:neighborhood_lig_expr-1", data))

                lr_connections_cols = list(product(X.columns, X.columns))
                lr_connections_cols.sort(key=lambda x: x[1])
                lr_connections_cols = [f"{i[0]}-{i[1]}_{lig}-{rec}" for i in lr_connections_cols]
                self.niche_mats[f"{lig}-{rec}"] = pd.DataFrame(lr_connections, columns=lr_connections_cols)
                self.niche_mats = {key: value for key, value in sorted(self.niche_mats.items())}

            # Define set of variables to regress on- genes downstream of the receptor. Can use custom provided list
            # or create the list from database:
            if rec_ds is not None:
                # If specific list of downstream genes (indirect connections to ligands) is provided, append:
                ds = rec_ds
            else:
                #  Optionally append all downstream genes from the database (direct connections to receptors,
                #  indirect connections to ligands):
                receptors = set([pair[1] for pair in pairs])
                signet = pd.read_csv(os.path.join(self.cci_dir, "human_mouse_signaling_network.csv"), index_col=0)
                if species == "axolotl":
                    species = "human"
                sig_net = signet[signet["species"] == species.title()]

                receiver_ds = set(list(sig_net.loc[sig_net["src"].isin(receptors)]["dest"].values))
                ds = list(receiver_ds)
                self.logger.info(
                    "List of receptor-downstream genes was not provided- found these genes from the "
                    f"provided receptors: {(', ').join(ds)}"
                )
            self.genes = ds
            self.X = pd.concat(self.niche_mats, axis=1)
            self.X.columns = self.X.columns.droplevel()

            # Minmax-scale columns to minimize the external impact of intercellular differences in ligand/receptor
            # expression:
            self.X = (self.X - self.X.min()) / (self.X.max() - self.X.min())

            self.param_labels = self.variable_names = self.X.columns

        else:
            self.logger.error("Invalid argument to 'mod_type'.")

        # Save model type as an attribute so it can be accessed by other methods:
        self.mod_type = mod_type

        # Filter gene names if specific gene names are provided. If not, use all genes referenced in .X:
        if self.genes is not None:
            self.genes = list(self.adata.var.index.intersection(self.genes))
        else:
            self.genes = list(self.adata.var.index)
        self.adata = self.adata[:, self.genes]

    def compute_spatial_weights(self):
        """
        Generates matrix of pairwise spatial distances, used in spatially-lagged models
        """
        # Choose how weights are computed:
        if self.weights_mode == "knn":
            self.w = weights.distance.KNN.from_array(self.adata.obsm[self.spatial_key], k=self.sp_kwargs["n_neighbors"])
        elif self.weights_mode == "kernel":
            self.w = weights.distance.Kernel.from_array(
                self.adata.obsm[self.spatial_key],
                bandwidth=self.sp_kwargs["bandwidth"],
                fixed=self.sp_kwargs["fixed"],
                k=self.sp_kwargs["n_neighbors_bandwidth"],
                function=self.sp_kwargs["kernel_function"],
            )
        elif self.weights_mode == "band":
            self.w = weights.distance.DistanceBand.from_array(
                self.adata.obsm[self.spatial_key], threshold=self.sp_kwargs["threshold"], alpha=self.sp_kwargs["alpha"]
            )
        else:
            self.logger.error("Invalid input to 'weights_mode'. Options: 'knn', 'kernel', 'band'.")

        # Row standardize spatial weights matrix:
        self.w.transform = "R"

    # ---------------------------------------------------------------------------------------------------
    # Computing parameters for spatially-aware and lagged models
    # ---------------------------------------------------------------------------------------------------
    def run_OLS(self, n_jobs: int = 30) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Wrapper for ordinary least squares regression.

        Args:
            n_jobs: For parallel processing, number of tasks to run at once

        Returns:
            coeffs: Contains fitted parameters for each feature
            reconst: Contains predicted expression for each feature
        """
        results = Parallel(n_jobs)(
            delayed(ols_fit_predict)(self.X, self.adata, self.variable_names, cur_g) for cur_g in self.genes
        )
        coeffs = [item[0] for item in results]
        reconst = [item[1] for item in results]

        coeffs = pd.DataFrame(coeffs, index=self.genes, columns=self.X.columns)
        for cn in coeffs.columns:
            self.adata.var.loc[:, cn] = coeffs[cn]
        # Nested list transforms into dataframe rows- instantiate and transpose to get to correct shape:
        reconst = pd.DataFrame(reconst, index=self.genes, columns=self.cell_names).T
        return coeffs, reconst

    def run_lasso_LS(self, n_jobs: int = 30) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Wrapper for Lasso-regularized ordinary least squares regression.

        Args:
            n_jobs: For parallel processing, number of tasks to run at once
            *TO INSERT: LASSO ARGUMENTS*

        Returns:
            coeffs: Contains fitted parameters for each feature
            reconst: Contains predicted expression for each feature
        """

    # ---------------------------------------------------------------------------------------------------
    # Downstream interpretation (mostly for interaction models)
    # ---------------------------------------------------------------------------------------------------
    def visualize_params(self, coeffs: pd.DataFrame):
        """
        Generates heatmap of parameter values for visualization

        coeffs: Contains coefficients from regression for each variable
        """

    def get_sender_receiver_effects(
        self,
        coeffs: pd.DataFrame,
        significance_threshold: float = 0.05,
        lr_pair: Union[None, str] = None,
        save_prefix: Union[None, str] = None,
    ):
        """
        For each predictor and each feature, determine if the influence of said predictor in predicting said feature is
        significant.

        Additionally, if the connections b/w categories are used as variables for regression,
        for each feature and each sender-receiver category pair, determines the log fold-change that the sender
        induces in the feature for the receiver.

        Only valid if the model specified uses the connections between categories as variables for the regression-
        thus can be applied to 'mod_type' "connections", "niche", or "niche_lr".

        Args:
            coeffs : Contains coefficients from regression for each variable
            significance_threshold: p-value needed to call a sender-receiver relationship significant
            lr_pair: Required if (and used only in the case that) coefficients came from a Niche-LR model; used to
                subset the coefficients array to the specific ligand-receptor pair of interest. Takes the form
                "{ligand}-{receptor}" and should match one of the keys in :dict `self.niche_mats`. If not given,
                will default to the first key in this dictionary.
            save_prefix: If provided, saves all relevant dataframes to :path `./regression_outputs` under the name
                `{prefix}_{coeffs/pvalues, etc.}.csv`. If not provided, will not save.
        """
        if not "connections" in self.mod_type or not "niche" in self.mod_type:
            self.logger.error(
                "Type coupling analysis only valid if connections between categories are used as the "
                "predictor variable."
            )

        # Save labels of indices and columns (correspond to features & parameters, respectively, for the coeffs
        # DataFrame, will be columns & indices respectively for the other arrays generated by this function):
        feature_labels = coeffs.index
        param_labels = coeffs.columns

        # Return only the numerical coefficients:
        coeffs_np = coeffs[[col for col in coeffs.columns if "coeff" in col]].values
        if "lag" in self.mod_type:
            # Remove the first column (the intercept):
            coeffs_np = coeffs_np[:, 1:]

        # Get inverse Fisher information matrix, with the y block containing all features that were used in regression):
        y = self.adata[:, self.genes].X
        inverse_fisher = get_fisher_inverse(self.X.values, y)

        # Compute significance for each parameter:
        is_significant, pvalues, qvalues = compute_wald_test(
            params=coeffs_np, fisher_inv=inverse_fisher, significance_threshold=significance_threshold
        )

        # If 'save_prefix' is given, save the complete coefficients, significance, p-value and q-value matrices:
        if save_prefix is not None:
            is_significant = pd.DataFrame(is_significant, index=param_labels, columns=feature_labels)
            pvalues = pd.DataFrame(pvalues, index=param_labels, columns=feature_labels)
            qvalues = pd.DataFrame(qvalues, index=param_labels, columns=feature_labels)

            if not os.path.exists("./regression_outputs"):
                os.makedirs("./regression_outputs")
            is_significant.to_csv(f"./regression_outputs/{save_prefix}_is_sign.csv")
            pvalues.to_csv(f"./regression_outputs/{save_prefix}_pvalues.csv")
            qvalues.to_csv(f"./regression_outputs/{save_prefix}_qvalues.csv")
            coeffs.to_csv(f"./regression_outputs/{save_prefix}_coeffs.csv")

        # If niche ligand lag model, extract the portion that corresponds to the interaction terms:
        if self.mod_type == "niche_ligand_lag":
            interaction_shape = np.int(self.n_features**2)
            is_significant = is_significant[:interaction_shape, :]
            pvalues = pvalues[:interaction_shape, :]
            qvalues = qvalues[:interaction_shape, :]

            # Compute the fold-change induced in the receiver by the sender, from the interaction terms:
            interaction_params = coeffs_np[:, :interaction_shape]
            # Split array such that an nxn matrix is created, where n is 'n_features' (the number of cell type
            # categories)
            self.fold_change = np.concatenate(
                np.expand_dims(
                    np.split(interaction_params.T, indices_or_sections=np.sqrt(interaction_params.T.shape[0]), axis=0),
                    axis=0,
                ),
                axis=0,
            )
        # If niche-LR model, extract the portion corresponding to the interaction terms for a specific L-R pair:
        elif self.mod_type == "niche_lr":
            if lr_pair is None:
                self.logger.warning(
                    "'lr_pair' not specified- defaulting to the first L:R pair that was used for the "
                    "model. For reference, all L:R pairs used for the "
                    f"model: {list(self.niche_mats.keys())}"
                )
                lr_pair = list(self.niche_mats.keys())[0]
            if lr_pair not in self.niche_mats.keys():
                self.logger.warning(
                    "Input to 'lr_pair' not recognized- proceeding with the first L:R pair that was "
                    "used for the model. For reference, all L:R pairs used for the "
                    f"model: {list(self.niche_mats.keys())}"
                )
                lr_pair = list(self.niche_mats.keys())[0]

            is_significant = is_significant.filter(lr_pair, axis="index").values
            pvalues = pvalues.filter(lr_pair, axis="index").values
            qvalues = qvalues.filter(lr_pair, axis="index").values

            # Coefficients, etc. will also be a subset of the complete array:
            coeffs_np = coeffs.filter(lr_pair, axis="columns").values
            # Significance, pvalues, qvalues filtered above

            self.fold_change = np.concatenate(
                np.expand_dims(
                    np.split(coeffs_np.T, indices_or_sections=np.sqrt(coeffs_np.T.shape[0]), axis=0), axis=0
                ),
                axis=0,
            )

        # Else if connection-based model, all regression coefficients already correspond to the interaction terms:
        else:
            self.fold_change = np.concatenate(
                np.expand_dims(
                    np.split(coeffs_np.T, indices_or_sections=np.sqrt(coeffs_np.T.shape[0]), axis=0), axis=0
                ),
                axis=0,
            )

        # Split array such that an nxn matrix is created, where n is 'n_features' (the number of cell type
        # categories)
        self.pvalues = np.concatenate(
            np.expand_dims(np.split(pvalues, indices_or_sections=np.sqrt(pvalues.shape[0]), axis=0), axis=0),
            axis=0,
        )
        self.qvalues = np.concatenate(
            np.expand_dims(np.split(qvalues, indices_or_sections=np.sqrt(qvalues.shape[0]), axis=0), axis=0),
            axis=0,
        )
        self.is_significant = np.concatenate(
            np.expand_dims(
                np.split(is_significant, indices_or_sections=np.sqrt(is_significant.shape[0]), axis=0), axis=0
            ),
            axis=0,
        )

    def type_coupling_analysis(
        self,
        cmap: str = "Reds",
        fontsize: Union[None, int] = None,
        figsize: Union[None, Tuple[int, int]] = None,
        save_show_or_return: Literal["save", "show", "return", "both", "all"] = "save",
        save_id: Union[None, str] = None,
        save_kwargs: dict = {},
    ):
        """
        Generates heatmap of spatially differentially-expressed features for each pair of sender and receiver
        categories. Only valid if the model specified uses the connections between categories as variables for the
        regression.

        A high number of differentially-expressed genes between a given sender-receiver pair means that the sender
        being in the neighborhood of the receiver tends to correlate with differential expression levels of many of
        the genes within the selection- much of the cellular variation in the receiver cell type can be attributed to
        being in proximity with the sender.

        Args:
            cmap: Name of Matplotlib color map to use
            fontsize: Size of figure title and axis labels
            figsize: Width and height of plotting window
            save_show_or_return: Options: "save", "show", "return", "both", "all"
                - "both" for save and show
            save_id: Name of the saved figure, without the extension
            save_kwargs: A dictionary that will passed to the save_fig function. By default it is an empty
                dictionary and the save_fig function will use the {"path": None, "prefix": 'scatter', "dpi": None,
                "ext": 'pdf', "transparent": True, "close": True, "verbose": True} as its parameters. But to change
                any of these parameters, this dictionary can be used to do so.
        """
        if fontsize is None:
            self.fontsize = rcParams.get("font.size")
        else:
            self.fontsize = fontsize
        if figsize is None:
            self.figsize = rcParams.get("figure.figsize")
        else:
            self.figsize = figsize

        path = os.path.join(os.getcwd(), "/figures", save_id)
        # Update save_kwargs with save path:
        save_kwargs["path"] = path

        if not hasattr(self, "is_sign"):
            self.logger.warn("Significance dataframe does not exist- please run `get_sender_receiver_effects` first.")

        sig_df = pd.DataFrame(np.sum(self.is_sign, axis=-1), columns=self.cell_names, index=self.cell_names)

        np.fill_diagonal(sig_df.values, 0)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        sns.heatmap(sig_df, square=True, linecolor="grey", linewidths=0.3, cmap=cmap)
        plt.xlabel("Sender")
        plt.ylabel("Receiver")
        plt.tight_layout()

        save_return_show_fig_utils(
            save_show_or_return=save_show_or_return,
            show_legend=True,
            background="white",
            prefix="type_coupling",
            save_kwargs=save_kwargs,
            total_panels=1,
            fig=fig,
            axes=ax,
            return_all=False,
            return_all_list=None,
        )

    def sender_effect(
        self,
        receiver: str,
        plot_mode: str = "fold_change",
        gene_subset: Union[None, List[str]] = None,
        significance_threshold: float = 0.05,
        cut_pvals: float = -5,
        fontsize: Union[None, int] = None,
        figsize: Union[None, Tuple[float, float]] = None,
        cmap: str = "seismic",
        save_show_or_return: Literal["save", "show", "return", "both", "all"] = "show",
        save_kwargs: Optional[dict] = {},
    ):
        """
        Evaluates and visualizes the effect that each sender cell type has on specific genes in the receiver
        cell type.

        Args:
            receiver: Receiver cell type label
            plot_mode: specifies what gets plotted.
                Options:
                    - "qvals": elements of the plot represent statistical significance of the interaction
                    - "fold_change": elements of the plot represent fold change induced in the receiver by the sender
            gene_subset: Names of genes to subset for plot. If None, will use all genes that were used in the
                regression.
            significance_threshold: Set non-significant fold changes to zero, where the threshold is given here
            cut_pvals: Minimum allowable log10(pval)- anything below this will be clipped to this value
            fontsize: Size of figure title and axis labels
            figsize: Width and height of plotting window
            cmap: Name of matplotlib colormap specifying colormap to use
            save_show_or_return: Whether to save, show or return the figure.
                If "both", it will save and plot the figure at the same time. If "all", the figure will be saved,
                displayed and the associated axis and other object will be return.
            save_kwargs: A dictionary that will passed to the save_fig function.
                By default it is an empty dictionary and the save_fig function will use the
                {"path": None, "prefix": 'scatter', "dpi": None, "ext": 'pdf', "transparent": True, "close": True,
                "verbose": True} as its parameters. Otherwise you can provide a dictionary that properly modifies those
                keys according to your needs.
        """
        logger = lm.get_main_logger()
        config_spateo_rcParams()

        if fontsize is None:
            self.fontsize = rcParams.get("font.size")
        else:
            self.fontsize = fontsize
        if figsize is None:
            self.figsize = rcParams.get("figure.figsize")
        else:
            self.figsize = figsize

        receiver_idx = self.celltype_names.index(receiver)

        if plot_mode == "qvals":
            # In the analysis process, the receiving cell types become aligned along the column axis:
            arr = np.log(self.qvalues[receiver_idx, :, :].copy())
            arr[arr < cut_pvals] = cut_pvals
            df = pd.DataFrame(arr, index=self.celltype_names, columns=self.genes)
            if gene_subset:
                df = df.drop(index=receiver)[gene_subset]

            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
            sns.heatmap(
                df.T,
                square=True,
                linecolor="grey",
                linewidths=0.3,
                cbar_kws={"label": "$\log_{10}$ FDR-corrected pvalues"},
                cmap=cmap,
                vmin=-5,
                vmax=0.0,
            )
        elif plot_mode == "fold_change":
            arr = self.fold_change[receiver_idx, :, :].copy()
            arr[np.where(self.qvalues[receiver_idx, :, :] > significance_threshold)] = 0
            df = pd.DataFrame(arr, index=self.celltype_names, columns=self.genes)
            if gene_subset:
                df = df.drop(index=receiver)[gene_subset]
            vmax = np.max(np.abs(df.values))

            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
            sns.heatmap(
                df.T,
                square=True,
                linecolor="grey",
                linewidths=0.3,
                cbar_kws={"label": "$\ln$ fold change", "location": "top"},
                cmap=cmap,
                vmin=-vmax,
                vmax=vmax,
            )
        else:
            logger.error("Invalid input to 'plot_mode'. Options: 'qvals', 'fold_change'.")

        plt.xlabel("Sender cell type")
        plt.title("Sender Effects on " + receiver)
        plt.tight_layout()

        save_return_show_fig_utils(
            save_show_or_return=save_show_or_return,
            show_legend=True,
            background="white",
            prefix="sender_effects_on_{}".format(receiver),
            save_kwargs=save_kwargs,
            total_panels=1,
            fig=fig,
            axes=ax,
            return_all=False,
            return_all_list=None,
        )

    def receiver_effect(
        self,
        sender: str,
        plot_mode: str = "fold_change",
        gene_subset: Union[None, List[str]] = None,
        significance_threshold: float = 0.05,
        cut_pvals: float = -5,
        fontsize: Union[None, int] = None,
        figsize: Union[None, Tuple[float, float]] = None,
        cmap: str = "seismic",
        save_show_or_return: Literal["save", "show", "return", "both", "all"] = "show",
        save_kwargs: Optional[dict] = {},
    ):
        """
        Evaluates and visualizes the effect that one specific sender cell type has on select genes in all possible
        receiver cell types.

        Args:
            sender: Sender cell type label
            plot_mode: specifies what gets plotted.
                Options:
                    - "qvals": elements of the plot represent statistical significance of the interaction
                    - "fold_change": elements of the plot represent fold change induced in the receiver by the sender
            gene_subset: Names of genes to subset for plot. If None, will use all genes that were used in the
                regression.
            significance_threshold: Set non-significant fold changes to zero, where the threshold is given here
            cut_pvals: Minimum allowable log10(pval)- anything below this will be clipped to this value
            fontsize: Size of figure title and axis labels
            figsize: Width and height of plotting window
            cmap: Name of matplotlib colormap specifying colormap to use
            save_show_or_return: Whether to save, show or return the figure.
                If "both", it will save and plot the figure at the same time. If "all", the figure will be saved,
                displayed and the associated axis and other object will be return.
            save_kwargs: A dictionary that will passed to the save_fig function.
                By default it is an empty dictionary and the save_fig function will use the
                {"path": None, "prefix": 'scatter', "dpi": None, "ext": 'pdf', "transparent": True, "close": True,
                "verbose": True} as its parameters. Otherwise you can provide a dictionary that properly modifies those
                keys according to your needs.
        """
        logger = lm.get_main_logger()
        config_spateo_rcParams()

        if fontsize is None:
            self.fontsize = rcParams.get("font.size")
        else:
            self.fontsize = fontsize
        if figsize is None:
            self.figsize = rcParams.get("figure.figsize")
        else:
            self.figsize = figsize

        sender_idx = self.celltype_names.index(sender)

        if plot_mode == "qvals":
            arr = np.log(self.qvalues[:, sender_idx, :].copy())
            arr[arr < cut_pvals] = cut_pvals
            df = pd.DataFrame(arr, index=self.celltype_names, columns=self.genes)
            if gene_subset:
                df = df.drop(index=sender)[gene_subset]

            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
            sns.heatmap(
                df.T,
                square=True,
                linecolor="grey",
                linewidths=0.3,
                cbar_kws={"label": "$\log_{10}$ FDR-corrected pvalues"},
                cmap=cmap,
                vmin=-5,
                vmax=0.0,
            )

        elif plot_mode == "fold_change":
            arr = self.fold_change[:, sender_idx, :].copy()
            arr[np.where(self.qvalues[:, sender_idx, :] > significance_threshold)] = 0
            df = pd.DataFrame(arr, index=self.celltype_names, columns=self.genes)
            if gene_subset:
                df = df.drop(index=sender)[gene_subset]
            vmax = np.max(np.abs(df.values))

            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
            sns.heatmap(
                df.T,
                square=True,
                linecolor="grey",
                linewidths=0.3,
                cbar_kws={"label": "$\ln$ fold change", "location": "top"},
                cmap=cmap,
                vmin=-vmax,
                vmax=vmax,
            )
        else:
            logger.error("Invalid input to 'plot_mode'. Options: 'qvals', 'fold_change'.")

        plt.xlabel("Receiving cell type")
        plt.title("{} effects on receivers".format(sender))
        plt.tight_layout()

        save_return_show_fig_utils(
            save_show_or_return=save_show_or_return,
            show_legend=True,
            background="white",
            prefix="{}_effects_on_receiver".format(sender),
            save_kwargs=save_kwargs,
            total_panels=1,
            fig=fig,
            axes=ax,
            return_all=False,
            return_all_list=None,
        )

    def sender_receiver_effect_volcanoplot(
        self,
        receiver: str,
        sender: str,
        significance_threshold: float = 0.05,
        fold_change_threshold: Union[None, float] = None,
        fontsize: Union[None, int] = None,
        figsize: Union[None, Tuple[float, float]] = (4.5, 7.0),
        save_show_or_return: Literal["save", "show", "return", "both", "all"] = "show",
        save_kwargs: Optional[dict] = {},
    ):
        """
        Volcano plot to identify differentially expressed genes of a given receiver cell type in the presence of a
        given sender cell type.

        Args:
            receiver: Receiver cell type label
            sender: Sender cell type label
            significance_threshold:  Set non-significant fold changes (given by q-values) to zero, where the
                threshold is given here
            fold_change_threshold: Set absolute value fold-change threshold beyond which observations are marked as
                interesting. If not given, will take the 95th percentile fold-change as
            fontsize: Size of figure title and axis labels
            figsize: Width and height of plotting window
            save_show_or_return: Whether to save, show or return the figure.
                If "both", it will save and plot the figure at the same time. If "all", the figure will be saved,
                displayed and the associated axis and other object will be return.
            save_kwargs: A dictionary that will passed to the save_fig function.
                By default it is an empty dictionary and the save_fig function will use the
                {"path": None, "prefix": 'scatter', "dpi": None, "ext": 'pdf', "transparent": True, "close": True,
                "verbose": True} as its parameters. Otherwise you can provide a dictionary that properly modifies those
                keys according to your needs.
        """
        logger = lm.get_main_logger()
        config_spateo_rcParams()

        if fontsize is None:
            self.fontsize = rcParams.get("font.size")
        else:
            self.fontsize = fontsize
        if figsize is None:
            self.figsize = rcParams.get("figure.figsize")
        else:
            self.figsize = figsize

        receiver_idx = self.celltype_names.index(receiver)
        sender_idx = self.celltype_names.index(sender)

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.grid(False)

        # All non-significant features:
        qval_filter = np.where(self.qvalues[receiver_idx, sender_idx, :] >= significance_threshold)
        vmax = np.max(np.abs(self.fold_change[receiver_idx, sender_idx, :]))

        sns.scatterplot(
            x=self.fold_change[receiver_idx, sender_idx, :][qval_filter],
            y=-np.log10(self.qvalues[receiver_idx, sender_idx, :])[qval_filter],
            color="white",
            edgecolor="black",
            s=100,
            ax=ax,
        )

        # Identify subset that may be significant, but which doesn't pass the fold-change threshold:
        qval_filter = np.where(self.qvalues[receiver_idx, sender_idx, :] < significance_threshold)
        x = self.fold_change[receiver_idx, sender_idx, :][qval_filter]
        fc_filter = np.where(x < fold_change_threshold)
        y = -np.nan_to_num(np.log10(self.qvalues[receiver_idx, sender_idx, :])[qval_filter])
        sns.scatterplot(x=x[fc_filter], y=y[fc_filter], color="darkgrey", edgecolor="black", s=100, ax=ax)

        # Identify subset that are significantly downregulated:
        dreg_color = matplotlib.cm.get_cmap("winter")(0)
        x = self.fold_change[receiver_idx, sender_idx, :][qval_filter]
        fc_filter = np.where(x <= -fold_change_threshold)
        y = -np.nan_to_num(np.log10(self.qvalues[receiver_idx, sender_idx, :])[qval_filter], neginf=-14.5)
        sns.scatterplot(x=x[fc_filter], y=y[fc_filter], color=dreg_color, edgecolor="black", s=100, ax=ax)

        # Identify subset that are significantly upregulated:
        ureg_color = matplotlib.cm.get_map("autumn")(0)
        x = self.fold_change[receiver_idx, sender_idx, :][qval_filter]
        fc_filter = np.where(x >= fold_change_threshold)
        y = -np.nan_to_num(np.log10(self.qvalues[receiver_idx, sender_idx, :])[qval_filter], neginf=-14.5)
        sns.scatterplot(x=x[fc_filter], y=y[fc_filter], color=ureg_color, edgecolor="black", s=100, ax=ax)

        # Plot configuration:
        ax.set_xlim((-vmax * 1.1, vmax * 1.1))
        ax.set_xlabel("$\ln$ fold change")
        ax.set_ylabel("$-\log_{10}$ FDR-corrected pvalues")
        plt.axvline(-fold_change_threshold, color="darkgrey", linestyle="--")
        plt.axvline(fold_change_threshold, color="darkgrey", linestyle="--")
        plt.axhline(-np.log10(significance_threshold), linestyle="--", color="darkgrey")

        plt.tight_layout()
        save_return_show_fig_utils(
            save_show_or_return=save_show_or_return,
            show_legend=True,
            background="white",
            prefix="effect_of_{}_on_{}".format(sender, receiver),
            save_kwargs=save_kwargs,
            total_panels=1,
            fig=fig,
            axes=ax,
            return_all=False,
            return_all_list=None,
        )


class Category_Interpreter(BaseInterpreter):
    """
    Wraps all necessary methods for data loading and preparation, model initialization, parameterization, evaluation and
    prediction when instantiating a model for spatially-aware (but not spatially lagged) regression using categorical
    variables (specifically, the prevalence of categories within spatial neighborhoods) to predict the value of gene
    expression.

    Arguments passed to :class `BaseInterpreter`. The only keyword argument that is used for this class is
    'n_neighbors'.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.group_key is not None, "Categorical labels required for this model."

        # Prepare data:
        self.prepare_data(mod_type="category")


class Niche_Interpreter(BaseInterpreter):
    """
    Wraps all necessary methods for data loading and preparation, model initialization, parameterization, evaluation and
    prediction when instantiating a model for spatially-aware regression using both the prevalence of and connections
    between categories within spatial neighborhoods to predict the value of gene expression.

    Arguments passed to :class `BaseInterpreter`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.group_key is not None, "Categorical labels required for this model."

        self.prepare_data(mod_type="niche")


class Ligand_Lagged_Interpreter(BaseInterpreter):
    """
    Wraps all necessary methods for data loading and preparation, model initialization, parameterization, evaluation and
    prediction when instantiating a model for spatially-lagged regression using the spatial lag of ligand genes to
    predict the regression target.

    Arguments passed to :class `BaseInterpreter`.

    Args:
        lig: Name(s) of ligands to use as predictors
        rec: Name(s) of receptors to use as regression targets. If not given, will search through database for all
            genes that correspond to the provided genes from 'ligands'.
        rec_ds: Name(s) of receptor-downstream genes to use as regression targets. If not given, will search through
            database for all genes that correspond to receptor-downstream genes.
        species: Specifies L:R database to use
        args: Additional positional arguments to :class `BaseInterpreter`
        kwargs: Additional keyword arguments to :class `BaseInterpreter`
    """

    def __init__(
        self,
        lig: Union[None, str, List[str]],
        rec: Union[None, str, List[str]] = None,
        rec_ds: Union[None, str, List[str]] = None,
        species: Literal["human", "mouse", "axolotl"] = "human",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.prepare_data(mod_type="ligand_lag", lig=lig, rec=rec, rec_ds=rec_ds, species=species)

    def run_GM_lag(self):
        """Runs spatially lagged two-stage least squares model"""
        if not hasattr(self, "w"):
            self.logger.info(
                "Called 'run_GM_lag' before computing spatial weights array- computing spatial weights "
                "array before proceeding..."
            )
            self.compute_spatial_weights()

        # Regress on one gene at a time:
        all_values, all_pred, all_resid = [], [], []
        for i in tqdm(range(len(self.genes))):
            cur_g = self.genes[i]
            values, pred, resid = self.single(
                cur_g, self.X, self.variable_names, self.param_labels, self.adata, self.w, self.layer
            )
            all_values.append(values)
            all_pred.append(pred)
            all_resid.append(resid)

        # Coefficients and their significance:
        coeffs = pd.DataFrame(np.vstack(all_values))
        coeffs.columns = self.adata.var.loc[self.genes, :].columns

        pred = pd.DataFrame(np.hstack(all_pred), index=self.adata.obs_names, columns=self.genes)
        resid = pd.DataFrame(np.hstack(all_resid), index=self.adata.obs_names, columns=self.genes)

        # Update AnnData object:
        self.adata.obsm["ypred"] = pred
        self.adata.obsm["resid"] = resid

        for cn in coeffs.columns:
            self.adata.var.loc[:, cn] = coeffs[cn]

        return coeffs, pred, resid

    def single(
        self,
        cur_g: str,
        X: pd.DataFrame,
        X_variable_names: List[str],
        param_labels: List[str],
        adata: AnnData,
        w: np.ndarray,
        layer: Union[None, str] = None,
    ):
        """
        Defines model run process for a single feature- not callable by the user, all arguments populated by
        arguments passed on instantiation of :class `BaseInterpreter`.

        Args:
            cur_g: Name of the feature to regress on
            X: Values used for the regression
            X_variable_names: Names of the variables used for the regression
            param_labels: Names of categories- each computed parameter corresponds to a single element in
                param_labels
            adata: AnnData object to store results in
            w: Spatial weights array
            layer: Specifies layer in AnnData to use- if None, will use .X.

        Returns:
            coeffs: Coefficients for each categorical group for each feature
            pred: Predicted values from regression for each feature
            resid: Residual values from regression for each feature
        """
        if layer is None:
            X["log_expr"] = adata[:, cur_g].X.A
        else:
            X["log_expr"] = adata[:, cur_g].layers[layer].A

        try:
            model = spreg.GM_Lag(
                X[["log_expr"]].values,
                X[X_variable_names].values,
                w=w,
                name_y="log_expr",
                name_x=X_variable_names,
            )
            self.logger.info(f"Printing model summary for regression on {cur_g}: \n")
            print(model.summary)
            y_pred = model.predy
            resid = model.u

            # Coefficients for each cell type:
            a = pd.DataFrame(model.betas, model.name_x + ["W_log_exp"], columns=["coef"])
            b = pd.DataFrame(
                model.z_stat,
                model.name_x + ["W_log_exp"],
                columns=["z_stat", "p_val"],
            )

            df = a.merge(b, left_index=True, right_index=True)

            for ind, g in enumerate(["const"] + param_labels + ["W_log_exp"]):
                adata.var.loc[cur_g, str(g) + "_GM_lag_coeff"] = df.iloc[ind, 0]
                adata.var.loc[cur_g, str(g) + "_GM_lag_zstat"] = df.iloc[ind, 1]
                adata.var.loc[cur_g, str(g) + "_GM_lag_pval"] = df.iloc[ind, 2]

        except:
            y_pred = np.full((X.shape[0],), np.nan)
            resid = np.full((X.shape[0],), np.nan)

            for ind, g in enumerate(["const"] + param_labels + ["W_log_exp"]):
                adata.var.loc[cur_g, str(g) + "_GM_lag_coeff"] = np.nan
                adata.var.loc[cur_g, str(g) + "_GM_lag_zstat"] = np.nan
                adata.var.loc[cur_g, str(g) + "_GM_lag_pval"] = np.nan

        # Outputs for a single gene:
        return adata.var.loc[cur_g, :].values, y_pred.reshape(-1, 1), resid.reshape(-1, 1)


class Niche_Ligand_Lagged_Interpreter(BaseInterpreter):
    """
    Wraps all necessary methods for data loading and preparation, model initialization, parameterization, evaluation and
    prediction when instantiating a model for spatially-lagged regression using the spatial lag of ligand genes as
    well as the (non-lagged) prevalence of and connections between categories within spatial neighborhoods to
    predict the regression target.

    Arguments passed to :class `BaseInterpreter`.

    Args:
        lig: Name(s) of ligands to use as predictors
        rec: Name(s) of receptors to use as regression targets. If not given, will search through database for all
            genes that correspond to the provided genes from 'ligands'.
        rec_ds: Name(s) of receptor-downstream genes to use as regression targets. If not given, will search through
            database for all genes that correspond to receptor-downstream genes.
        species: Specifies L:R database to use
        niche_lr_r_lag: Only used if 'mod_type' is "niche_lr". Uses the spatial lag of the receptor as the
            dependent variable rather than each spot's unique receptor expression. Defaults to True.
        args: Additional positional arguments to :class `BaseInterpreter`
        kwargs: Additional keyword arguments to :class `BaseInterpreter`
    """

    def __init__(
        self,
        lig: Union[None, str, List[str]],
        rec: Union[None, str, List[str]] = None,
        rec_ds: Union[None, str, List[str]] = None,
        species: Literal["human", "mouse", "axolotl"] = "human",
        niche_lr_r_lag: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.prepare_data(
            mod_type="niche_ligand_lag", lig=lig, rec=rec, rec_ds=rec_ds, species=species, niche_lr_r_lag=niche_lr_r_lag
        )

    # Custom version of the GM lag model defined in the base class:
    def run_GM_lag(self):
        """Runs spatially lagged two-stage least squares model for the ligand-lag case"""
        self.logger.info("Running niche ligand lag model...")

        if not hasattr(self, "w"):
            self.logger.info(
                "Called 'run_GM_lag' before computing spatial weights array- computing spatial weights array before "
                "proceeding..."
            )
            self.compute_spatial_weights()

        # Regress on one gene at a time:
        all_values, all_pred, all_resid = [], [], []
        for i in tqdm(range(len(self.genes))):
            cur_g = self.genes[i]
            values, pred, resid = self.single(
                cur_g, self.X, self.variable_names, self.param_labels, self.adata, self.w, self.layer
            )
            all_values.append(values)
            all_pred.append(pred)
            all_resid.append(resid)

        # Coefficients and their significance:
        coeffs = pd.DataFrame(np.vstack(all_values))
        coeffs.columns = self.adata.var.loc[self.genes, :].columns

        pred = pd.DataFrame(np.hstack(all_pred), index=self.adata.obs_names, columns=self.genes)
        resid = pd.DataFrame(np.hstack(all_resid), index=self.adata.obs_names, columns=self.genes)

        # Update AnnData object:
        self.adata.obsm["ypred"] = pred
        self.adata.obsm["resid"] = resid

        for cn in coeffs.columns:
            self.adata.var.loc[:, cn] = coeffs[cn]

        return coeffs, pred, resid

    def single(
        self,
        cur_g: str,
        X: pd.DataFrame,
        X_lag_vars: List[str],
        param_labels: List[str],
        adata: AnnData,
        w: np.ndarray,
        layer: Union[None, str] = None,
    ):
        """
        Defines model run process for a single feature- not callable by the user, all arguments populated by
        arguments passed on instantiation of :class `BaseInterpreter`.

        Args:
            cur_g: Name of the feature to regress on
            X: Values used for the regression, in the form of a dataframe
            X_lag_vars: Names of the variables for which to compute spatial lag
            param_labels: Names of categories- each computed parameter corresponds to a single element in
                param_labels
            adata: AnnData object to store results in
            w: Spatial weights array
            layer: Specifies layer in AnnData to use- if None, will use .X.

        Returns:
            coeffs: Coefficients for each categorical group for each feature
            pred: Predicted values from regression for each feature
            resid: Residual values from regression for each feature
        """
        if layer is None:
            X["log_expr"] = adata[:, cur_g].X.A
        else:
            X["log_expr"] = adata[:, cur_g].layers[layer].A

        try:
            model = LR_GM_lag(
                df=X,
                y_col="log_expr",
                sp_lag_feats=X_lag_vars,
                w=w,
                name_x=list(X.columns),
            )
            self.logger.info(f"Printing model summary for regression on {cur_g}: \n")
            print(model.summary)
            y_pred = model.predy
            resid = model.u

            # Coefficients for each cell type:
            a = pd.DataFrame(model.betas, model.name_x + ["W_log_exp"], columns=["coef"])
            b = pd.DataFrame(
                model.z_stat,
                model.name_x + ["W_log_exp"],
                columns=["z_stat", "p_val"],
            )

            df = a.merge(b, left_index=True, right_index=True)

            for ind, g in enumerate(["const"] + param_labels + ["W_log_exp"]):
                # Parameter may or may not be lagged:
                if g in X_lag_vars:
                    suffix = "_GM_lag"
                else:
                    suffix = "_GM"
                adata.var.loc[cur_g, str(g) + f"{suffix}_coeff"] = df.iloc[ind, 0]
                adata.var.loc[cur_g, str(g) + f"{suffix}_zstat"] = df.iloc[ind, 1]
                adata.var.loc[cur_g, str(g) + f"{suffix}_pval"] = df.iloc[ind, 2]

        except:
            y_pred = np.full((X.shape[0],), np.nan)
            resid = np.full((X.shape[0],), np.nan)

            for ind, g in enumerate(["const"] + param_labels + ["W_log_exp"]):
                # Parameter may or may not be lagged:
                if g in X_lag_vars:
                    suffix = "_GM_lag"
                else:
                    suffix = "_GM"
                adata.var.loc[cur_g, str(g) + f"{suffix}_coeff"] = np.nan
                adata.var.loc[cur_g, str(g) + f"{suffix}_zstat"] = np.nan
                adata.var.loc[cur_g, str(g) + f"{suffix}_pval"] = np.nan

        # Outputs for a single gene:
        return adata.var.loc[cur_g, :].values, y_pred, resid


class Niche_LR_Interpreter(BaseInterpreter):
    """
    Wraps all necessary methods for data loading and preparation, model initialization, parameterization, evaluation and
    prediction when instantiating a model for spatially-aware regression using the prevalence of and connections
    between categories within spatial neighborhoods and the cell type-specific expression of ligands and receptors to
    predict the regression target.

    Arguments passed to :class `BaseInterpreter`.

    Args:
        lig: Name(s) of ligands to use as predictors
        rec: Name(s) of receptors to use as regression targets. If not given, will search through database for all
            genes that correspond to the provided genes from 'ligands'
        rec_ds: Name(s) of receptor-downstream genes to use as regression targets. If not given, will search through
            database for all genes that correspond to receptors
        species: Specifies L:R database to use
        niche_lr_r_lag: Only used if 'mod_type' is "niche_lr". Uses the spatial lag of the receptor as the
            dependent variable rather than each spot's unique receptor expression. Defaults to True.
        args: Additional positional arguments to :class `BaseInterpreter`
        kwargs: Additional keyword arguments to :class `BaseInterpreter`
    """

    def __init__(
        self,
        lig: Union[None, str, List[str]],
        rec: Union[None, str, List[str]] = None,
        rec_ds: Union[None, str, List[str]] = None,
        species: Literal["human", "mouse", "axolotl"] = "human",
        niche_lr_r_lag: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.prepare_data(
            mod_type="niche_lr", lig=lig, rec=rec, rec_ds=rec_ds, species=species, niche_lr_r_lag=niche_lr_r_lag
        )


# ---------------------------------------------------------------------------------------------------
# Regression Metrics
# ---------------------------------------------------------------------------------------------------
def mae(y_true, y_pred):
    """
    Mean absolute error- in this context, actually log1p mean absolute error

    Args:
        y_true: Regression model output
        y_pred: Observed values for the dependent variable

    Returns:
        mae: Mean absolute error value across all samples
    """
    abs = np.abs(y_true - y_pred)
    mean = np.mean(abs)
    return mean


def mse(y_true, y_pred):
    """
    Mean squared error- in this context, actually log1p mean squared error

    Args:
        y_true: Regression model output
        y_pred: Observed values for the dependent variable

    Returns:
        mse: Mean squared error value across all samples
    """
    se = np.square(y_true - y_pred)
    se = np.mean(se, axis=-1)
    return se


# NOTE: NLL from here: https://github.com/tensorchiefs/dl_book/blob/master/chapter_06/nb_ch06_02.ipynb
def nll(y_true, y_pred):
    """
    Negative log likelihood

    Args:
        y_true: Regression model output
        y_pred: Observed values for the dependent variable

    Returns:
        neg_ll: Negative log likelihood across all samples
    """
    n = len(y_true)
    sigma_hat_2 = (n - 1.0) / (n - 2.0) * np.var(y_true - y_pred.flatten(), ddof=1)
    nll = 0.5 * np.log(2 * np.pi * sigma_hat_2) + 0.5 * np.mean((y_true - y_pred.flatten()) ** 2) / sigma_hat_2
    return nll


def r_squared(y_true, y_pred):
    """
    Compute custom r squared- in this context, actually log1p R^2

    Args:
        y_true: Regression model output
        y_pred: Observed values for the dependent variable

    Returns:
        r2: Coefficient of determination
    """
    resid = np.sum(np.square(y_true - y_pred))
    total = np.sum(np.square(y_true - np.sum(y_true)))
    r2 = 1.0 - resid / total
    return r2


# ---------------------------------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------------------------------

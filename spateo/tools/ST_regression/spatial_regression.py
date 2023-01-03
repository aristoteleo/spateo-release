"""
Suite of tools for spatially-aware as well as spatially-lagged linear regression

Also performs downstream characterization following spatially-informed regression to characterize niche impact on gene
expression

Note to self: current set up --> each of the spatial regression classes can be called either through cell_interaction (
e.g. st.cell_interaction.NicheModel) or standalone (e.g. st.NicheModel)- the same is true for all
functions besides the general regression ones (e.g. fit_glm, which must be called w/ st.fit_glm).
"""
import os
import time
from itertools import product
from random import sample
from typing import List, Optional, Tuple, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from anndata import AnnData
from joblib import Parallel, delayed
from matplotlib import rcParams
from patsy import dmatrix
from pysal.lib import weights
from pysal.model import spreg
from scipy.sparse import diags, issparse
from tqdm import tqdm

from ...configuration import config_spateo_rcParams, shiftedColorMap
from ...logging import logger_manager as lm
from ...plotting.static.utils import save_return_show_fig_utils
from ...preprocessing.normalize import normalize_total
from ...preprocessing.transform import log1p
from ...tools.find_neighbors import construct_nn_graph, transcriptomic_connectivity
from ...tools.utils import update_dict
from .generalized_lm import fit_glm
from .regression_utils import compute_wald_test, get_fisher_inverse


# ---------------------------------------------------------------------------------------------------
# Wrapper classes for model running
# ---------------------------------------------------------------------------------------------------
class Base_Model:
    """Basis class for all spatially-aware and spatially-lagged regression models that can be implemented through this
    toolkit. Includes necessary methods for data loading and preparation, computation of spatial weights matrices,
    computation of evaluation metrics and more.

    Args:
        adata: object of class `anndata.AnnData`
        group_key: Key in .obs where group (e.g. cell type) information can be found
        spatial_key: Key in .obsm where x- and y-coordinates are stored
        distr: Can optionally provide distribution family to specify the type of model that should be fit at the time
            of initializing this class rather than after calling :func `GLMCV_fit_predict`- can be "gaussian",
            "poisson", "softplus", "neg-binomial", or "gamma". Case sensitive.
        genes: Subset to genes of interest: will be used as dependent variables in non-ligand-based regression analyses,
            will be independent variables in ligand-based regression analyses
        drop_dummy: Name of the category to be dropped (the "dummy variable") in the regression. The dummy category
            can aid in interpretation as all model coefficients can be taken to be in reference to the dummy
            category. If None, will randomly select a few samples to constitute the dummy group.
        layer: Entry in .layers to use instead of .X when fitting model- all other operations will use .X.
        cci_dir: Full path to the directory containing cell-cell communication databases. Only used in the case of
            models that use ligands for prediction.
        normalize: Perform library size normalization, to set total counts in each cell to the same number (adjust
            for cell size)
        smooth: To correct for dropout effects, leverage gene expression neighborhoods to smooth expression
        log_transform: Set True if log-transformation should be applied to expression (otherwise, will assume
            preprocessing/log-transform was computed beforehand)
        niche_compute_indicator: Only used if 'mod_type' is "niche" or "niche_lr". If True, for the "niche" model, for
            the connections array encoding the cell type-cell type interactions that occur within each niche,
            threshold all nonzero values to 1, to reflect the presence of a pairwise cell type interaction.
            Otherwise, will fit on the normalized number of pairwise interactions within each niche. For the
            "niche_lr" model, for the cell type pair interactions array, threshold all nonzero values to 1 to reflect
            the presence of an interaction between the two cell types within each niche. Otherwise, will fit on
            normalized data.
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
        distr: Union[None, Literal["gaussian", "poisson", "softplus", "neg-binomial", "gamma"]] = None,
        group_key: Union[None, str] = None,
        genes: Union[None, List] = None,
        drop_dummy: Union[None, str] = None,
        layer: Union[None, str] = None,
        cci_dir: Union[None, str] = None,
        normalize: bool = True,
        smooth: bool = False,
        log_transform: bool = False,
        niche_compute_indicator: bool = True,
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
        self.distr = distr
        self.group_key = group_key
        self.genes = genes
        self.logger.info(
            "Note: argument provided to 'genes' represents the dependent variables for non-ligand-based "
            "analysis, but are used as independent variables for ligand-based analysis."
        )
        self.drop_dummy = drop_dummy
        self.layer = layer
        self.cci_dir = cci_dir
        self.normalize = normalize
        self.smooth = smooth
        self.log_transform = log_transform
        self.niche_compute_indicator = niche_compute_indicator
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

    def preprocess_data(
        self,
        normalize: Union[None, bool] = None,
        smooth: Union[None, bool] = None,
        log_transform: Union[None, bool] = None,
    ):
        """Normalization and transformation of input data. Can manually specify whether to normalize, scale,
        etc. data- any arguments not given this way will default to values passed on instantiation of the Interpreter
        object.

        Returns:
            None, all preprocessing operates inplace on the object's input AnnData.
        """
        if normalize is None:
            normalize = self.normalize
        if smooth is None:
            smooth = self.smooth
        if log_transform is None:
            log_transform = self.log_transform

        if self.distr in ["poisson", "softplus", "neg-binomial"]:
            if normalize or smooth or log_transform:
                self.logger.info(
                    f"With a {self.distr} assumption, it is recommended to fit to raw counts. Computing normalizations "
                    f"and transforms if applicable, but storing the results for later and fitting to the raw counts."
                )
                self.adata.layers["raw"] = self.adata.X

        # Normalize to size factor:
        if normalize:
            if self.distr not in ["poisson", "softplus", "neg-binomial"]:
                self.logger.info("Setting total counts in each cell to 1e4 inplace.")
                normalize_total(self.adata)
            else:
                self.logger.info("Setting total counts in each cell to 1e4, storing in adata.layers['X_norm'].")
                dat = normalize_total(self.adata, inplace=False)
                self.adata.layers["X_norm"] = dat["X"]
                self.adata.obs["norm_factor"] = dat["norm_factor"]
                self.adata.layers["stored_processed"] = dat["X"]

        # Smooth data if 'smooth' is True and log-transform data matrix if 'log_transform' is True:
        if smooth:
            if self.distr not in ["poisson", "softplus", "neg-binomial"]:
                self.logger.info("Smoothing gene expression inplace...")
                # Compute connectivity matrix if not already existing:
                try:
                    conn = self.adata.obsp["expression_connectivities"]
                except:
                    _, adata = transcriptomic_connectivity(self.adata, n_neighbors_method="ball_tree")
                    conn = adata.obsp["expression_connectivities"]
                adata_smooth_norm, _ = calc_1nd_moment(self.adata.X, conn, normalize_W=True)
                self.adata.layers["M_s"] = adata_smooth_norm

                # Use smoothed layer for downstream processing:
                self.adata.layers["raw"] = self.adata.X
                self.adata.X = self.adata.layers["M_s"]

            else:
                self.logger.info(
                    "Smoothing gene expression inplace and storing in in adata.layers['M_s'] or "
                    "adata.layers['normed_M_s'] if normalization was first performed."
                )
                adata_temp = self.adata.copy()
                # Check if normalized expression is present- if 'distr' is one of the indicated distributions AND
                # 'normalize' is True, AnnData will not have been updated in place, with the normalized array
                # instead being stored in the object.
                try:
                    adata_temp.X = adata_temp.layers["X_norm"]
                    norm = True
                except:
                    norm = False
                    pass

                try:
                    conn = self.adata.obsp["expression_connectivities"]
                except:
                    _, adata = transcriptomic_connectivity(adata_temp, n_neighbors_method="ball_tree")
                    conn = adata.obsp["expression_connectivities"]
                adata_smooth_norm, _ = calc_1nd_moment(adata_temp.X, conn, normalize_W=True)
                if norm:
                    self.adata.layers["norm_M_s"] = adata_smooth_norm
                else:
                    self.adata.layers["M_s"] = adata_smooth_norm
                self.adata.layers["stored_processed"] = adata_smooth_norm

        if log_transform:
            if self.distr not in ["poisson", "softplus", "neg-binomial"]:
                self.logger.info("Log-transforming expression inplace...")
                log1p(self.adata)
            else:
                self.logger.info(
                    "Log-transforming expression and storing in adata.layers['X_log1p'], "
                    "adata.layers['X_norm_log1p'], adata.layers['X_M_s_log1p'], or adata.layers["
                    "'X_norm_M_s_log1p'], depending on the normalizations and transforms that were "
                    "specified."
                )
                adata_temp = self.adata.copy()
                # Check if normalized expression is present- if 'distr' is one of the indicated distributions AND
                # 'normalize' and/or 'smooth' is True, AnnData will not have been updated in place,
                # with the normalized array instead being stored in the object.
                if "norm_M_s" in adata_temp.layers.keys():
                    layer = "norm_M_s"
                    adata_temp.X = adata_temp.layers["norm_M_s"]
                    norm, smoothed = True, True
                elif "M_s" in adata_temp.layers.keys():
                    layer = "M_s"
                    adata_temp.X = adata_temp.layers["M_s"]
                    norm, smoothed = False, True
                elif "X_norm" in adata_temp.layers.keys():
                    layer = "X_norm"
                    adata_temp.X = adata_temp.layers["X_norm"]
                    norm, smoothed = True, False
                else:
                    layer = None
                    norm, smoothed = False, False

                if layer is not None:
                    log1p(adata_temp.layers[layer])
                else:
                    log1p(adata_temp)

                if norm and smoothed:
                    self.adata.layers["X_norm_M_s_log1p"] = adata_temp.X
                elif norm:
                    self.adata.layers["X_norm_log1p"] = adata_temp.X
                elif smoothed:
                    self.adata.layers["X_M_s_log1p"] = adata_temp.X
                else:
                    self.adata.layers["X_log1p"] = adata_temp.X
                self.adata.layers["stored_processed"] = adata_temp.X

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
        """Handles any necessary data preparation, starting from given source AnnData object

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
                "human", "mouse", "axolotl".
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

        self.preprocess_data()

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
                construct_nn_graph(
                    self.adata,
                    spatial_key=self.spatial_key,
                    n_neighbors=self.sp_kwargs["n_neighbors"],
                    exclude_self=True,
                )
                self.logger.info(f"Computed adjacency matrix, time elapsed: {time.time() - start}s.")

                # Create 'neighbors' directory, if necessary:
                if not os.path.exists("./neighbors"):
                    os.makedirs("./neighbors")
                # And save computed adjacency matrix:
                self.logger.info(f"Saving adjacency matrix to path neighbors/{self.data_id}_neighbors.csv")
                adj = pd.DataFrame(self.adata.obsm["adj"], index=self.adata.obs_names, columns=self.adata.obs_names)
                adj.to_csv(os.path.join(os.getcwd(), f"neighbors/{self.data_id}_neighbors.csv"))
        else:
            self.logger.info(f"Path to pre-computed adjacency matrix not given. Computing adjacency matrix.")
            start = time.time()
            construct_nn_graph(
                self.adata,
                spatial_key=self.spatial_key,
                n_neighbors=self.sp_kwargs["n_neighbors"],
                exclude_self=True,
            )
            self.logger.info(f"Computed adjacency matrix, time elapsed: {time.time() - start}s.")

            # Create 'neighbors' directory, if necessary:
            if not os.path.exists("./neighbors"):
                os.makedirs("./neighbors")
            # And save computed adjacency matrix:
            self.logger.info(f"Saving adjacency matrix to path neighbors/{self.data_id}_neighbors.csv")
            adj = pd.DataFrame(self.adata.obsm["adj"], index=self.adata.obs_names, columns=self.adata.obs_names)
            adj.to_csv(os.path.join(os.getcwd(), f"neighbors/{self.data_id}_neighbors.csv"))

        adj = self.adata.obsm["adj"]

        # Construct category adjacency matrix (n_samples x n_categories array that records how many neighbors of
        # each category are present within the neighborhood of each sample):
        dmat_neighbors = (adj > 0).astype("int").dot(X.values)

        # Construct the category interaction matrix (1D array w/ n_categories ** 2 elements, encodes the niche of
        # each sample by documenting the category-category spatial connections within the niche- specifically,
        # for each sample, records the category identity of its neighbors in space):
        data = {"categories": X, "dmat_neighbours": dmat_neighbors}
        connections = np.asarray(dmatrix("categories:dmat_neighbours-1", data))

        # Set connections array to indicator array:
        if self.niche_compute_indicator:
            connections[connections > 1] = 1
        else:
            connections = (connections - connections.min()) / (connections - connections.max())

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

            # Construct category adjacency matrix (n_samples x n_categories array that records how many neighbors of
            # each category are present within the neighborhood of each sample):
            dmat_neighbors = (adj > 0).astype("int").dot(X.values)
            self.X = pd.DataFrame(dmat_neighbors, columns=X.columns, index=self.adata.obs_names)
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

        elif mod_type == "niche" or mod_type == "niche_lag":
            # If mod_type is 'niche' or 'niche_lag', use the connections matrix as independent variables in the
            # regression:
            connections_cols = list(product(X.columns, X.columns))
            connections_cols.sort(key=lambda x: x[1])
            connections_cols = [f"{i[0]}-{i[1]}" for i in connections_cols]
            self.X = pd.DataFrame(connections, columns=connections_cols, index=self.adata.obs_names)

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
                ligands = [g for g in self.genes if g in lig_available]
            else:
                # Filter provided ligands to those that can be found in the database:
                ligands = [l for l in ligands if l in lig_available]
                self.logger.info("Proceeding with analysis using ligands {}".format(",".join(ligands)))

            # Filter ligands to those that can be found in the database:
            ligands = [l for l in ligands if l in self.adata.var_names]
            if len(ligands) == 0:
                self.logger.error(
                    "None of the ligands could be found in AnnData variable names. "
                    "Check that AnnData index names match database entries. "
                    "Also possible to have selected only ligands that can't be found in AnnData- "
                    "select different ligands."
                )
            self.n_ligands = len(ligands)

            ligands_expr = pd.DataFrame(
                self.adata[:, ligands].X.toarray() if scipy.sparse.issparse(self.adata.X) else self.adata[:, ligands].X,
                index=self.adata.obs_names,
                columns=ligands,
            )

            self.X = ligands_expr

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

            # All ligands will have associated parameters and be used as variables in the model
            self.param_labels = self.variable_names = ligands

        elif mod_type == "niche_lr":
            # Load LR database based on input to 'species':
            if species == "human":
                lr_network = pd.read_csv(os.path.join(self.cci_dir, "lr_network_human.csv"), index_col=0)
            elif species == "mouse":
                lr_network = pd.read_csv(os.path.join(self.cci_dir, "lr_network_mouse.csv"), index_col=0)
            elif species == "axolotl":
                lr_network = pd.read_csv(os.path.join(self.cci_dir, "lr_network_axolotl.csv"), index_col=0)
            else:
                self.logger.error("Invalid input given to 'species'. Options: 'human', 'mouse', or 'axolotl'.")

            if lig is None:
                self.logger.error("For 'mod_type' = 'niche_lr', ligands must be provided.")

            # If no receptors are given, search database for matches w/ the ligand:
            if rec is None:
                rec = set(list(lr_network.loc[lr_network["from"].isin(lig)]["to"].values))

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

                if not any(receptor in possible_receptors for receptor in rec):
                    self.logger.error(
                        "No record of {} interaction with any of {}. Ensure provided lists contain "
                        "paired ligand-receptors.".format(ligand, (",".join(rec)))
                    )
                found_receptors = list(set(possible_receptors).intersection(set(rec)))
                lig_pairs = list(product([ligand], found_receptors))
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
                    "Regression model has many predictors- consider measuring fewer ligands and receptors."
                )

            # Each ligand-receptor pair will have an associated niche matrix:
            self.niche_mats = {}

            # Copy of AnnData to avoid modifying in-place:
            expr = self.adata.copy()
            # Look for normalized and/or transformed values if "poisson", "softplus" or "neg-binomial" were given as the
            # distribution to fit to- Niche LR dependent variables draw from the gene expression:
            try:
                expr.X = expr.layers["stored_processed"]
            except:
                pass

            for lr_pair in pairs:
                lig, rec = lr_pair[0], lr_pair[1]
                lig_expr_values = expr[:, lig].X.toarray() if scipy.sparse.issparse(expr.X) else expr[:, lig].X
                rec_expr_values = expr[:, rec].X.toarray() if scipy.sparse.issparse(expr.X) else expr[:, rec].X
                # Optionally, compute the spatial lag of the receptor:
                if niche_lr_r_lag:
                    if not hasattr(self, "w"):
                        self.compute_spatial_weights()
                    rec_lag = spreg.utils.lag_spatial(self.w, rec_expr_values)
                    expr.obs[f"{rec}_lag"] = rec_lag
                # Multiply one-hot category array by the expression of select receptor within that cell:
                if not niche_lr_r_lag:
                    rec_vals = expr[:, rec].X.toarray() if scipy.sparse.issparse(expr.X) else expr[:, rec].X
                else:
                    rec_vals = expr.obs[f"{rec}_lag"].values
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
                # Swap sending & receiving cell types because we're looking at receptor expression in the "source" cell
                # and ligand expression in the surrounding cells.
                lr_connections_cols = [f"{i[1]}-{i[0]}_{lig}-{rec}" for i in lr_connections_cols]
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
            self.X.index = self.adata.obs_names
            # Drop all-zero columns (represent cell type pairs with no spatial coupled L/R expression):
            self.X = self.X.loc[:, (self.X != 0).any(axis=0)]

            if self.niche_compute_indicator:
                self.X[self.X > 0] = 1
            else:
                # Minmax-scale columns to minimize the external impact of intercellular differences in ligand/receptor
                # expression:
                self.X = (self.X - self.X.min()) / (self.X.max() - self.X.min())

            self.param_labels = self.variable_names = self.X.columns

        else:
            self.logger.error("Invalid argument to 'mod_type'.")

        # Save model type as an attribute so it can be accessed by other methods:
        self.mod_type = mod_type

        # If 'genes' is given, can take the minimum necessary portion of AnnData object- otherwise, use all genes:
        if self.genes is not None:
            self.genes = list(self.adata.var.index.intersection(self.genes))
        else:
            self.genes = list(self.adata.var.index)
        # self.adata = self.adata[:, self.genes]

    def compute_spatial_weights(self):
        """Generates matrix of pairwise spatial distances, used in spatially-lagged models"""
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
    # Computing parameters for spatially-aware and lagged models- generalized linear models
    # ---------------------------------------------------------------------------------------------------
    def GLMCV_fit_predict(
        self,
        gs_params: Union[None, dict] = None,
        n_gs_cv: Union[None, int] = None,
        n_jobs: int = 30,
        cat_key: Union[None, str] = None,
        categories: Union[None, str, List[str]] = None,
        **kwargs,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Wrapper for fitting predictive generalized linear regression model.

        Args:
            gs_params: Optional dictionary where keys are variable names for the regressor and
                values are lists of potential values for which to find the best combination using grid search.
                Classifier parameters should be given in the following form: 'classifier__{parameter name}'.
            n_gs_cv: Number of folds for grid search cross-validation, will only be used if gs_params is not None. If
                None, will default to a 5-fold cross-validation.
            n_jobs: For parallel processing, number of tasks to run at once
            cat_key: Optional, name of key in .obs containing categorical (e.g. cell type) information
            categories: Optional, names of categories to subset to for the regression. In cases where the exogenous
                block is exceptionally heterogenous, can be used to narrow down the search space.
            kwargs: Additional named arguments that will be provided to :class `GLMCV`.

        Returns:
            coeffs: Contains fitted parameters for each feature
            reconst: Contains predicted expression for each feature
        """
        X = self.X[self.variable_names]
        kwargs["distr"] = self.distr

        if categories is not None:
            self.categories = categories
            # Flag indicating that resultant parameters matrix is not pairwise (i.e. that there's not one parameter
            # for each cell type combination):
            self.square = False
            if not isinstance(self.categories, list):
                self.categories = [self.categories]

            if cat_key is None:
                self.logger.error(
                    ":func `GLMCV_fit_predict` error: 'Categories' were given, but not 'cat_key' "
                    "specifying where in .obs to look."
                )
            # Filter adata for rows annotated as being any category in 'categories', and X block for columns annotated with
            # any of the categories in 'categories'.
            self.adata = self.adata[self.adata.obs[cat_key].isin(self.categories)]
            self.cell_names = self.adata.obs_names
            X = X.filter(regex="|".join(self.categories))
            X = X.loc[self.adata.obs_names]
        else:
            self.square = True

        # Set preprocessing parameters to False- :func `prepare_data` handles these steps.
        results = Parallel(n_jobs)(
            delayed(fit_glm)(
                X,
                self.adata,
                cur_g,
                calc_first_moment=False,
                log_transform=False,
                gs_params=gs_params,
                n_gs_cv=n_gs_cv,
                return_model=False,
                **kwargs,
            )
            for cur_g in self.genes
        )
        intercepts = [item[0] for item in results]
        coeffs = [item[1] for item in results]
        opt_scores = [item[2] for item in results]
        reconst = [item[3] for item in results]

        coeffs = pd.DataFrame(coeffs, index=self.genes, columns=X.columns)
        for cn in coeffs.columns:
            self.adata.var.loc[:, cn] = coeffs[cn]
        self.adata.uns["pseudo_r2"] = dict(zip(self.genes, opt_scores))
        self.adata.uns["intercepts"] = dict(zip(self.genes, intercepts))
        # Nested list transforms into dataframe rows- instantiate and transpose to get to correct shape:
        reconst = pd.DataFrame(reconst, index=self.genes, columns=self.cell_names).T
        return coeffs, reconst

    # ---------------------------------------------------------------------------------------------------
    # Downstream interpretation
    # ---------------------------------------------------------------------------------------------------
    def visualize_params(
        self,
        coeffs: pd.DataFrame,
        subset_cols: Union[None, str, List[str]] = None,
        cmap: str = "autumn",
        zero_center_cmap: bool = False,
        mask_threshold: Union[None, float] = None,
        mask_zero: bool = True,
        transpose: bool = False,
        title: Union[None, str] = None,
        xlabel: Union[None, str] = None,
        ylabel: Union[None, str] = None,
        figsize: Union[None, Tuple[float, float]] = None,
        annot_kws: dict = {},
        save_show_or_return: Literal["save", "show", "return", "both", "all"] = "save",
        save_kwargs: dict = {},
    ):
        """Generates heatmap of parameter values for visualization

        Args:
            coeffs: Contains coefficients (and any other relevant statistics that were computed) from regression for
                each variable
            subset_cols: String or list of strings that can be used to subset coeffs DataFrame such that only columns
                with names containing the provided key strings are displayed on heatmap. For example, can use "coeff" to
                plot only the linear regression coefficients, "zstat" for the z-statistic, etc. Or can use the full
                name of the column to select specific columns.
            cmap: Name of the colormap to use
            zero_center_cmap: Set True to set colormap intensity midpoint to zero.
            mask_threshold: Optional, sets lower absolute value thresholds for parameters to be assigned color in
                heatmap (will compare absolute value of each element against this threshold)
            mask_zero: Set True to not assign color to zeros (representing neither a positive or negative interaction)
            transpose: Set True to reverse the dataframe's orientation before plotting
            title: Optional, provides title for plot. If not given, will use default "Spatial Parameters".
            xlabel: Optional, provides label for x-axis. If not given, will use default "Predictor Features".
            ylabel: Optional, provides label for y-axis. If not given, will use default "Target Features".
            figsize: Can be used to set width and height of figure window, in inches. If not given, will use Spateo
                default.
            annot_kws: Optional dictionary that can be used to set qualities of the axis/tick labels. For example,
                can set 'size': 9, 'weight': 'bold', etc.
            save_show_or_return: Whether to save, show or return the figure.
                If "both", it will save and plot the figure at the same time. If "all", the figure will be saved,
                displayed and the associated axis and other object will be return.
            save_kwargs: A dictionary that will passed to the save_fig function.
                By default it is an empty dictionary and the save_fig function will use the
                {"path": None, "prefix": 'scatter', "dpi": None, "ext": 'pdf', "transparent": True, "close": True,
                "verbose": True} as its parameters. Otherwise you can provide a dictionary that properly modifies those
                keys according to your needs.
        """
        config_spateo_rcParams()
        if figsize is None:
            self.figsize = rcParams.get("figure.figsize")
        else:
            self.figsize = figsize
        if len(annot_kws) == 0:
            annot_kws = {"size": 6, "weight": "bold"}

        # Reformat column names for better visual:
        coeffs.columns = coeffs.columns.str.replace("group_", "")
        coeffs.columns = coeffs.columns.str.replace("_", ":")

        if subset_cols is not None:
            if isinstance(subset_cols, str):
                subset_cols = [subset_cols]
            col_subset = [col for col in coeffs.columns if any(key in col for key in subset_cols)]
            coeffs = coeffs[col_subset]
        # Remove rows with no nonzero values:
        coeffs = coeffs.loc[(coeffs != 0).any(axis=1)]

        if transpose:
            coeffs = coeffs.T

        if mask_threshold is not None:
            mask = np.abs(coeffs) < mask_threshold
            # Drop columns in which all elements fail to meet mask threshold criteria (then recompute mask w/
            # potentially smaller dataframe):
            coeffs = coeffs.loc[:, (mask == 0).any(axis=0)]
            mask = np.abs(coeffs) < mask_threshold
        elif mask_zero:
            mask = coeffs == 0
            # Drop columns in which all elements fail to meet mask threshold criteria (then recompute mask w/
            # potentially smaller dataframe):
            coeffs = coeffs.loc[:, (mask == 0).any(axis=0)]
            mask = coeffs == 0
        else:
            mask = None

        # If "zero_center_cmap", find percentile corresponding to zero and set colormap midpoint to this value:
        if zero_center_cmap:
            cmap = plt.get_cmap(cmap)
            coeffs_max = np.max(coeffs.values)
            zero_point = 1 - coeffs_max / (coeffs_max + abs(np.min(coeffs.values)))
            print(zero_point)
            cmap = shiftedColorMap(cmap, midpoint=zero_point)

        xtick_labels = list(coeffs.columns)
        ytick_labels = list(coeffs.index)

        fig, ax = plt.subplots(1, 1, figsize=self.figsize)
        res = sns.heatmap(
            coeffs,
            cmap=cmap,
            square=True,
            yticklabels=ytick_labels,
            linecolor="grey",
            linewidths=0.3,
            annot_kws=annot_kws,
            xticklabels=xtick_labels,
            mask=mask,
            ax=ax,
        )
        # Outer frame:
        for _, spine in res.spines.items():
            spine.set_visible(True)
            spine.set_linewidth(0.75)

        plt.title(title if title is not None else "Spatial Parameters")
        if xlabel is not None:
            plt.xlabel(xlabel, size=6)
        if ylabel is not None:
            plt.ylabel(ylabel, size=6)
        ax.set_xticklabels(xtick_labels, rotation=90, ha="center")
        plt.tight_layout()

        save_return_show_fig_utils(
            save_show_or_return=save_show_or_return,
            show_legend=True,
            background="white",
            prefix="parameters",
            save_kwargs=save_kwargs,
            total_panels=1,
            fig=fig,
            axes=ax,
            return_all=False,
            return_all_list=None,
        )

    def compute_coeff_significance(
        self,
        coeffs: pd.DataFrame,
        significance_threshold: float = 0.05,
        only_positive: bool = False,
        only_negative: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Computes statistical significance for fitted coefficients.

        Args:
            coeffs: Contains coefficients from regression for each variable
            significance_threshold: p-value needed to call a sender-receiver relationship significant
            only_positive: Set True to find significance/pvalues/qvalues only for the subset of coefficients that is
                positive (representing possible mechanisms of positive regulation).
            only_negative: Set True to find significance/pvalues/qvalues only for the subset of coefficients that is
                negative (representing possible mechanisms of positive regulation).

        Returns:
            is_significant: Dataframe of identical shape to coeffs, where each element is True or False if it meets the
            threshold for significance
            pvalues: Dataframe of identical shape to coeffs, where each element is a p-value for that instance of that
                feature
            qvalues: Dataframe of identical shape to coeffs, where each element is a q-value for that instance of that
                feature
        """
        # If Poisson or softplus, use log-transformed values for downstream applications (model ultimately uses a
        # linear combination of independent variables to predict log-transformed dependent):
        if self.distr in ["poisson", "softplus"]:
            try:
                log_key = [key for key in self.adata.layers.keys() if "log1p" in key][0]
                self.adata.X = self.adata.layers[log_key]
                self.logger.info(
                    "With Poisson distribution assumed for dependent variable, using log-transformed data "
                    f"to compute sender-receiver effects...log key found in adata under key {log_key}."
                )
            except:
                self.logger.info(
                    "With Poisson distribution assumed for dependent variable, using log-transformed data "
                    "to compute sender-receiver effects...log key not found in adata, manually computing."
                )
                log1p(self.adata)
                self.logger.info("Data log-transformed.")

        coeffs_np = coeffs.values
        # Save labels of indices and columns (correspond to features & parameters, respectively, for the coeffs
        # DataFrame, will be columns & indices respectively for the other arrays generated by this function):
        feature_labels = coeffs.index
        param_labels = coeffs.columns

        # Get inverse Fisher information matrix, with the y block containing all features that were used in regression):
        y = (
            self.adata[:, self.genes].X.toarray()
            if scipy.sparse.issparse(self.adata.X)
            else self.adata[:, self.genes].X
        )
        inverse_fisher = get_fisher_inverse(self.X.values, y)

        # Compute significance for each parameter:
        is_significant, pvalues, qvalues = compute_wald_test(
            params=coeffs_np, fisher_inv=inverse_fisher, significance_threshold=significance_threshold
        )

        is_significant = pd.DataFrame(is_significant, index=param_labels, columns=feature_labels)
        pvalues = pd.DataFrame(pvalues, index=param_labels, columns=feature_labels)
        qvalues = pd.DataFrame(qvalues, index=param_labels, columns=feature_labels)

        # If 'only_positive' or 'only_negative' are set, set all elements corresponding to negative or positive
        # coefficients (respectively) to False and all pvalues/qvalues to 1:
        if only_positive:
            is_significant[coeffs.T <= 0] = False
            pvalues[coeffs.T <= 0] = 1
            qvalues[coeffs.T <= 0] = 1
        elif only_negative:
            is_significant[coeffs.T >= 0] = False
            pvalues[coeffs.T >= 0] = 1
            qvalues[coeffs.T >= 0] = 1

        return is_significant, pvalues, qvalues

    def get_effect_sizes(
        self,
        coeffs: pd.DataFrame,
        only_positive: bool = False,
        only_negative: bool = False,
        significance_threshold: float = 0.05,
        lr_pair: Union[None, str] = None,
        save_prefix: Union[None, str] = None,
    ):
        """For each predictor and each feature, determine if the influence of said predictor in predicting said
        feature is significant.

        Additionally, for each feature and each sender-receiver category pair, determines the effect size that
        the sender induces in the feature for the receiver.

        Only valid if the model specified uses the connections between categories as variables for the regression-
        thus can be applied to 'mod_type' "niche", or "niche_lr".

        Args:
            coeffs: Contains coefficients from regression for each variable
            only_positive: Set True to find significance/pvalues/qvalues only for the subset of coefficients that is
                positive (representing possible mechanisms of positive regulation).
            only_negative: Set True to find significance/pvalues/qvalues only for the subset of coefficients that is
                negative (representing possible mechanisms of positive regulation).
            significance_threshold: p-value needed to call a sender-receiver relationship significant
            lr_pair: Required if (and used only in the case that) coefficients came from a Niche-LR model; used to
                subset the coefficients array to the specific ligand-receptor pair of interest. Takes the form
                "{ligand}-{receptor}" and should match one of the keys in :dict `self.niche_mats`. If not given,
                will default to the first key in this dictionary.
            save_prefix: If provided, saves all relevant dataframes to :path `./regression_outputs` under the name
                `{prefix}_{coeffs/pvalues, etc.}.csv`. If not provided, will not save.
        """
        # If "Poisson" given as the distributional assumption, check for log-transformed data:
        if self.distr == "poisson":
            if not any("log1p" in key for key in self.adata.layers.keys()):
                self.logger.info(
                    "With Poisson distribution assumed for dependent variable, using log-transformed data "
                    "to compute sender-receiver effects...log key not found in adata, manually computing."
                )
                self.preprocess_data(log_transform=True)

        if not "niche" in self.mod_type:
            self.logger.error(
                "Type coupling analysis only valid if connections between categories are used as the "
                "predictor variable."
            )

        coeffs_np = coeffs.values

        is_significant, pvalues, qvalues = self.compute_coeff_significance(
            coeffs,
            only_positive=only_positive,
            only_negative=only_negative,
            significance_threshold=significance_threshold,
        )

        # If 'save_prefix' is given, save the complete coefficients, significance, p-value and q-value matrices:
        if save_prefix is not None:
            if not os.path.exists("./regression_outputs"):
                os.makedirs("./regression_outputs")
            is_significant.to_csv(f"./regression_outputs/{save_prefix}_is_sign.csv")
            pvalues.to_csv(f"./regression_outputs/{save_prefix}_pvalues.csv")
            qvalues.to_csv(f"./regression_outputs/{save_prefix}_qvalues.csv")
            coeffs.to_csv(f"./regression_outputs/{save_prefix}_coeffs.csv")

        # If niche-LR model, extract the portion corresponding to the interaction terms for a specific L-R pair:
        if self.mod_type == "niche_lr":
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

            is_significant = is_significant.filter(lr_pair, axis="index")
            pvalues = pvalues.filter(lr_pair, axis="index")
            qvalues = qvalues.filter(lr_pair, axis="index")

            # Coefficients, etc. will also be a subset of the complete array:
            coeffs = coeffs.filter(lr_pair, axis="columns")
            coeffs_np = coeffs.values
            # Significance, pvalues, qvalues filtered above

            # If the 'square' flag is set- that is, if the original parameters constitute at least one pairwise
            # combination of cell types (which is not true if 'categories' are given to the fit function):
            if self.square:
                self.effect_size = np.concatenate(
                    np.expand_dims(
                        np.split(coeffs_np.T, indices_or_sections=np.sqrt(coeffs_np.T.shape[0]), axis=0), axis=0
                    ),
                    axis=0,
                )
            else:
                self.effect_size = coeffs

        # Else if connection-based model, all regression coefficients already correspond to the interaction terms:
        else:
            if self.square:
                self.effect_size = np.concatenate(
                    np.expand_dims(
                        np.split(coeffs_np.T, indices_or_sections=np.sqrt(coeffs_np.T.shape[0]), axis=0), axis=0
                    ),
                    axis=0,
                )
            else:
                self.effect_size = coeffs

        if self.square:
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
        else:
            self.pvalues = pvalues.T
            self.qvalues = qvalues.T
            self.is_significant = is_significant.T

    def type_coupling(
        self,
        cmap: str = "Reds",
        fontsize: Union[None, int] = None,
        figsize: Union[None, Tuple[float, float]] = None,
        ignore_self: bool = True,
        save_show_or_return: Literal["save", "show", "return", "both", "all"] = "save",
        save_kwargs: dict = {},
    ):
        """Generates heatmap of spatially differentially-expressed features for each pair of sender and receiver
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
            ignore_self: If True, will ignore the effect of cell type in proximity to other cells of the same type-
                will record the number of DEGs only if the two cell types are different.
            save_show_or_return: Whether to save, show or return the figure.
                If "both", it will save and plot the figure at the same time. If "all", the figure will be saved,
                displayed and the associated axis and other object will be return.
            save_kwargs: A dictionary that will passed to the save_fig function.
                By default it is an empty dictionary and the save_fig function will use the
                {"path": None, "prefix": 'scatter', "dpi": None, "ext": 'pdf', "transparent": True, "close": True,
                "verbose": True} as its parameters. Otherwise you can provide a dictionary that properly modifies those
                keys according to your needs.
        """
        if fontsize is None:
            self.fontsize = rcParams.get("font.size")
        else:
            self.fontsize = fontsize
        if figsize is None:
            self.figsize = rcParams.get("figure.figsize")
        else:
            self.figsize = figsize

        if not hasattr(self, "is_significant"):
            self.logger.warning("Significance dataframe does not exist- please run :func `get_effect_sizes` " "first.")

        if not hasattr(self, "square"):
            self.logger.error(
                ":func `type_coupling_analysis` can only be run if the design matrix can be made square- that is, "
                "if all pairwise combinations of cell types are represented."
            )

        sig_df = pd.DataFrame(
            np.sum(self.is_significant, axis=-1), columns=self.celltype_names, index=self.celltype_names
        )
        if ignore_self:
            np.fill_diagonal(sig_df.values, 0)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        res = sns.heatmap(sig_df, square=True, linecolor="grey", linewidths=0.3, cmap=cmap, mask=(sig_df == 0), ax=ax)
        # Outer frame:
        for _, spine in res.spines.items():
            spine.set_visible(True)
            spine.set_linewidth(0.75)
        plt.xlabel("Receiving Cell")
        plt.ylabel("Sending Cell")

        title = (
            "Niche-Associated Differential Expression"
            if self.mod_type == "niche"
            else "Cell Type-Specific Ligand:Receptor-Associated Differential Expression"
        )
        plt.title(title)
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

    def sender_effect_on_all_receivers(
        self,
        sender: str,
        plot_mode: str = "effect_size",
        gene_subset: Union[None, List[str]] = None,
        significance_threshold: float = 0.05,
        cut_pvals: float = -5,
        fontsize: Union[None, int] = None,
        figsize: Union[None, Tuple[float, float]] = None,
        cmap: str = "seismic",
        save_show_or_return: Literal["save", "show", "return", "both", "all"] = "show",
        save_kwargs: Optional[dict] = {},
    ):
        """Evaluates and visualizes the effect that the given sender cell type has on expression/abundance in each
        possible receiver cell type.

        Args:
            sender: sender cell type label
            plot_mode: specifies what gets plotted.
                Options:
                    - "qvals": elements of the plot represent statistical significance of the interaction
                    - "effect_size": elements of the plot represent numerical expression change induced in the
                        sender by the sender
            gene_subset: Names of genes to subset for plot. If None, will use all genes that were used in the
                regression.
            significance_threshold: Set non-significant effect sizes to zero, where the threshold is given here
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

        if self.square:
            sender_idx = self.celltype_names.index(sender)

            if plot_mode == "qvals":
                # In the analysis process, the receiving cell types become aligned along the column axis:
                arr = np.log(self.qvalues[sender_idx, :, :].copy())
                arr[arr < cut_pvals] = cut_pvals
                df = pd.DataFrame(arr, index=self.celltype_names, columns=self.genes)
                if gene_subset:
                    df = df.drop(index=sender)[gene_subset]

                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=self.figsize)
                qv = sns.heatmap(
                    df.T,
                    square=True,
                    linecolor="grey",
                    linewidths=0.3,
                    cbar_kws={"label": "$\log_{10}$ FDR-corrected pvalues", "location": "top"},
                    cmap=cmap,
                    vmin=-5,
                    vmax=0.0,
                    ax=ax,
                )

                # Outer frame:
                for _, spine in qv.spines.items():
                    spine.set_visible(True)
                    spine.set_linewidth(0.75)

            elif plot_mode == "effect_size":
                arr = self.effect_size[sender_idx, :, :].copy()
                arr[np.where(self.qvalues[sender_idx, :, :] > significance_threshold)] = 0
                df = pd.DataFrame(arr, index=self.celltype_names, columns=self.genes)
                if gene_subset:
                    df = df.drop(index=sender)[gene_subset]
                vmax = np.max(np.abs(df.values))

                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=self.figsize)
                es = sns.heatmap(
                    df.T,
                    square=True,
                    linecolor="grey",
                    linewidths=0.3,
                    cbar_kws={"label": "Effect size", "location": "top"},
                    cmap=cmap,
                    vmin=-vmax,
                    vmax=vmax,
                    ax=ax,
                )

                # Outer frame:
                for _, spine in es.spines.items():
                    spine.set_visible(True)
                    spine.set_linewidth(0.75)

            else:
                logger.error("Invalid input to 'plot_mode'. Options: 'qvals', 'effect_size'.")

        else:
            if sender not in self.categories:
                self.logger.error(
                    "Adata was subset to categories of interest and fit on those categories, "
                    "but the group provided to 'sender' is not one of those categories."
                )

            sender_cols = [col for col in self.effect_size.columns if sender in col.split("-")[1]]
            # (note that what should be considered the "receiver" is the first cell type listed)

            if plot_mode == "qvals":
                df = np.log(self.qvalues[sender_cols].copy())
                # Reformat columns for visual purposes:
                receivers = [ct[0] for ct in df.columns.str.split("-")]
                df.columns = [col.split("_")[1] for col in receivers]
                df[df < cut_pvals] = cut_pvals
                if gene_subset:
                    df = df.loc[gene_subset]

                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=self.figsize)
                qv = sns.heatmap(
                    df,
                    square=True,
                    linecolor="grey",
                    linewidths=0.3,
                    cbar_kws={"label": "$\log_{10}$ FDR-corrected pvalues", "location": "top"},
                    cmap=cmap,
                    vmin=-5,
                    vmax=0.0,
                    ax=ax,
                )

                # Outer frame:
                for _, spine in qv.spines.items():
                    spine.set_visible(True)
                    spine.set_linewidth(0.75)

            elif plot_mode == "effect_size":
                df = self.effect_size[sender_cols].copy()
                # Reformat columns for visual purposes:
                receivers = [ct[0] for ct in df.columns.str.split("-")]
                df.columns = [col.split("_")[1] for col in receivers]
                df.values[np.where(self.qvalues[sender_cols] > significance_threshold)] = 0
                if gene_subset:
                    df = df.loc[gene_subset]
                vmax = np.max(np.abs(df.values))

                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=self.figsize)
                es = sns.heatmap(
                    df,
                    square=True,
                    linecolor="grey",
                    linewidths=0.3,
                    cbar_kws={"label": "Effect size", "location": "top"},
                    cmap=cmap,
                    vmin=-vmax,
                    vmax=vmax,
                    ax=ax,
                )

                # Outer frame:
                for _, spine in es.spines.items():
                    spine.set_visible(True)
                    spine.set_linewidth(0.75)

            else:
                logger.error("Invalid input to 'plot_mode'. Options: 'qvals', 'effect_size'.")

        plt.xlabel("Receiver cell type", fontsize=9)
        plt.title("{} effects on receivers".format(sender), fontsize=9)
        plt.tight_layout()

        save_return_show_fig_utils(
            save_show_or_return=save_show_or_return,
            show_legend=True,
            background="white",
            prefix="{}_effects_on_receivers".format(sender),
            save_kwargs=save_kwargs,
            total_panels=1,
            fig=fig,
            axes=ax,
            return_all=False,
            return_all_list=None,
        )

    def all_senders_effect_on_receiver(
        self,
        receiver: str,
        plot_mode: str = "effect_size",
        gene_subset: Union[None, List[str]] = None,
        significance_threshold: float = 0.05,
        cut_pvals: float = -5,
        fontsize: Union[None, int] = None,
        figsize: Union[None, Tuple[float, float]] = None,
        cmap: str = "seismic",
        save_show_or_return: Literal["save", "show", "return", "both", "all"] = "show",
        save_kwargs: Optional[dict] = {},
    ):
        """Evaluates and visualizes the effect that each possible sender cell type has on expression/abundance in a
        selected receiver cell type.

        Args:
            receiver: Receiver cell type label
            plot_mode: specifies what gets plotted.
                Options:
                    - "qvals": elements of the plot represent statistical significance of the interaction
                    - "effect_size": elements of the plot represent effect size induced in the receiver by the sender
            gene_subset: Names of genes to subset for plot. If None, will use all genes that were used in the
                regression.
            significance_threshold: Set non-significant effect sizes to zero, where the threshold is given here
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

        if self.square:
            receiver_idx = self.celltype_names.index(receiver)

            if plot_mode == "qvals":
                arr = np.log(self.qvalues[:, receiver_idx, :].copy())
                arr[arr < cut_pvals] = cut_pvals
                df = pd.DataFrame(arr, index=self.celltype_names, columns=self.genes)
                if gene_subset:
                    df = df.drop(index=receiver)[gene_subset]

                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=self.figsize)
                qv = sns.heatmap(
                    df.T,
                    square=True,
                    linecolor="grey",
                    linewidths=0.3,
                    cbar_kws={"label": "$\log_{10}$ FDR-corrected pvalues", "location": "top"},
                    cmap=cmap,
                    vmin=-5,
                    vmax=0.0,
                    ax=ax,
                )

                # Outer frame:
                for _, spine in qv.spines.items():
                    spine.set_visible(True)
                    spine.set_linewidth(0.75)

            elif plot_mode == "effect_size":
                arr = self.effect_size[:, receiver_idx, :].copy()
                arr[np.where(self.qvalues[:, receiver_idx, :] > significance_threshold)] = 0
                df = pd.DataFrame(arr, index=self.celltype_names, columns=self.genes)
                if gene_subset:
                    df = df.drop(index=receiver)[gene_subset]
                vmax = np.max(np.abs(df.values))

                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=self.figsize)
                es = sns.heatmap(
                    df.T,
                    square=True,
                    linecolor="grey",
                    linewidths=0.3,
                    cbar_kws={"label": "Effect size", "location": "top"},
                    cmap=cmap,
                    vmin=-vmax,
                    vmax=vmax,
                    ax=ax,
                )

                # Outer frame:
                for _, spine in es.spines.items():
                    spine.set_visible(True)
                    spine.set_linewidth(0.75)

            else:
                logger.error("Invalid input to 'plot_mode'. Options: 'qvals', 'effect_size'.")

        else:
            if receiver not in self.categories:
                self.logger.error(
                    "Adata was subset to categories of interest and fit on those categories, "
                    "but the provided group to 'receiver' is not one of those categories."
                )

            receiver_cols = [col for col in self.effect_size.columns if receiver in col.split("-")[0]]
            # (note that what should be considered the "receiver" is the first cell type listed)

            if plot_mode == "qvals":
                df = np.log(self.qvalues[receiver_cols].copy())
                # Reformat columns for visual purposes:
                senders = [ct[1] for ct in df.columns.str.split("-")]
                df.columns = [col.split("_")[1] for col in senders]
                df[df < cut_pvals] = cut_pvals
                if gene_subset:
                    df = df.loc[gene_subset]

                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=self.figsize)
                qv = sns.heatmap(
                    df,
                    square=True,
                    linecolor="grey",
                    linewidths=0.3,
                    cbar_kws={"label": "$\log_{10}$ FDR-corrected pvalues", "location": "top"},
                    cmap=cmap,
                    vmin=-5,
                    vmax=0.0,
                    ax=ax,
                )

                # Outer frame:
                for _, spine in qv.spines.items():
                    spine.set_visible(True)
                    spine.set_linewidth(0.75)

            elif plot_mode == "effect_size":
                df = self.effect_size[receiver_cols].copy()
                # Reformat columns for visual purposes:
                senders = [ct[1] for ct in df.columns.str.split("-")]
                df.columns = [col.split("_")[1] for col in senders]
                df.values[np.where(self.qvalues[receiver_cols] > significance_threshold)] = 0
                if gene_subset:
                    df = df.loc[gene_subset]
                vmax = np.max(np.abs(df.values))

                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=self.figsize)
                es = sns.heatmap(
                    df,
                    square=True,
                    linecolor="grey",
                    linewidths=0.3,
                    cbar_kws={"label": "Effect size", "location": "top"},
                    cmap=cmap,
                    vmin=-vmax,
                    vmax=vmax,
                    ax=ax,
                )

                # Outer frame:
                for _, spine in es.spines.items():
                    spine.set_visible(True)
                    spine.set_linewidth(0.75)

            else:
                logger.error("Invalid input to 'plot_mode'. Options: 'qvals', 'effect_size'.")

        plt.xlabel("Sender cell type", fontsize=9)
        plt.title("Sender Effects on " + receiver, fontsize=9)
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

    def sender_receiver_effect_volcanoplot(
        self,
        receiver: str,
        sender: str,
        significance_threshold: float = 0.05,
        effect_size_threshold: Union[None, float] = None,
        fontsize: Union[None, int] = None,
        figsize: Union[None, Tuple[float, float]] = (4.5, 7.0),
        save_show_or_return: Literal["save", "show", "return", "both", "all"] = "show",
        save_kwargs: Optional[dict] = {},
    ):
        """Volcano plot to identify differentially expressed genes of a given receiver cell type in the presence of a
        given sender cell type.

        Args:
            receiver: Receiver cell type label
            sender: Sender cell type label
            significance_threshold:  Set non-significant effect sizes (given by q-values) to zero, where the
                threshold is given here
            effect_size_threshold: Set absolute value effect-size threshold beyond which observations are marked as
                interesting. If not given, will take the 95th percentile fold-change as the cutoff.
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

        # Set fold-change threshold if not already provided:
        if effect_size_threshold is None:
            effect_size_threshold = np.percentile(self.effect_size, 95)

        fig, ax = plt.subplots(1, 1, figsize=self.figsize)
        ax.grid(False)

        if self.square:
            receiver_idx = self.celltype_names.index(receiver)
            sender_idx = self.celltype_names.index(sender)

            # All non-significant features:
            qval_filter = np.where(self.qvalues[sender_idx, receiver_idx, :] >= significance_threshold)
            vmax = np.max(np.abs(self.effect_size[sender_idx, receiver_idx, :]))

            if qval_filter[0].size > 0:
                sns.scatterplot(
                    x=self.effect_size[sender_idx, receiver_idx, :][qval_filter],
                    y=-np.log10(self.qvalues[sender_idx, receiver_idx, :])[qval_filter],
                    color="white",
                    edgecolor="black",
                    s=50,
                    ax=ax,
                )

            # Identify subset that may be significant, but which doesn't pass the fold-change threshold:
            qval_filter = np.where(self.qvalues[sender_idx, receiver_idx, :] < significance_threshold)
            x = self.effect_size[sender_idx, receiver_idx, :][qval_filter]
            y = -np.nan_to_num(
                np.log10(self.qvalues[sender_idx, receiver_idx, :])[qval_filter], posinf=14.5, neginf=-14.5
            )
            fc_filter = np.where(x < effect_size_threshold)
            if qval_filter[0].size > 0:
                sns.scatterplot(x=x[fc_filter], y=y[fc_filter], color="darkgrey", edgecolor="black", s=50, ax=ax)

            # Identify subset that are significantly downregulated:
            dreg_color = matplotlib.cm.get_cmap("winter")(0)
            y = -np.nan_to_num(
                np.log10(self.qvalues[sender_idx, receiver_idx, :])[qval_filter], posinf=14.5, neginf=-14.5
            )
            fc_filter = np.where(x <= -effect_size_threshold)
            if qval_filter[0].size > 0:
                sns.scatterplot(x=x[fc_filter], y=y[fc_filter], color=dreg_color, edgecolor="black", s=50, ax=ax)

            # Identify subset that are significantly upregulated:
            ureg_color = matplotlib.cm.get_cmap("autumn")(0)
            fc_filter = np.where(x >= effect_size_threshold)
            if qval_filter[0].size > 0:
                sns.scatterplot(x=x[fc_filter], y=y[fc_filter], color=ureg_color, edgecolor="black", s=50, ax=ax)

        else:
            if sender not in self.categories and receiver not in self.categories:
                self.logger.error(
                    "Adata was subset to categories of interest and fit on those categories, "
                    "but neither the sender nor the receiver group are of those categories."
                )

            # All non-significant features:
            sender_receiver_cols = [
                col for col in self.effect_size.columns if sender in col.split("-")[1] and receiver in col.split("-")[0]
            ]
            qval_filter = np.where(self.qvalues[sender_receiver_cols] >= significance_threshold)
            vmax = np.max(np.abs(self.effect_size[sender_receiver_cols].values))

            if qval_filter[0].size > 0:
                sns.scatterplot(
                    x=self.effect_size[sender_receiver_cols].values[qval_filter],
                    y=-np.log10(self.qvalues[sender_receiver_cols].values)[qval_filter],
                    color="white",
                    edgecolor="black",
                    s=50,
                    ax=ax,
                )

            qval_filter = np.where(self.qvalues[sender_receiver_cols] < significance_threshold)
            x = self.effect_size[sender_receiver_cols].values[qval_filter]
            y = -np.nan_to_num(
                np.log10(self.qvalues[sender_receiver_cols].values)[qval_filter], posinf=14.5, neginf=-14.5
            )

            # Identify subset that may be significant, but which doesn't pass the effect size threshold:
            fc_filter = np.where(x < effect_size_threshold)
            if qval_filter[0].size > 0:
                sns.scatterplot(x=x[fc_filter], y=y[fc_filter], color="darkgrey", edgecolor="black", s=50, ax=ax)

            # Identify subset that are significantly downregulated:
            dreg_color = matplotlib.cm.get_cmap("winter")(0)
            fc_filter = np.where(x <= -effect_size_threshold)
            y = -np.nan_to_num(
                np.log10(self.qvalues[sender_receiver_cols].values)[qval_filter], posinf=14.5, neginf=-14.5
            )
            if qval_filter[0].size > 0:
                sns.scatterplot(x=x[fc_filter], y=y[fc_filter], color=dreg_color, edgecolor="black", s=50, ax=ax)

            # Identify subset that are significantly upregulated:
            ureg_color = matplotlib.cm.get_cmap("autumn")(0)
            fc_filter = np.where(x >= effect_size_threshold)
            if qval_filter[0].size > 0:
                sns.scatterplot(x=x[fc_filter], y=y[fc_filter], color=ureg_color, edgecolor="black", s=50, ax=ax)

        # Plot configuration:
        ax.set_xlim((-vmax * 1.1, vmax * 1.1))
        ax.set_xlabel("Effect size", fontsize=9)
        ax.set_ylabel("$-\log_{10}$ FDR-corrected pvalues", fontsize=9)
        ax.tick_params(axis="both", labelsize=8)
        plt.axvline(-effect_size_threshold, color="darkgrey", linestyle="--", linewidth=0.9)
        plt.axvline(effect_size_threshold, color="darkgrey", linestyle="--", linewidth=0.9)
        plt.axhline(-np.log10(significance_threshold), linestyle="--", color="darkgrey", linewidth=0.9)

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


class Category_Model(Base_Model):
    """Wraps all necessary methods for data loading and preparation, model initialization, parameterization,
    evaluation and prediction when instantiating a model for spatially-aware (but not spatially lagged) regression
    using categorical variables (specifically, the prevalence of categories within spatial neighborhoods) to predict
    the value of gene expression.

    Arguments passed to :class `Base_Model`. The only keyword argument that is used for this class is
    'n_neighbors'.

    Args:
        args: Positional arguments to :class `Base_Model`
        kwargs: Keyword arguments to :class `Base_Model`
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.group_key is not None, "Categorical labels required for this model."

        # Prepare data:
        self.prepare_data(mod_type="category")


class Niche_Model(Base_Model):
    """Wraps all necessary methods for data loading and preparation, model initialization, parameterization,
    evaluation and prediction when instantiating a model for spatially-aware regression using both the prevalence of
    and connections between categories within spatial neighborhoods to predict the value of gene expression.

    Arguments passed to :class `Base_Model`.

    Args:
        args: Positional arguments to :class `Base_Model`
        kwargs: Keyword arguments to :class `Base_Model`
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.group_key is not None, "Categorical labels required for this model."

        self.prepare_data(mod_type="niche")


class Lagged_Model(Base_Model):
    """Wraps all necessary methods for data loading and preparation, model initialization, parameterization,
    evaluation and prediction when instantiating a model for spatially-lagged regression.

    Can specify one of two models: "ligand", which uses the spatial lag of ligand genes and the spatial lag of the
    regression target to predict the regression target, or "niche", which uses the spatial lag of cell type
    colocalization and the spatial lag of the regression target to predict the regression target.

    If "ligand" is specified, arguments to `lig` must be given, and it is recommended to provide `species` as well-
    default for this is human.

    Arguments passed to :class `Base_Model`.

    Args:
        model_type: Either "ligand" or "niche", specifies whether to fit a model that incorporates the spatial lag of
            ligand expression or the spatial lag of cell type colocalization.
        lig: Name(s) of ligands to use as predictors
        rec: Name(s) of receptors to use as regression targets. If not given, will search through database for all
            genes that correspond to the provided genes from 'ligands'.
        rec_ds: Name(s) of receptor-downstream genes to use as regression targets. If not given, will search through
            database for all genes that correspond to receptor-downstream genes.
        species: Specifies L:R database to use
        normalize: Perform library size normalization, to set total counts in each cell to the same number (adjust
            for cell size)
        smooth: To correct for dropout effects, leverage gene expression neighborhoods to smooth expression
        log_transform: Set True if log-transformation should be applied to expression (otherwise, will assume
            preprocessing/log-transform was computed beforehand)
        args: Additional positional arguments to :class `Base_Model`
        kwargs: Additional keyword arguments to :class `Base_Model`
    """

    def __init__(
        self,
        model_type: str = "ligand",
        lig: Union[None, str, List[str]] = None,
        rec: Union[None, str, List[str]] = None,
        rec_ds: Union[None, str, List[str]] = None,
        species: Literal["human", "mouse", "axolotl"] = "human",
        normalize: bool = True,
        smooth: bool = False,
        log_transform: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if self.distr != "gaussian":
            self.logger.info(
                "We recommend applying spatially-lagged models to processed data, for which normality "
                "can be assumed- in this case `distr` can be set to 'gaussian'."
            )

        if model_type == "ligand":
            if lig is None:
                self.logger.error(
                    "From instantiation of :class `Lagged_Model`: `model_type` was given as 'ligand', "
                    "but ligands were not provided using parameter 'lig'."
                )
            # Optional data preprocessing:
            self.preprocess_data(normalize, smooth, log_transform)
            self.prepare_data(mod_type="ligand_lag", lig=lig, rec=rec, rec_ds=rec_ds, species=species)
        elif model_type == "niche":
            # Optional data preprocessing:
            self.preprocess_data(normalize, smooth, log_transform)
            self.prepare_data(mod_type="niche_lag")

    def run_GM_lag(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Defines model run process for a single feature- not callable by the user, all arguments populated by
        arguments passed on instantiation of :class `Base_Model`.

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


class Niche_LR_Model(Base_Model):
    """Wraps all necessary methods for data loading and preparation, model initialization, parameterization,
    evaluation and prediction when instantiating a model for spatially-aware regression using the prevalence of and
    connections between categories within spatial neighborhoods and the cell type-specific expression of ligands and
    receptors to predict the regression target.

    Arguments passed to :class `Base_Model`.

    Args:
        lig: Name(s) of ligands to use as predictors
        rec: Name(s) of receptors to use as regression targets. If not given, will search through database for all
            genes that correspond to the provided genes from 'ligands'
        rec_ds: Name(s) of receptor-downstream genes to use as regression targets. If not given, will search through
            database for all genes that correspond to receptors
        species: Specifies L:R database to use
        niche_lr_r_lag: Only used if 'mod_type' is "niche_lr". Uses the spatial lag of the receptor as the
            dependent variable rather than each spot's unique receptor expression. Defaults to True.
        args: Additional positional arguments to :class `Base_Model`
        kwargs: Additional keyword arguments to :class `Base_Model`
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
        self.logger.info(
            "Predictor arrays for :class `Niche_LR_Model` are extremely sparse. It is recommended "
            "to provide categories to subset for :func `GLMCV_fit_predict`."
        )

        self.prepare_data(
            mod_type="niche_lr", lig=lig, rec=rec, rec_ds=rec_ds, species=species, niche_lr_r_lag=niche_lr_r_lag
        )


def calc_1nd_moment(X, W, normalize_W=True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if normalize_W:
        if type(W) == np.ndarray:
            d = np.sum(W, 1).flatten()
        else:
            d = np.sum(W, 1).A.flatten()
        W = diags(1 / d) @ W if issparse(W) else np.diag(1 / d) @ W
        return W @ X, W
    else:
        return W @ X

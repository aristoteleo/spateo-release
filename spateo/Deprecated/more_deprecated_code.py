"""
Functions to perform regression over smaller subsets of the tissue.
"""
import os
from typing import Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import linalg
from scipy import interpolate
from scipy.signal import argrelextrema
from scipy.sparse import csgraph
from sklearn.cluster import KMeans, SpectralClustering

from ...logging import logger_manager as lm
from ..find_neighbors import calculate_affinity

# ---------------------------------------------------------------------------------------------------
# Utility for subsetting tissue into smaller regions
# ---------------------------------------------------------------------------------------------------
VALID_CURVE = ["convex", "concave"]
VALID_DIRECTION = ["increasing", "decreasing"]


class KneePoint(object):
    """
    Once instantiated, this class attempts to find the point of maximum curvature on a line. The knee is accessible
    via the `.knee` attribute.

    Args:
        x: List or 1D array of shape (`number_of_y_values`,) containing feature values, must be the same length as y.
        y: List or 1D array of shape (`number_of_y_values`,) containing y values, must be the same length as x.
        S: Sensitivity, defaults to 1.0
        curve: If 'concave', algorithm will detect knees. If 'convex', it will detect elbows.
        direction: one of {"increasing", "decreasing"}
        interp_method: one of {"interp1d", "polynomial"}
        polynomial_degree: The degree of the fitting polynomial. Only used when interp_method="polynomial".
            This argument is passed to numpy polyfit `deg` parameter.
    """

    def __init__(
        self,
        x: Iterable[float],
        y: Iterable[float],
        S: float = 1.0,
        curve: str = "concave",
        direction: str = "increasing",
        interp_method: str = "interp1d",
        polynomial_degree: int = 7,
    ):
        self.logger = lm.get_main_logger()

        self.x = np.array(x)
        self.y = np.array(y)
        self.curve = curve
        self.direction = direction
        self.N = len(self.x)
        self.S = S
        self.all_knees = set()
        self.all_norm_knees = set()
        self.all_knees_y = []
        self.all_norm_knees_y = []
        self.polynomial_degree = polynomial_degree

        valid_curve = self.curve in VALID_CURVE
        valid_direction = self.direction in VALID_DIRECTION
        if not all((valid_curve, valid_direction)):
            raise ValueError("Please check that the curve and direction arguments are valid.")

        # Step 1: fit a smooth line
        if interp_method == "interp1d":
            uspline = interpolate.interp1d(self.x, self.y)
            self.Ds_y = uspline(self.x)
        elif interp_method == "polynomial":
            p = np.poly1d(np.polyfit(x, y, self.polynomial_degree))
            self.Ds_y = p(x)
        else:
            raise ValueError(
                "{} is an invalid interp_method parameter, use either 'interp1d' or 'polynomial'".format(interp_method)
            )

        # Step 2: normalize values
        self.x_normalized = self.normalize(self.x)
        self.y_normalized = self.normalize(self.Ds_y)

        # Step 3: Calculate the Difference curve
        self.y_normalized = self.transform_y(self.y_normalized, self.direction, self.curve)
        # normalized difference curve
        self.y_difference = self.y_normalized - self.x_normalized
        self.x_difference = self.x_normalized.copy()

        # Step 4: Identify local maxima/minima
        # local maxima
        self.maxima_indices = argrelextrema(self.y_difference, np.greater_equal)[0]
        self.x_difference_maxima = self.x_difference[self.maxima_indices]
        self.y_difference_maxima = self.y_difference[self.maxima_indices]

        # local minima
        self.minima_indices = argrelextrema(self.y_difference, np.less_equal)[0]
        self.x_difference_minima = self.x_difference[self.minima_indices]
        self.y_difference_minima = self.y_difference[self.minima_indices]

        # Step 5: Calculate thresholds
        self.Tmx = self.y_difference_maxima - (self.S * np.abs(np.diff(self.x_normalized).mean()))

        # Step 6: find knee
        self.knee, self.norm_knee = self.find_knee()

        # Step 7: If we have a knee, extract data about it
        self.knee_y = self.norm_knee_y = None
        if self.knee:
            self.knee_y = self.y[self.x == self.knee][0]
            self.norm_knee_y = self.y_normalized[self.x_normalized == self.norm_knee][0]

    @staticmethod
    def normalize(a: Iterable[float]) -> Iterable[float]:
        """Normalize an array by minmax scaling.

        Args:
            a: The array to normalize
        """
        return (a - min(a)) / (max(a) - min(a))

    @staticmethod
    def transform_y(y: Iterable[float], direction: str, curve: str) -> float:
        """Transform y to concave, increasing based on given direction and curve"""
        # Convert elbows to knees
        if direction == "decreasing":
            if curve == "concave":
                y = np.flip(y)
            elif curve == "convex":
                y = y.max() - y
        elif direction == "increasing" and curve == "convex":
            y = np.flip(y.max() - y)

        return y

    def find_knee(self):
        """It identifies the knee value and sets the instance attributes."""
        if not self.maxima_indices.size:
            self.logger.warn(
                "No local maxima found in the difference curve\n"
                "The line is probably not polynomial, try plotting\n"
                "the difference curve with plt.plot(knee.x_difference, knee.y_difference)\n"
                "Also check that you aren't mistakenly setting the curve argument",
                RuntimeWarning,
            )
            return None, None
        # Placeholder for which threshold region i is located in.
        maxima_threshold_index = 0
        minima_threshold_index = 0
        traversed_maxima = False
        # Traverse the difference curve
        for i, x in enumerate(self.x_difference):
            # Skip points on the curve before the first local maxima
            if i < self.maxima_indices[0]:
                continue

            j = i + 1

            # Reached the end of the curve
            if x == 1.0:
                break

            # If we're at a local max, increment the maxima threshold index and continue
            if (self.maxima_indices == i).any():
                threshold = self.Tmx[maxima_threshold_index]
                threshold_index = i
                maxima_threshold_index += 1
            # Values in difference curve are at or after a local minimum
            if (self.minima_indices == i).any():
                threshold = 0.0
                minima_threshold_index += 1

            if self.y_difference[j] < threshold:
                if self.curve == "convex":
                    if self.direction == "decreasing":
                        knee = self.x[threshold_index]
                        norm_knee = self.x_normalized[threshold_index]
                    else:
                        knee = self.x[-(threshold_index + 1)]
                        norm_knee = self.x_normalized[threshold_index]

                elif self.curve == "concave":
                    if self.direction == "decreasing":
                        knee = self.x[-(threshold_index + 1)]
                        norm_knee = self.x_normalized[threshold_index]
                    else:
                        knee = self.x[threshold_index]
                        norm_knee = self.x_normalized[threshold_index]

                # Add the y value at the knee
                y_at_knee = self.y[self.x == knee][0]
                y_norm_at_knee = self.y_normalized[self.x_normalized == norm_knee][0]
                if knee not in self.all_knees:
                    self.all_knees_y.append(y_at_knee)
                    self.all_norm_knees_y.append(y_norm_at_knee)

                # Now add the knee
                self.all_knees.add(knee)
                self.all_norm_knees.add(norm_knee)

        if self.all_knees == set():
            return None, None

        return knee, norm_knee

    def plot_knee_normalized(
        self,
        figsize: Optional[Tuple[int, int]] = None,
        title: str = "Normalized Knee Point",
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
    ):
        """Plot the normalized curve, the difference curve (x_difference, y_normalized) and the knee, if it exists.

        Args:
            figsize: The figure size of the plot. Example (12, 8)
            title: Title of the visualization, defaults to "Normalized Knee Point"
            xlabel: x-axis label
            ylabel: y-axis label
        """
        if figsize is None:
            figsize = (6, 6)

        plt.figure(figsize=figsize)
        plt.title(title)
        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)
        plt.plot(self.x_normalized, self.y_normalized, "b", label="normalized curve")
        plt.plot(self.x_difference, self.y_difference, "r", label="difference curve")
        plt.xticks(np.arange(self.x_normalized.min(), self.x_normalized.max() + 0.1, 0.1))
        plt.yticks(np.arange(self.y_difference.min(), self.y_normalized.max() + 0.1, 0.1))

        plt.vlines(
            self.norm_knee,
            plt.ylim()[0],
            plt.ylim()[1],
            linestyles="--",
            label="knee/elbow",
        )
        plt.legend(loc="best")

    def plot_knee(
        self,
        figsize: Optional[Tuple[int, int]] = None,
        title: str = "Knee Point",
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        save_id: Optional[str] = None,
    ):
        """
        Plot the curve and the knee, if it exists

        Args:
            figsize: The figure size of the plot. Example (12, 8)
            title: Title of the visualization, defaults to "Knee Point"
            xlabel: x-axis label
            ylabel: y-axis label
            save_id: Optional identifier that can be used to save plot
        """
        if figsize is None:
            figsize = (6, 6)

        plt.figure(figsize=figsize)
        plt.title(title)
        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)
        plt.plot(self.x, self.y, "b", label="data")
        plt.vlines(self.knee, plt.ylim()[0], plt.ylim()[1], linestyles="--", label="knee/elbow")
        plt.legend(loc="best")

        if save_id is not None:
            if not os.path.exists("./figures"):
                os.makedirs("./figures")
            plt.savefig(f"./figures/{save_id}_knee_point.png")

        plt.show()

    # Niceties for users working with elbows rather than knees
    @property
    def elbow(self):
        return self.knee

    @property
    def norm_elbow(self):
        return self.norm_knee

    @property
    def elbow_y(self):
        return self.knee_y

    @property
    def norm_elbow_y(self):
        return self.norm_knee_y

    @property
    def all_elbows(self):
        return self.all_knees

    @property
    def all_norm_elbows(self):
        return self.all_norm_knees

    @property
    def all_elbows_y(self):
        return self.all_knees_y

    @property
    def all_norm_elbows_y(self):
        return self.all_norm_knees_y


def compute_kmeans(
    data: pd.DataFrame,
    k_custom: Optional[int] = None,
    k_max: int = 10,
    plot_knee: bool = False,
    save_id: Optional[str] = None,
    **kwargs,
) -> np.ndarray:
    """Finds optimum number of data clusters using K-means clustering, from DataFrame.

    Args:
        data: Data to cluster
        k_custom: Optional, can be used to directly specify the number of clusters to find
        k_max: Maximum number of clusters to test. Only used if 'k_custom' is not given.
        plot_knee: Set True to plot knee point. If False, 'save_id' will not be used.
        save_id: Optional identifier that can be used to save plot. If 'plot_knee' is not True, this will not be
            used, even if an argument is provided.
        kwargs: Additional keyword arguments to :class `~sklearn.cluster.KMeans`

    Returns:
        clust_predicted: Array containing predicted cluster for each sample (row) of 'data' following computation of
            the ideal number of clusters
    """
    logger = lm.get_main_logger()
    kwargs["random_seed"] = kwargs.get("random_seed", 888)

    if k_custom is None:
        k_range = range(1, k_max)
        inertias = []
        for k in k_range:
            km = KMeans(n_clusters=k, **kwargs)
            km.fit(data)
            inertias.append(km.inertia_)
        y = np.zeros(len(inertias))

        # Compute optimal number of clusters:
        kn = KneePoint(k_range, inertias, curve="convex", direction="decreasing")
        if kn.knee is None:
            logger.info(f"No knee point found. Using the highest tested number of regions: {k_max}")
            elbow_point = k_max
        else:
            elbow_point = kn.knee - 1

        if plot_knee:
            kn.plot_knee(ylabel="Sum of squared error", xlabel="k value", save_id=save_id)

        # K-means algorithm using the computed optimal number of clusters:
        km = KMeans(n_clusters=elbow_point, **kwargs)
    else:
        km = KMeans(n_clusters=k_custom, **kwargs)

    clust_predicted = km.fit_predict(data)

    return clust_predicted


def eigen_decomposition(
    affinity: np.ndarray, plot: bool = True, max_k: int = 10, top_k: int = 1
) -> Tuple[int, np.ndarray, np.ndarray]:
    """For the purposes of finding the optimal cluster number by analyzing the eigenvectors.
    Source for math: Zelnik-Manor, L., & Perona, P. (2004). Self-tuning spectral clustering.
    Advances in neural information processing systems, 17.
    https://proceedings.neurips.cc/paper/2004/file/40173ea48d9567f1f393b20c855bb40b-Paper.pdf

    Args:
        affinity: Affinity matrix
        plot: Set True to plot the eigenvalues
        max_k: Maximum number of clusters to query
        top_k: Find the top k options for the number of clusters from the eigen decomposition

    Returns:
        n_clusters: Optimal number(s) of clusters, depending on input provided to 'top_k'
        eigenvalues: Array of eigenvalues
        eigenvectors: Array of eigenvectors
    """
    logger = lm.get_main_logger()

    if top_k > max_k:
        logger.error(
            f"Cannot find the top {top_k} clusters when {max_k} is the maximum cluster number- adjust " f"'max_k'."
        )

    L = csgraph.laplacian(affinity, normed=True)
    n_components = affinity.shape[0]

    eigenvalues, eigenvectors = linalg.eig(L)
    eigenvalues = eigenvalues[:max_k]
    eigenvectors = eigenvectors[:max_k, :max_k]

    if plot:
        plt.title("Largest eigenvalues of input matrix")
        plt.xlabel("Number of eigenvalues")
        plt.ylabel("Eigenvalue magnitude")
        plt.scatter(np.arange(len(eigenvalues)), eigenvalues)
        plt.show()

    # Identify the optimal number of clusters as the index corresponding to the largest gap between successive
    # eigenvalues:
    index_largest_gap = np.argsort(np.diff(eigenvalues))[::-1][:top_k]
    nb_clusters = index_largest_gap + 1

    return nb_clusters, eigenvalues, eigenvectors


def compute_spectral_clustering(
    data: pd.DataFrame,
    k_custom: Optional[int] = None,
    k_max: int = 10,
    plot_eigenvalues: bool = False,
    save_id: Optional[str] = None,
    distance_metric: str = "euclidean",
    n_neighbors: int = 10,
    **kwargs,
) -> np.ndarray:
    """Finds optimum number of data clusters using spectral clustering, from DataFrame.

    Args:
        data: DataFrame containing coordinates for each point
        k_custom: Optional, can be used to directly specify the number of clusters to find
        k_max: Maximum number of clusters to test. Only used if 'k_custom' is not given.
        plot_eigenvalues: Set True to plot number vs. value of eigenvalues. If False, 'save_id' will not be used.
        save_id: Optional identifier that can be used to save plot. If 'plot_eigenvalues' is not True, this will not be
            used, even if an argument is provided.
        distance_metric: Metric used to compute pairwise distance matrix. Options: ‘braycurtis’, ‘canberra’,
            ‘chebyshev’, ‘cityblock’, ‘correlation’, ‘cosine’, ‘dice’, ‘euclidean’, ‘hamming’, ‘jaccard’,
            ‘jensenshannon’,  ‘kulczynski1’, ‘mahalanobis’, ‘matching’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’,
            ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’.
        n_neighbors: Number of nearest neighbors for computing the affinity matrix
        kwargs: Additional keyword arguments to :class `~sklearn.cluster.SpectralClustering`

    Returns:
        clust_predicted: Array containing predicted cluster for each sample (row) of 'data' following computation of
            the ideal number of clusters
    """
    kwargs["random_seed"] = kwargs.get("random_seed", 888)

    if k_custom is None:
        affinity_matrix = calculate_affinity(position=data, dist_metric=distance_metric, n_neighbors=n_neighbors)
        n_clust, _, _ = eigen_decomposition(affinity_matrix, plot=plot_eigenvalues, max_k=k_max)

        # K-means algorithm using the computed optimal number of clusters:
        spec = SpectralClustering(n_clusters=n_clust, **kwargs)
    else:
        spec = SpectralClustering(n_clusters=k_custom, **kwargs)

    clust_predicted = spec.fit_predict(data)

    return clust_predicted


def L1_L2_penalty(
    alpha: float,
    beta: np.ndarray,
    Tau: Union[None, np.ndarray] = None,
) -> float:
    """
    Combination of the L1 and L2 penalties.

    Args:
        alpha: The weighting between L1 penalty (alpha=1.) and L2 penalty (alpha=0.) term of the loss function.
        beta: Array of shape [n_features,]; learned model coefficients
        Tau: optional array of shape [n_features, n_features]; the Tikhonov matrix for ridge regression. If not
        provided, Tau will default to the identity matrix.

    Returns:
        P: Value for the regularization parameter
    """
    P = 0.5 * (1 - alpha) * L2_penalty(beta, Tau) + alpha * L1_penalty(beta)
    return P


class STGWR:
    """Geographically weighted regression on spatial omics data with parallel processing.

    Args:
        n_pool_processes: Number of processes to use for parallel processing using Python's class `:Pool`. If None,
            use all available processes if :param `MPI_comm` is also None.
        bw: Used to provide previously obtained bandwidth for the spatial kernel. Consists of either a distance
            value or N for the number of nearest neighbors. Can be obtained using BW_Selector or some other
            user-defined method. Pass "np.inf" if all other points should have the same spatial weight.
        coords: Array-like of shape [n_samples, 2]; list of coordinates (x, y) for each sample
        y: Array-like of shape [n_samples]; response variable
        X: Array-like of shape [n_samples, n_features]; independent variables
        distr: Distribution family for the dependent variable; one of "gaussian", "poisson", "nb"
        kernel: Type of kernel function used to weight observations; one of "bisquare", "exponential", "gaussian",
            "quadratic", "triangular" or "uniform".
        fixed_bw: Set True for distance-based kernel function and False for nearest neighbor-based kernel function
        fit_intercept: Set True to include intercept in the model and False to exclude intercept
    """

    def __init__(
        self,
        n_processes: Optional[int],
        bw: Union[float, int],
        coords: Optional[np.ndarray],
        y: Optional[np.ndarray],
        X: Optional[np.ndarray],
        distr: str = "gaussian",
        kernel: str = "bisquare",
        fixed_bw: bool = False,
        fit_intercept: bool = True,
    ):
        self.logger = lm.get_main_logger()

        self.n_processes = n_processes
        self.coords = coords
        self.y = y
        self.X = X
        self.bw = bw
        self.distr = distr
        self.kernel = kernel
        self.fixed_bw = fixed_bw
        self.fit_intercept = fit_intercept
        # Model fitting parameters:
        self.init_params = {}

        self.n_samples = self.X.shape[0]

    def _get_wi(self, i: int, bw: Union[float, int]) -> np.ndarray:
        """Get spatial weights for each sample.

        Args:
            i: Index of sample for which weights are to be calculated to all other samples in the dataset
            bw: Bandwidth for the spatial kernel

        Returns:
            wi: Array of weights for sample of interest
        """

        if bw == np.inf:
            wi = np.ones(self.n_samples)
            return wi

        try:
            wi = Kernel(i, self.coords, bw, fixed=self.fixed_bw, function=self.kernel, subset_idxs=None).kernel
        except:
            self.logger.error(f"Error in getting weights for sample {i}")
        return wi

    def _local_fit(self, i: int):
        """Fit a local regression model for each sample.

        Args:
            i: Index of sample for which local regression model is to be fitted
        """
        wi = self._get_wi(i, self.bw).reshape(-1, 1)

        if self.distr == "gaussian":
            betas, influence_matrix = compute_betas_local(self.y, self.X, wi)
            y_hat = np.dot(self.X[i], betas)[0]
            residual = self.y[i] - y_hat

            # Effect of deleting sample i from the dataset on the estimated coefficients
            influence_i = np.dot(self.X[i], influence_matrix[:, i])
            w = 1

        elif self.distr == "poisson" or self.distr == "nb":
            # init_betas (initial coefficients) to be incorporated at runtime:
            betas, y_hat, n_iter, spatial_weights, linear_predictor, adjusted_predictor, influence_matrix = iwls(
                self.y,
                self.X,
                distr=self.distr,
                init_betas=self.init_params["init_betas"],
            )

    # For now, use a dummy class for Comm:
    class Comm:
        def __init__(self):
            self.rank = 0
            self.size = 1

    Comm_obj = Comm()

    test_model = STGWR(Comm_obj, parser)
    """
    print(test_model.adata[:, "SDC1"].X)
    #print(test_model.cell_categories)
    print(test_model.ligands_expr)
    print(test_model.receptors_expr)
    print(test_model.targets_expr)

    # See if the correct numbers show up:
    print(test_model.all_spatial_weights[121])
    print(test_model.all_spatial_weights[121].shape)
    neighbors = np.argpartition(test_model.all_spatial_weights[121].toarray().ravel(), -10)[-10:]

    print(neighbors)
    print(test_model.receptors_expr["SDC1"].iloc[121])
    print(test_model.ligands_expr["TNC"].iloc[neighbors])
    print(test_model.ligands_expr["TNC"].iloc[103])"""

    test_model._adjust_x()
    # print(test_model.X[121])

    # test_model_2 = GWRGRN(Comm_obj, parser)
    # print(test_model_2.distr)

    """
    print(test_model.adata[:, "SDC1"].X)
    # print(test_model.cell_categories)
    print(test_model.ligands_expr)
    print(test_model.receptors_expr)
    print(test_model.targets_expr)

    test_model._adjust_x()
    print(test_model.signaling_types)
    print(test_model.feature_names)

    # Multicollinearity check test:
    print(test_model.X.shape)
    test_model_df = pd.DataFrame(test_model.X, columns=test_model.feature_names)
    test = multicollinearity_check(test_model_df, thresh=5.0)
    print(test.shape)"""

    """
    # See if the correct numbers show up:
    print(test_model.all_spatial_weights[121])
    print(test_model.all_spatial_weights[121].shape)
    neighbors = np.argpartition(test_model.all_spatial_weights[121].toarray().ravel(), -10)[-10:]

    print(neighbors)
    print(test_model.receptors_expr["SDC1"].iloc[121])
    print(test_model.ligands_expr["TNC"].iloc[neighbors])
    print(test_model.ligands_expr["TNC"].iloc[103])

    test_model._adjust_x()
    print(test_model.X[121])"""

    """
    start = MPI.Wtime()

    # Check to see if multiscale model is specified:
    if parser.parse_args().multiscale:
        # SPACE FOR LATER
        "filler"
    else:
        STGWR(comm, parser).fit()

    end = MPI.Wtime()

    wt = comm.gather(end - start, root=0)
    if rank == 0:
        print("Total Time Elapsed:", np.round(max(wt), 2), "seconds")
        print("-" * 60)"""


# Old niche matrix code:
# in self._compute_niche_mat():
# Encode the "niche" or each sample by taking into account each sample's own cell type:
data = {"categories": self.cell_categories, "dmat_neighbors": dmat_neighbors}
niche_mat = np.asarray(dmatrix("categories:dmat_neighbors-1", data))
connections_cols = list(product(self.cell_categories.columns, self.cell_categories.columns))
connections_cols.sort(key=lambda x: x[1])
return niche_mat, connections_cols

# in self._adjust_x():
# If feature names doesn't already exist, create it:
if not hasattr(self, "feature_names"):
    self.feature_names = [f"{i[0]}-{i[1]}" for i in connections_cols]


# Old slice model code:
rec_expr = np.multiply(self.cell_categories, np.tile(rec_expr_values.toarray(), self.cell_categories.shape[1]))

# Construct the category interaction matrix (1D array w/ n_categories ** 2 elements, encodes the
# ligand-receptor niches of each sample by documenting the cell type-specific L:R enrichment within
# the niche:
data = {"category_rec_expr": rec_expr, "neighborhood_lig_expr": nbhd_lig_expr}
lr_connections = np.asarray(dmatrix("category_rec_expr:neighborhood_lig_expr-1", data))

lr_connections_cols = list(product(self.cell_categories.columns, self.cell_categories.columns))
lr_connections_cols.sort(key=lambda x: x[1])
n_connections_pairs = len(lr_connections_cols)
# Swap sending & receiving cell types because we're looking at receptor expression in the "source" cell
# and ligand expression in the surrounding cells.
lr_connections_cols = [f"{i[1]}-{i[0]}:{lig}-{rec}" for i in lr_connections_cols]
niche_mats[f"{lig}-{rec}"] = pd.DataFrame(lr_connections, columns=lr_connections_cols)
niche_mats = {key: value for key, value in sorted(niche_mats.items())}

# If list of L:R labels (secreted vs. membrane-bound vs. ECM) doesn't already exist, create it:
if not hasattr(self, "self.signaling_types"):
    query = re.compile(r"\w+-\w+:(\w+-\w+)")
    self.signaling_types = []
    for col in self.feature_names:
        ligrec = re.search(query, col).group(1)
        result = self.lr_db.loc[
            (self.lr_db["from"] == ligrec.split("-")[0]) & (self.lr_db["to"] == ligrec.split("-")[1]),
            "type",
        ].iloc[0]

        self.signaling_types.append(result)


elif self.distr == "poisson":
    for j in range(self.n_features):
        # Compute diagonal of the Hessian matrix:
        hessian_chunk[j, j] = -np.sum(partial_hat[:, :, j] * self.X[:, j].reshape(-1, 1) * self.X[:, j] * y_pred)
        # Compute off-diagonal of the Hessian matrix:
        for k in range(j + 1, self.n_features):
            hessian_chunk[j, k] = -np.sum(partial_hat[:, :, j] * self.X[:, j].reshape(-1, 1) * self.X[:, k] * y_pred)
            hessian_chunk[k, j] = hessian_chunk[j, k]

    return ENP_chunk, hessian_chunk

# Hessian for NB:
else:
    for j in range(self.n_features):
        # Compute diagonal of the Hessian matrix:
        X_j = self.X[:, j].reshape(-1, 1)
        psi_deriv = special.digamma(self.distr_obj.variance.disp + np.dot(X_j**2, y_pred)) - special.digamma(
            self.distr_obj.variance.disp + np.dot(X_j, y_pred)
        )
        hessian_chunk = np.sum(partial_hat[:, :, j] * X_j**2 * y_pred * psi_deriv)

        # Compute off-diagonal of the Hessian matrix:
        for k in range(j + 1, self.n_features):
            X_k = self.X[:, k].reshape(-1, 1)
            psi_deriv = special.digamma(self.distr_obj.variance.disp + np.dot(X_j * X_k, y_pred)) - special.digamma(
                self.distr_obj.variance.disp + np.dot(X_j, y_pred)
            )
            hessian_chunk = np.sum(partial_hat[:, :, j] * X_j * X_k * y_pred * psi_deriv)

    return ENP_chunk, hessian_chunk

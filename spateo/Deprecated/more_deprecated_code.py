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

if self.distr == "gaussian":
    hessian = np.dot(X.T, X)
elif self.distr == "poisson":
    hessian = np.dot(X.T, np.dot(np.diag(fitted), X))
elif self.distr == "nb":
    hessian = np.dot(X.T, np.dot(np.diag(fitted * (1 + fitted / self.distr_obj.variance.disp)), X))
return hessian


def mpi_fit(
    self,
    y: Optional[np.ndarray],
    X: Optional[np.ndarray],
    y_label: str,
    bw: Union[float, int],
    final: bool = False,
    multiscale: bool = False,
    fit_predictor: bool = False,
) -> Union[None, np.ndarray]:
    """Fit local regression model for each sample in parallel, given a specified bandwidth.

    Args:
        y: Response variable
        X: Independent variable array- if not given, will default to :attr `X`. Note that if object was initialized
            using an AnnData object, this will be overridden with :attr `X` even if a different array is given.
        y_label: Used to provide a unique ID for the dependent variable for saving purposes and to query keys
            from various dictionaries
        bw: Bandwidth for the spatial kernel
        final: Set True to indicate that no additional parameter selection needs to be performed; the model can
            be fit and more stats can be returned.
        multiscale: Set True to fit a multiscale GWR model where the independent-dependent relationships can vary
            over different spatial scales
        fit_predictor: Set True to indicate that dependent variable to fit is a linear predictor rather than a
            true response variable
    """
    # If model to be run is a "niche", "lr" or "slice" model, update the spatial weights and then update X given
    # the current value of the bandwidth:
    if hasattr(self, "adata"):
        self.all_spatial_weights = self._compute_all_wi(bw)
        self.all_spatial_weights = self.comm.bcast(self.all_spatial_weights, root=0)
        self.logger.info(f"Adjusting X for new bandwidth: {bw}")
        self._adjust_x()
        self.X = self.comm.bcast(self.X, root=0)
        self.X_df = self.comm.bcast(self.X_df, root=0)
        self.logger(f"Using adjusted X array for {self.mod_type} model.")
        X = self.X

    if X.shape[1] != self.n_features:
        n_features = X.shape[1]
        n_features = self.comm.bcast(n_features, root=0)
    else:
        n_features = self.n_features

    if self.grn:
        self.all_spatial_weights = self._compute_all_wi(bw)
        # Row standardize spatial weights so as to ensure results aren't biased by the number of neighbors of
        # each cell:
        self.all_spatial_weights = self.all_spatial_weights / self.all_spatial_weights.sum(axis=1)[:, None]
        y, X = self._adjust_x_nbhd_convolve(y, X)
        y = self.comm.bcast(y, root=0)
        X = self.comm.bcast(X, root=0)

    # If subsampled, take the subsampled portion of the X array- if :attr `multiscale` is True, this subsampling
    # will be performed before calling :func `mpi_fit`:
    if self.subsampled:
        indices = self.subsampled_indices[y_label]
        n_samples = self.n_samples_subset[y_label]
        X = X[indices, :]
        y = y[indices]
    else:
        n_samples = self.n_samples

    if final:
        if multiscale:
            local_fit_outputs = np.empty((self.x_chunk.shape[0], n_features), dtype=np.float64)
        else:
            local_fit_outputs = np.empty((self.x_chunk.shape[0], 2 * n_features + 3), dtype=np.float64)

        # Fitting for each location, or each location that is among the subsampled points:
        pos = 0
        for i in self.x_chunk:
            local_fit_outputs[pos] = self.local_fit(
                i, y, X, y_label=y_label, bw=bw, final=final, multiscale=multiscale, fit_predictor=fit_predictor
            )
            pos += 1

        # Gather data to the central process such that an array is formed where each sample has its own
        # measurements:
        all_fit_outputs = self.comm.gather(local_fit_outputs, root=0)
        # For non-MGWR:
        # Column 0: Index of the sample
        # Column 1: Diagnostic (residual for Gaussian, fitted response value for Poisson/NB)
        # Column 2: Contribution of each sample to its own value
        # Columns 3-n_feats+3: Estimated coefficients
        # Columns n_feats+3-end: Canonical correlations
        # All columns are betas for MGWR

        # If multiscale, do not need to fit using fixed bandwidth:
        if multiscale:
            # At final iteration, for MGWR, this function is only needed to get parameters:
            all_fit_outputs = self.comm.bcast(all_fit_outputs, root=0)
            all_fit_outputs = np.vstack(all_fit_outputs)
            return all_fit_outputs

        if self.comm.rank == 0:
            all_fit_outputs = np.vstack(all_fit_outputs)
            self.logger.info(f"Computing metrics for GWR using bandwidth: {bw}")

            # Residual sum of squares for Gaussian model:
            if self.distr == "gaussian":
                RSS = np.sum(all_fit_outputs[:, 1] ** 2)
                # Total sum of squares:
                TSS = np.sum((y - np.mean(y)) ** 2)
                r_squared = 1 - RSS / TSS

                # Note: trace of the hat matrix and effective number of parameters (ENP) will be used
                # interchangeably:
                ENP = np.sum(all_fit_outputs[:, 2])
                # Residual variance:
                sigma_squared = RSS / (n_samples - ENP)
                # Corrected Akaike Information Criterion:
                aicc = self.compute_aicc_linear(RSS, ENP, n_samples=X.shape[0])
                # Scale the leverages by their variance to compute standard errors of the predictor:
                all_fit_outputs[:, -n_features:] = np.sqrt(all_fit_outputs[:, -n_features:] * sigma_squared)

                # For saving outputs:
                header = "index,residual,influence,"
            else:
                r_squared = None

            if self.distr == "poisson" or self.distr == "nb":
                # Deviance:
                deviance = self.distr_obj.deviance(y, all_fit_outputs[:, 1])
                # Log-likelihood:
                ll = self.distr_obj.log_likelihood(y, all_fit_outputs[:, 1])
                # Reshape if necessary:
                if self.n_features > 1:
                    ll = ll.reshape(-1, 1)
                # ENP:
                ENP = np.sum(all_fit_outputs[:, 2])
                # Corrected Akaike Information Criterion:
                aicc = self.compute_aicc_glm(ll, ENP, n_samples=n_samples)
                # To obtain standard errors for each coefficient, take the square root of the diagonal elements
                # of the covariance matrix:
                # Compute the covariance matrix using the Hessian- first compute the estimate for dispersion of
                # the NB distribution:
                if self.distr == "nb":
                    theta = 1 / self.distr_obj.variance(all_fit_outputs[:, 1])
                    weights = self.distr_obj.weights(all_fit_outputs[:, 1])
                    deviance = 2 * np.sum(
                        weights
                        * (
                            y * np.log(y / all_fit_outputs[:, 1])
                            + (theta - 1) * np.log(1 + all_fit_outputs[:, 1] / (theta - 1))
                        )
                    )
                    dof = len(y) - self.X.shape[1]
                    self.distr_obj.variance.disp = deviance / dof

                hessian = self.hessian(all_fit_outputs[:, 1], X=X)
                cov_matrix = np.linalg.inv(hessian)
                all_fit_outputs[:, -n_features:] = np.sqrt(np.diag(cov_matrix))

                # For saving outputs:
                header = "index,prediction,influence,"
            else:
                deviance = None

            # Save results:
            varNames = self.feature_names
            # Columns for the possible intercept, coefficients and squared canonical coefficients:
            for x in varNames:
                header += "b_" + x + ","
            for x in varNames:
                header += "se_" + x + ","

            # Return output diagnostics and save result:
            self.output_diagnostics(aicc, ENP, r_squared, deviance)
            self.save_results(all_fit_outputs, header, label=y_label)

        return

    # If not the final run:
    if self.distr == "gaussian" or fit_predictor:
        # Compute AICc using the sum of squared residuals:
        RSS = 0
        trace_hat = 0

        for i in self.x_chunk:
            fit_outputs = self.local_fit(i, y, X, y_label=y_label, bw=bw, fit_predictor=fit_predictor, final=False)
            err_sq, hat_i = fit_outputs[0], fit_outputs[1]
            RSS += err_sq
            trace_hat += hat_i

        # Send data to the central process:
        RSS_list = self.comm.gather(RSS, root=0)
        trace_hat_list = self.comm.gather(trace_hat, root=0)

        if self.comm.rank == 0:
            RSS = np.sum(RSS_list)
            trace_hat = np.sum(trace_hat_list)
            aicc = self.compute_aicc_linear(RSS, trace_hat, n_samples=n_samples)
            if not multiscale:
                self.logger.info(f"Bandwidth: {bw:.3f}, Linear AICc: {aicc:.3f}")
            return aicc

    elif self.distr == "poisson" or self.distr == "nb":
        # Compute AICc using the fitted and observed values, using the linear predictor for multiscale models and
        # the predicted response otherwise:
        trace_hat = 0
        pos = 0
        y_pred = np.empty(self.x_chunk.shape[0], dtype=np.float64)

        for i in self.x_chunk:
            fit_outputs = self.local_fit(i, y, X, y_label=y_label, bw=bw, fit_predictor=fit_predictor, final=False)
            y_pred_i, hat_i = fit_outputs[0], fit_outputs[1]
            y_pred[pos] = y_pred_i
            trace_hat += hat_i
            pos += 1

        # Send data to the central process:
        all_y_pred = self.comm.gather(y_pred, root=0)
        trace_hat_list = self.comm.gather(trace_hat, root=0)

        if self.comm.rank == 0:
            ll = self.distr_obj.log_likelihood(y, all_y_pred)
            trace_hat = np.sum(trace_hat_list)
            aicc = self.compute_aicc_glm(ll, trace_hat, n_samples=n_samples)
            self.logger.info(f"Bandwidth: {bw:.3f}, GLM AICc: {aicc:.3f}")

            return aicc

    return


def hessian(self, fitted: np.ndarray, X: Optional[np.ndarray] = None) -> np.ndarray:
    """Compute the Hessian matrix for the given spatially-weighted model, representing the confidence in the
    parameter estimates. Note that this is formed such that the Hessian matrix is computed for each cell.

    Args:
        fitted: Array of shape [n_samples,]; fitted mean response variable (link function evaluated
            at the linear predicted values)
        X: Independent variable array of shape [n_samples, n_features]

    Returns:
        hessian: Hessian matrix
    """
    if X is None:
        X = self.X

    hessian = np.zeros((self.n_samples, self.n_features, self.n_features))

    for i in range(self.n_samples):
        if self.distr == "gaussian":
            hessian[i] = np.outer(X[i], X[i])
        elif self.distr == "poisson":
            hessian[i] = np.outer(X[i], X[i] * fitted[i])
        elif self.distr == "nb":
            hessian[i] = np.outer(X[i], X[i] * fitted[i] * (1 + fitted[i] / self.distr_obj.variance.disp))

    return hessian


def multiscale_backfitting(
    self,
    y: Optional[pd.DataFrame] = None,
    X: Optional[np.ndarray] = None,
    init_betas: Optional[Dict[str, np.ndarray]] = None,
):
    """
    Backfitting algorithm for MGWR, obtains parameter estimates and variate-specific bandwidths by iterating one
    predictor while holding all others constant. Run before :func `fit` to obtain initial covariate-specific
    bandwidths.

    Reference: Fotheringham et al. 2017. Annals of AAG.

    Args:
        y: Optional dataframe, can be used to provide dependent variable array directly to the fit function. If
            None, will use :attr `targets_expr` computed using the given AnnData object to create this (each
            individual column will serve as an independent variable). Needed to be given as a dataframe so that
            column(s) are labeled, so each result can be associated with a labeled dependent variable.
        X: Optional array, can be used to provide dependent variable array directly to the fit function. If
            None, will use :attr `X` computed using the given AnnData object and the type of the model to create.
        init_betas: Optional dictionary containing arrays with initial values for the coefficients. Keys should
            correspond to target genes and values should be arrays of shape [n_features, 1].
    """
    if self.comm.rank == 0:
        self.logger.info("Multiscale Backfitting...")

    if not self.set_up:
        self.logger.info("Model has not yet been set up to run, running :func `SWR._set_up_model()` now...")
        self._set_up_model()

    if self.comm.rank == 0:
        self.logger.info("Initialization complete.")

    self.all_bws_history = {}
    self.params_all_targets = {}
    self.errors_all_targets = {}
    self.predictions_all_targets = {}
    # For linear models:
    self.all_RSS = {}
    self.all_TSS = {}

    # For GLM models:
    self.all_deviances = {}
    self.all_log_likelihoods = {}

    # Optional, to save the dispersion parameter for negative binomial fitted to each target:
    self.nb_disp_dict = {}

    if y is None:
        y_arr = self.targets_expr if hasattr(self, "targets_expr") else self.target
    else:
        y_arr = y
        y_arr = self.comm.bcast(y_arr, root=0)

    for target in y_arr.columns:
        y = y_arr[target].to_frame()
        y_label = target

        if self.subsampled:
            n_samples = self.n_samples_subset[y_label]
            indices = self.subsampled_indices[y_label]
        else:
            n_samples = self.n_samples
            indices = np.arange(self.n_samples)

        # Initialize parameters, with a uniform initial bandwidth for all features- set fit_predictor False to
        # fit model under the assumption that y is a Poisson-distributed dependent variable:
        self.logger.info(f"Finding uniform initial bandwidth for all features for target {target}...")
        all_betas, self.all_bws_init = self.fit(
            y, X, multiscale=True, fit_predictor=False, init_betas=init_betas, verbose=False
        )
        # If applicable, i.e. if model if one of the signaling models for which the X array varies with
        # bandwidth, update X with the up-to-date version that leverages the most recent bandwidth estimations:
        if X is None:
            X = self.X
        X = self.comm.bcast(X, root=0)

        # Initial values- multiply input by the array corresponding to the correct target- note that this is
        # denoted as the predicted dependent variable, but is actually the linear predictor in the case of GLMs:
        y_pred_init = X * all_betas[target]
        all_y_pred = np.sum(y_pred_init, axis=1)

        if self.distr != "gaussian":
            y_true = self.distr_obj.get_predictors(y.values).reshape(-1)
        else:
            y_true = y.values.reshape(-1)

        error = y_true - all_y_pred.reshape(-1)
        self.logger.info(f"Initial RSS: {np.sum(error ** 2):.3f}")
        if self.distr != "gaussian":
            # Small errors <-> large negatives in log space, but in reality these are negligible- set these to 0:
            error[error < 0] = 0
            error = self.distr_obj.get_predictors(error)
            error[error < -1] = 0
        # error = np.zeros(y.values.shape[0])

        bws = [None] * self.n_features
        bw_plateau_counter = 0
        bw_history = []
        error_history = []
        y_pred_history = []
        score_history = []

        n_iters = max(200, self.max_iter)
        for iter in range(1, n_iters + 1):
            new_ys = np.empty(y_pred_init.shape, dtype=np.float64)
            new_betas = np.empty(y_pred_init.shape, dtype=np.float64)

            for n_feat in range(self.n_features):
                if self.adata_path is not None:
                    signaling_type = self.signaling_types[n_feat]
                else:
                    signaling_type = None
                # Use each individual feature to predict the response- note y is set up as a DataFrame because in
                # other cases column names/target names are taken from y:
                y_mod = y_pred_init[:, n_feat] + error
                temp_y = pd.DataFrame(y_mod.reshape(-1, 1), columns=[target])

                # Check if the bandwidth has plateaued for all features in this iteration:
                if bw_plateau_counter > self.patience:
                    # If applicable, i.e. if model if one of the signaling models for which the X array varies with
                    # bandwidth, update X with the up-to-date version that leverages the most recent bandwidth
                    # estimations:
                    if X is None:
                        temp_X = (self.X[:, n_feat]).reshape(-1, 1)
                    else:
                        temp_X = X[:, n_feat].reshape(-1, 1)
                    # Use the bandwidths from the previous iteration before plateau was determined to have been
                    # reached:
                    bw = bws[n_feat]
                    betas = self.mpi_fit(
                        temp_y.values,
                        temp_X,
                        y_label=target,
                        bw=bw,
                        final=True,
                        fit_predictor=True,
                        multiscale=True,
                    )
                else:
                    betas, bw_dict = self.fit(
                        temp_y,
                        X,
                        n_feat=n_feat,
                        init_betas=init_betas,
                        multiscale=True,
                        fit_predictor=True,
                        signaling_type=signaling_type,
                        verbose=False,
                    )
                    # Get coefficients for this particular target:
                    betas = betas[target]

                # If applicable, i.e. if model if one of the signaling models for which the X array varies with
                # bandwidth, update X with the up-to-date version that leverages the most recent bandwidth
                # estimations:
                if X is None:
                    temp_X = (self.X[:, n_feat]).reshape(-1, 1)
                else:
                    temp_X = X[:, n_feat].reshape(-1, 1)
                # Update the dependent prediction (again not for GLMs, this quantity is instead the linear
                # predictor) and betas:
                new_y = (temp_X * betas).reshape(-1)
                error = y_mod - new_y
                new_ys[:, n_feat] = new_y
                new_betas[:, n_feat] = betas.reshape(-1)
                # Update running list of bandwidths for this feature:
                bws[n_feat] = bw_dict[target]

            # Check if ALL bandwidths remain the same between iterations:
            if (iter > 1) and np.all(bw_history[-1] == bws):
                bw_plateau_counter += 1
            else:
                bw_plateau_counter = 0

            # Compute normalized sum-of-squared-errors-of-prediction using the updated predicted values:
            bw_history.append(deepcopy(bws))
            error_history.append(deepcopy(error))
            y_pred_history.append(deepcopy(new_ys))
            SSE = np.sum((new_ys - y_pred_init) ** 2) / n_samples
            TSS = np.sum(np.sum(new_ys, axis=1) ** 2)
            rmse = (SSE / TSS) ** 0.5
            score_history.append(rmse)

            if self.comm.rank == 0:
                self.logger.info(f"Target: {target}, Iteration: {iter}, Score: {rmse:.5f}")
                self.logger.info(f"Bandwidths: {bws}")

            if rmse < self.tolerance:
                self.logger.info(f"For target {target}, multiscale optimization converged after {iter} iterations.")
                break

            # Check for local minimum:
            if iter > 2:
                if score_history[-3] >= score_history[-2] and score_history[-1] >= score_history[-2]:
                    self.logger.info(f"Local minimum reached for target {target} after {iter} iterations.")
                    new_ys = y_pred_history[-2]
                    error = error_history[-2]
                    bw_history = bw_history[:-1]
                    rmse = score_history[-2]
                    self.logger.info(f"Target: {target}, Iteration: {iter-1}, Score: {rmse:.5f}")
                    break

            # Use the new predicted values as the initial values for the next iteration:
            y_pred_init = new_ys

        # Final estimated values:
        y_pred = new_ys
        y_pred = y_pred.sum(axis=1)

        bw_history = np.array(bw_history)
        self.all_bws_history[target] = bw_history

        # Compute diagnostics for current target using the final errors:
        if self.distr == "gaussian":
            RSS = np.sum(error**2)
            self.all_RSS[target] = RSS
            # Total sum of squares:
            TSS = np.sum((y.values - np.mean(y.values)) ** 2)
            self.all_TSS[target] = TSS
            r_squared = 1 - RSS / TSS

            # For saving outputs:
            header = "index,residual,"
        else:
            r_squared = None

        if self.distr == "poisson" or self.distr == "nb":
            # Map linear predictors to the response variable:
            y_pred = self.distr_obj.predict(y_pred)
            error = y.values.reshape(-1) - y_pred
            self.logger.info(f"Final RSS: {np.sum(error ** 2):.3f}")

            # Set dispersion for negative binomial:
            if self.distr == "nb":
                theta = 1 / self.distr_obj.variance(y_pred)
                weights = self.distr_obj.weights(y_pred)
                deviance = 2 * np.sum(
                    weights
                    * (y.values * np.log(y.values / y_pred) + (theta - 1) * np.log(1 + y_pred[:, 1] / (theta - 1)))
                )
                dof = len(y.values) - X.shape[1]
                self.nb_disp_dict[target] = deviance / dof

            # Deviance:
            deviance = self.distr_obj.deviance(y.values.reshape(-1), y_pred)
            self.all_deviances[target] = deviance
            ll = self.distr_obj.log_likelihood(y.values.reshape(-1), y_pred)
            # Reshape if necessary:
            if self.n_features > 1:
                ll = ll.reshape(-1, 1)
            self.all_log_likelihoods[target] = ll

            # For saving outputs:
            header = "index,deviance,"
        else:
            deviance = None
        # Store some of the final values of interest:
        self.params_all_targets[target] = new_betas
        self.errors_all_targets[target] = error
        self.predictions_all_targets[target] = y_pred

        # Save results without standard errors or influence measures:
        if self.comm.rank == 0 and self.multiscale_params_only:
            varNames = self.feature_names
            # Save intercept and parameter estimates:
            for x in varNames:
                header += "b_" + x + ","

            # Return output diagnostics and save result:
            self.output_diagnostics(None, None, r_squared, deviance)
            output = np.hstack([indices.reshape(-1, 1), error.reshape(-1, 1), self.params_all_targets[target]])
            self.save_results(header, output, label=y_label)


def multiscale_compute_metrics(self, X: Optional[np.ndarray] = None, n_chunks: int = 2):
    """Compute multiscale inference and output results.

    Args:
        X: Optional array, can be used to provide dependent variable array directly to the fit function. If
            None, will use :attr `X` computed using the given AnnData object and the type of the model to create.
            Must be the same X array as was used to fit the model (i.e. the same X given to :func
            `multiscale_backfitting`).
        n_chunks: Number of partitions comprising each covariate-specific hat matrix.
    """
    if X is None:
        X = self.X

    if self.multiscale_params_only:
        self.logger.warning(
            "Chunked computations will not be performed because `multiscale_params_only` is set to True, "
            "so only parameter values (and no other metrics) will be saved."
        )
        return

    # Check that initial bandwidths and bandwidth history are present (e.g. that :func `multiscale_backfitting` has
    # been called):
    if not hasattr(self, "all_bws_history"):
        raise ValueError(
            "Initial bandwidths must be computed before calling `multiscale_fit`. Run :func "
            "`multiscale_backfitting` first."
        )

    if self.comm.rank == 0:
        self.logger.info(f"Computing model metrics, using {n_chunks} chunks...")

    self.n_chunks = self.comm.size * n_chunks
    self.chunks = np.arange(self.comm.rank * n_chunks, (self.comm.rank + 1) * n_chunks)

    y_arr = self.targets_expr if hasattr(self, "targets_expr") else self.target
    for target_label in y_arr.columns:
        # sample_names = self.sample_names if not self.subsampled else self.subsampled_sample_names[target_label]
        # Fitted coefficients, errors and predictions:
        parameters = self.params_all_targets[target_label]
        predictions = self.predictions_all_targets[target_label]
        print(predictions.shape)
        errors = self.errors_all_targets[target_label]
        y_label = target_label

        # If subsampling was done, check for the number of fitted samples for the right target:
        if self.subsampled:
            self.n_samples = self.n_samples_subset[target_label]
            self.indices = self.subsampled_indices[target_label]
            self.coords = self.coords[self.indices, :]
            X = X[self.indices, :]
        else:
            self.indices = np.arange(self.n_samples)

        # Lists to store the results of each chunk for this variable (lvg list only used if Gaussian,
        # Hessian list only used if non-Gaussian):
        ENP_list = []
        lvg_list = []

        for chunk in self.chunks:
            if self.distr == "gaussian":
                ENP_chunk, lvg_chunk = self.chunk_compute_metrics(X, chunk_id=chunk, target_label=target_label)
                ENP_list.append(ENP_chunk)
                lvg_list.append(lvg_chunk)
            else:
                ENP_chunk = self.chunk_compute_metrics(X, chunk_id=chunk, target_label=target_label)
                ENP_list.append(ENP_chunk)

        # Gather results from all chunks:
        ENP_list = np.array(self.comm.gather(ENP_list, root=0))
        if self.distr == "gaussian":
            lvg_list = np.array(self.comm.gather(lvg_list, root=0))

        if self.comm.rank == 0:
            # Compile results from all chunks to get the estimated number of parameters for this response variable:
            ENP = np.sum(np.vstack(ENP_list), axis=0)
            # Total estimated parameters:
            ENP_total = np.sum(ENP)

            if self.distr == "gaussian":
                # Compile results from all chunks to get the leverage matrix for this response variable:
                lvg = np.sum(np.vstack(lvg_list), axis=0)

                # Get sums-of-squares corresponding to this feature:
                RSS = self.all_RSS[target_label]
                TSS = self.all_TSS[target_label]
                # Residual variance:
                sigma_squared = RSS / (self.n_samples - ENP)
                # R-squared:
                r_squared = 1 - RSS / TSS
                # Corrected Akaike Information Criterion:
                aicc = self.compute_aicc_linear(RSS, ENP_total, n_samples=self.n_samples)
                # Scale leverages by the residual variance to compute standard errors:
                standard_error = np.sqrt(lvg * sigma_squared)
                self.output_diagnostics(aicc, ENP_total, r_squared=r_squared, deviance=None, y_label=y_label)

                header = "index,residual,"
                outputs = np.hstack([self.indices, errors.reshape(-1, 1), parameters, standard_error])

            if self.distr == "poisson" or self.distr == "nb":
                # Get deviances corresponding to this feature:
                deviance = self.all_deviances[target_label]
                ll = self.all_log_likelihoods[target_label]

                # Corrected Akaike Information Criterion:
                aicc = self.compute_aicc_glm(ll, ENP_total, n_samples=self.n_samples)
                # Compute standard errors using the covariance:
                if self.distr == "nb":
                    self.distr_obj.variance.disp = self.nb_disp_dict[target_label]

                # Standard errors using the Hessian:
                all_standard_errors = []
                hessian = self.hessian(predictions, X=X)
                print(hessian.shape)
                for i in range(self.n_samples):
                    try:
                        cov_matrix = np.linalg.inv(hessian[i])
                        standard_error = np.sqrt(np.diag(cov_matrix))
                    except:
                        standard_error = np.full((self.n_features,), np.nan)
                    all_standard_errors.append(standard_error)
                standard_error = np.vstack(all_standard_errors)
                self.output_diagnostics(aicc, ENP_total, r_squared=None, deviance=deviance, y_label=y_label)

                header = "index,prediction,"
                outputs = np.hstack(
                    [self.indices.reshape(-1, 1), predictions.reshape(-1, 1), parameters, standard_error]
                )

            varNames = self.feature_names
            # Save intercept and parameter estimates:
            for x in varNames:
                header += "b_" + x + ","
            for x in varNames:
                header += "se_" + x + ","

            self.save_results(outputs, header, label=y_label)


theta = 1 / self.distr_obj.variance(all_fit_outputs[:, 1])
weights = self.distr_obj.weights(all_fit_outputs[:, 1])
deviance = 2 * np.sum(
    weights * (y * np.log(y / all_fit_outputs[:, 1]) + (theta - 1) * np.log(1 + all_fit_outputs[:, 1] / (theta - 1)))
)
dof = len(y) - self.X.shape[1]
self.distr_obj.variance.disp = deviance / dof


# Get list of regulatory factors from among the most highly spatially-variable genes, indicative of
# potentially interesting spatially-enriched signal:
self.logger.info("Preparing data: getting list of regulators from among the most highly spatially-variable genes.")
if "m_degs" not in locals():
    m_degs = moran_i(self.adata)
m_filter_genes = m_degs[m_degs.moran_q_val < 0.05].sort_values(by=["moran_i"], ascending=False).index
regulators = [g for g in m_filter_genes if g in database_tfs]

# If no significant spatially-variable receptors are found, use the top 100 most spatially-variable TFs:
if len(regulators) == 0:
    self.logger.info(
        "No significant spatially-variable regulatory factors found. Using top 100 most " "spatially-variable TFs."
    )
    m_filter_genes = m_degs.sort_values(by=["moran_i"], ascending=False).index
    regulators = [g for g in m_filter_genes if g in database_tfs][:100]


zero_y = np.where(y == 0)[0]
if np.any(zero_y):
    # Find the max distance between any given point and its closest neighbor with nonzero y:
    self.minbw = np.max(
        [np.min(cdist(self.coords[[i]], self.coords[zero_y])) for i in range(self.n_samples) if y[i] != 0]
    )
else:
    "filler"

# Optionally, subsample particular cell types of interest:
if self.group_subset is not None:
    subset = self.adata.obs[self.group_key].isin(self.group_subset)
    self.fitted_indices = [self.sample_names.get_loc(name) for name in subset.index]
    self.fitted_sample_names = subset.index
    self.n_samples_fitted = len(subset)
    # Add cells that are neighboring cells of the chosen type, but which are not of the chosen type:
    get_wi_partial = partial(
        get_wi,
        n_samples=self.n_samples,
        coords=self.coords,
        fixed_bw=False,
        exclude_self=True,
        kernel=self.kernel,
        bw=10,
        threshold=0.01,
        sparse_array=True,
    )

    with Pool() as pool:
        weights = pool.map(get_wi_partial, self.fitted_indices)
    w_subset = scipy.sparse.vstack(weights)
    rows, cols = w_subset.nonzero()
    unique_indices = set(rows)
    names_all_neighbors = self.sample_names[unique_indices]
    subset = self.adata[self.adata.obs[self.group_key].isin(names_all_neighbors)]
    self.subsampled_indices = [self.sample_names.get_loc(name) for name in subset.obs_names]
    self.n_samples_subset = len(subset)

    self.neighboring_unsampled = None


# If model is not multiscale model, ensure all signaling type labels are the same in terms of the assumed
# length scale:
if hasattr(self, "self.signaling_types"):
    # If all features are assumed to operate on the same length scale, there should not be a mix of secreted
    # and membrane-bound-mediated signaling:
    if not self.multiscale_flag:
        # Secreted + ECM-receptor can diffuse across larger distances, but membrane-bound interactions are
        # limited by non-diffusivity. Therefore, it is not advisable to include a mixture of membrane-bound with
        # either of the other two categories in the same model.
        if ("Cell-Cell Contact" in set(self.signaling_types) and "Secreted Signaling" in set(self.signaling_types)) or (
            "Cell-Cell Contact" in set(self.signaling_types) and "ECM-Receptor" in set(self.signaling_types)
        ):
            raise ValueError(
                "It is not advisable to include a mixture of membrane-bound with either secreted or "
                "ECM-receptor in the same model because the valid distance scales over which they operate "
                "is different."
            )

        self.signaling_types = set(self.signaling_types)
        if "Secred Signaling" in self.signaling_types or "ECM-Receptor" in self.signaling_types:
            self.signaling_types = "Diffusive Signaling"
        else:
            self.signaling_types = "Cell-Cell Contact"
    self.signaling_types = self.comm.bcast(self.signaling_types, root=0)

if self.adata_path is not None:
    if signaling_type is None:
        signaling_type = self.signaling_types

    # Check whether the signaling types defined are membrane-bound or are composed of soluble molecules:
    if signaling_type == "Cell-Cell Contact":
        # Signaling is limited to occurring between only the nearest neighbors of each cell:
        if self.bw_fixed:
            distances = cdist(self.coords, self.coords)
            # Set max bandwidth to the average distance to the 20 nearest neighbors:
            nearest_idxs_all = np.argpartition(distances, 21, axis=1)[:, 1:21]
            nearest_distances = np.take_along_axis(distances, nearest_idxs_all, axis=1)
            self.maxbw = np.mean(nearest_distances, axis=1)

            if self.minbw is None:
                # Set min bandwidth to the average distance to the 5 nearest neighbors:
                nearest_idxs_all = np.argpartition(distances, 6, axis=1)[:, 1:6]
                nearest_distances = np.take_along_axis(distances, nearest_idxs_all, axis=1)
                self.minbw = np.mean(nearest_distances, axis=1)
        else:
            self.maxbw = 20

            if self.minbw is None:
                self.minbw = 5

        if self.minbw >= self.maxbw:
            raise ValueError(
                "The minimum bandwidth must be less than the maximum bandwidth. Please adjust the `minbw` "
                "parameter accordingly."
            )
        return

    # If the bandwidth is defined by a fixed spatial distance:
    if self.bw_fixed:
        max_dist = np.max(np.array([np.max(cdist([self.coords[i]], self.coords)) for i in range(self.n_samples)]))
        # Set max bandwidth higher to twice the max distance between any two given samples:
        self.maxbw = max_dist * 2

        # Set minimum bandwidth to ensure at least one "negative" example is included in spatially-weighted
        # calculation for each cell:
        if self.minbw is None:
            # Set minimum bandwidth to the distance to 3x the smallest distance between neighboring points:
            min_dist = np.min(
                np.array([np.min(np.delete(cdist(self.coords[[i]], self.coords), i)) for i in range(self.n_samples)])
            )
            self.minbw = min_dist * 3

    # If the bandwidth is defined by a fixed number of neighbors (and thus adaptive in terms of radius):
    else:
        if self.maxbw is None:
            self.maxbw = 100

        if self.minbw is None:
            self.minbw = 5

    if self.minbw >= self.maxbw:
        raise ValueError(
            "The minimum bandwidth must be less than the maximum bandwidth. Please adjust the `minbw` "
            "parameter accordingly."
        )

if self.adata_path is not None:
    if n_feat < len(self.signaling_types):
        signaling_type = self.signaling_types[n_feat]
    else:
        signaling_type = None
else:
    signaling_type = None


# Compute initial spatial weights for all samples- use the arbitrarily defined five times the min distance as
# initial bandwidth if not provided (for fixed bw) or 10 nearest neighbors (for adaptive bw):
if self.bw is None:
    if self.bw_fixed:
        init_bw = (
            np.min(
                np.array([np.min(np.delete(cdist([self.coords[i]], self.coords), 0)) for i in range(self.n_samples)])
            )
            * 5
        )
    else:
        init_bw = 10
else:
    init_bw = self.bw


(gene_counts_stats, gene_fano_params) = get_highvar_genes_sparse(self.adata.X, numgenes=2000)
high_variance_genes_filter = list(self.adata.var.index[gene_counts_stats.high_var.values])
self.targets_expr = pd.DataFrame(
    self.adata[:, high_variance_genes_filter].X.toarray()
    if scipy.sparse.issparse(self.adata.X)
    else self.adata[:, high_variance_genes_filter].X,
    index=self.sample_names,
    columns=high_variance_genes_filter,
)

# Make a note of whether ligands are secreted or membrane-bound:
self.signaling_types = self.lr_db.loc[self.lr_db["from"].isin([x[0] for x in self.lr_pairs]), "type"].tolist()


def huber_weights(self, fitted: np.ndarray, residuals: np.ndarray, threshold: float = None):
    """Compute Huber weights for the IWLS algorithm.

    Args:
        fitted: Array of shape [n_samples,]; transformed mean response variable
        residuals: Array of shape [n_samples,]; residuals between observed and predicted values
        threshold: Float; the threshold for switching between squared loss and linear loss

    Returns:
        w: Array of shape [n_samples,]; computed Huber weights for the IWLS algorithm
    """
    huber_w = np.ones_like(residuals)
    mask = np.abs(residuals) > threshold
    huber_w[mask] = threshold / np.abs(residuals[mask])

    w = huber_w / (self.link.deriv(fitted) ** 2 * self.variance(fitted))
    return w


betas, y_hat, _, _ = iwls(
    y_binary,
    X,
    distr="binomial",
    tol=self.tolerance,
    max_iter=self.max_iter,
    link=None,
    ridge_lambda=self.ridge_lambda,
)

# Zero-inflated local logistic model:
if not self.no_hurdle:
    tf.random.set_seed(888)
    y_binary = np.where(y > 0, 1, 0).reshape(-1, 1).astype(np.float32)
    # Network architecture:
    model = tf.keras.Sequential()
    layer_dims = []
    if X.shape[1] >= 8:
        layer_dims.append(np.ceil(X.shape[1] / 4))
        while layer_dims[-1] / 4 > 8:
            layer_dims.append(layer_dims[-1] / 4)
        layer_dims.append(1)
    else:
        layer_dims.append(1)

    if len(layer_dims) > 1:
        model.add(tf.keras.layers.Dense(layer_dims[0], activation="relu", input_shape=(X.shape[1],)))
        for layer_dim in layer_dims[1:-1]:
            model.add(tf.keras.layers.Dense(layer_dim, activation="relu"))
        model.add(tf.keras.layers.Dense(layer_dims[-1], activation="sigmoid"))
    else:
        model.add(tf.keras.layers.Dense(layer_dims[0], activation="sigmoid", input_shape=(X.shape[1],)))

    # Compile model:
    proportion_true_pos = tf.reduce_mean(y_binary)
    proportion_true_neg = 1 - proportion_true_pos
    model.compile(
        optimizer="adam",
        loss=lambda y_binary, y_pred: weighted_binary_crossentropy(
            y_binary, y_pred, weight_0=proportion_true_pos, weight_1=proportion_true_neg
        ),
        metrics=["accuracy"],
    )

    model.fit(X, y, epochs=100, batch_size=8, verbose=0)
    predictions = model.predict(X)
    obj_function = lambda threshold: logistic_objective(threshold=threshold, proba=predictions, y_true=y_binary)
    optimal_threshold = golden_section_search(obj_function, a=0.0, b=1.0, tol=self.tolerance)
    pred_binary = (predictions >= optimal_threshold).astype(int)

    log_pred_v_true = np.hstack((predictions, pred_binary, y_binary, y))

    if not os.path.exists(os.path.join(os.path.dirname(self.output_path), "logistic_predictions")):
        os.makedirs(os.path.join(os.path.dirname(self.output_path), "logistic_predictions"))
    predictions_df = pd.DataFrame(
        log_pred_v_true,
        columns=["predictions", "predicted_binary_value", "true binary value", "true expression"],
        index=self.sample_names[self.x_chunk],
    )
    predictions_df.to_csv(
        os.path.join(os.path.dirname(self.output_path), f"logistic_predictions/logistic_predictions_{target}.csv")
    )

    # Mask indices where variable is predicted to be nonzero but is actually zero (based on inferences,
    # these will result in likely underestimation of the effect)- we infer that the effect of the
    # independent variables is similar for these observations (and the zero is either technical or due to
    # an external factor):
    mask_indices = np.where((pred_binary != 0) & (y == 0))[0]
else:
    "filler"


#     if self.subsampled:
#         sample_index = (
#             self.subsampled_indices[y_label][i] if not self.subset else self.subsampled_indices[i]
#         )
#     elif self.subset:
#         sample_index = self.subset_indices[i]
#     else:
#         sample_index = i

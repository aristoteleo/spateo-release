"""
Auxiliary functions to aid in the interpretation functions for the spatial and spatially-lagged regression models.
"""
import os
from typing import Iterable, Optional, Tuple, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import statsmodels.stats.multitest
from anndata import AnnData
from numpy import linalg
from scipy import interpolate
from scipy.signal import argrelextrema
from scipy.sparse import csgraph
from sklearn.cluster import KMeans, SpectralClustering
from statsmodels.stats.outliers_influence import variance_inflation_factor

from ...configuration import SKM
from ...logging import logger_manager as lm
from ...preprocessing.transform import log1p
from ..find_neighbors import calculate_affinity


# ---------------------------------------------------------------------------------------------------
# Nonlinearity
# ---------------------------------------------------------------------------------------------------
def softplus(z):
    """Numerically stable version of log(1 + exp(z))."""
    nl = z.copy()
    nl[z > 35] = z[z > 35]
    nl[z < -10] = np.exp(z[z < -10])
    nl[(z >= -10) & (z <= 35)] = log1p(np.exp(z[(z >= -10) & (z <= 35)]))
    return nl


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


# ---------------------------------------------------------------------------------------------------
# Check multicollinearity
# ---------------------------------------------------------------------------------------------------
def multicollinearity_check(X: pd.DataFrame, thresh: float = 5.0):
    """Checks for multicollinearity in dependent variable array, and drops the most multicollinear features until
    all features have VIF less than a given threshold.

    Args:
        X: Dependent variable array, in dataframe format
        thresh: VIF threshold; features with values greater than this value will be removed from the regression

    Returns:
        X: Dependent variable array following filtering
    """
    logger = lm.get_main_logger()

    int_cols = X.select_dtypes(
        include=["int", "int16", "int32", "int64", "float", "float16", "float32", "float64"]
    ).shape[1]
    total_cols = X.shape[1]

    if int_cols != total_cols:
        logger.error("All columns should be integer or float.")
    else:
        variables = list(range(X.shape[1]))
        dropped = True
        logger.info(
            f"Iterating through features and calculating respective variance inflation factors (VIF). Will "
            "iteratively drop the highest VIF features until all features have VIF less than the threshold "
            "value of {thresh}"
        )
        while dropped:
            dropped = False
            vif = [variance_inflation_factor(X.iloc[:, variables].values, ix) for ix in variables]
            maxloc = vif.index(max(vif))
            if max(vif) > thresh:
                logger.info("Dropping '" + X.iloc[:, variables].columns[maxloc] + "' at index: " + str(maxloc))
                X.drop(X.columns[variables[maxloc]], 1, inplace=True)
                variables = list(range(X.shape[1]))
                dropped = True

        logger.info(f"\n\nRemaining variables:\n {X.columns[variables]}")
        return X


# ---------------------------------------------------------------------------------------------------
# Regularization
# ---------------------------------------------------------------------------------------------------
def L1_penalty(beta: np.ndarray) -> float:
    """
    Implementation of the L1 penalty that penalizes based on absolute value of coefficient magnitude.

    Args:
        beta: Array of shape [n_features,]; learned model coefficients

    Returns:
        L1penalty: float, value for the regularization parameter (typically stylized by lambda)
    """
    # Lasso-like penalty- max(sum(abs(beta), axis=0))
    L1penalty = np.linalg.norm(beta, 1)
    return L1penalty


def L2_penalty(beta: np.ndarray, Tau: Union[None, np.ndarray] = None) -> float:
    """Implementation of the L2 penalty that penalizes based on the square of coefficient magnitudes.

    Args:
        beta: Array of shape [n_features,]; learned model coefficients
        Tau: optional array of shape [n_features, n_features]; the Tikhonov matrix for ridge regression. If not
        provided, Tau will default to the identity matrix.
    """
    if Tau is None:
        # Ridge=like penalty
        L2penalty = np.linalg.norm(beta, 2) ** 2
    else:
        # Tikhonov penalty
        if Tau.shape[0] != beta.shape[0] or Tau.shape[1] != beta.shape[0]:
            raise ValueError("Tau should be (n_features x n_features)")
        else:
            L2penalty = np.linalg.norm(np.dot(Tau, beta), 2) ** 2

    return L2penalty


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


# ---------------------------------------------------------------------------------------------------
# Significance Testing
# ---------------------------------------------------------------------------------------------------
def get_fisher_inverse(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Computes the Fisher matrix that measures the amount of information each feature in x provides about y- that is,
    whether the log-likelihood is sensitive to change in the parameter x.

    Function from diffxpy: https://github.com/theislab/diffxpy

    Args:
        x: Independent variable array
        y: Dependent variable array

    Returns:
        inverse_fisher : np.ndarray
    """

    var = np.var(y, axis=0)
    fisher = np.expand_dims(np.matmul(x.T, x), axis=0) / np.expand_dims(var, axis=[1, 2])

    fisher = np.nan_to_num(fisher)

    inverse_fisher = np.array([np.linalg.pinv(fisher[i, :, :]) for i in range(fisher.shape[0])])
    return inverse_fisher


def wald_test(theta_mle: np.ndarray, theta_sd: np.ndarray, theta0: Union[float, np.ndarray] = 0) -> np.ndarray:
    """Perform single-coefficient Wald test, informing whether a given coefficient deviates significantly from the
    supplied reference value (theta0), based on the standard deviation of the posterior of the parameter estimate.

    Function from diffxpy: https://github.com/theislab/diffxpy

    Args:
        theta_mle: Maximum likelihood estimation of given parameter by feature
        theta_sd: Standard deviation of the maximum likelihood estimation
        theta0: Value(s) to test theta_mle against. Must be either a single number or an array w/ equal number of
            entries to theta_mle.

    Returns:
        pvals : np.ndarray
    """

    if np.size(theta0) == 1:
        theta0 = np.broadcast_to(theta0, theta_mle.shape)

    if theta_mle.shape[0] != theta_sd.shape[0]:
        raise ValueError("stats.wald_test(): theta_mle and theta_sd have to contain the same number of entries")
    if theta0.shape[0] > 1:
        if theta_mle.shape[0] != theta0.shape[0]:
            raise ValueError("stats.wald_test(): theta_mle and theta0 have to contain the same number of entries")

    theta_sd = np.nextafter(0, np.inf, out=theta_sd, where=theta_sd < np.nextafter(0, np.inf))
    wald_statistic = np.abs(np.divide(theta_mle - theta0, theta_sd))
    pvals = 2 * (1 - scipy.stats.norm(loc=0, scale=1).cdf(wald_statistic))  # two-tailed test
    return pvals


def multitesting_correction(pvals: np.ndarray, method: str = "fdr_bh", alpha: float = 0.05) -> np.ndarray:
    """In the case of testing multiple hypotheses from the same experiment, perform multiple test correction to adjust
    q-values.

    Function from diffxpy: https://github.com/theislab/diffxpy

    Args:
    pvals: Uncorrected p-values; must be given as a one-dimensional array
    method: Method to use for correction. Available methods can be found in the documentation for
        statsmodels.stats.multitest.multipletests(), and are also listed below (in correct case) for convenience:
            - Named methods:
                - bonferroni
                - sidak
                - holm-sidak
                - holm
                - simes-hochberg
                - hommel
            - Abbreviated methods:
                - fdr_bh: Benjamini-Hochberg correction
                - fdr_by: Benjamini-Yekutieli correction
                - fdr_tsbh: Two-stage Benjamini-Hochberg
                - fdr_tsbky: Two-stage Benjamini-Krieger-Yekutieli method
    alpha: Family-wise error rate (FWER)

    Returns
        qval: p-values post-correction
    """

    qval = np.zeros([pvals.shape[0]]) + np.nan
    qval[np.isnan(pvals) == False] = statsmodels.stats.multitest.multipletests(
        pvals=pvals[np.isnan(pvals) == False], alpha=alpha, method=method, is_sorted=False, returnsorted=False
    )[1]

    return qval


def get_p_value(variables: np.array, fisher_inv: np.array, coef_loc: int) -> np.ndarray:
    """Computes p-value for differential expression for a target feature

    Function from diffxpy: https://github.com/theislab/diffxpy

    Args:
        variables: Array where each column corresponds to a feature
        fisher_inv: Inverse Fisher information matrix
        coef_loc: Numerical column of the array corresponding to the coefficient to test

    Returns:
        pvalues: Array of identical shape to variables, where each element is a p-value for that instance of that
            feature
    """

    theta_mle = variables[coef_loc]
    theta_sd = fisher_inv[:, coef_loc, coef_loc]
    theta_sd = np.nextafter(0, np.inf, out=theta_sd, where=theta_sd < np.nextafter(0, np.inf))
    theta_sd = np.sqrt(theta_sd)

    pvalues = wald_test(theta_mle, theta_sd, theta0=0.0)
    return pvalues


def compute_wald_test(
    params: np.ndarray, fisher_inv: np.ndarray, significance_threshold: float = 0.01
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Function from diffxpy: https://github.com/theislab/diffxpy

    Args:
        params: Array of shape [n_features, n_params]
        fisher_inv: Inverse Fisher information matrix
        significance_threshold: Upper threshold to be considered significant

    Returns:
        significance: Array of identical shape to variables, where each element is True or False if it meets the
            threshold for significance
        pvalues: Array of identical shape to variables, where each element is a p-value for that instance of that
            feature
        qvalues: Array of identical shape to variables, where each element is a q-value for that instance of that
            feature
    """

    pvalues = []

    # Compute p-values for each feature, store in temporary list:
    for idx in range(params.T.shape[0]):
        pvals = get_p_value(params.T, fisher_inv, idx)
        pvalues.append(pvals)

    pvalues = np.concatenate(pvalues)
    # Multiple testing correction w/ Benjamini-Hochberg procedure and FWER 0.05
    qvalues = multitesting_correction(pvalues)
    pvalues = np.reshape(pvalues, (-1, params.T.shape[1]))
    qvalues = np.reshape(qvalues, (-1, params.T.shape[1]))
    significance = qvalues < significance_threshold

    return significance, pvalues, qvalues


# ---------------------------------------------------------------------------------------------------
# Regression Metrics
# ---------------------------------------------------------------------------------------------------
def mae(y_true, y_pred) -> float:
    """Mean absolute error- in this context, actually log1p mean absolute error

    Args:
        y_true: Regression model output
        y_pred: Observed values for the dependent variable

    Returns:
        mae: Mean absolute error value across all samples
    """
    abs = np.abs(y_true - y_pred)
    mean = np.mean(abs)
    return mean


def mse(y_true, y_pred) -> float:
    """Mean squared error- in this context, actually log1p mean squared error

    Args:
        y_true: Regression model output
        y_pred: Observed values for the dependent variable

    Returns:
        mse: Mean squared error value across all samples
    """
    se = np.square(y_true - y_pred)
    se = np.mean(se, axis=-1)
    return se


# ---------------------------------------------------------------------------------------------------
# Testing Model Accuracy
# ---------------------------------------------------------------------------------------------------
@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE, "adata")
def plot_prior_vs_data(
    reconst: pd.DataFrame,
    adata: AnnData,
    kind: str = "barplot",
    target_name: Union[None, str] = None,
    title: Union[None, str] = None,
    figsize: Union[None, Tuple[float, float]] = None,
    save_show_or_return: Literal["save", "show", "return", "both", "all"] = "save",
    save_kwargs: dict = {},
):
    """Plots distribution of observed vs. predicted counts in the form of a comparative density barplot.

    Args:
        reconst: DataFrame containing values for reconstruction/prediction of targets of a regression model
        adata: AnnData object containing observed counts
        kind: Kind of plot to generate. Options: "barplot", "scatterplot". Case sensitive, defaults to "barplot".
        target_name: Optional, can be:
                - Column name in DataFrame/AnnData object: name of gene to subset to
                - "sum": computes sum over all features present in 'reconst' to compare to the corresponding subset of
                'adata'.
                - "mean": computes mean over all features present in 'reconst' to compare to the corresponding subset of
                'adata'.
            If not given, will subset AnnData to features in 'reconst' and flatten both arrays to compare all values.

            If not given, will compute the sum over all
            features present in 'reconst' and compare to the corresponding subset of 'adata'.
        save_show_or_return: Whether to save, show or return the figure.
            If "both", it will save and plot the figure at the same time. If "all", the figure will be saved,
            displayed and the associated axis and other object will be return.
        save_kwargs: A dictionary that will passed to the save_fig function.
            By default it is an empty dictionary and the save_fig function will use the
            {"path": None, "prefix": 'scatter', "dpi": None, "ext": 'pdf', "transparent": True, "close": True,
            "verbose": True} as its parameters. Otherwise you can provide a dictionary that properly modifies those
            keys according to your needs.
    """
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    from ...configuration import config_spateo_rcParams
    from ...plotting.static.utils import save_return_show_fig_utils

    logger = lm.get_main_logger()

    config_spateo_rcParams()
    if figsize is None:
        figsize = rcParams.get("figure.figsize")

    if target_name == "sum":
        predicted = reconst.sum(axis=1).values.reshape(-1, 1)
        observed = (
            adata[:, reconst.columns].X.toarray() if scipy.sparse.issparse(adata.X) else adata[:, reconst.columns].X
        )
        observed = np.sum(observed, axis=1).reshape(-1, 1)
    elif target_name == "mean":
        predicted = reconst.mean(axis=1).values.reshape(-1, 1)
        observed = (
            adata[:, reconst.columns].X.toarray() if scipy.sparse.issparse(adata.X) else adata[:, reconst.columns].X
        )
        observed = np.mean(observed, axis=1).reshape(-1, 1)
    elif target_name is not None:
        observed = adata[:, target_name].X.toarray() if scipy.sparse.issparse(adata.X) else adata[:, target_name].X
        observed = observed.reshape(-1, 1)
        predicted = reconst[target_name].values.reshape(-1, 1)
    else:
        # Flatten arrays:
        observed = (
            adata[:, reconst.columns].X.toarray() if scipy.sparse.issparse(adata.X) else adata[:, reconst.columns].X
        )
        observed = observed.flatten().reshape(-1, 1)
        predicted = reconst.values.flatten().reshape(-1, 1)

    obs_pred = np.hstack((observed, predicted))
    # Upper limit along the x-axis (99th percentile to prevent outliers from affecting scale too badly):
    xmax = np.percentile(obs_pred, 99)
    # Lower limit along the x-axis:
    xmin = np.min(observed)
    # Divide x-axis into pieces for purposes of setting x labels:
    xrange, step = np.linspace(xmin, xmax, num=10, retstep=True)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    if target_name is None:
        target_name = "Total Counts"

    if kind == "barplot":
        ax.hist(
            obs_pred,
            xrange,
            alpha=0.7,
            label=[f"Observed {target_name}", f"Predicted {target_name}"],
            density=True,
            color=["#FFA07A", "#20B2AA"],
        )

        plt.legend(loc="upper right", fontsize=9)

        ax.set_xticks(ticks=[i + 0.5 * step for i in xrange[:-1]], labels=[np.round(l, 3) for l in xrange[:-1]])
        plt.xlabel("Counts", size=9)
        plt.ylabel("Normalized Proportion of Cells", size=9)
        if title is not None:
            plt.title(title, size=9)
        plt.tight_layout()

    elif kind == "scatterplot":
        from scipy.stats import spearmanr

        observed = observed.flatten()
        predicted = predicted.flatten()
        slope, intercept = np.polyfit(observed, predicted, 1)

        # Extract residuals:
        predicted_model = np.polyval([slope, intercept], observed)
        observed_mean = np.mean(observed)
        predicted_mean = np.mean(predicted)
        n = observed.size  # number of samples
        m = 2  # number of parameters
        dof = n - m  # degrees of freedom
        # Students statistic of interval confidence:
        t = scipy.stats.t.ppf(0.975, dof)
        residual = observed - predicted_model
        # Standard deviation of the error:
        std_error = (np.sum(residual**2) / dof) ** 0.5

        # Calculate spearman correlation and coefficient of determination:
        s = spearmanr(observed, predicted)[0]
        numerator = np.sum((observed - observed_mean) * (predicted - predicted_mean))
        denominator = (np.sum((observed - observed_mean) ** 2) * np.sum((predicted - predicted_mean) ** 2)) ** 0.5
        correlation_coef = numerator / denominator
        r2 = correlation_coef**2

        # Plot best fit line:
        observed_line = np.linspace(np.min(observed), np.max(observed), 100)
        predicted_line = np.polyval([slope, intercept], observed_line)

        # Confidence interval and prediction interval:
        ci = (
            t
            * std_error
            * (1 / n + (observed_line - observed_mean) ** 2 / np.sum((observed - observed_mean) ** 2)) ** 0.5
        )
        pi = (
            t
            * std_error
            * (1 + 1 / n + (observed_line - observed_mean) ** 2 / np.sum((observed - observed_mean) ** 2)) ** 0.5
        )

        ax.plot(observed, predicted, "o", ms=3, color="royalblue", alpha=0.7)
        ax.plot(observed_line, predicted_line, color="royalblue", alpha=0.7)
        ax.fill_between(
            observed_line, predicted_line + pi, predicted_line - pi, color="lightcyan", label="95% prediction interval"
        )
        ax.fill_between(
            observed_line, predicted_line + ci, predicted_line - ci, color="skyblue", label="95% confidence interval"
        )
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        ax.set_xlabel(f"Observed {target_name}")
        ax.set_ylabel(f"Predicted {target_name}")
        title = title if title is not None else "Observed and Predicted {}".format(target_name)
        ax.set_title(title)

        # Display r^2, Spearman correlation, mean absolute error on plot as well:
        r2s = str(np.round(r2, 2))
        spearman = str(np.round(s, 2))
        ma_err = mae(observed, predicted)
        mae_s = str(np.round(ma_err, 2))

        # Place text at slightly above the minimum x_line value and maximum y_line value to avoid obscuring the plot:
        ax.text(
            1.01 * np.min(observed),
            1.01 * np.max(predicted),
            "$r^2$ = " + r2s + ", Spearman $r$ = " + spearman + ", MAE = " + mae_s,
            fontsize=8,
        )
        plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.4), fontsize=8)

    else:
        logger.info(
            ":func `plot_prior_vs_data` error: Invalid input given to 'kind'. Options: 'barplot', " "'scatterplot'."
        )

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

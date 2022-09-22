"""
Auxiliary functions to aid in the interpretation functions for the spatial and spatially-lagged regression models.
"""
from typing import Tuple, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np
import pandas as pd
import scipy
import sklearn
import statsmodels.stats.multitest
from anndata import AnnData

from ...configuration import SKM
from ...logging import logger_manager as lm


# ------------------------------------------- Significance Testing ------------------------------------------- #
def get_fisher_inverse(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Computes the Fisher matrix that measures the amount of information each feature in x provides about y- that is,
    whether the log-likelihood is sensitive to change in the parameter x.

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
    """
    Perform single-coefficient Wald test, informing whether a given coefficient deviates significantly from the
    supplied reference value (theta0), based on the standard deviation of the posterior of the parameter estimate.

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
    """
    In the case of testing multiple hypotheses from the same experiment, perform multiple test correction to adjust
    q-values.

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


def _get_p_value(variables: np.array, fisher_inv: np.array, coef_loc_totest: int) -> np.ndarray:
    """
    Computes p-values for differential expression for each feature

    Args:
        variables: Array where each column corresponds to a feature
        fisher_inv: Inverse Fisher information matrix
        coef_loc_totest: Numerical column of the array corresponding to the coefficient to test

    Returns:
        pvalues: Array of identical shape to variables, where each element is a p-value for that instance of that
            feature
    """

    theta_mle = variables[coef_loc_totest]
    theta_sd = fisher_inv[:, coef_loc_totest, coef_loc_totest]
    theta_sd = np.nextafter(0, np.inf, out=theta_sd, where=theta_sd < np.nextafter(0, np.inf))
    theta_sd = np.sqrt(theta_sd)

    pvalues = wald_test(theta_mle, theta_sd, theta0=0.0)
    return pvalues


def compute_wald_test(
    params: np.ndarray, fisher_inv: np.ndarray, significance_threshold: float = 0.01
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Args:
        params: Array of shape [n_features, n_params]
        fisher_inv: Inverse Fisher information matrix
        significance_threshold: Upper threshold to be considered significant

    Returns
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
        pvals = _get_p_value(params.T, fisher_inv, idx)
        pvalues.append(pvals)

    pvalues = np.concatenate(pvalues)
    # Multiple testing correction w/ Benjamini-Hochberg procedure and FWER 0.05
    qvalues = multitesting_correction(pvalues)
    pvalues = np.reshape(pvalues, (-1, params.T.shape[1]))
    qvalues = np.reshape(qvalues, (-1, params.T.shape[1]))
    significance = qvalues < significance_threshold

    return significance, pvalues, qvalues


# ------------------------------------------- Comparison ------------------------------------------- #
@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE, "adata")
def plot_prior_vs_data(
    reconst: pd.DataFrame,
    adata: AnnData,
    target_name: Union[None, str] = None,
    title: Union[None, str] = None,
    figsize: Union[None, Tuple[float, float]] = None,
    save_show_or_return: Literal["save", "show", "return", "both", "all"] = "save",
    save_kwargs: dict = {},
):
    """
    For diagnostics, plots distribution of observed vs. predicted counts. Note that this is most effective for counts
    after a log transformation. Assumed

    Args:
        reconst: DataFrame containing values for reconstruction/prediction of targets of a regression model
        adata: AnnData object containing observed counts
        target_name: Optional, name of the column in the DataFrame/variable name in the AnnData object corresponding
            to the target gene. Assumed to be the same in both if given. If not given, will compute the mean over all
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

    config_spateo_rcParams()
    if figsize is None:
        figsize = rcParams.get("figure.figsize")

    if target_name is not None:
        observed = adata[:, target_name].X.toarray() if scipy.sparse.issparse(adata.X) else adata[:, target_name].X
        observed = observed.reshape(-1, 1)
        predicted = reconst[target_name].values.reshape(-1, 1)
    else:
        predicted = reconst.mean(axis=1).values.reshape(-1, 1)
        observed = (
            adata[:, reconst.columns].X.toarray() if scipy.sparse.issparse(adata.X) else adata[:, reconst.columns].X
        )
        observed = np.mean(observed, axis=1).reshape(-1, 1)

    obs_pred = np.hstack((observed, predicted))
    # Upper limit along the x-axis (99th percentile to prevent outliers from affecting scale too badly):
    xmax = np.percentile(obs_pred, 99)
    # Divide x-axis into pieces for purposes of setting x labels:
    xrange, step = np.linspace(0, xmax, num=10, retstep=True)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
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

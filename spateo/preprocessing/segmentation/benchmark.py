"""Functions to help segmentation benchmarking, specifically to compare
two sets of segmentation labels.
"""
from typing import Tuple

import numpy as np
import pandas as pd
from anndata import AnnData
from sklearn import metrics

from ...configuration import SKM
from ...logging import logger_manager as lm


def adjusted_rand_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the Adjusted Rand Score (ARS).

    Re-implementation to deal with over/underflow that is common with large
    datasets.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Adjusted Rand Score
    """
    (tn, fp), (fn, tp) = metrics.pair_confusion_matrix(y_true, y_pred)
    tn, tp, fp, fn = int(tn), int(tp), int(fp), int(fn)
    if fn == 0 and fp == 0:
        return 1.0
    return 2.0 * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))


def classification_stats(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Tuple[float, float, float, float, float, float, float]:
    """Calculate pixel classification statistics by considering labeled pixels
    as occupied (1) and unlabled pixels as unoccupied (0).

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        A 7-element tuple containing the following values:
            * true negative rate
            * false positive rate
            * false negative rate
            * true positive rate (a.k.a. recall)
            * precision
            * accuracy
            * F1 score
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    y_true_bool = y_true > 0
    y_pred_bool = y_pred > 0
    pos = y_true_bool.sum()
    neg = (~y_true_bool).sum()

    tn, fp, fn, tp = metrics.confusion_matrix(y_true_bool, y_pred_bool).ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return (tn / neg, fp / neg, fn / pos, recall, precision, accuracy, f1)


def labeling_stats(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float, float]:
    """Calculate labeling (cluster) statistics.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        A 4-element tuple containing the following values:
            * adjusted rand score
            * homogeneity
            * completeness
            * v score
    """
    ars = adjusted_rand_score(y_true, y_pred)
    homogeneity, completeness, v = metrics.homogeneity_completeness_v_measure(y_true, y_pred)
    return ars, homogeneity, completeness, v


def compare(
    adata: AnnData, true_layer: str, pred_layer: str, data_layer: str = SKM.X_LAYER, umi_pixels_only: bool = True
) -> pd.DataFrame:
    """Compute segmentation statistics.

    Args:
        adata: Input Anndata
        true_layer: Layer containing true labels
        pred_layer: Layer containing predicted labels
        data_layer: Layer containing UMIs
        umi_pixels_only: Whether or not to only consider pixels that have at least
            one UMI captured (as determined by `data_layer`).

    Returns:
        Pandas DataFrame containing classification and labeling statistics
    """
    y_true = SKM.select_layer_data(adata, true_layer)
    y_pred = SKM.select_layer_data(adata, pred_layer)

    if umi_pixels_only:
        lm.main_info("Ignoring pixels with zero detected UMIs.")
        X = SKM.select_layer_data(adata, data_layer, make_dense=True)
        umi_mask = X > 0
        y_true = y_true[umi_mask]
        y_pred = y_pred[umi_mask]

    lm.main_info("Computing classification statistics.")
    tn, fp, fn, tp, precision, accuracy, f1 = classification_stats(y_true, y_pred)
    lm.main_info("Computing label statistics.")
    both_labeled = (y_true > 0) & (y_pred > 0)
    ars, homogeneity, completeness, v = labeling_stats(y_true[both_labeled], y_pred[both_labeled])
    return pd.DataFrame(
        {"value": [tn, fp, fn, tp, precision, accuracy, f1, ars, homogeneity, completeness, v]},
        index=[
            "True negative",
            "False positive",
            "False negative",
            "True positive",
            "Precision",
            "Accuracy",
            "F1 score",
            "Adjusted rand score",
            "Homogeneity",
            "Completeness",
            "V measure",
        ],
    )

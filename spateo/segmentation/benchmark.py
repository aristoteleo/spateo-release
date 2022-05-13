"""Functions to help segmentation benchmarking, specifically to compare
two sets of segmentation labels.
"""
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse
from sklearn import metrics

from ..configuration import SKM
from ..logging import logger_manager as lm
from . import utils
from .qc import _generate_random_labels


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


def iou(labels1: np.ndarray, labels2: np.ndarray) -> sparse.csr_matrix:
    """Compute intersection-over-union (IOU).

    Args:
        labels1: First set of labels
        labels2: Second set of labels

    Returns:
        Sparse matrix where the first axis corresponds to the first set of
            labels and vice-versa.
    """
    areas1 = np.bincount(labels1.flatten())
    areas2 = np.bincount(labels2.flatten())
    overlaps = utils.label_overlap(labels1, labels2).astype(float)
    for i, j in zip(*overlaps.nonzero()):
        overlap = overlaps[i, j]
        overlaps[i, j] = overlap / (areas1[i] + areas2[j] - overlap)
    return overlaps


def average_precision(iou: sparse.csr_matrix, tau: float = 0.5) -> float:
    """Compute average precision (AP).

    Args:
        iou: IOU of true and predicted labels
        tau: IOU threshold to determine whether a prediction is correct

    Returns:
        Average precision
    """
    tp = (iou > tau).sum()
    fp = iou.shape[1] - tp - 1
    fn = iou.shape[0] - tp - 1
    return tp / (tp + fn + fp)


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


@SKM.check_adata_is_type(SKM.ADATA_AGG_TYPE)
def compare(
    adata: AnnData,
    true_layer: str,
    pred_layer: str,
    data_layer: str = SKM.X_LAYER,
    umi_pixels_only: bool = True,
    random_background: bool = True,
    ap_taus: Tuple[int, ...] = tuple(np.arange(0.5, 1, 0.05)),
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Compute segmentation statistics.

    Args:
        adata: Input Anndata
        true_layer: Layer containing true labels
        pred_layer: Layer containing predicted labels
        data_layer: Layer containing UMIs
        umi_pixels_only: Whether or not to only consider pixels that have at least
            one UMI captured (as determined by `data_layer`).
        random_background: Simulate random background by randomly permuting the
            `pred_layer` labels and computing the same statistics against
            `true_layer`. The returned DataFrame will have an additional column
            for these statistics.
        ap_taus: Tau thresholds to calculate average precision. Defaults to
            0.05 increments starting at 0.5 and ending at (and including) 0.95.
        seed: Random seed.

    Returns:
        Pandas DataFrame containing classification and labeling statistics
    """

    def _stats(y_true, y_pred):
        tn, fp, fn, tp, precision, accuracy, f1 = classification_stats(y_true, y_pred)
        both_labeled = (y_true > 0) & (y_pred > 0)
        ars, homogeneity, completeness, v = labeling_stats(y_true[both_labeled], y_pred[both_labeled])
        return tn, fp, fn, tp, precision, accuracy, f1, ars, homogeneity, completeness, v

    def _ap(y_true, y_pred, taus):
        aps = []
        _iou = None
        for tau in taus:
            if _iou is None:
                _iou = iou(y_true, y_pred)
            aps.append(average_precision(_iou, tau))
        return aps

    y_true = SKM.select_layer_data(adata, true_layer)
    y_pred = SKM.select_layer_data(adata, pred_layer)

    if umi_pixels_only:
        lm.main_debug("Ignoring pixels with zero detected UMIs.")
        X = SKM.select_layer_data(adata, data_layer, make_dense=True)
        umi_mask = X > 0
        y_true = y_true[umi_mask]
        y_pred = y_pred[umi_mask]

    lm.main_info("Computing statistics.")
    pred_stats = list(_stats(y_true, y_pred)) + _ap(y_true, y_pred, ap_taus)
    data = {pred_layer: pred_stats}
    if random_background:
        lm.main_info("Computing background statistics.")
        bincount = np.bincount(y_pred.flatten())
        y_random = _generate_random_labels(y_pred.shape, bincount[1:], seed)

        random_stats = list(_stats(y_true, y_random)) + _ap(y_true, y_random, ap_taus)
        data["background"] = random_stats
    return pd.DataFrame(
        data,
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
        ]
        + [f"Average precision ({tau:.2f})" for tau in ap_taus],
    )

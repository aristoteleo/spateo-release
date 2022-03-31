"""Functions to help segmentation benchmarking, specifically to compare
two sets of segmentation labels.
"""
from typing import Tuple

import numpy as np
from sklearn import metrics


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

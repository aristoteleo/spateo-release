"""
Auxiliary functions to aid in the interpretation functions for the spatially-lagged regression model.
"""
import numpy as np

def get_fisher_inverse(x: np.ndarray, y: np.ndarray):
    """
    Computes the Fisher matrix that measures the amount of information each feature in x provides about y- that is,
    whether the log-likelihood is sensitive to change in the parameter x.

    Args:
        x : np.ndarray
            Independent variable array
        y : np.ndarray
            Dependent variable array

    Returns:
        inverse_fisher : np.ndarray
    """
    var = np.var(y, axis=0)
    fisher = np.expand_dims(np.matmul(x.T, x), axis=0) / np.expand_dims(var, axis=[1, 2])

    fisher = np.nan_to_num(fisher)

    inverse_fisher = np.array([
        np.linalg.pinv(fisher[i, :, :])
        for i in range(fisher.shape[0])
    ])
    return inverse_fisher


def _get_p_value(variables: np.array, fisher_inv: np.array, coef_loc_totest: int):
    """
    Computes p-value for differential expression

    Args:
        variables : np.ndarray

        fisher_inv : np.ndarray
            Inverse Fisher information matrix
        coef_loc_totest : int
            Numerical column of the array corresponding to the coefficient to test
    :return:
    """
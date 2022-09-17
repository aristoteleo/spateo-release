"""
Spatial lag model for the purpose of predicting and quantifying ligand:receptor interactions.

Builds from the implementation in spreg: https://spreg.readthedocs.io/en/latest/,
Authors: Luc Anselin, David C. Folch
"""
from typing import List, Union

import numpy as np
import pandas as pd
import spreg.summary_output as SUMMARY
from spreg import user_output as USER
from spreg.sputils import sphstack
from spreg.twosls import BaseTSLS
from spreg.utils import get_lags, lag_spatial, set_warn, sp_att

from ...logging import logger_manager as lm


def set_endog(
    y: np.ndarray,
    x: np.ndarray,
    lag_idx: List[int],
    w: "libpysal.weights.W",
    yend: Union[None, np.ndarray] = None,
    q: Union[None, np.ndarray] = None,
    w_lags: int = 1,
    lag_q: bool = False,
):
    """
    Computes spatial lag for the selected independent variables and set them as exogenous variables.

    Args:
        y : np.ndarray
            nx1 array for dependent variable
        x : np.ndarray
            Two dimensional array containing all independent variable values, excluding the constant term
        lag_idx : list of int
            Columnar indices corresponding to columns of X for which to compute spatial lag
        w : pysal W object
            Spatial weights object
        yend : optional np.ndarray
            Two dimensional array with n rows and one column for each endogenous variable
        q : optional np.ndarray
            Two dimensional array with n rows and one column for each external exogenous variable to use as
            instruments (note: this should not contain any variables from x)
        w_lags : int, default 1
            Order(s) of W to include as instruments for the spatially lagged dependent variable. For example,
            w_lags=1, then instruments are WX; if w_lags=2, then WX, WWX; and so on.
        lag_q : bool, default False
            If True, then include spatial lags of the additional instruments (q)
    """
    logger = lm.get_main_logger()

    # Create spatial lag of y
    yl = lag_spatial(w, y)

    # If separate yend is provided:
    if issubclass(type(yend), np.ndarray):
        if lag_q:
            lag_vars = sphstack(x[:, lag_idx], q)
        else:
            lag_vars = x[:, lag_idx]
        spatial_inst = get_lags(w, lag_vars, w_lags)
        q = sphstack(q, spatial_inst)
        yend = sphstack(yend, yl)

    # If the only endogenous variable is the dependent lag:
    elif yend == None:
        x = x[:, lag_idx]
        q = get_lags(w, x, w_lags)
        yend = yl

    else:
        logger.error("Invalid value passed to 'yend'.")

    return yend, q


class LR_BaseGM_Lag(BaseTSLS):
    """
    Spatial two stage least squares (S2SLS) (note: no consistency checks, diagnostics or constant added);
    Original implementation from Anselin, L. (1988). Spatial econometrics: methods and models (Vol. 4). Springer
    Science & Business Media.

    See documentation of :class `spreg.BaseGM_Lag` (https://github.com/pysal/spreg/blob/master/spreg/twosls_sp.py) for
    original docstring.

    Changes: takes dataframe as an argument and column names to define x, y, and which features spatial lag
    computation is applied to.

    Args:
        df : pd.DataFrame
            Dataframe containing dependent variable and all independent variables
        y_col : str
            Name of column corresponding to the dependent variable. Will assume all other columns contain independent
            variables if 'yend' and 'q' are not provided.
        sp_lag_feats : list of str
            Names of columns containing features to compute spatial lag for
        yend : list of str
            Name of columns corresponding to endogenous variables
        q : list of str
            Names of columns corresponding to external exogenous variables (note: this should not contain any variables
            from x); cannot be used in combination with h
        w : Pysal weights matrix
            Spatial weights matrix
        w_lags : int
            Orders of W to include as instruments for the spatially lagged dependent variable. For example,
            w_lags=1, then instruments are WX; if w_lags=2, then WX, WWX; and so on.
        lag_q : boolean
            If True, then include spatial lags of the additional instruments (q) also as exogenous variables
        robust : string
            For standard errors of heteroskedasticity: if 'white', then a White consistent estimator of the
            variance-covariance matrix is given.  If 'hac', then a HAC consistent estimator of the variance-covariance
            matrix is given. Default set to None.
        gwk : pysal W object
            Kernel spatial weights needed for HAC estimation. Note: matrix must have ones along the main diagonal.
        sig2n_k : boolean
            If True, then use n-k to estimate sigma^2. If False, use n.
    """

    # All two dimensional
    def __init__(
        self,
        df: pd.DataFrame,
        y_col: str,
        sp_lag_feats: List[str],
        w: "libpysal.weights.W",
        yend_cols: Union[None, List[str]] = None,
        q_cols: Union[None, List[str]] = None,
        w_lags: int = 1,
        lag_q: bool = True,
        robust: Union[None, str] = None,
        gwk: Union[None, "libpysal.weights.W"] = None,
        sig2n_k: bool = False,
    ):
        if not isinstance(yend_cols, list):
            yend_cols = [yend_cols]
        if not isinstance(q_cols, list):
            q_cols = [q_cols]

        # Extract arrays from provided dataframe + column names:
        y = df[y_col].values.reshape(-1, 1)
        yend = df[yend_cols].values if len(yend_cols) > 1 else df[yend_cols].values.reshape(-1, 1)
        q = df[q_cols].values if len(q_cols) > 1 else df[q_cols].values.reshape(-1, 1)

        # Extract the independent variables, including all features for which to compute spatial lag alongside all
        # features to leave as is:
        x = df.loc[:, df.columns != y_col].values

        # Indices of independent variables for which to compute spatial lag:
        splag_idx = df.columns.get_indexer(sp_lag_feats)

        # Compute spatial lag for selected endogenous and exogenous variables:
        yend_set, q_set = set_endog(y, x, splag_idx, w, yend, q, w_lags, lag_q)

        # Attach constant to independent array:
        x_constant, name_x, warn = USER.check_constant(x)
        self.x_constant = x_constant
        # Set y, x, q, yend variables as attributes- will also compute predicted y values and save as attribute:
        BaseTSLS.__init__(self, y=y, x=x_constant, yend=yend_set, q=q_set, robust=robust, gwk=gwk, sig2n_k=sig2n_k)


class LR_GM_lag(LR_BaseGM_Lag):
    """
    Spatial two stage least squares (S2SLS) with results and diagnostics; original implementation from Anselin,
    L. (1988). Spatial econometrics: methods and models (Vol. 4). Springer Science & Business Media.

    See documentation of :class `spreg.GM_Lag` (https://github.com/pysal/spreg/blob/master/spreg/twosls_sp.py) for
    original docstring.

    Changes: takes dataframe as an argument and column names to define x, y, and which features spatial lag
    computation is applied to.

    Args:
        df: Dataframe containing dependent variable and all independent variables
        y_col: Name of column corresponding to the dependent variable. Will assume all other columns contain independent
            variables if 'yend' and 'q' are not provided.
        sp_lag_feats: Names of columns containing features to compute spatial lag for
        yend_cols: Name of columns corresponding to endogenous variables
        q_cols: Names of columns corresponding to external exogenous variables (note: this should not contain any
            variables from x); cannot be used in combination with h
        w: Spatial weights matrix (Pysal weights matrix)
        w_lags: Orders of W to include as instruments for the spatially lagged dependent variable. For example,
            w_lags=1, then instruments are WX; if w_lags=2, then WX, WWX; and so on.
        lag_q: If True, then include spatial lags of the additional instruments (q) also as exogenous variables
        robust: For standard errors of heteroskedasticity: if 'white', then a White consistent estimator of the
            variance-covariance matrix is given.  If 'hac', then a HAC consistent estimator of the variance-covariance
            matrix is given. Default set to None.
        gwk: Kernel spatial weights needed for HAC estimation (Pysal weights matrix). Note: matrix must have ones along
            the main diagonal.
        sig2n_k: If True, then use n-k to estimate sigma^2. If False, use n.
        spat_diag: If True, then compute Anselin-Kelejian test
        vm: If True, include variance-covariance matrix in summary results
        name_ds: Name to assign dataset (for summary/printing purposes)
        name_x: Name to assign independent variables (for summary/printing purposes)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        y_col: str,
        sp_lag_feats: List[str],
        w: "libpysal.weights.W",
        yend_cols: Union[None, List[str]] = None,
        q_cols: Union[None, List[str]] = None,
        w_lags: int = 1,
        lag_q: bool = True,
        robust: Union[None, str] = None,
        gwk: Union[None, "libpysal.weights.W"] = None,
        sig2n_k: bool = False,
        spat_diag: bool = False,
        vm: bool = False,
        name_ds: Union[None, str] = None,
        name_x: Union[None, List[str]] = None,
    ):
        # Check arrays before proceeding:
        y = df[y_col].values.reshape(-1, 1)
        yend = df[yend_cols].values if len(yend_cols) > 1 else df[yend_cols].values.reshape(-1, 1)
        q = df[q_cols].values if len(q_cols) > 1 else df[q_cols].values.reshape(-1, 1)
        x = df.loc[:, df.columns != y_col].values

        n = USER.check_arrays(x, yend, q)
        y = USER.check_y(y, n)
        USER.check_weights(w, y, w_required=True)
        USER.check_robust(robust, gwk)

        LR_BaseGM_Lag.__init__(
            self,
            df=df,
            y_col=y_col,
            sp_lag_feats=sp_lag_feats,
            w=w,
            yend_cols=yend_cols,
            q_cols=q_cols,
            w_lags=w_lags,
            robust=robust,
            gwk=gwk,
            lag_q=lag_q,
            sig2n_k=sig2n_k,
        )

        self.rho = self.betas[-1]
        self.predy_e, self.e_pred, warn = sp_att(w, self.y, self.predy, self.yend[:, -1].reshape(self.n, 1), self.rho)
        set_warn(self, warn)

        self.title = "SPATIAL TWO STAGE LEAST SQUARES"
        self.name_ds = USER.set_name_ds(name_ds)
        self.name_y = USER.set_name_y(y_col)
        self.name_x = USER.set_name_x(name_x, self.x_constant)
        # Set explicitly provided endogenous variables (if any):
        self.name_yend = USER.set_name_yend(yend_cols, yend)
        # And combine with implicitly defined endogenous variable from spatial lag of dependent variable:
        self.name_yend.append(USER.set_name_yend_sp(self.name_y))
        self.name_z = self.name_x + self.name_yend
        # Set explicitly provided exogenous variables (if any):
        self.name_q = USER.set_name_q(q_cols, q)
        # And combine with implicitly defined endogenous variables from spatial lag of chosen independent variables:
        self.name_q.extend(USER.set_name_q_sp(sp_lag_feats, w_lags, self.name_q, lag_q))
        self.name_h = USER.set_name_h(self.name_x, self.name_q)
        self.robust = USER.set_robust(robust)
        SUMMARY.GM_Lag(reg=self, w=w, vm=vm, spat_diag=spat_diag)

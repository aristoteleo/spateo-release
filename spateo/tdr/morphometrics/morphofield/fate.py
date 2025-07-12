import itertools
from multiprocessing.dummy import Pool as ThreadPool
from typing import Callable, Dict, List, Optional, Tuple, Union, Any

import numpy as np
from anndata import AnnData

# from ..vectorfield import vector_field_function
from spateo.logging import logger_manager as lm
from scipy import interpolate
from scipy.integrate import solve_ivp
import scipy.sparse as sp
from tqdm import tqdm
from .traj_class import Trajectory


def fate(
    adata: AnnData,
    init_cells: list,
    init_states: Optional[np.ndarray] = None,
    basis: Optional[None] = None,
    layer: str = "X",
    dims: Optional[Union[int, List[int], Tuple[int], np.ndarray]] = None,
    genes: Optional[List] = None,
    t_end: Optional[float] = None,
    direction: str = "both",
    interpolation_num: int = 250,
    average: bool = False,
    sampling: str = "arc_length",
    VecFld_true: Callable = None,
    cores: int = 1,
    ivp_args: Optional[Tuple] = None,
    **kwargs: dict,
) -> AnnData:
    """Predict the historical and future cell transcriptomic states over arbitrary time scales.

     This is achieved by integrating the reconstructed vector field function from one or a set of initial cell state(s).
     Note that this function is designed so that there is only one trajectory (based on averaged cell states if multiple
     initial states are provided) will be returned. `dyn.tl._fate` can be used to calculate multiple cell states.

    Args:
        adata: AnnData object that contains the reconstructed vector field function in the `uns` attribute.
        init_cells: Cell name or indices of the initial cell states for the historical or future cell state prediction with
            numerical integration. If the names in init_cells not found in the adata.obs_name, it will be treated as
            cell indices and must be integers.
        init_states: Initial cell states for the historical or future cell state prediction with numerical integration.
        basis: The embedding data to use for predicting cell fate. If `basis` is either `umap` or `pca`, the reconstructed
            trajectory will be projected back to high dimensional space via the `inverse_transform` function.
        layer: Which layer of the data will be used for predicting cell fate with the reconstructed vector field function.
            The layer once provided, will override the `basis` argument and then predicting cell fate in high
            dimensional space.
        dims: The dimensions that will be selected for fate prediction.
        genes: The gene names whose gene expression will be used for predicting cell fate. By default (when genes is set to
            None), the genes used for velocity embedding (var.use_for_transition) will be used for vector field
            reconstruction. Note that the genes to be used need to have velocity calculated and corresponds to those
            used in the `dyn.tl.VectorField` function.
        t_end: The length of the time period from which to predict cell state forward or backward over time. This is used
            by the odeint function.
        direction: The direction to predict the cell fate. One of the `forward`, `backward` or `both` string.
        interpolation_num: The number of uniformly interpolated time points.
        average: The method to calculate the average cell state at each time step, can be one of `origin` or `trajectory`. If
            `origin` used, the average expression state from the init_cells will be calculated and the fate prediction
            is based on this state. If `trajectory` used, the average expression states of all cells predicted from the
            vector field function at each time point will be used. If `average` is `False`, no averaging will be
            applied. If `average` is True, `origin` will be used.
        sampling: Methods to sample points along the integration path, one of `{'arc_length', 'logspace', 'uniform_indices'}`.
            If `logspace`, we will sample time points linearly on log space. If `uniform_indices`, the sorted unique set
            of all time points from all cell states' fate prediction will be used and then evenly sampled up to
            `interpolation_num` time points. If `arc_length`, we will sample the integration path with uniform arc
            length.
        VecFld_true: The true ODE function, useful when the data is generated through simulation. Replace VecFld argument when
            this has been set.
        cores: Number of cores to calculate path integral for predicting cell fate. If cores is set to be > 1,
            multiprocessing will be used to parallel the fate prediction.
        kwargs: Additional parameters that will be passed into the fate function.

    Returns:
        AnnData object that is updated with the dictionary Fate (includes `t` and `prediction` keys) in uns attribute.
    """

    if basis is not None:
        fate_key = "fate_" + basis
    else:
        fate_key = "fate" if layer == "X" else "fate_" + layer

    init_states, VecFld, t_end, valid_genes = fetch_states(
        adata,
        init_states,
        init_cells,
        basis,
        layer,
        average,
        t_end,
    )

    if np.isscalar(dims):
        init_states = init_states[:, :dims]
    elif dims is not None:
        init_states = init_states[:, dims]
    if VecFld_true is None:
        from dynamo.vectorfield import vector_field_function
        vf = (lambda x: scale * vector_field_function(x=x, vf_dict=VecFld, dim=dims))
    else:
        vf = VecFld_true
    print(interpolation_num)
    t, prediction = _fate(
        vf,
        init_states,
        t_end=t_end,
        direction=direction,
        interpolation_num=interpolation_num,
        average=True if average == "trajectory" else False,
        sampling=sampling,
        cores=cores,
        ivp_args=ivp_args,
        **kwargs,
    )

    exprs = None

    adata.uns[fate_key] = {
        "init_states": init_states,
        "init_cells": list(init_cells),
        "average": average,
        "t": t,
        "prediction": prediction,
        "genes": valid_genes,
    }
    if exprs is not None:
        adata.uns[fate_key]["exprs"] = exprs

    return adata


def _fate(
    VecFld: Callable,
    init_states: np.ndarray,
    t_end: Optional[float] = None,
    step_size: Optional[float] = None,
    direction: str = "both",
    interpolation_num: int = 250,
    average: bool = True,
    sampling: str = "arc_length",
    cores: int = 1,
    ivp_args: Optional[Tuple] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict the historical and future cell transcriptomic states over arbitrary time scales by integrating vector
    field functions from one or a set of initial cell state(s).

    Args:
        VecFld: Functional form of the vector field reconstructed from sparse single cell samples. It is applicable to the
            entire transcriptomic space.
        init_states: Initial cell states for the historical or future cell state prediction with numerical integration.
        t_end: The length of the time period from which to predict cell state forward or backward over time. This is used
            by the odeint function.
        step_size: Step size for integrating the future or history cell state, used by the odeint function. By default it is
            None, and the step_size will be automatically calculated to ensure 250 total integration time-steps will be
            used.
        direction: The direction to predict the cell fate. One of the `forward`, `backward`or `both` string.
        interpolation_num: The number of uniformly interpolated time points.
        average: A boolean flag to determine whether to smooth the trajectory by calculating the average cell state at each
            time step.
        sampling: Methods to sample points along the integration path, one of `{'arc_length', 'logspace', 'uniform_indices'}`.
            If `logspace`, we will sample time points linearly on log space. If `uniform_indices`, the sorted unique set
            of all time points from all cell states' fate prediction will be used and then evenly sampled up to
            `interpolation_num` time points. If `arc_length`, we will sample the integration path with uniform arc
            length.
        cores: Number of cores to calculate path integral for predicting cell fate. If cores is set to be > 1,
            multiprocessing will be used to parallel the fate prediction.

    Returns:
        A tuple containing two elements:
            t: The time at which the cell state are predicted.
            prediction: Predicted cells states at different time points. Row order corresponds to the element order in
                t. If init_states corresponds to multiple cells, the expression dynamics over time for each cell is
                concatenated by rows. That is, the final dimension of prediction is (len(t) * n_cells, n_features).
                n_cells: number of cells; n_features: number of genes or number of low dimensional embeddings.
                Of note, if the average is set to be True, the average cell state at each time point is calculated for
                all cells.
    """

    if sampling == "uniform_indices":
        lm.main_warning(
            f"Uniform_indices method sample data points from all time points. The multiprocessing will be disabled."
        )
        cores = 1

    # t_linspace = getTseq(init_states, t_end, step_size)
    # print(len(t_linspace))
    if cores == 1:
        t, prediction = integrate_vf_ivp(
            init_states,
            t_end,
            direction,
            VecFld,
            interpolation_num=interpolation_num,
            average=average,
            sampling=sampling,
            args=ivp_args,
        )
    else:
        pool = ThreadPool(cores)
        res = pool.starmap(
            integrate_vf_ivp,
            zip(
                init_states,
                itertools.repeat(t_end),
                itertools.repeat(direction),
                itertools.repeat(VecFld),
                itertools.repeat(()),
                itertools.repeat(interpolation_num),
                itertools.repeat(average),
                itertools.repeat(sampling),
                itertools.repeat(False),
                itertools.repeat(True),
            ),
        )  # disable tqdm when using multiple cores.
        pool.close()
        pool.join()
        t_, prediction_ = zip(*res)
        t, prediction = [i[0] for i in t_], [i[0] for i in prediction_]

    if init_states.shape[0] > 1 and average:
        t_stack, prediction_stack = np.hstack(t), np.hstack(prediction)
        n_cell, n_feature = init_states.shape

        t_len = int(len(t_stack) / n_cell)
        avg = np.zeros((n_feature, t_len))

        for i in range(t_len):
            avg[:, i] = np.mean(prediction_stack[:, np.arange(n_cell) * t_len + i], 1)

        prediction = [avg]
        t = [np.sort(np.unique(t))]

    return t, prediction


def fetch_states(
    adata: AnnData,
    init_states: np.ndarray,
    init_cells: Union[str, List],
    basis: str,
    layer: str,
    average: Union[str, bool],
    t_end: float,
) -> Tuple[np.ndarray, Dict[str, Any], float, Optional[List[str]]]:
    """Fetch initial states for the vector field modeling of single-cell data.

    This function retrieves the initial states for the vector field modeling of single-cell data from the provided
    `adata` object. It allows providing either the `init_states` directly or the `init_cells` names and the `basis`
    (e.g., pca) from which the initial states should be derived.

    Args:
        adata: An AnnData object containing the single-cell data.
        init_states: The initial states to use for the vector field modeling. If not provided, `init_cells` and `basis`
            should be used to derive the initial states.
        init_cells: The cell names to use for deriving the initial states.
        basis: The basis to use for deriving the initial states.
        layer: The layer of the data to use for deriving the initial states.
        average: Determines how to handle multiple initial states when provided. If "origin" or True, the initial states
            will be averaged to a single state. If "trajectory", the initial states will be kept as separate states.
        t_end: The end time point for the vector field modeling.

    Returns:
        A tuple containing the following:
            - init_states: the derived initial states for the vector field modeling.
            - VecFld: a dictionary containing information about the vector field.
            - t_end: the end time point for the vector field modeling.
            - valid_genes: a list of valid gene names used for the vector field modeling,
    """
    if basis is not None:
        vf_key = "VecFld_" + basis
    else:
        vf_key = "VecFld"
    VecFld = adata.uns[vf_key]
    X = VecFld["X"]
    valid_genes = None

    if init_states is None and init_cells is None:
        raise Exception("Either init_state or init_cells should be provided.")
    elif init_states is None and init_cells is not None:
        if type(init_cells) == str:
            init_cells = [init_cells]
        intersect_cell_names = sorted(
            set(init_cells).intersection(adata.obs_names),
            key=lambda x: list(init_cells).index(x),
        )
        _cell_names = init_cells if len(intersect_cell_names) == 0 else intersect_cell_names
        print(adata)
        if basis is not None:
            init_states = adata[_cell_names].obsm["X_" + basis].copy()
            if len(_cell_names) == 1:
                init_states = init_states.reshape((1, -1))
            VecFld = adata.uns["VecFld_" + basis]
            X = adata.obsm["X_" + basis]

            valid_genes = [basis + "_" + str(i) for i in np.arange(init_states.shape[1])]
        else:
            raise Exception("basis is not provided")

    if init_states.shape[0] > 1 and average in ["origin", True]:
        init_states = init_states.mean(0).reshape((1, -1))

    if t_end is None:
        t_end = getTend(X, VecFld["V"])

    if sp.issparse(init_states):
        init_states = init_states.A

    return init_states, VecFld, t_end, valid_genes


def getTseq(init_states: np.ndarray, t_end: float, step_size: Optional[Union[int, float]] = None) -> np.ndarray:
    """Generate a time sequence for the vector field modeling.

    Args:
        init_states: The initial states for the vector field modeling as a 2D numpy array.
        t_end: The end time for the vector field modeling.
        step_size: The time step size between each time point in the sequence.

    Returns:
        An array containing the time sequence for the vector field modeling.
    """
    if step_size is None:
        max_steps = int(max(7 / (init_states.shape[1] / 300), 4)) if init_states.shape[1] > 300 else 7
        t_linspace = np.linspace(0, t_end, 10 ** (np.min([int(np.log10(t_end)), max_steps])))
    else:
        t_linspace = np.arange(0, t_end + step_size, step_size)

    return t_linspace


def integrate_vf_ivp(
    init_states: np.ndarray,
    t_end: float,
    integration_direction: str,
    f: Callable,
    args: Optional[Tuple] = None,
    interpolation_num: int = 250,
    average: bool = True,
    sampling: str = "arc_length",
    verbose: bool = False,
    disable: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Integrating along vector field function using the initial value problem solver from scipy.integrate.

    Args:
        init_states: Initial states of the system.
        t_end: End time of the integration.
        integration_direction: The direction of integration.
        f: The vector field function of the system.
        args: Additional arguments to pass to the vector field function.
        interpolation_num: Number of time points to interpolate the trajectories over.
        average: Whether to average the trajectories.
        sampling: The method of sampling points along a trajectory.
        verbose: Whether to print the integration time.
        disable: Whether to disable the progress bar.

    Returns:
        The time and trajectories of the system.
    """

    # TODO: rewrite this function with the Trajectory class
    if init_states.ndim == 1:
        init_states = init_states[None, :]
    n_cell, n_feature = init_states.shape
    max_step = np.abs(t_end) / interpolation_num

    T, Y, SOL = [], [], []

    if interpolation_num is not None and integration_direction == "both":
        interpolation_num = interpolation_num * 2

    for i in tqdm(range(n_cell), desc="integration with ivp solver", disable=disable):
        y0 = init_states[i, :]
        ivp_f, ivp_f_event = (
            lambda t, x: f(x),
            lambda t, x: np.all(abs(f(x)) < 1e-5) - 1,
            # np.linalg.norm(np.abs(f(x))) - 1e-5 if velocity on all dimension is less than 1e-5
        )
        ivp_f_event.terminal = True

        if verbose:
            print("\nintegrating cell ", i, "; Initial state: ", init_states[i, :])
        if integration_direction == "forward":
            y_ivp = solve_ivp(
                ivp_f,
                [0, t_end],
                y0,
                events=ivp_f_event,
                args=args,
                # max_step=max_step,
                dense_output=True,
                t_eval=np.linspace(0, t_end, interpolation_num),
            )
            y, t_trans, sol = y_ivp.y, y_ivp.t, y_ivp.sol
        elif integration_direction == "backward":
            y_ivp = solve_ivp(
                ivp_f,
                [0, -t_end],
                y0,
                events=ivp_f_event,
                args=args,
                max_step=max_step,
                dense_output=True,
            )
            y, t_trans, sol = y_ivp.y, y_ivp.t, y_ivp.sol
        elif integration_direction == "both":
            y_ivp_f = solve_ivp(
                ivp_f,
                [0, t_end],
                y0,
                events=ivp_f_event,
                args=args,
                max_step=max_step,
                dense_output=True,
            )
            y_ivp_b = solve_ivp(
                ivp_f,
                [0, -t_end],
                y0,
                events=ivp_f_event,
                args=args,
                max_step=max_step,
                dense_output=True,
            )
            y, t_trans = (
                np.hstack((y_ivp_b.y[::-1, :], y_ivp_f.y)),
                np.hstack((y_ivp_b.t[::-1], y_ivp_f.t)),
            )
            sol = [y_ivp_b.sol, y_ivp_f.sol]
        else:
            raise Exception("both, forward, backward are the only valid direction argument strings")

        T.append(t_trans)
        Y.append(y)
        SOL.append(sol)

        if verbose:
            print("\nintegration time: ", len(t_trans))

    trajs = [Trajectory(X=Y[i], t=T[i], sort=False) for i in range(n_cell)]

    if sampling == "arc_length":
        for i in tqdm(
            range(n_cell),
            desc="uniformly sampling points along a trajectory",
            disable=disable,
        ):
            trajs[i].archlength_sampling(
                SOL[i],
                interpolation_num=interpolation_num,
                integration_direction=integration_direction,
            )

        t, Y = [traj.t for traj in trajs], [traj.X for traj in trajs]
    elif sampling == "logspace":
        for i in tqdm(
            range(n_cell),
            desc="sampling points along a trajectory in logspace",
            disable=disable,
        ):
            trajs[i].logspace_sampling(
                SOL[i],
                interpolation_num=interpolation_num,
                integration_direction=integration_direction,
            )

        t, Y = [traj.t for traj in trajs], [traj.X for traj in trajs]
    elif sampling == "uniform_indices":
        t_uniq = np.unique(np.hstack(T))
        if len(t_uniq) > interpolation_num:
            valid_t_trans = np.hstack([0, np.sort(t_uniq)])[
                (np.linspace(0, len(t_uniq), interpolation_num)).astype(int)
            ]

            if len(valid_t_trans) != interpolation_num:
                n_missed = interpolation_num - len(valid_t_trans)
                tmp = np.zeros(n_missed)

                for i in range(n_missed):
                    tmp[i] = (valid_t_trans[i] + valid_t_trans[i + 1]) / 2

                valid_t_trans = np.sort(np.hstack([tmp, valid_t_trans]))
        else:
            neg_tau, pos_tau = t_uniq[t_uniq < 0], t_uniq[t_uniq >= 0]
            t_0, t_1 = (
                -np.linspace(min(t_uniq), 0, interpolation_num),
                np.linspace(0, max(t_uniq), interpolation_num),
            )

            valid_t_trans = np.hstack((t_0, t_1))

        _Y = None
        if integration_direction == "both":
            neg_t_len = sum(valid_t_trans < 0)
        for i in tqdm(
            range(n_cell),
            desc="calculate solutions on the sampled time points in logspace",
            disable=disable,
        ):
            cur_Y = (
                SOL[i](valid_t_trans)
                if integration_direction != "both"
                else np.hstack(
                    (
                        SOL[i][0](valid_t_trans[:neg_t_len]),
                        SOL[i][1](valid_t_trans[neg_t_len:]),
                    )
                )
            )
            _Y = cur_Y if _Y is None else np.hstack((_Y, cur_Y))

        t, Y = valid_t_trans, _Y

        # TODO: this part is buggy, need to fix
        if n_cell > 1 and average:
            t_len = int(len(t) / n_cell)
            avg = np.zeros((n_feature, t_len))

            for i in range(t_len):
                avg[:, i] = np.mean(Y[:, np.arange(n_cell) * t_len + i], 1)
            Y = avg

        t = [t] * n_cell
        subarray_width = Y.shape[1] // n_cell
        Y = [Y[:, i * subarray_width : (i + 1) * subarray_width] for i in range(n_cell)]
    
    else:
        t = T

    return t, Y



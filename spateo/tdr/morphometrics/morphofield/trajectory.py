from typing import Optional, Union

from anndata import AnnData

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


def morphopath(
    adata: AnnData,
    vf_key: str = "VecFld_morpho",
    key_added: str = "fate_morpho",
    layer: str = "X",
    direction: str = "forward",
    interpolation_num: int = 250,
    t_end: Optional[Union[int, float]] = None,
    average: bool = False,
    cores: int = 1,
    inplace: bool = True,
    **kwargs,
) -> Optional[AnnData]:
    """
    Prediction of cell developmental trajectory based on reconstructed vector field.

    Args:
        adata: AnnData object that contains the reconstructed vector field function in the ``.uns`` attribute.
        vf_key: The key in ``.uns`` that corresponds to the reconstructed vector field.
        key_added: The key under which to add the dictionary Fate (includes ``t`` and ``prediction`` keys).
        layer: Which layer of the data will be used for predicting cell fate with the reconstructed vector field function.
        direction: The direction to predict the cell fate. One of the ``forward``, ``backward`` or ``both`` string.
        interpolation_num:  The number of uniformly interpolated time points.
        t_end: The length of the time period from which to predict cell state forward or backward over time.
        average: The method to calculate the average cell state at each time step, can be one of ``origin`` or
                 ``trajectory``. If ``origin`` used, the average expression state from the init_cells will be calculated and
                 the fate prediction is based on this state. If ``trajectory`` used, the average expression states of all
                 cells predicted from the vector field function at each time point will be used. If ``average`` is
                 ``False``, no averaging will be applied.
        cores: Number of cores to calculate path integral for predicting cell fate. If cores is set to be > 1,
               multiprocessing will be used to parallel the fate prediction.
        inplace: Whether to copy adata or modify it inplace.
        **kwargs: Additional parameters that will be passed into the ``fate`` function.

    Returns:

        An ``AnnData`` object is updated/copied with the ``key_added`` dictionary in the ``.uns`` attribute.

        The  ``key_added``  dictionary which contains:

            t: The time at which the cell state are predicted.
            prediction: Predicted cells states at different time points. Row order corresponds to the element order in
                        t. If init_states corresponds to multiple cells, the expression dynamics over time for each cell
                        is concatenated by rows. That is, the final dimension of prediction is (len(t) * n_cells,
                        n_features). n_cells: number of cells; n_features: number of genes or number of low dimensional
                        embeddings. Of note, if the average is set to be True, the average cell state at each time point
                        is calculated for all cells.
    """
    from dynamo.prediction.fate import fate

    adata = adata if inplace else adata.copy()
    fate_adata = adata.copy()
    if vf_key not in fate_adata.uns_keys():
        raise Exception(
            f"The {vf_key} that corresponds to the reconstructed vector field is not in ``anndata.uns``."
            f"Please run ``st.tdr.morphofield_gp`` or ``st.tdr.morphofield_sparsevfc`` before fate prediction."
        )
    if adata.uns[vf_key]["method"] not in ["gaussian_process", "sparsevfc"]:
        raise Exception(
            f"The method for vector field  reconstruction is not in avaliable."
            f"Please re-run ``st.tdr.morphofield_gp`` or ``st.tdr.morphofield_sparsevfc`` before fate prediction."
        )
    if f"VecFld_{key_added}" not in fate_adata.uns_keys():
        fate_adata.uns[f"VecFld_{key_added}"] = fate_adata.uns[vf_key]
    if f"X_{key_added}" not in fate_adata.obsm_keys():
        fate_adata.obsm[f"X_{key_added}"] = fate_adata.uns[f"VecFld_{key_added}"]["X"]

    method = adata.uns[vf_key]["method"]
    if method == "gaussian_process":
        from .gaussian_process import _gp_velocity

        fate(
            fate_adata,
            init_cells=fate_adata.obs_names.tolist(),
            basis=key_added,
            layer=layer,
            interpolation_num=interpolation_num,
            t_end=t_end,
            direction=direction,
            average=average,
            cores=cores,
            VecFld_true=lambda X: _gp_velocity(X=X, vf_dict=fate_adata.uns[vf_key]),
            **kwargs,
        )
    elif method == "sparsevfc":
        fate(
            fate_adata,
            init_cells=fate_adata.obs_names.tolist(),
            basis=key_added,
            layer=layer,
            interpolation_num=interpolation_num,
            t_end=t_end,
            direction=direction,
            average=average,
            cores=cores,
            **kwargs,
        )
    adata.uns[key_added] = fate_adata.uns[f"fate_{key_added}"].copy()

    cells_states = adata.uns[key_added]["prediction"]
    cells_times = adata.uns[key_added]["t"]
    adata.uns[key_added]["prediction"] = {i: cell_states.T for i, cell_states in enumerate(cells_states)}
    adata.uns[key_added]["t"] = {i: cell_times for i, cell_times in enumerate(cells_times)}

    return None if inplace else adata

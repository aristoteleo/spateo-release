from typing import Optional

import numpy as np
from anndata import AnnData

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


def _generate_vf_class(
    adata: AnnData,
    vf_key: str,
    method: str = "gaussian_process",
    nonrigid_only: bool = False,
):
    if vf_key in adata.uns.keys():
        if method == "gaussian_process":
            from .GPVectorField import GPVectorField

            vector_field_class = GPVectorField()
            vector_field_class.from_adata(adata, vf_key=vf_key, nonrigid_only=nonrigid_only)
        elif method == "sparsevfc":
            from dynamo.vectorfield.scVectorField import SvcVectorField

            vector_field_class = SvcVectorField()
            vector_field_class.from_adata(adata, basis=None, vf_key=vf_key)
        else:
            raise Exception(
                f"The {method} is not in ``anndata.uns[{vf_key}]``."
                f"Please re-run ``st.tdr.morphofield_gp`` or ``st.tdr.morphofield_sparsevfc`` before running this function."
            )
    else:
        raise Exception(
            f"The {vf_key} that corresponds to the reconstructed vector field is not in ``anndata.uns``."
            f"Please run ``st.align.morpho_align(adata, vecfld_key_added='{vf_key}')`` before running this function."
        )
    return vector_field_class


def morphofield_velocity(
    adata: AnnData,
    vf_key: str = "VecFld_morpho",
    key_added: str = "velocity",
    nonrigid_only: bool = False,
    inplace: bool = True,
) -> Optional[AnnData]:
    """
    Calculate the velocity for each cell with the reconstructed vector field function.

    Args:
        adata: AnnData object that contains the reconstructed vector field.
        vf_key: The key in ``.uns`` that corresponds to the reconstructed vector field.
        key_added: The key that will be used for the velocity key in ``.obsm``.
        nonrigid_only: If True, only the nonrigid part of the vector field will be calculated.
        inplace: Whether to copy adata or modify it inplace.

    Returns:
        An ``AnnData`` object is updated/copied with the ``key_added`` in the ``.obsm`` attribute which contains velocities.
    """
    adata = adata if inplace else adata.copy()
    vector_field_class = _generate_vf_class(
        adata=adata, vf_key=vf_key, method=adata.uns[vf_key]["method"], nonrigid_only=nonrigid_only
    )

    init_states = adata.uns[vf_key]["X"]
    adata.obsm[key_added] = vector_field_class.func(init_states)

    return None if inplace else adata


def morphofield_acceleration(
    adata: AnnData,
    vf_key: str = "VecFld_morpho",
    key_added: str = "acceleration",
    method: str = "analytical",
    nonrigid_only: bool = False,
    inplace: bool = True,
) -> Optional[AnnData]:
    """
    Calculate acceleration for each cell with the reconstructed vector field function.

    Args:
        adata: AnnData object that contains the reconstructed vector field.
        vf_key: The key in ``.uns`` that corresponds to the reconstructed vector field.
        key_added: The key that will be used for the acceleration key in ``.obs`` and ``.obsm``.
        method: The method that will be used for calculating acceleration field, either ``'analytical'`` or ``'numerical'``.

                ``'analytical'`` method uses the analytical expressions for calculating acceleration while ``'numerical'``
                method uses numdifftools, a numerical differentiation tool, for computing acceleration. ``'analytical'``
                method is much more efficient.
        nonrigid_only: If True, only the nonrigid part of the vector field will be calculated.
        inplace: Whether to copy adata or modify it inplace.

    Returns:
        An ``AnnData`` object is updated/copied with the ``key_added`` in the ``.obs`` and ``.obsm`` attribute.

        The  ``key_added`` in the ``.obs`` which contains acceleration.
        The  ``key_added`` in the ``.obsm`` which contains acceleration vectors.
    """

    adata = adata if inplace else adata.copy()
    vector_field_class = _generate_vf_class(
        adata=adata, vf_key=vf_key, method=adata.uns[vf_key]["method"], nonrigid_only=nonrigid_only
    )

    X, V = vector_field_class.get_data()
    adata.obs[key_added], adata.obsm[key_added] = vector_field_class.compute_acceleration(X=X, method=method)

    return None if inplace else adata


def morphofield_curvature(
    adata: AnnData,
    vf_key: str = "VecFld_morpho",
    key_added: str = "curvature",
    formula: int = 2,
    method: str = "analytical",
    nonrigid_only: bool = False,
    inplace: bool = True,
) -> Optional[AnnData]:
    """
    Calculate curvature for each cell with the reconstructed vector field function.

    Args:
        adata: AnnData object that contains the reconstructed vector field.
        vf_key: The key in ``.uns`` that corresponds to the reconstructed vector field.
        key_added: The key that will be used for the curvature key in ``.obs`` and ``.obsm``.
        formula: Which formula of curvature will be used, there are two formulas, so formula can be either ``{1, 2}``.
                 By default it is 2 and returns both the curvature vectors and the norm of the curvature. The formula
                 one only gives the norm of the curvature.
        method: The method that will be used for calculating curvature field, either ``'analytical'`` or ``'numerical'``.

                ``'analytical'`` method uses the analytical expressions for calculating curvature while ``'numerical'``
                method uses numdifftools, a numerical differentiation tool, for computing curvature. ``'analytical'``
                method is much more efficient.
        nonrigid_only: If True, only the nonrigid part of the vector field will be calculated.
        inplace: Whether to copy adata or modify it inplace.

    Returns:
        An ``AnnData`` object is updated/copied with the ``key_added`` in the ``.obs`` and ``.obsm`` attribute.

        The  ``key_added`` in the ``.obs`` which contains curvature.
        The  ``key_added`` in the ``.obsm`` which contains curvature vectors.

    """

    adata = adata if inplace else adata.copy()
    vector_field_class = _generate_vf_class(
        adata=adata, vf_key=vf_key, method=adata.uns[vf_key]["method"], nonrigid_only=nonrigid_only
    )

    X, V = vector_field_class.get_data()
    adata.obs[key_added], adata.obsm[key_added] = vector_field_class.compute_curvature(
        X=X, formula=formula, method=method
    )

    return None if inplace else adata


def morphofield_curl(
    adata: AnnData,
    vf_key: str = "VecFld_morpho",
    key_added: str = "curl",
    method: str = "analytical",
    nonrigid_only: bool = False,
    inplace: bool = True,
) -> Optional[AnnData]:
    """
    Calculate curl for each cell with the reconstructed vector field function.

    Args:
        adata: AnnData object that contains the reconstructed vector field.
        vf_key: The key in ``.uns`` that corresponds to the reconstructed vector field.
        key_added: The key that will be used for the torsion key in ``.obs``.
        method: The method that will be used for calculating torsion field, either ``'analytical'`` or ``'numerical'``.

                ``'analytical'`` method uses the analytical expressions for calculating torsion while ``'numerical'``
                method uses numdifftools, a numerical differentiation tool, for computing torsion. ``'analytical'``
                method is much more efficient.
        nonrigid_only: If True, only the nonrigid part of the vector field will be calculated.
        inplace: Whether to copy adata or modify it inplace.

    Returns:
        An ``AnnData`` object is updated/copied with the ``key_added`` in the ``.obs`` and ``.obsm`` attribute.

        The  ``key_added`` in the ``.obs`` which contains magnitude of curl.
        The  ``key_added`` in the ``.obsm`` which contains curl vectors.
    """

    adata = adata if inplace else adata.copy()
    vector_field_class = _generate_vf_class(
        adata=adata, vf_key=vf_key, method=adata.uns[vf_key]["method"], nonrigid_only=nonrigid_only
    )

    X, V = vector_field_class.get_data()
    curl = vector_field_class.compute_curl(X=X, method=method)
    curl_mag = np.array([np.linalg.norm(i) for i in curl])

    adata.obs[key_added] = curl_mag
    adata.obsm[key_added] = curl

    return None if inplace else adata


def morphofield_torsion(
    adata: AnnData,
    vf_key: str = "VecFld_morpho",
    key_added: str = "torsion",
    method: str = "analytical",
    nonrigid_only: bool = False,
    inplace: bool = True,
) -> Optional[AnnData]:
    """
    Calculate torsion for each cell with the reconstructed vector field function.

    Args:
        adata: AnnData object that contains the reconstructed vector field.
        vf_key: The key in ``.uns`` that corresponds to the reconstructed vector field.
        key_added: The key that will be used for the torsion key in ``.obs`` and ``.obsm``.
        method: The method that will be used for calculating torsion field, either ``'analytical'`` or ``'numerical'``.

                ``'analytical'`` method uses the analytical expressions for calculating torsion while ``'numerical'``
                method uses numdifftools, a numerical differentiation tool, for computing torsion. ``'analytical'``
                method is much more efficient.
        nonrigid_only: If True, only the nonrigid part of the vector field will be calculated.
        inplace: Whether to copy adata or modify it inplace.

    Returns:
        An ``AnnData`` object is updated/copied with the ``key_added`` in the ``.obs`` and ``.uns`` attribute.

        The  ``key_added`` in the ``.obs`` which contains torsion.
        The  ``key_added`` in the ``.uns`` which contains torsion matrix.
    """

    adata = adata if inplace else adata.copy()
    vector_field_class = _generate_vf_class(
        adata=adata, vf_key=vf_key, method=adata.uns[vf_key]["method"], nonrigid_only=nonrigid_only
    )

    X, V = vector_field_class.get_data()
    torsion_mat = vector_field_class.compute_torsion(X=X, method=method)
    torsion = np.array([np.linalg.norm(i) for i in torsion_mat])

    adata.obs[key_added] = torsion
    adata.uns[key_added] = torsion_mat

    return None if inplace else adata


def morphofield_divergence(
    adata: AnnData,
    vf_key: str = "VecFld_morpho",
    key_added: str = "divergence",
    method: str = "analytical",
    vectorize_size: Optional[int] = 1000,
    nonrigid_only: bool = False,
    inplace: bool = True,
) -> Optional[AnnData]:
    """
    Calculate divergence for each cell with the reconstructed vector field function.

    Args:
        adata: AnnData object that contains the reconstructed vector field.
        vf_key: The key in ``.uns`` that corresponds to the reconstructed vector field.
        key_added: The key that will be used for the acceleration key in ``.obs`` and ``.obsm``.
        method: The method that will be used for calculating acceleration field, either ``'analytical'`` or ``'numerical'``.

                ``'analytical'`` method uses the analytical expressions for calculating acceleration while ``'numerical'``
                method uses numdifftools, a numerical differentiation tool, for computing acceleration. ``'analytical'``
                method is much more efficient.
        vectorize_size: vectorize_size is used to control the number of samples computed in each vectorized batch.

                * If vectorize_size = 1, there's no vectorization whatsoever.
                * If vectorize_size = None, all samples are vectorized.
        nonrigid_only: If True, only the nonrigid part of the vector field will be calculated.
        inplace: Whether to copy adata or modify it inplace.

    Returns:
        An ``AnnData`` object is updated/copied with the ``key_added`` in the ``.obs`` attribute.

        The  ``key_added`` in the ``.obs`` which contains divergence.
    """

    adata = adata if inplace else adata.copy()
    vector_field_class = _generate_vf_class(
        adata=adata, vf_key=vf_key, method=adata.uns[vf_key]["method"], nonrigid_only=nonrigid_only
    )

    X, V = vector_field_class.get_data()
    adata.obs[key_added] = vector_field_class.compute_divergence(X=X, method=method, vectorize_size=vectorize_size)

    return None if inplace else adata


def morphofield_jacobian(
    adata: AnnData,
    vf_key: str = "VecFld_morpho",
    key_added: str = "jacobian",
    method: str = "analytical",
    nonrigid_only: bool = False,
    inplace: bool = True,
) -> Optional[AnnData]:
    """
    Calculate jacobian for each cell with the reconstructed vector field function.

    Args:
        adata: AnnData object that contains the reconstructed vector field.
        vf_key: The key in ``.uns`` that corresponds to the reconstructed vector field.
        key_added: The key that will be used for the jacobian key in ``.obs`` and ``.obsm``.
        method: The method that will be used for calculating jacobian field, either ``'analytical'`` or ``'numerical'``.

                ``'analytical'`` method uses the analytical expressions for calculating jacobian while ``'numerical'``
                method uses numdifftools, a numerical differentiation tool, for computing jacobian. ``'analytical'``
                method is much more efficient.
        nonrigid_only: If True, only the nonrigid part of the vector field will be calculated.
        inplace: Whether to copy adata or modify it inplace.

    Returns:
        An ``AnnData`` object is updated/copied with the ``key_added`` in the ``.obs`` and ``.uns`` attribute.

        The  ``key_added`` in the ``.obs`` which contains jacobian.
        The  ``key_added`` in the ``.uns`` which contains jacobian tensor.
    """

    adata = adata if inplace else adata.copy()
    vector_field_class = _generate_vf_class(
        adata=adata, vf_key=vf_key, method=adata.uns[vf_key]["method"], nonrigid_only=nonrigid_only
    )
    X, V = vector_field_class.get_data()
    Jac_func = vector_field_class.get_Jacobian(method=method)

    cell_idx = np.arange(adata.n_obs)
    Js = Jac_func(x=X[cell_idx])
    Js_det = [np.linalg.det(Js[:, :, i]) for i in np.arange(Js.shape[2])]

    adata.obs[key_added] = Js_det
    adata.uns[key_added] = Js

    return None if inplace else adata

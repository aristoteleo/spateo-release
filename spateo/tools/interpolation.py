import numpy as np
from dynamo.vectorfield.scVectorField import SparseVFC

def interpolation_SparseVFC(adata, 
                            genes = None,
                            grid_num = 50,
                            lambda_ = 0.02,
                            lstsq_method = "scipy",
                            **kwargs
):
    """
    predict missing location’s gene expression and learn a continuous gene expression pattern over space

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object that contains spatial (numpy.ndarray) in the `obsm` attribute.
        genes: `list` (default None)
            Gene list that needs to interpolate.
        grid_num: 'int' (default 50)
            Number of grid to generate. Default is 50. Must be non-negative. 
        lambda_: 'float' (default: 0.02)
            Represents the trade-off between the goodness of data fit and regularization. Larger Lambda_ put more weights
            on regularization.
        lstsq_method: 'str' (default: `scipy`)
           The name of the linear least square solver, can be either 'scipy` or `douin`.
        **kwargs：
        Additional parameters that will be passed to SparseVFC function.

    Returns
    -------
    Res: 'dict'
        A dictionary which contains:
            X: Current location.
            valid_ind: The indices of cells that have finite expression values.
            X_ctrl: Sample control points of current location.
            ctrl_idx: Indices for the sampled control points.
            Y: expression estimates in delta t.
            beta: Parameter of the Gaussian Kernel for the kernel matrix (Gram matrix).
            V: Prediction of expression of X.
            C: Finite set of the coefficients for the
            P: Posterior probability Matrix of inliers.
            VFCIndex: Indexes of inliers found by sparseVFC.
            sigma2: Energy change rate.
            grid: Grid of current location.
            grid_V: Prediction of expression of the grid.
            iteration: Number of the last iteration.
            tecr_vec: Vector of relative energy changes rate comparing to previous step.
            E_traj: Vector of energy at each iteration,
        where V = f(X), P is the posterior probability and VFCIndex is the indexes of inliers found by sparseVFC.
        Note that V = `con_K(Grid, X_ctrl, beta).dot(C)` gives the prediction of expression on Grid (but can also be any
        point in the gene expression location space).

    """

    X, V = adata.obsm['spatial'], adata[:, genes].X
    
    # Generate grid 
    min_vec, max_vec = (
        X.min(0),
        X.max(0),
    )
    min_vec = min_vec - 0.01 * np.abs(max_vec - min_vec)
    max_vec = max_vec + 0.01 * np.abs(max_vec - min_vec)
    Grid_list = np.meshgrid(
        *[np.linspace(i, j, grid_num) for i, j in zip(min_vec, max_vec)]
    )
    Grid = np.array([i.flatten() for i in Grid_list]).T
    
    res = SparseVFC(X, V, Grid,lambda_= lambda_,lstsq_method = lstsq_method,**kwargs)

    return res

import os

import numpy as np
import pandas as pd
from patsy import dmatrix

'''
test1 = np.random.uniform(0, 2, (5, 5))
test1[test1 < 1] = 0
test1[test1 > 1] = 1
print(test1)

test2 = np.eye(4)[np.random.choice(4, 5)]
print(test2)

print(test1 > 0.0)
test3 = (test1 > 0.0).astype("int").dot(test2)
print(test3)


X = {
'a': [2, 3, 4, 5],
'b': [1, 0, 0, 1],
'c': [0, 1, 1, 0],
'd': [1, 0, 1, 0],
'e': [0, 1, 0, 1]
}

X = pd.DataFrame(X)
formula = '0+a:c+a:d+a:e+b:c+b:d+b:e'

m = dmatrix(formula, data=X, return_type='dataframe')
print(m)'''

'''
test1 = np.random.uniform(0, 5, (5, 5))
print(test1)

test2 = test1.flatten()
print(test2)'''


import spateo as st
import warnings
from random import sample
from typing import Tuple

import anndata

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import geopandas
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pysal.lib import weights
from pysal.model import spreg

'''
def GM_lag_model(
    adata: anndata.AnnData,
    group: str,
    spatial_key: str = "spatial",
    genes: Tuple[None, list] = None,
    drop_dummy: Tuple[None, str] = None,
    n_neighbors: int = 5,
    layer: Tuple[None, str] = None,
    copy: bool = False,
    n_jobs=30,
):
    """Spatial lag model with spatial two stage least squares (S2SLS) with results and diagnostics; Anselin (1988).

    :math: `\log{P_i} = \alpha + \rho \log{P_{lag-i}} + \sum_k \beta_k X_{ki} + \epsilon_i`

    Reference:
        https://geographicdata.science/book/notebooks/11_regression.html
        http://darribas.org/gds_scipy16/ipynb_md/08_spatial_regression.html

    Args:
        adata: An adata object that has spatial information (via `spatial_key` key in adata.obsm).
        group: The key to the cell group in the adata object.
        spatial_key: The spatial key of the spatial coordinate of each bucket.
        genes: The gene that will be used for S2SLS analyses, must be included in the data.
        drop_dummy: The name of the dummy group.
        n_neighbors: The number of nearest neighbors of each bucket that will be used in calculating the spatial lag.
        layer: The key to the layer. If it is None, adata.X will be used by default.
        copy: Whether to copy the adata object.

    Returns:
        Depend on the `copy` argument, return a deep copied adata object (when `copy = True`) or inplace updated adata
        object. The result adata will include the following new columns in `adata.var`:
            {*}_GM_lag_coeff: coefficient of GM test for each cell group (denoted by {*})
            {*}_GM_lag_zstat: z-score of GM test for each cell group (denoted by {*})
            {*}_GM_lag_pval: p-value of GM test for each cell group (denoted by {*})

    Examples:
    >>> import spateo as st
    >>> st.tl.GM_lag_model(adata, group='simpleanno')
    >>> coef_cols = adata.var.columns[adata.var.columns.str.endswith('_GM_lag_coeff')]
    >>> adata.var.loc[["Hbb-bt", "Hbb-bh1", "Hbb-y", "Hbb-bs"], :].T
    >>>     for i in coef_cols[1:-1]:
    >>>         print(i)
    >>>         top_markers = adata.var.sort_values(i, ascending=False).index[:5]
    >>>         st.pl.space(adata, basis='spatial', color=top_markers, ncols=5, pointsize=0.1, alpha=1)
    >>>         st.pl.space(adata.copy(), basis='spatial', color=['simpleanno'],
    >>>             highlights=[i.split('_GM_lag_coeff')[0]], pointsize=0.1, alpha=1, show_legend='on data')
    """
    group_num = adata.obs[group].value_counts()
    max_group, min_group, min_group_ncells = (
        group_num.index[0],
        group_num.index[-1],
        group_num.values[-1],
    )

    group_name = adata.obs[group]
    db = pd.DataFrame({"group": group_name})
    categories = np.array(adata.obs[group].unique().tolist() + ["others"])
    db["group"] = pd.Categorical(db["group"], categories=categories)
    print(db)

    if drop_dummy is None:
        db.iloc[sample(np.arange(adata.n_obs).tolist(), min_group_ncells), :] = "others"
        drop_columns = ["group_others"]  # ?
    elif drop_dummy in categories:
        group_inds = np.where(db["group"] == drop_dummy)[0]
        db.iloc[group_inds, :] = "others"
        drop_columns = ["group_others", "group_" + str(drop_dummy)]  # ?
    else:
        raise ValueError(f"drop_dummy, {drop_dummy} you provided is not in the adata.obs[{group}].")

    X = pd.get_dummies(data=db, drop_first=False)
    variable_names = X.columns.difference(drop_columns).to_list()

    uniq_g, group_name = (
        set(group_name).difference([drop_dummy]),
        group_name.to_list(),
    )

    uniq_g = list(np.sort(list(uniq_g)))  # sort and convert to list

    # Generate W from the GeoDataFrame
    knn = weights.distance.KNN.from_array(adata.obsm[spatial_key], k=n_neighbors)
    knn.transform = "R"

    if genes is None:
        genes = adata.var.index[adata.var.use_for_pca]
    else:
        genes = adata.var.index.intersection(genes)

    for i in ["const"] + uniq_g + ["W_log_exp"]:
        adata.var[str(i) + "_GM_lag_coeff"] = None
        adata.var[str(i) + "_GM_lag_zstat"] = None
        adata.var[str(i) + "_GM_lag_pval"] = None

    def _single(
        cur_g,
        genes,
        X,
        adata,
        knn,
    ):
        coeff_df = pd.DataFrame()
        for i in ["const"] + uniq_g + ["W_log_exp"]:
            coeff_df[str(i) + "_GM_lag_coeff"] = None
            coeff_df[str(i) + "_GM_lag_zstat"] = None
            coeff_df[str(i) + "_GM_lag_pval"] = None
        col_names = coeff_df.columns

        if layer is None:
            X["log_exp"] = adata[:, cur_g].X.A
        else:
            X["log_exp"] = np.log1p(adata[:, cur_g].layers[layer].A)

        try:
            model = spreg.GM_Lag(
                X[["log_exp"]].values,
                X[variable_names].values,
                w=knn,
                name_y="log_exp",
                name_x=variable_names,
            )
            a = pd.DataFrame(model.betas, model.name_x + ["W_log_exp"], columns=["coef"])

            b = pd.DataFrame(
                model.z_stat,
                model.name_x + ["W_log_exp"],
                columns=["z_stat", "p_val"],
            )  # ?

            df = a.merge(b, left_index=True, right_index=True)

            # Store in the temporary df created above:
            for ind, g in enumerate(["const"] + uniq_g + ["W_log_exp"]):
                coeff_df.loc[:, str(g) + "_GM_lag_coeff"] = df.iloc[ind, 0]
                coeff_df.loc[:, str(g) + "_GM_lag_zstat"] = df.iloc[ind, 1]
                coeff_df.loc[:, str(g) + "_GM_lag_pval"] = df.iloc[ind, 2]

                adata.var.loc[cur_g, str(g) + "_GM_lag_coeff"] = df.iloc[ind, 0]
                adata.var.loc[cur_g, str(g) + "_GM_lag_zstat"] = df.iloc[ind, 1]
                adata.var.loc[cur_g, str(g) + "_GM_lag_pval"] = df.iloc[ind, 2]
        except:
            for ind, g in enumerate(["const"] + uniq_g + ["W_log_exp"]):
                adata.var.loc[cur_g, str(g) + "_GM_lag_coeff"] = np.nan
                adata.var.loc[cur_g, str(g) + "_GM_lag_zstat"] = np.nan
                adata.var.loc[cur_g, str(g) + "_GM_lag_pval"] = np.nan
        params = adata.var.loc[cur_g, :].values
        return params, col_names

    results = Parallel(n_jobs)(
        delayed(_single)(
            cur_g,
            genes,
            X,
            adata,
            knn,
        )
        for cur_g in genes
    )
    res = [item[0] for item in results]
    names = [item[1] for item in results]
    res = pd.DataFrame(res, index=genes)
    res.columns = adata.var.loc[genes, :].columns
    print(res['W_log_exp_GM_lag_coeff'])
    print(names)
    return res'''



brain_test = st.read_h5ad('/mnt/d/SCData/Stereo-seq/Mouse_brain/adata_X_labels_cluster.h5ad')
#GM_lag_model(brain_test, group='Celltype', genes=['Lamp5', 'Pvalb', 'Sst'], drop_dummy='OLIG')
#print(brain_test.var['AMY neuron_GM_lag_coeff'])


from spatial_regression import Category_Interpreter
test = Category_Interpreter(brain_test, group_key='Celltype', genes=['Lamp5', 'Pvalb', 'Sst'], drop_dummy='OLIG',
                            log_transform=True, data_id='mouse_brain_stereo')

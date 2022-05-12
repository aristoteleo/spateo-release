"""Spatial markers.
"""
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
from pysal import explore
from pysal.lib import weights
from pysal.model import spreg
from tqdm import tqdm

from ..configuration import SKM
from ..utils import copy_adata


@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE)
def lisa_geo_df(
    adata: anndata.AnnData,
    gene: str,
    spatial_key: str = "spatial",
    n_neighbors: int = 8,
    layer: Tuple[None, str] = None,
) -> geopandas.GeoDataFrame:
    """Perform Local Indicators of Spatial Association (LISA) analyses on specific genes and prepare a geopandas
    dataframe for downstream lisa plots to reveal the quantile plots and the hotspot, coldspot, doughnut and
    diamond regions.

    Args:
        adata: An adata object that has spatial information (via `spatial_key` key in adata.obsm).
        gene: The gene that will be used for lisa analyses, must be included in the data.
        spatial_key: The spatial key of the spatial coordinate of each bucket.
        n_neighbors: The number of nearest neighbors of each bucket that will be used in calculating the spatial lag.
        layer: the key to the layer. If it is None, adata.X will be used by default.

    Returns:
        df: a geopandas dataframe that includes the coordinate (`x`, `y` columns), expression (`exp` column) and lagged
        expression (`w_exp` column), z-score (`exp_zscore`, `w_exp_zscore`) and the LISA (`Is` column).
        score.
    """
    coords = adata.obsm[spatial_key]

    # Generate W from the GeoDataFrame
    w = weights.distance.KNN.from_array(coords, k=n_neighbors)
    # Row-standardization
    w.transform = "R"

    df = pd.DataFrame(coords, columns=["x", "y"])

    if layer is None:
        df["exp"] = adata[:, gene].X.A.flatten()
    else:
        df["exp"] = np.log1p(adata[:, gene].layers[layer].A.flatten())

    df["w_exp"] = weights.spatial_lag.lag_spatial(w, df["exp"])
    df["exp_zscore"] = (df["exp"] - df["exp"].mean()) / df["exp"].std()
    df["w_exp_zscore"] = (df["w_exp"] - df["w_exp"].mean()) / df["w_exp"].std()

    df["exp"] = df["exp"].astype(np.float64)
    lisa = explore.esda.moran.Moran_Local(df["exp"], w)
    df = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df.x, df.y))
    df = df.assign(Is=lisa.Is)

    q_labels = ["Q1", "Q2", "Q3", "Q4"]
    labels = [q_labels[i - 1] for i in lisa.q]
    df = df.assign(labels=labels)

    sig = 1 * (lisa.p_sim < 0.05)
    df = df.assign(sig=sig)

    hotspot = 1 * (sig * lisa.q == 1)
    coldspot = 3 * (sig * lisa.q == 3)
    doughnut = 2 * (sig * lisa.q == 2)
    diamond = 4 * (sig * lisa.q == 4)
    spots = hotspot + coldspot + doughnut + diamond
    spot_labels = ["0 ns", "1 hot spot", "2 doughnut", "3 cold spot", "4 diamond"]
    group = [spot_labels[i] for i in spots]
    df = df.assign(group=group)

    return (lisa, df)


@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE)
def local_moran_i(
    adata: anndata.AnnData,
    group: str,
    spatial_key: str = "spatial",
    genes: Tuple[None, list] = None,
    layer: Tuple[None, str] = None,
    n_neighbors: int = 5,
    copy: bool = False,
    n_jobs: int = 30,
):
    """Identify cell type specific genes with local Moran's I test.

    Args:
        adata: An adata object that has spatial information (via `spatial_key` key in adata.obsm).
        group: The key to the cell group in the adata.obs.
        spatial_key: The spatial key of the spatial coordinate of each bucket.
        genes: The gene that will be used for lisa analyses, must be included in the data.
        layer: the key to the layer. If it is None, adata.X will be used by default.
        n_neighbors: The number of nearest neighbors of each bucket that will be used in calculating the spatial lag.
        copy: Whether to copy the adata object.

    Returns:
        Depend on the `copy` argument, return a deep copied adata object (when `copy = True`) or inplace updated adata
        object. The resultant adata will include the following new columns in `adata.var`:
            {*}_num_val: The maximum number of categories (`{"hotspot", "coldspot", "doughnut", "diamond"}) across all
                         cell groups
            {*}_frac_val: The maximum fraction of categories across all cell groups
            {*}_spec_val: The maximum specificity of categories across all cell groups
            {*}_num_group: The corresponding cell group with the largest number of each category (this can be affect by
                           the cell group size).
            {*}_frac_group: The corresponding cell group with the highest fraction of each category.
            {*}_spec_group: The corresponding cell group with the highest specificity of each category.
        {*} can be one of `{"hotspot", "coldspot", "doughnut", "diamond"}`.

    Examples:
    >>> import spateo as st
    >>> markers_df = pd.DataFrame(adata.var).query("hotspot_frac_val > 0.05 & mean > 0.05").\
    >>> groupby(['hotspot_spec_group'])['hotspot_spec_val'].nlargest(5)
    >>> markers = markers_df.index.get_level_values(1)
    >>>
    >>> for i in adata.obs[group].unique():
    >>>     if i in markers_df.index.get_level_values(0):
    >>>         print(markers_df[i])
    >>>         dyn.pl.space(adata, color=group, highlights=[i], pointsize=0.1, alpha=1, figsize=(12, 8))
    >>>         st.pl.space(adata, color=markers_df[i].index, pointsize=0.1, alpha=1, figsize=(12, 8))
    """
    group_num = adata.obs[group].value_counts()
    group_name = adata.obs[group]
    uniq_g, group_name = group_name.unique(), group_name.to_list()

    # Generate W from the GeoDataFrame
    w = weights.distance.KNN.from_array(adata.obsm[spatial_key], k=n_neighbors)

    # Row-standardization
    w.transform = "R"

    if genes is None:
        genes = adata.var.index[adata.var.use_for_pca]
    else:
        genes = adata.var.index.intersection(genes)

    db = pd.DataFrame(adata.obsm[spatial_key], columns=["x", "y"])

    suffix = ["_num", "_frac", "_spec"]

    # hotspot: HH; coldspot: LL; doughnut: HL, diamond: LH; the first one is the query point
    # while the second the neighbors. Order on the quantile plot is 1, 3, 2, 4
    def _assign_columns(type):
        cat_group_list = [type + i + "_group" for i in suffix]
        cat_val_list = [type + i + "_val" for i in suffix]
        adata.var[cat_group_list[0]], adata.var[cat_group_list[1]], adata.var[cat_group_list[2]] = None, None, None
        adata.var[cat_val_list[0]], adata.var[cat_val_list[1]], adata.var[cat_val_list[2]] = None, None, None

    for i in ["hotspot", "coldspot", "doughnut", "diamond"]:
        _assign_columns(i)

    the_tuple = (
        np.zeros(len(uniq_g)),
        np.zeros(len(uniq_g)),
        np.zeros(len(uniq_g)),
    )
    hotspot_num, hotspot_frac, hotspot_spec = the_tuple
    coldspot_num, coldspot_frac, coldspot_spec = the_tuple
    doughnut_num, doughnut_frac, doughnut_spec = the_tuple
    diamond_num, diamond_frac, diamond_spec = the_tuple

    valid_inds_list = [np.array(group_name) == g for g in uniq_g]

    ###
    def _single(
        cur_g,
        db,
        w,
        adata,
        uniq_g,
        valid_inds_list,
        hotspot_num,
        hotspot_frac,
        hotspot_spec,
        coldspot_num,
        coldspot_frac,
        coldspot_spec,
        doughnut_num,
        doughnut_frac,
        doughnut_spec,
        diamond_num,
        diamond_frac,
        diamond_spec,
    ):
        if layer is None:
            db["exp"] = adata[:, cur_g].X.A.flatten()
        else:
            db["exp"] = np.log1p(adata[:, cur_g].layers[layer].A.flatten())

        db["w_exp"] = weights.spatial_lag.lag_spatial(w, db["exp"])

        db["exp"] = db["exp"].astype(np.float64)
        lisa = explore.esda.moran.Moran_Local(db["exp"], w, permutations=199)

        # find significant cells
        sig = 1 * (lisa.p_sim < 0.05)

        # get quantiles (z-score of cells within the neighborhood and that of the smoothed expression)
        hotspot = 1 * (sig * lisa.q == 1)
        coldspot = 3 * (sig * lisa.q == 3)
        doughnut = 2 * (sig * lisa.q == 2)
        diamond = 4 * (sig * lisa.q == 4)

        # calculate the value
        def _get_nums(spots, valid_inds, g):
            return (
                sum(spots[valid_inds] > 0),  # number of {*} (like hotspot, colospot, etc.) in this cell group
                sum(spots[valid_inds] > 0) / group_num[g],  # fraction of {*} in this cell group
                sum(spots[valid_inds] > 0) / sum(spots > 0),  # specificity of {*} in this cell group
            )

        # calculate the maximum val across all cell groups
        def _get_group_max(num, frac, spec):
            return (np.max(num), np.max(frac), np.max(spec))

        # get the group name with the maximum
        def _get_max_group_name(num, frac, spec):
            return (
                uniq_g[np.argsort(num)[-1]],
                uniq_g[np.argsort(frac)[-1]],
                uniq_g[np.argsort(spec)[-1]],
            )

        for ind, g in enumerate(uniq_g):
            valid_inds = valid_inds_list[ind]
            hotspot_num[ind], hotspot_frac[ind], hotspot_spec[ind] = _get_nums(hotspot, valid_inds, g)
            coldspot_num[ind], coldspot_frac[ind], coldspot_spec[ind] = _get_nums(coldspot, valid_inds, g)
            doughnut_num[ind], doughnut_frac[ind], doughnut_spec[ind] = _get_nums(doughnut, valid_inds, g)
            diamond_num[ind], diamond_frac[ind], diamond_spec[ind] = _get_nums(diamond, valid_inds, g)

        (
            adata.var.loc[cur_g, "hotspot_num_val"],
            adata.var.loc[cur_g, "hotspot_frac_val"],
            adata.var.loc[cur_g, "hotspot_spec_val"],
        ) = _get_group_max(hotspot_num, hotspot_frac, hotspot_spec)
        (
            adata.var.loc[cur_g, "coldspot_num_val"],
            adata.var.loc[cur_g, "coldspot_frac_val"],
            adata.var.loc[cur_g, "coldspot_spec_val"],
        ) = _get_group_max(coldspot_num, coldspot_frac, coldspot_spec)
        (
            adata.var.loc[cur_g, "doughnut_num_val"],
            adata.var.loc[cur_g, "doughnut_frac_val"],
            adata.var.loc[cur_g, "doughnut_spec_val"],
        ) = _get_group_max(doughnut_num, doughnut_frac, doughnut_spec)
        (
            adata.var.loc[cur_g, "diamond_num_val"],
            adata.var.loc[cur_g, "diamond_frac_val"],
            adata.var.loc[cur_g, "diamond_spec_val"],
        ) = _get_group_max(diamond_num, diamond_frac, diamond_spec)

        (
            adata.var.loc[cur_g, "hotspot_num_group"],
            adata.var.loc[cur_g, "hotspot_frac_group"],
            adata.var.loc[cur_g, "hotspot_spec_group"],
        ) = _get_max_group_name(hotspot_num, hotspot_frac, hotspot_spec)
        (
            adata.var.loc[cur_g, "coldspot_num_group"],
            adata.var.loc[cur_g, "coldspot_frac_group"],
            adata.var.loc[cur_g, "coldspot_spec_group"],
        ) = _get_max_group_name(coldspot_num, coldspot_frac, coldspot_spec)
        (
            adata.var.loc[cur_g, "doughnut_num_group"],
            adata.var.loc[cur_g, "doughnut_frac_group"],
            adata.var.loc[cur_g, "doughnut_spec_group"],
        ) = _get_max_group_name(doughnut_num, doughnut_frac, doughnut_spec)
        (
            adata.var.loc[cur_g, "diamond_num_group"],
            adata.var.loc[cur_g, "diamond_frac_group"],
            adata.var.loc[cur_g, "diamond_spec_group"],
        ) = _get_max_group_name(diamond_num, diamond_frac, diamond_spec)
        return adata.var.loc[cur_g, :].values

    # parallel computing
    res = Parallel(n_jobs)(
        delayed(_single)(
            cur_g,
            db,
            w,
            adata,
            uniq_g,
            valid_inds_list,
            hotspot_num,
            hotspot_frac,
            hotspot_spec,
            coldspot_num,
            coldspot_frac,
            coldspot_spec,
            doughnut_num,
            doughnut_frac,
            doughnut_spec,
            diamond_num,
            diamond_frac,
            diamond_spec,
        )
        for cur_g in genes
    )
    res = pd.DataFrame(res, index=genes)
    res = res.drop(columns=0)
    res.columns = adata.var.loc[genes, :].columns.drop("mt")
    return res


@SKM.check_adata_is_type(SKM.ADATA_UMI_TYPE)
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

    if drop_dummy is None:
        db.iloc[sample(np.arange(adata.n_obs).tolist(), min_group_ncells), :] = "others"
        drop_columns = ["group_others"]  # ?
    elif drop_dummy in group_name:
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

            for ind, g in enumerate(["const"] + uniq_g + ["W_log_exp"]):
                adata.var.loc[cur_g, str(g) + "_GM_lag_coeff"] = df.iloc[ind, 0]
                adata.var.loc[cur_g, str(g) + "_GM_lag_zstat"] = df.iloc[ind, 1]
                adata.var.loc[cur_g, str(g) + "_GM_lag_pval"] = df.iloc[ind, 2]
        except:
            for ind, g in enumerate(["const"] + uniq_g + ["W_log_exp"]):
                adata.var.loc[cur_g, str(g) + "_GM_lag_coeff"] = np.nan
                adata.var.loc[cur_g, str(g) + "_GM_lag_zstat"] = np.nan
                adata.var.loc[cur_g, str(g) + "_GM_lag_pval"] = np.nan
        return adata.var.loc[cur_g, :].values

    res = Parallel(n_jobs)(
        delayed(_single)(
            cur_g,
            genes,
            X,
            adata,
            knn,
        )
        for cur_g in genes
    )
    res = pd.DataFrame(res, index=genes)
    res.columns = adata.var.loc[genes, :].columns
    return res

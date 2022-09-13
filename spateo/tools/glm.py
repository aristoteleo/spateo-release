import re
from typing import Optional

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from anndata import AnnData
from patsy import bs, cr, dmatrix
from scipy import stats
from scipy.sparse import issparse
from statsmodels.sandbox.stats.multicomp import multipletests

from ..logging import logger_manager as lm


def glm_degs(
    adata: AnnData,
    X_data: Optional[np.ndarray] = None,
    genes: Optional[list] = None,
    layer: Optional[str] = None,
    key_added: str = "glm_degs",
    fullModelFormulaStr: str = "~cr(time, df=3)",
    reducedModelFormulaStr: str = "~1",
    qval_threshold: Optional[float] = 0.05,
    llf_threshold: Optional[float] = -2000,
    ci_alpha: float = 0.05,
    inplace: bool = True,
) -> Optional[AnnData]:
    """Differential genes expression tests using generalized linear regressions. Here only size factor normalized gene
    expression matrix can be used, and SCT/pearson residuals transformed gene expression can not be used.

    Tests each gene for differential expression as a function of integral time (the time estimated via the reconstructed
    vector field function) or pseudo-time using generalized additive models with natural spline basis. This function can
    also use other co-variates as specified in the full (i.e `~clusters`) and reduced model formula to identify differentially
    expression genes across different categories, group, etc.
    glm_degs relies on statsmodels package and is adapted from the `differentialGeneTest` function in Monocle. Note that
    glm_degs supports performing deg analysis for any layer or normalized data in your adata object. That is you can either
    use the total, new, unspliced or velocity, etc. for the differential expression analysis.

    Args:
        adata: An Anndata object. The anndata object must contain a size factor normalized gene expression matrix.
        X_data: The user supplied data that will be used for differential expression analysis directly.
        genes: The list of genes that will be used to subset the data for differential expression analysis. If ``genes = None``, all genes will be used.
        layer: The layer that will be used to retrieve data for dimension reduction and clustering. If ``layer = None``, ``.X`` is used.
        key_added: The key that will be used for the glm_degs key in ``.uns``.
        fullModelFormulaStr: A formula string specifying the full model in differential expression tests (i.e. likelihood ratio tests) for each gene/feature.
        reducedModelFormulaStr: A formula string specifying the reduced model in differential expression tests (i.e. likelihood ratio tests) for each gene/feature.
        qval_threshold: Only keep the glm test results whose qval is less than the ``qval_threshold``.
        llf_threshold: Only keep the glm test results whose log-likelihood is less than the ``llf_threshold``.
        ci_alpha: The significance level for the confidence interval. The default ``ci_alpha = .05`` returns a 95% confidence interval.
        inplace: Whether to copy adata or modify it inplace.

    Returns:
        An ``AnnData`` object is updated/copied with the ``key_added`` dictionary in the ``.uns`` attribute, storing the differential
        expression test results after the GLM test.
    """

    adata = adata if inplace else adata.copy()

    if X_data is None:
        from dynamo.tools.utils import fetch_X_data

        genes, X_data = fetch_X_data(adata, genes, layer)
    else:
        assert (
            genes is not None
        ), "When providing X_data, a list of genes name that corresponds to the columns of X_data must be provided."
        assert (
            len(genes) == X_data.shape[1]
        ), "When providing X_data, the number of genes must be equal the columns of X_data."
    lm.main_warning(
        "Gene expression matrix must be normalized by the size factor, please check if the input gene expression matrix is correct."
        "If you don't have the size factor normalized gene expression matrix, please run `dynamo.pp.normalize_cell_expr_by_size_factors(skip_log = True)`."
    )

    md = patsy.ModelDesc.from_formula(fullModelFormulaStr)
    termlist = md.rhs_termlist + md.lhs_termlist
    factors = []
    for term in termlist:
        for factor in term.factors:
            factor_name = re.findall(r"cr\((.+?), df=\d\)", factor.name())
            factors.append(factor.name() if factor_name == [] else factor_name[0])
    assert set(factors).issubset(
        set(adata.obs.columns)
    ), f"adata object doesn't include the factors from the model formula {fullModelFormulaStr} you provided."
    df_factors = adata.obs[factors]

    sparse = issparse(X_data)
    deg_df = pd.DataFrame(index=genes, columns=["status", "family", "log-likelihood", "pval"])
    deg_dict = {}
    for i in lm.progress_logger(
        range(len(genes)), progress_name="Detecting genes via Generalized Additive Models (GAMs)"
    ):
        gene = genes[i]
        expression = X_data[:, i].A if sparse else X_data[:, i]
        df_factors["expression"] = expression
        try:
            nb2_full, nb2_null = glm_test(df_factors, fullModelFormulaStr, reducedModelFormulaStr)

            pval = lrt(nb2_full, nb2_null)
            deg_df.iloc[i, :] = ("ok", "NB2", nb2_full.llf, pval)

            df_factors_gene = df_factors.copy()
            df_factors_gene["mu"] = nb2_full.mu
            # df_factors_gene["fitted_expression"] = nb2_full.predict() this is equal to nb2_full.mu
            df_factors_gene["resid_deviance"] = nb2_full.resid_deviance
            df_factors_gene["resid_pearson"] = nb2_full.resid_pearson
            df_factors_gene[["ci_lower", "ci_upper"]] = nb2_full.get_prediction().conf_int(alpha=ci_alpha)
            deg_dict[gene] = df_factors_gene
        except:
            deg_df.iloc[i, :] = ("fail", "NB2", "None", 1)

    deg_df["qval"] = multipletests(deg_df["pval"], method="fdr_bh")[1]
    deg_df = deg_df[deg_df["log-likelihood"] != "None"]
    deg_df.dropna(axis=0, how="any", inplace=True)
    deg_df[["log-likelihood", "pval", "qval"]] = deg_df[["log-likelihood", "pval", "qval"]].astype(np.float32)
    deg_df = deg_df.sort_values(by=["qval", "pval", "log-likelihood"], ascending=[True, True, True])

    if not (qval_threshold is None and llf_threshold is None):
        cut_deg_df = deg_df[deg_df["qval"] <= qval_threshold] if not (qval_threshold is None) else deg_df
        cut_deg_df = (
            cut_deg_df[cut_deg_df["log-likelihood"] <= llf_threshold] if not (llf_threshold is None) else cut_deg_df
        )
        cut_deg_dict = {gene: deg_dict[gene] for gene in cut_deg_df.index}
        adata.uns[key_added] = {"glm_result": cut_deg_df, "correlation": cut_deg_dict}
    else:
        adata.uns[key_added] = {"glm_result": deg_df, "correlation": deg_dict}
    return None if inplace else adata


def glm_test(
    data,
    fullModelFormulaStr="~cr(time, df=3)",
    reducedModelFormulaStr="~1",
):
    transformed_x = dmatrix(fullModelFormulaStr, data, return_type="dataframe")
    transformed_x_null = dmatrix(reducedModelFormulaStr, data, return_type="dataframe")

    nb2_family = sm.families.NegativeBinomial()  # (alpha=aux_olsr_results.params[0])
    nb2_full = sm.GLM(data["expression"], transformed_x, family=nb2_family).fit()
    nb2_null = sm.GLM(data["expression"], transformed_x_null, family=nb2_family).fit()
    return nb2_full, nb2_null


def lrt(full, restr):
    llf_full = full.llf
    llf_restr = restr.llf
    df_full = full.df_resid
    df_restr = restr.df_resid
    lrdf = df_restr - df_full
    lrstat = -2 * (llf_restr - llf_full)
    lr_pvalue = stats.chi2.sf(lrstat, df=lrdf)

    return lr_pvalue

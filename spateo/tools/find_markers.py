# after cluster(the cluster method contains spatial positions), find spatial
# different expression genes(DEG).


def findmarkers(
    adata,
    test_group,
    neighbor_groups,
    method: str = "wilcox_test",
):
    """Finds markers (differentially expressed genes) for identity clusters for
    each cluster, find statistical test different genes between one cluster
    and the rest clusters using wilcox_test or others
    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            an Annodata object
        test_group: one cluster (defult all clusters),
        neighbor_groupss: the neighbor clusters which detect the neighbors as
            "SpaGCN" (or rest of clusters)
        method: 'str'
            Denotes which test to use. Available options are:
            "wilcox" : Identifies differentially expressed genes between two
                groups of cells using a Wilcoxon Rank Sum test (default)

            "bimod" : Likelihood-ratio test for single cell gene expression,
                (McDavid et al., Bioinformatics, 2013)

            "roc" : Identifies 'markers' of gene expression using ROC analysis.
                For each gene, evaluates (using AUC) a classifier built on that
                gene alone, to classify between two groups of cells.
                An AUC value of 1 means that expression values for this gene
                alone can perfectly classify the two groupings (i.e. Each of
                the cells in cells.1 exhibit a higher level than each of the
                cells in cells.2). An AUC value of 0 also means there is
                perfect classification, but in the other direction. A value
                of 0.5 implies that the gene has no predictive power to
                classify the two groups. Returns a 'predictive power'
                (abs(AUC-0.5) * 2) ranked matrix of putative differentially
                expressed genes.

            "t" : Identify differentially expressed genes between two groups of
                cells using the Student's t-test.

            "negbinom" : Identifies differentially expressed genes between two
                groups of cells using a negative binomial generalized linear
                model. Use only for UMI-based datasets

            "poisson" : Identifies differentially expressed genes between two
                groups of cells using a poisson generalized linear model.
                Use only for UMI-based datasets

            "LR" : Uses a logistic regression framework to determine
                differentially expressed genes. Constructs a logistic
                regression model predicting group membership based on each
                feature individually and compares this to a null model with
                a likelihood ratio test.
    Returns
    -------
        DEG: 'pd.DataFrame'
            Returns a 'pd.DataFrame', each row of the DataFrame correspond to a
            gene, columns contains statistic results(e.g gene,p_value, group
            avg_fold_change et.al)
    """


# before cluster, detect the highly variable genes(HVG).


def moran_i(
    adata,
    genes,
    weighted,
    k=5,
    assumption="permutation",
):
    """Identify genes with strong spatial autocorrelation with Moran's I test.

    Parameters
    ----------
        adata: class:`~anndata.AnnData`
            An Annodata object
        genes: `list` or None (default: `None`)
            The list of genes that will be used to subset the data for dimension
            reduction and clustering. If `None`, all genes will be used.
        k: 'int'
            The number of neighbors around a given cell or bin.
        weighted: 'matrix'
            A spatial weights matrix.
        assumption: `str`(defult: `permutation` )
            Monte Carlo test  (a permutation bootstrap test)

    Returns
    -------
        I: `list`
            The Moran'I for given genes, each elements of list a 'float'
            between (-1,1)
    """


def Findspatialvariablegenes(
    adata,
    model: str = "Gaussion Process regression",
):
    """Detect the genes which expression highly variable in the whole domains.

    Parameters
     ----------
        adata: class:`~anndata.AnnData`
            An Annodata object
        model: 'str'
            Denotes which model to use. Available options are:
            "Gaussion Process regression" : the gene expression profiles
                y = (y1, … , yN) for a given gene across spatial coordinates
                X = (x1, … , xN) using a multivariate normal model. methods
                contain 'spatialDE'(Svensson et.al 2018),'spatialDE2'
                (Kats et.al 2021)

            "Generalized linear spatial model" : 'Spark'(Sun et.al 2020)

            "Regularized negative binomial regression": the sum of all molecules
                assigned to a cell as a proxy for sequencing depth and use this
                cell attribute in a regression model with negative binomial (NB)
                error distribution and log link function(Hafemeister et.al 2019)
     Returns
     -------
         HVG: `pd.DataFrame`
             The spatial highly variable genes.
    """

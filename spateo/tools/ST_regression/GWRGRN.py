"""
Modeling putative gene regulatory networks using a regression model that is considerate of the spatial heterogeneity of
(and thus the context-dependency of the relationships of) the response variable.
"""
import argparse
import sys

from mpi4py import MPI

# For now, add Spateo working directory to sys path so compiler doesn't look in the installed packages:
sys.path.insert(0, "/mnt/c/Users/danie/Desktop/Github/Github/spateo-release-main")

from spateo.logging import logger_manager as lm


# ---------------------------------------------------------------------------------------------------
# GWR for inferring gene regulatory networks
# ---------------------------------------------------------------------------------------------------
class GWRGRN:
    """
    Construct regulatory networks, taking prior knowledge network and spatial expression patterns into account.

    Args:
        MPI_comm: MPI communicator object initialized with mpi4py, to control parallel processing operations
        parser: ArgumentParser object initialized with argparse, to parse command line arguments for arguments
            pertinent to modeling.

    Attributes:
        adata_path: Path to the AnnData object from which to extract data for modeling
        normalize: Set True to Perform library size normalization, to set total counts in each cell to the same
            number (adjust for cell size). It is advisable not to do this if performing Poisson or negative binomial
            regression.
        smooth: Set True to correct for dropout effects by leveraging gene expression neighborhoods to smooth
            expression. It is advisable not to do this if performing Poisson or negative binomial regression.
        log_transform: Set True if log-transformation should be applied to expression. It is advisable not to do
            this if performing Poisson or negative binomial regression.


        custom_lig_path: Optional path to a .txt file containing a list of ligands for the model, separated by
            newlines. If not provided, will select ligands using a threshold based on expression
            levels in the data.
        custom_rec_path: Optional path to a .txt file containing a list of receptors for the model, separated by
            newlines. If not provided, will select receptors using a threshold based on expression
            levels in the data.
        custom_pathways_path: Rather than  providing a list of receptors, can provide a list of signaling pathways-
            all receptors with annotations in this pathway will be included in the model.
        custom_tf_path: Optional path to a .txt file containing a list of transcription factors for the model. If not
            provided, will select transcription factors using a threshold based on expression levels in the data.


        cci_dir: Full path to the directory containing cell-cell communication databases
        species: Selects the cell-cell communication database the relevant ligands will be drawn from. Options:
                "human", "mouse".
        output_path: Full path name for the .csv file in which results will be saved.


        coords_key: Key in .obsm of the AnnData object that contains the coordinates of the cells
        bw: Used to provide previously obtained bandwidth for the spatial kernel. Consists of either a distance
            value or N for the number of nearest neighbors. Can be obtained using BW_Selector or some other
            user-defined method. Pass "np.inf" if all other points should have the same spatial weight. Defaults to
            1000 if not provided.
        minbw: For use in automated bandwidth selection- the lower-bound bandwidth to test.
        maxbw: For use in automated bandwidth selection- the upper-bound bandwidth to test.


        distr: Distribution family for the dependent variable; one of "gaussian", "poisson", "nb"
        kernel: Type of kernel function used to weight observations; one of "bisquare", "exponential", "gaussian",
            "quadratic", "triangular" or "uniform".


        bw_fixed: Set True for distance-based kernel function and False for nearest neighbor-based kernel function
        exclude_self: If True, ignore each sample itself when computing the kernel density estimation
        fit_intercept: Set True to include intercept in the model and False to exclude intercept
    """

    def __init__(self, comm: MPI.Comm, parser: argparse.ArgumentParser):
        self.logger = lm.get_main_logger()

        self.comm = comm
        self.parser = parser

        self.species = None
        self.ligands = None
        self.receptors = None
        self.tfs = None
        self.normalize = None
        self.smooth = None
        self.log_transform = None

        self.coords = None
        self.y = None
        self.X = None

        self.bw = None
        self.minbw = None
        self.maxbw = None

        self.distr = None
        self.kernel = None
        self.n_samples = None
        self.n_features = None
        # Number of STGWR runs to go through:
        self.n_runs_alls = None

        self.parse_stgwr_args()

    def parse_stgwr_args(self):
        """
        Parse command line arguments for arguments pertinent to modeling.
        """
        arg_retrieve = self.parser.parse_args()

    def grn_load_and_process(self):
        "filler"

    def grn_mpi_fit(self):
        "filler"

    def grn_fit(self):
        "filler"

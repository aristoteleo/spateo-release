import argparse

from mpi4py import MPI
from spatial_regression import STGWR

if __name__ == "__main__":
    # From the command line, run spatial GWR

    # Initialize MPI

    parser = argparse.ArgumentParser(description="Spatial GWR")
    parser.add_argument("-data_path", type=str)
    parser.add_argument("-mod_type", default="niche", type=str)
    parser.add_argument("-cci_dir", type=str)
    parser.add_argument("-species", type=str, default="human")
    parser.add_argument("-output_path", default="./output/stgwr_results.csv", type=str)
    parser.add_argument("-custom_lig_path", type=str)
    parser.add_argument("-custom_rec_path", type=str)
    parser.add_argument("-targets_path", type=str)

    parser.add_argument("-normalize", action="store_true")
    parser.add_argument("-smooth", action="store_true")
    parser.add_argument("-log_transform", action="store_true")

    parser.add_argument("-coords_key", default="spatial", type=str)
    parser.add_argument("-group_key", default="cell_type", type=str)

    parser.add_argument("-bw")
    parser.add_argument("-minbw")
    parser.add_argument("-maxbw")
    parser.add_argument("-bw_fixed", action="store_true")
    parser.add_argument("-kernel", default="gaussian", type=str)

    parser.add_argument("-distr", default="gaussian", type=str)
    parser.add_argument("-fit_intercept", action="store_true")

    # For now, use a dummy class for Comm:
    class Comm:
        def __init__(self):
            self.rank = 0
            self.size = 1

    Comm_obj = Comm()

    test_model = STGWR(Comm_obj, parser)
    print(test_model.cell_categories)
    print(test_model.ligands_expr)
    print(test_model.receptors_expr)

import argparse

import numpy as np
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
    parser.add_argument("-custom_tf_path", type=str)
    parser.add_argument("-custom_pathways_path", type=str)
    parser.add_argument("-targets_path", type=str)
    parser.add_argument("-init_betas_path", type=str)

    parser.add_argument("-normalize", action="store_true")
    parser.add_argument("-smooth", action="store_true")
    parser.add_argument("-log_transform", action="store_true")
    parser.add_argument("-target_expr_threshold", default=0.2, type=float)

    parser.add_argument("-coords_key", default="spatial", type=str)
    parser.add_argument("-group_key", default="cell_type", type=str)

    parser.add_argument("-bw")
    parser.add_argument("-minbw")
    parser.add_argument("-maxbw")
    parser.add_argument("-bw_fixed", action="store_true")
    parser.add_argument("-exclude_self", action="store_true")
    parser.add_argument("-kernel", default="gaussian", type=str)

    parser.add_argument("-distr", default="gaussian", type=str)
    parser.add_argument("-fit_intercept", action="store_true")
    parser.add_argument("-tolerance", default=1e-5, type=float)
    parser.add_argument("-max_iter", default=500, type=int)
    parser.add_argument("-alpha", type=float)

    # For now, use a dummy class for Comm:
    class Comm:
        def __init__(self):
            self.rank = 0
            self.size = 1

    Comm_obj = Comm()

    test_model = STGWR(Comm_obj, parser)
    """
    print(test_model.adata[:, "SDC1"].X)
    #print(test_model.cell_categories)
    print(test_model.ligands_expr)
    print(test_model.receptors_expr)
    print(test_model.targets_expr)

    # See if the correct numbers show up:
    print(test_model.all_spatial_weights[121])
    print(test_model.all_spatial_weights[121].shape)
    neighbors = np.argpartition(test_model.all_spatial_weights[121].toarray().ravel(), -10)[-10:]

    print(neighbors)
    print(test_model.receptors_expr["SDC1"].iloc[121])
    print(test_model.ligands_expr["TNC"].iloc[neighbors])
    print(test_model.ligands_expr["TNC"].iloc[103])"""

    test_model._adjust_x()
    # print(test_model.X[121])

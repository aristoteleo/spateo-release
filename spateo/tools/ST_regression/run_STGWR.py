"""
Enables STGWR to be run using the "run" command rather than needing to navigate to and call the main file
(STGWR_mpi.py).
"""
import os
import sys

import click

# For now, add Spateo working directory to sys path so compiler doesn't look in the installed packages:
sys.path.insert(0, "/mnt/c/Users/danie/Desktop/Github/Github/spateo-release-main")
import spateo.tools.ST_regression as fast_stgwr


@click.group()
@click.version_option("0.3.2")
def main():
    pass


@main.command()
@click.option(
    "n_processes",
    default=2,
    help="Number of processes to use. Note the max number of processes is " "determined by the number of CPUs.",
    required=True,
)
@click.option("adata_path")
@click.option(
    "csv_path",
    help="Can be used to provide a .csv file, containing gene expression data or any other kind of data. "
    "Assumes the first three columns contain x- and y-coordinates and then dependent variable "
    "values, in that order.",
)
@click.option("mgwr", default=False, required=False, is_flag=True)
@click.option(
    "mod_type",
    default="niche",
    required=False,
    help="If adata_path is provided, one of the STGWR models " "will be used. Options: 'niche', 'lr', 'slice'.",
)
def run_STGWR():
    "filler"


"""
parser.add_argument("-adata_path", type=str)
parser.add_argument(
    "-csv_path",
    type=str,
    help="Can be used to provide a .csv file, containing gene expression data or any other kind of data. "
         "Assumes the first three columns contain x- and y-coordinates and then dependent variable "
         "values, in that order.",
)
parser.add_argument("-mgwr", action="store_true")
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
parser.add_argument(
    "-target_expr_threshold",
    default=0.2,
    type=float,
    help="For automated selection, the threshold proportion of cells for which transcript "
         "needs to be expressed in to be selected as a target of interest.",
)

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
parser.add_argument("-alpha", type=float)"""

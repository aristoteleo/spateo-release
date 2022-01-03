"""Assign spots to individual cells.

Todo:
    * Instead of writing a new matrix file with the assigned cell labels,
        the `cell_labels` and `cens` should be used directly by an IO function
        to create an AnnData. In this sense, this file is probably going to be
        removed, and its functionality split across `pp.segmentation.label`
        and `io`.
"""
from collections import Counter
from typing import Dict, Tuple

import numpy as np
from skimage.draw import disk
from skimage.measure import regionprops


def find_cell_centroids(cell_labels: np.ndarray) -> Dict[int, Tuple[int, int]]:
    """Find coordinates of centroid of all cells.

    Args:
        cell_labels: Numpy array containing cell labels as integers

    Returns:
        Dictionary of cell labels as keys and (rounded) centroid coordinates as
        a tuple
    """
    props = regionprops(cell_labels)
    cens = {}
    for p in props:
        cens[p.label] = [round(i) for i in p.centroid]
    return cens


def get_circle_per_dict(x: int, y: int, radius: int, cell_labels: np.ndarray):
    """"""
    rr, cc = disk((y, x), radius, shape=cell_labels.shape)  # Y X
    all_labels = cell_labels[rr, cc]
    all_labels = [i for i in all_labels if i > 0]
    if len(all_labels) == 0:
        return None
    else:
        cell_labels_per_dict = Counter(all_labels)
        for key in cell_labels_per_dict:
            cell_labels_per_dict[key] /= len(all_labels)
        return cell_labels_per_dict


def assign_point(
    long_matrix_file: str,
    cell_labels: np.ndarray,
    cens: Dict[int, Tuple[int, int]],
    xmin: int,
    ymin: int,
    radius: int = 10,  # radius = 1 => no effect
):
    """Assign non-cell spots to its adjacent cells according to overlap between
    a circle with target spot as center and adjacent cells. Signal of target
    spot is allocated to different cells according to the proportion of
    overlapping area with different cells.

    Args:
        long_matrix_file: File contains information about cell labels.
        cell_labels: Numpy array containing cell labels as integers
        cens: Dictionary containing cell centroid coordinates, as returned by
            :func:`find_cell_centroids`
        xmin: The x for [0,0] element in `cell_labels`.
        ymin: The y for [0,0] element in `cell_labels`.
        adius: The radius of the target spot to be considered.
    """
    assigns = {}
    o = open("expressionCellLabelsAssign.matrix", "wt")
    with open(long_matrix_file, "rt") as f:
        o.write(f.readline())
        for line in f:
            lines = line.strip().split("\t")
            gene, x, y, umi, cell_label = (
                lines[0],
                int(lines[1]) - xmin,
                int(lines[2]) - ymin,
                int(lines[3]),
                int(lines[4]),
            )
            if cell_label == 0:
                cell_labels_per_dict = get_circle_per_dict(x, y, radius, cell_labels)
                if cell_labels_per_dict == None:
                    o.write(line.strip() + "\n")
                else:
                    # print(cell_labels_per_dict)
                    for cell in cell_labels_per_dict:
                        if cell not in assigns:
                            assigns[cell] = {}
                        if gene not in assigns[cell]:
                            assigns[cell][gene] = 0
                        assigns[cell][gene] += cell_labels_per_dict[cell] * umi
            elif (
                cell_label > 0 and y == cens[cell_label][0] and x == cens[cell_label][1]
            ):
                if cell_label not in assigns:
                    assigns[cell_label] = {}
                if gene not in assigns[cell_label]:
                    assigns[cell_label][gene] = 0
                assigns[cell_label][gene] += umi
            else:
                o.write(f"{line.strip()}\n")

    for cell in assigns:
        for gene in assigns[cell]:
            o.write(
                f"{gene}\t{cens[cell][1]+xmin}\t{cens[cell][0]+ymin}\t{assigns[cell][gene]}\t{cell}\t{cens[cell][0]+ymin}\t{cens[cell][1]+xmin}\n"
            )
    o.close()

import copy
from typing import List, Tuple, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np
import scipy
from anndata import AnnData

from ..logging import logger_manager as lm

# --------------------------------------- Normalizing sparse arrays --------------------------------------- #


def row_normalize(
    graph: scipy.sparse.csr_matrix,
    copy: bool = False,
    verbose: bool = True,
) -> scipy.sparse.csr_matrix:
    """Normalize a compressed sparse row (CSR) matrix by row- written for sparse pairwise distance arrays, but can be
    applied to any sparse matrix.

    Args:
        graph: Sparse array of shape [n_samples, n_features]. If pairwise distance array, shape [n_samples, n_samples].
        copy: If True, create a copy of the graph before computations so that the original is preserved.
        verbose: If True, prints number of nonzero entries.

    Returns:
        graph: Input array (or the copy of the input array) post-normalization.
    """
    logger = lm.get_main_logger()

    if copy:
        logger.info(
            "Deep copying AnnData object and working on the new copy. Original AnnData object will not be modified.",
            indent_level=1,
        )
        graph = graph.copy()

    data = graph.data

    for start_ptr, end_ptr in zip(graph.indptr[:-1], graph.indptr[1:]):

        row_sum = data[start_ptr:end_ptr].sum()

        if row_sum != 0:
            data[start_ptr:end_ptr] /= row_sum

        if verbose:
            logger.info(
                f"Computed normalized sum from ptr {start_ptr} to {end_ptr}. "
                f"Total entries: {end_ptr - start_ptr}, sum: {np.sum(graph.data[start_ptr:end_ptr])}"
            )

    return graph


# --------------------------------------- Label class --------------------------------------- #


class Label(object):
    """Given categorizations for a set of points, wrap into a Label class.

    labels_dense: Numerical labels.
    str_map: Optional mapping of numerical labels (keys) to strings (values).
    verbose: whether to print running info of row_normalize.
    """

    def __init__(
        self,
        labels_dense: Union[np.ndarray, list],
        str_map: Union[None, dict] = None,
        verbose: bool = False,
    ) -> None:
        logger = lm.get_main_logger()

        # Check type, dimensions, ensure all elements non-negative
        if isinstance(labels_dense, list):
            labels_dense = np.asarray(labels_dense, dtype=np.int32)
        elif isinstance(labels_dense, np.ndarray):
            pass
        else:
            logger.error(f"Labels provided are of type {type(labels_dense)}. Should be list or 1-dimensional ndarray.")
            raise TypeError(
                f"Labels provided are of type {type(labels_dense)}. "
                f"Should be list or 1-dimensional numpy ndarray.\n"
            )

        if labels_dense.ndim != 1:
            logger.error(f"Label array has {labels_dense.ndim} dimensions, should be 1-dimensional.")
            raise ValueError(f"Label array has {labels_dense.ndim} dimensions, " f"should be 1-dimensional.")

        if not np.issubdtype(labels_dense.dtype, np.integer):
            logger.error(f"Label array data type is {labels_dense.dtype}, should be integer.")
            raise TypeError(f"Label array data type is {labels_dense.dtype}, " f"should be integer.")

        if np.amin(labels_dense) < 0:
            logger.error(f"Some of the labels have negative values. All labels must be 0 or positive integers.")
            raise ValueError(
                f"Some of the labels have negative values.\n" f"All labels must be 0 or positive integers.\n"
            )

        # Initialize attributes
        self.dense = labels_dense
        # Total number of data-points with label (e.g. number of cells)
        self.num_samples = len(labels_dense)

        # Number of instances of each integer up to maximum label id
        self.bins = np.bincount(self.dense)
        # Unique labels (all non-negative integers)
        self.ids = np.nonzero(self.bins)[0]
        # Counts per label (same order as self.ids)
        self.counts = self.bins[self.ids]
        # Highest integer id for a label
        self.max_id = np.amax(self.ids)
        # Total number of labels
        self.num_labels = len(self.ids)
        # Verbose
        self.verbose = verbose

        # Mapping from numerical labels to strings
        if str_map is not None:
            self.str_map = str_map
            self.str_labels = list(map(self.str_map.get, labels_dense))
            self.str_ids = list(map(self.str_map.get, self.ids))

        self.onehot = None
        self.normalized_onehot = None

    def __repr__(self) -> str:
        return f"{self.num_labels} labels, {self.num_samples} samples, " f"ids: {self.ids}, counts: {self.counts}"

    def __str__(self) -> str:
        return (
            f"Label object:\n"
            f"Number of labels: {self.num_labels}, "
            f"number of samples: {self.num_samples}\n"
            f"ids: {self.ids}, counts: {self.counts},\n"
        )

    def get_onehot(self) -> scipy.sparse.csr_matrix:
        """return one-hot sparse array of labels.
        If not already computed, generate the sparse array from dense label array
        """
        if self.onehot is None:
            self.onehot = self.generate_onehot()

        return self.onehot

    def get_normalized_onehot(self) -> scipy.sparse.csr_matrix:
        """Return normalized one-hot sparse array of labels."""
        if self.normalized_onehot is None:
            self.normalized_onehot = self.generate_normalized_onehot()

        return self.normalized_onehot

    def generate_normalized_onehot(self) -> scipy.sparse.csr_matrix:
        """Generate a normalized onehot matrix where each row is normalized by the count of that label
        e.g. a row [0 1 1 0 0] will be converted to [0 0.5 0.5 0 0]
        """
        return row_normalize(self.get_onehot().astype(np.float64), verbose=self.verbose, copy=True)

    def generate_onehot(self) -> scipy.sparse.csr_matrix:
        """Convert an array of labels to a num_labels x num_samples sparse one-hot matrix
        Labels MUST be integers starting from 0, but can have gaps in between e.g. [0,1,5,9]
        """
        logger = lm.get_main_logger()

        # Initialize the fields of the CSR
        indptr = np.zeros((self.num_labels + 1,), dtype=np.int32)
        indices = np.zeros((self.num_samples,), dtype=np.int32)
        data = np.ones_like(indices, dtype=np.int32)

        logger.info(
            f"\n--- {self.num_labels} labels, "
            f"{self.num_samples} samples ---\n"
            f"initalized {indptr.shape} index ptr: {indptr}\n"
            f"initalized {indices.shape} indices: {indices}\n"
            f"initalized {data.shape} data: {data}\n"
        )

        # Update index pointer and indices row by row
        for n, label in enumerate(self.ids):
            label_indices = np.nonzero(self.dense == label)[0]
            label_count = len(label_indices)

            previous_ptr = indptr[n]
            current_ptr = previous_ptr + label_count
            indptr[n + 1] = current_ptr

            if self.verbose:
                logger.info(
                    f"indices for label {label}: {label_indices}\n"
                    f"previous pointer: {previous_ptr}, "
                    f"current pointer: {current_ptr}\n"
                )

            if current_ptr > previous_ptr:
                indices[previous_ptr:current_ptr] = label_indices

        return scipy.sparse.csr_matrix((data, indices, indptr), shape=(self.num_labels, self.num_samples))


# --------------------------------------- Label Curation and Label Processing --------------------------------------- #


def _rand_binary_array(array_length, num_onbits):
    array = np.zeros(array_length, dtype=np.int32)
    array[:num_onbits] = 1
    np.random.shuffle(array)
    return array


def expand_labels(
    label: Label,
    max_label_id: int,
    sort_labels: bool = False,
) -> Label:
    """Spread out label IDs such that they range evenly from 0 to max_label_id, e.g. [0 1 2] -> [0 5 10]
    Useful if you need to be consistent with other label sets with many more label IDs.
    This spreads labels out along the color spectrum/map so that the colors are not too similar to each other.
    Use sort_labels if the list of IDs are not already sorted (although IDs are typically already sorted)
    """
    logger = lm.get_main_logger()
    logger.info(f"Expanding labels with ids: {label.ids} so that ids range from 0 to {max_label_id}")

    if sort_labels:
        ids = np.sort(copy.copy(label.ids))
    else:
        ids = copy.copy(label.ids)

    # Make sure smallest label ID is zero
    ids_zeroed = ids - np.amin(label.ids)
    num_extra_labels = max_label_id - np.amax(ids_zeroed)
    multiple, remainder = np.divmod(num_extra_labels, label.num_labels - 1)

    # Insert regular spaces between each id
    inserted = np.arange(label.num_labels) * multiple
    # Insert remaining spaces so that max label id equals given max_id
    extra = _rand_binary_array(label.num_labels - 1, remainder)
    expanded_ids = ids_zeroed + inserted
    expanded_ids[1:] += np.cumsum(extra)  # only add to 2nd label and above

    logger.info(
        f"Label ids zerod: {ids_zeroed}.\n"
        f"{multiple} to be inserted between each id: {inserted}\n"
        f"{remainder} extra rows to be randomly inserted: {extra}\n"
        f"New ids: {expanded_ids}"
    )

    expanded_dense = (expanded_ids @ label.get_onehot()).astype(np.int32)

    return Label(expanded_dense)


def match_labels(
    labels_1: Label,
    labels_2: Label,
    extra_labels_assignment: str = "random",
    verbose: bool = False,
) -> Label:
    """Match second set of labels to first, returning a new Label object
    Uses scipy's version of the Hungarian algorithm (linear_sum_assigment)
    """
    logger = lm.get_main_logger()
    max_id = max(labels_1.max_id, labels_2.max_id)
    num_extra_labels = labels_2.num_labels - labels_1.num_labels

    logger.info(
        f"Matching {labels_2.num_labels} labels against {labels_1.num_labels} labels.\n"
        f"highest label ID in both is {max_id}.\n"
    )

    onehot_1, onehot_2 = labels_1.get_onehot(), labels_2.get_onehot()
    cost_matrix = (onehot_1 @ onehot_2.T).toarray()

    labels_match_1, labels_match_2 = scipy.optimize.linear_sum_assignment(cost_matrix, maximize=True)

    logger.info("\nMatches:\n", list(zip(labels_match_1, labels_match_2)))

    # Temporary list keeping track of which labels are still available for use
    available_labels = list(range(max_id + 1))
    # List to be filled with new label ids
    relabeled_ids = -1 * np.ones((labels_2.num_labels,), dtype=np.int32)

    # Reassign labels
    for index_1, index_2 in zip(labels_match_1, labels_match_2):
        label_1 = labels_1.ids[index_1]
        label_2 = labels_2.ids[index_2]

        if verbose:
            logger.info(
                f"Assigning first set's {label_1} to " f"second set's {label_2}.\n" f"labels_left: {available_labels}"
            )

        relabeled_ids[index_2] = label_1
        available_labels.remove(label_1)

    # Assign remaining labels (if 2nd has more labels than 1st)
    if num_extra_labels > 0:
        unmatched_indices = np.nonzero(relabeled_ids == -1)[0]

        assert num_extra_labels == len(unmatched_indices), (
            f"number of unmatched label IDs {len(unmatched_indices)} does not match mumber of "
            f"extra labels in second set {num_extra_labels}.\n"
        )

        if extra_labels_assignment == "random":
            relabeled_ids[unmatched_indices] = np.random.choice(available_labels, size=num_extra_labels, replace=False)

        elif extra_labels_assignment == "greedy":

            def _insert_label(
                array: np.ndarray,
                max_length: int,
                added_labels: list = [],
            ) -> Tuple[np.ndarray, int, list]:
                """
                Insert a label in the middle of the largest interval
                Assumes array is already sorted!
                """
                if len(array) >= max_length:
                    return array, max_length, added_labels
                else:
                    intervals = array[1:] - array[:-1]
                    max_interval_index = np.argmax(intervals)
                    increment = intervals[max_interval_index] // 2
                    label_to_add = array[max_interval_index] + increment
                    inserted_array = np.insert(
                        array,
                        max_interval_index + 1,
                        label_to_add,
                    )
                    added_labels.append(label_to_add)
                    return _insert_label(inserted_array, max_length, added_labels)

            sorted_matched = np.sort(relabeled_ids[relabeled_ids != -1])

            logger.info(f"already matched ids (sorted): {sorted_matched}")

            _, _, added_labels = _insert_label(sorted_matched, labels_2.num_labels)
            relabeled_ids[unmatched_indices] = np.random.choice(added_labels, size=num_extra_labels, replace=False)

        else:
            logger.error(f"Extra labels assignment method not recognised, should be random or greedy.")

        logger.info(f"\nRelabeled labels: {relabeled_ids}\n")

    relabeled_dense = (relabeled_ids @ onehot_2).astype(np.int32)
    return Label(relabeled_dense)


def match_label_series(
    label_list: List[Label],
    least_labels_first: bool = True,
    extra_labels_assignment: str = "greedy",
) -> Tuple[List[Label], int]:
    """Match a list of labels to each other, one after another in order of increasing (if least_labels_first is true)
    or decreasing (least_labels_first set to false) number of label ids.

    Returns the relabeled list in original order.
    """
    logger = lm.get_main_logger()
    num_label_list = [label.num_labels for label in label_list]
    max_num_labels = max(num_label_list)
    sort_indices = np.argsort(num_label_list)

    logger.info(
        f"\nMaximum number of labels across all datasets = {max_num_labels}\n"
        f"Indices of sorted list: {sort_indices}\n"
    )

    ordered_relabels = []

    if least_labels_first:
        ordered_relabels.append(expand_labels(label_list[sort_indices[0]], max_num_labels - 1))
        logger.info(f"First label, expanded label ids: {ordered_relabels[0]}")
    else:
        # Argsort is in ascending order, reverse it
        sort_indices = sort_indices[:, :, -1]
        # Already has max number of labels, no need to expand
        ordered_relabels.append(label_list[sort_indices[0]])

    for index in sort_indices[1:]:
        current_label = label_list[index]
        previous_label = ordered_relabels[-1]
        logger.info(f"\nRelabeling:\n{current_label}\n" f"with reference to\n{previous_label}\n" + "-" * 70 + "\n")

        relabeled = match_labels(previous_label, current_label, extra_labels_assignment=extra_labels_assignment)

        ordered_relabels.append(relabeled)

    sort_indices_list = list(sort_indices)
    original_order_relabels = [ordered_relabels[sort_indices_list.index(n)] for n in range(len(label_list))]

    return original_order_relabels, max_num_labels


def interlabel_connections(
    label: Label,
    weights_matrix: Union[scipy.sparse.csr_matrix, np.ndarray],
) -> np.ndarray:
    """Compute connections strength between labels (based on pairwise distances), normalized by counts of each label

    Args:
        class: Instance of class 'Label', with one-hot dense label matrix in "dense", list of unique labels in "ids",
            counts per label in "counts", etc.
        weights_matrix: Pairwise adjacency matrix, weighted by e.g. spatial distance between points.

    Returns:
        connections: Pairwise connection strength array, shape [n_labels, n_labels].
    """
    logger = lm.get_main_logger()

    if weights_matrix.ndim != 2:
        logger.error(f"Weights matrix has {weights_matrix.ndim} dimensions, should be 2.")

    if weights_matrix.shape[0] != weights_matrix.shape[1] != label.num_samples:
        logger.error(f"Weights matrix dimensions do not match number of samples.")

    normalized_onehot = label.generate_normalized_onehot()

    logger.info(
        f"Matrix multiplying labels x weights x labels-transpose, shape {normalized_onehot.shape} x "
        f"{weights_matrix.shape} x {normalized_onehot.T.shape}."
    )

    connections = normalized_onehot @ weights_matrix @ normalized_onehot.T

    if scipy.sparse.issparse(connections):
        connections = connections.toarray()

    return connections


def create_label_class(
    adata: AnnData,
    cat_key: Union[str, List[str]],
) -> Union[Label, List[Label]]:
    """Wraps categorical labels into custom Label class for downstream processing.

    Args:
        adata: An anndata object.
        cat_key: Keys in .obs containing categorical labels. This function and the Label class provide the most utility
            when this is used in conjunction with the results of multiple different runs of the Louvain algorithm.

    Returns:
        label: Either an object of Label class or a list where each element is an object of Label class. Will return a
            list if given multiple arguments to 'cat_key'.
    """
    # Convert categorical labels to numerical and save mapping to have both numerical and categorical labels:
    if isinstance(cat_key, str):
        str_cat = np.unique(adata.obs[cat_key].values)
        num_cat = range(len(str_cat))
        map_dict = dict(zip(num_cat, str_cat))
        all_num_labels = adata.obs[cat_key].replace(str_cat, num_cat)

        label = Label(all_num_labels.to_numpy(), str_map=map_dict)
        return label
    else:
        all_labels = []
        for key in cat_key:
            str_cat = np.unique(adata.obs[key].values)
            num_cat = range(len(str_cat))
            map_dict = dict(zip(num_cat, str_cat))
            all_num_labels = adata.obs[key].replace(str_cat, num_cat)

            label = Label(all_num_labels.to_numpy(), str_map=map_dict)
            all_labels.append(label)
        return all_labels

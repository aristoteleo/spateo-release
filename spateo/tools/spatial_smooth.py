import functools
import gc
from multiprocessing import Pool, cpu_count
from typing import Optional, Tuple, Union

import numpy as np
import psutil
import scipy

from ..logging import logger_manager as lm


# ---------------------------------------------------------------------------------------------------
# Spatial smoothing
# ---------------------------------------------------------------------------------------------------
def smooth(
    X: Union[np.ndarray, scipy.sparse.csr_matrix],
    W: Union[np.ndarray, scipy.sparse.csr_matrix],
    ct: Optional[np.ndarray] = None,
    gene_expr_subset: Optional[Union[np.ndarray, scipy.sparse.csr_matrix]] = None,
    min_jaccard: Optional[float] = 0.05,
    manual_mask: Optional[np.ndarray] = None,
    normalize_W: bool = True,
    return_discrete: bool = False,
    smoothing_threshold: Optional[float] = None,
    n_subsample: Optional[int] = None,
    return_W: bool = False,
) -> Tuple[scipy.sparse.csr_matrix, Optional[Union[np.ndarray, scipy.sparse.csr_matrix]], Optional[np.ndarray]]:
    """Leverages neighborhood information to smooth gene expression.

    Args:
        X: Gene expression array or sparse matrix (shape n x m, where n is the number of cells and m is the number of
            genes)
        W: Spatial weights matrix (shape n x n)
        ct: Optional, indicates the cell type label for each cell (shape n x 1). If given, will smooth only within each
            cell type.
        gene_expr_subset: Optional, array corresponding to the expression of select genes (shape n x k,
            where k is the number of genes in the subset). If given, will smooth only over cells that largely match
            the expression patterns over these genes (assessed using a Jaccard index threshold that is greater than
            the median score).
        min_jaccard: Optional, and only used if 'gene_expr_subset' is also given. Minimum Jaccard similarity score to
            be considered "nonzero".
        manual_mask: Optional, binary array of shape n x n. For each cell (row), manually indicate which neighbors (
            if any) to use for smoothing.
        normalize_W: Set True to scale the rows of the weights matrix to sum to 1. Use this to smooth by taking an
            average over the entire neighborhood, including zeros. Set False to take the average over only the
            nonzero elements in the neighborhood.
        return_discrete: Set True to return
        smoothing_threshold: Optional, sets the threshold for smoothing in terms of the number of neighboring cells
            that must express each gene for a cell to be smoothed for that gene. The more gene-expressing neighbors,
            the more confidence in the biological signal. Can be given as a float between 0 and 1, in which case it
            will be interpreted as a proportion of the total number of neighbors.
        n_subsample: Optional, sets the number of random neighbor samples to use in the smoothing. If not given,
            will use all neighbors (nonzero weights) for each cell.
        return_W: Set True to return the weights matrix post-processing

    Returns:
        x_new: Smoothed gene expression array or sparse matrix
        W: If return_W is True, returns the weights matrix post-processing
        d: Only if normalize_W is True, returns the row sums of the weights matrix
    """
    logger = lm.get_main_logger()
    logger.info(
        "Warning- this can be quite memory intensive when 'gene_expr_subset' is provided, depending on the "
        "size of the AnnData object. If this is the case it is recommended only to use cell types."
    )

    if scipy.sparse.isspmatrix_csr(X):
        logger.info(f"Initial sparsity of array: {X.count_nonzero()}")
    else:
        logger.info(f"Initial sparsity of array: {np.count_nonzero(X)}")

    # Subsample weights array if applicable:
    if n_subsample is not None:
        # Threshold for smoothing (check that a sufficient number of neighbors express a given gene for increased
        # confidence of biological signal- must be greater than or equal to this threshold for smoothing):
        if scipy.sparse.isspmatrix_csr(W):
            W = subsample_neighbors_sparse(W, n_subsample)
        else:
            W = subsample_neighbors_dense(W, n_subsample)

    if smoothing_threshold is not None:
        threshold = smoothing_threshold
    else:
        # Threshold of zero means all neighboring cells get incorporated into the smoothing operation
        threshold = 0

    # If mask is manually given, no need to use cell type & expression information:
    if manual_mask is not None:
        logger.info(
            "Manual mask provided. Will use this to smooth, ignoring inputs to 'ct' and 'gene_expr_subset' if "
            "provided."
        )
        W = W.multiply(manual_mask)
    else:
        # Incorporate cell type information
        if ct is not None:
            if not isinstance(ct, np.ndarray):
                ct = np.array(ct)
            if ct.ndim != 1:
                ct = ct.flatten()

            logger.info(
                "Conditioning smoothing on cell type- only information from cells of the same type will be used."
            )
            rows, cols = np.where(ct[:, None] == ct)
            sparse_ct_matrix = scipy.sparse.coo_matrix((np.ones_like(rows), (rows, cols)), shape=(len(ct), len(ct)))
            ct_masks = sparse_ct_matrix.tocsr()
            logger.info("Modifying spatial weights considering cell type...")
            W = W.multiply(ct_masks)

            # ct_masks can be quite large- delete to free up memory once it's no longer necessary
            del ct_masks

        # Incorporate gene expression information
        if gene_expr_subset is not None:
            logger.info(
                "Conditioning smoothing on gene expression- only information from cells with similar gene "
                "expression patterns will be used."
            )
            jaccard_mat = compute_jaccard_similarity_matrix(gene_expr_subset, min_jaccard=min_jaccard)
            logger.info("Computing median Jaccard score from nonzero entries only")
            if scipy.sparse.isspmatrix_csr(jaccard_mat):
                jaccard_threshold = sparse_matrix_median(jaccard_mat, nonzero_only=True)
            else:
                jaccard_threshold = np.percentile(jaccard_mat[jaccard_mat != 0], 50)
            logger.info(f"Threshold Jaccard score: {jaccard_threshold}")
            # Generate a mask where Jaccard similarities are greater than the threshold, and apply to the weights
            # matrix:
            jaccard_mask = jaccard_mat >= jaccard_threshold
            W = W.multiply(jaccard_mask)

    if scipy.sparse.isspmatrix_csr(W):
        # Calculate the average number of non-zero weights per row for a sparse matrix
        row_nonzeros = W.getnnz(axis=1)  # Number of non-zero elements per row
        average_nonzeros = row_nonzeros.mean()  # Average over all rows
    else:
        # Calculate the average number of non-zero weights per row for a dense matrix
        row_nonzeros = (W != 0).sum(axis=1)  # Boolean matrix where True=1 for non-zeros, then sum per row
        average_nonzeros = row_nonzeros.mean()  # Average over all rows
    logger.info(f"Average number of non-zero weights per cell: {average_nonzeros}")
    # If threshold is given as a float, interpret as a proportion of the average number of non-zero weights
    if 0 < threshold < 1:
        threshold = int(average_nonzeros * threshold)
        logger.info(f"Threshold set to {threshold} based on the average number of non-zero weights.")

    # Original nonzero entries and values (keep these around):
    initial_nz_rows, initial_nz_cols = X.nonzero()
    if scipy.sparse.isspmatrix_csr(X):
        initial_nz_vals = X[initial_nz_rows, initial_nz_cols].A1
    else:
        initial_nz_vals = X[initial_nz_rows, initial_nz_cols]

    if normalize_W:
        if type(W) == np.ndarray:
            d = np.sum(W, 1).flatten()
        else:
            d = np.sum(W, 1).A.flatten()
        W = scipy.sparse.diags(1 / d) @ W if scipy.sparse.isspmatrix_csr(W) else np.diag(1 / d) @ W
        # Note that W @ X already returns sparse in this scenario, csr_matrix is just used to convert to common format
        x_new = scipy.sparse.csr_matrix(W @ X) if scipy.sparse.isspmatrix_csr(X) else W @ X

        if return_discrete:
            if scipy.sparse.isspmatrix_csr(x_new):
                data = x_new.data
                data[:] = np.where((0 < data) & (data < 1), 1, np.round(data))
            else:
                x_new = np.where((0 < x_new) & (x_new < 1), 1, np.round(x_new))
        if scipy.sparse.isspmatrix_csr(x_new):
            logger.info(f"Sparsity of smoothed array: {x_new.count_nonzero()}")
        else:
            logger.info(f"Sparsity of smoothed array: {np.count_nonzero(x_new)}")

        if return_W:
            return x_new, W, d
        else:
            return x_new, d
    else:
        processor_func = functools.partial(smooth_process_column, X=X, W=W, threshold=threshold)
        pool = Pool(cpu_count())
        mod = pool.map(processor_func, range(X.shape[1]))
        x_new = scipy.sparse.hstack(mod)

        # Add back in the original nonzero entries:
        if scipy.sparse.isspmatrix_csr(X):
            orig_values = scipy.sparse.csr_matrix((initial_nz_vals, (initial_nz_rows, initial_nz_cols)), shape=X.shape)
            x_new = x_new + orig_values
        else:
            x_new[initial_nz_rows, initial_nz_cols] = initial_nz_vals

        # Convert to CSR format for more efficient storage:
        if scipy.sparse.isspmatrix_csr(x_new):
            logger.info(f"Sparsity of smoothed array: {x_new.count_nonzero()}")
        else:
            logger.info(f"Sparsity of smoothed array: {np.count_nonzero(x_new)}")

        if return_discrete:
            if scipy.sparse.isspmatrix_csr(x_new):
                data = x_new.data
                data[:] = np.round(data)
            else:
                x_new = np.round(x_new)

        if return_W:
            return x_new, W
        else:
            return x_new


def compute_jaccard_similarity_matrix(
    data: Union[np.ndarray, scipy.sparse.csr_matrix], chunk_size: int = 1000, min_jaccard: float = 0.1
) -> np.ndarray:
    """Compute the Jaccard similarity matrix for input data with rows corresponding to samples and columns
    corresponding to features, processing in chunks for memory efficiency.

    Args:
        data: A dense numpy array or a sparse matrix in CSR format, with rows as features
        chunk_size: The number of rows to process in a single chunk
        min_jaccard: Minimum Jaccard similarity to be considered "nonzero"

    Returns:
        jaccard_matrix: A square matrix of Jaccard similarity coefficients
    """
    n_samples = data.shape[0]
    jaccard_matrix = np.zeros((n_samples, n_samples))

    if scipy.sparse.isspmatrix_csr(data):  # Check if input is a sparse matrix
        # Ensure the matrix is in CSR format for efficient row slicing
        data = scipy.sparse.csr_matrix(data)
        data_bool = data.astype(bool).astype(int)
        row_sums = data_bool.sum(axis=1)
        data_bool_T = data_bool.T
        row_sums_T = row_sums.T
    else:
        data_bool = (data > 0).astype(int)
        row_sums = data_bool.sum(axis=1).reshape(-1, 1)
        data_bool_T = data_bool.T
        row_sums_T = row_sums.T

    # Compute Jaccard similarities in chunks
    for chunk_start in range(0, n_samples, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_samples)
        print(f"Pairwise Jaccard similarity computation: processing rows {chunk_start} to {chunk_end}...")

        # Compute similarities for the current chunk
        chunk_intersection = data_bool[chunk_start:chunk_end].dot(data_bool_T)
        chunk_union = row_sums[chunk_start:chunk_end] + row_sums_T - chunk_intersection
        chunk_similarity = chunk_intersection / np.maximum(chunk_union, 1)
        chunk_similarity[chunk_similarity < min_jaccard] = 0.0

        jaccard_matrix[chunk_start:chunk_end] = chunk_similarity

        # Print available memory
        memory_stats = psutil.virtual_memory()
        print(f"Available memory: {memory_stats.available / (1024 * 1024):.2f} MB")

        print(f"Completed rows {chunk_start} to {chunk_end}.")

        # Clean up memory for the next chunk
        del chunk_intersection, chunk_union, chunk_similarity
        gc.collect()

    print("Pairwise Jaccard similarity computation: all chunks processed. Converting to final format...")
    if np.any(np.isnan(jaccard_matrix)) or np.any(np.isinf(jaccard_matrix)):
        raise ValueError("jaccard_matrix contains NaN or Inf values")

    if scipy.sparse.isspmatrix_csr(data):
        jaccard_matrix = scipy.sparse.csr_matrix(jaccard_matrix)
    print("Returning pairwise Jaccard similarity matrix...")

    return jaccard_matrix


def sparse_matrix_median(spmat: scipy.sparse.csr_matrix, nonzero_only: bool = False) -> scipy.sparse.csr_matrix:
    """Computes the median value of a sparse matrix, used here for determining a threshold value for Jaccard similarity.

    Args:
        spmat: The sparse matrix to compute the median value of
        nonzero_only: If True, only consider nonzero values in the sparse matrix

    Returns:
        median_value: The median value of the sparse matrix
    """
    # Flatten the sparse matrix data and sort it
    spmat_data_sorted = np.sort(spmat.data)

    if nonzero_only:
        # If the number of non-zero elements is even, take the average of the two middle values
        middle_index = spmat.nnz // 2
        if spmat.nnz % 2 == 0:  # Even number of non-zero elements
            median_value = (spmat_data_sorted[middle_index - 1] + spmat_data_sorted[middle_index]) / 2
        else:  # Odd number of non-zero elements, take the middle one
            median_value = spmat_data_sorted[middle_index]
    else:
        # Get the total number of elements in the matrix
        total_elements = spmat.shape[0] * spmat.shape[1]
        # Calculate the number of zeros
        num_zeros = total_elements - spmat.nnz
        # Find the number of sorted elements that need to be sorted through to find the median
        median_idx = total_elements // 2

        # If there are more zeros than the index for the median, then the median is zero
        if num_zeros > median_idx:
            median_value = 0
        else:
            median_idx_non_zero = median_idx - num_zeros
            median_value = spmat_data_sorted[median_idx_non_zero]

    return median_value


def smooth_process_column(
    i: int,
    X: Union[np.ndarray, scipy.sparse.csr_matrix],
    W: Union[np.ndarray, scipy.sparse.csr_matrix],
    threshold: float,
) -> scipy.sparse.csr_matrix:
    """Helper function for parallelization of smoothing via probabilistic selection of expression values.

    Args:
        i: Index of the column to be processed
        X: Dense or sparse array input data matrix
        W: Dense or sparse array pairwise spatial weights matrix
        threshold: Threshold value for the number of feature-expressing neighbors for a given row to be included in
            the smoothing.
        random_state: Optional, set a random seed for reproducibility

    Returns:
        smoothed_column: Processed column after probabilistic smoothing
    """
    feat = X[:, i].toarray().flatten() if scipy.sparse.isspmatrix_csr(X) else X[:, i]
    # Find the rows that meet the threshold criterion:
    eligible_rows = get_eligible_rows(W, feat, threshold)
    # Sample over eligible rows:
    sampled_values = sample_from_eligible_neighbors(W, feat, eligible_rows)
    # Create a sparse matrix with the sampled values:
    smoothed_column = scipy.sparse.csr_matrix(sampled_values.reshape(-1, 1))
    return smoothed_column


def get_eligible_rows(
    W: Union[np.ndarray, scipy.sparse.csr_matrix],
    feat: Union[np.ndarray, scipy.sparse.csr_matrix],
    threshold: float,
) -> np.ndarray:
    """Helper function for parallelization of smoothing via probabilistic selection of expression values.

    Args:
        W: Dense or sparse array pairwise spatial weights matrix
        feat: 1D array of feature expression values
        threshold: Threshold value for the number of feature-expressing neighbors for a given row to be included in
            the smoothing.

    Returns:
        eligible_rows: Array of row indices that meet the threshold criterion
    """
    if feat.ndim == 1:
        feat = feat.reshape(-1, 1)
    feat_sp = scipy.sparse.csr_matrix(feat).transpose()

    if scipy.sparse.isspmatrix_csr(W):
        W = W.multiply(feat_sp)
        # Check the number of non-zero weights per row after scaling
        nnz_new = W.getnnz(axis=1)
        # Find the rows that meet the threshold criterion
        eligible_rows = np.where(nnz_new > threshold)[0]
    else:
        W = W * feat_sp
        # Compute the number of nonzero weights per row in the updated array:
        nnz_new = (W != 0).sum(axis=1)
        # Find the rows that meet the threshold criterion:
        eligible_rows = np.where(nnz_new >= threshold)[0]

    # Remove rows that were nonzero in the original array (these do not need to be smoothed):
    eligible_rows = np.setdiff1d(eligible_rows, np.where(feat != 0)[0])

    return eligible_rows


def sample_from_eligible_neighbors(
    W: Union[np.ndarray, scipy.sparse.csr_matrix],
    feat: Union[np.ndarray, scipy.sparse.csr_matrix],
    eligible_rows: np.ndarray,
):
    """Sample feature values probabilistically based on weights matrix W.

    Args:
        W: Dense or sparse array pairwise spatial weights matrix
        feat: 1D array of feature expression values
        eligible_rows: Array of row indices that meet a prior-determined threshold criterion

    Returns:
        sampled_values: Array of sampled values
    """
    sampled_values = np.zeros(W.shape[0])

    if scipy.sparse.isspmatrix_csr(W):
        for row in eligible_rows:
            start_index = W.indptr[row]
            end_index = W.indptr[row + 1]
            indices = W.indices[start_index:end_index]
            data = W.data[start_index:end_index]

            # Filter based on nonzero feature values
            valid_mask = feat[indices] != 0
            valid_indices = indices[valid_mask]
            valid_data = data[valid_mask]

            # Sample from the valid entries:
            if valid_data.size > 0:
                probabilities = valid_data / valid_data.sum()
                sampled_index = np.random.choice(valid_indices, p=probabilities)
                sampled_values[row] = feat[sampled_index]
    else:
        for row in eligible_rows:
            valid_mask = (W[row, :] != 0) & (feat != 0)
            valid_data = W[row, valid_mask]
            valid_indices = np.where(valid_mask)[0]

            # Sample from the valid entries:
            if valid_data.size > 0:
                probabilities = valid_data / valid_data.sum()
                sampled_index = np.random.choice(valid_indices, p=probabilities)
                sampled_values[row] = feat[sampled_index]

    return sampled_values


def subsample_neighbors_dense(W: np.ndarray, n: int, verbose: bool = False) -> np.ndarray:
    """Given dense spatial weights matrix W and number of random neighbors n to take, perform subsampling.

    Parameters:
        W: Spatial weights matrix
        n: Number of neighbors to keep for each row
        verbose: Set True to print warnings for cells with fewer than n neighbors

    Returns:
        W_new: Subsampled spatial weights matrix
    """
    logger = lm.get_main_logger()

    W_new = W.copy()
    # Calculate the number of non-zero elements per row
    num_nonzeros = np.count_nonzero(W_new, axis=1)
    # Identify rows that need subsampling
    rows_to_subsample = np.where(num_nonzeros > n)[0]

    for i in rows_to_subsample:
        # Find the non-zero indices for the current row, shuffle and select the first n:
        nonzero_indices = np.flatnonzero(W_new[i])
        np.random.shuffle(nonzero_indices)
        indices_to_zero = nonzero_indices[n:]
        W_new[i, indices_to_zero] = 0

    if verbose:
        for i in np.where(num_nonzeros <= n)[0]:
            logger.warning(f"Cell {i} has fewer than {n} neighbors to sample from. Subsampling not performed.")
    return W_new


def subsample_neighbors_sparse(W: scipy.sparse.csr_matrix, n: int, verbose: bool = False) -> scipy.sparse.csr_matrix:
    """Given sparse spatial weights matrix W and number of random neighbors n to take, perform subsampling.

    Parameters:
        W: Spatial weights matrix
        n: Number of neighbors to keep for each row
        verbose: Set True to print warnings for cells with fewer than n neighbors

    Returns:
        W_new: Subsampled spatial weights matrix
    """
    logger = lm.get_main_logger()

    W_new = W.copy().tocsr()
    # Determine the count of nonzeros in each row
    row_nnz = W_new.getnnz(axis=1)
    # Find rows with more non-zero elements than 'n'
    rows_to_subsample = np.where(row_nnz > n)[0]

    for row in rows_to_subsample:
        # Get all column indices for the current 'row' with non-zero entries
        cols = W_new.indices[W_new.indptr[row] : W_new.indptr[row + 1]]
        # Randomly choose 'n' indices to keep and set the rest to zero
        np.random.shuffle(cols)
        cols_to_keep = cols[:n]
        # Create a mask for the columns to keep
        mask = np.isin(cols, cols_to_keep, assume_unique=True, invert=True)
        # Zero out the masked elements
        W_new.data[W_new.indptr[row] : W_new.indptr[row + 1]][mask] = 0

    if verbose:
        for i in np.where(row_nnz <= n)[0]:
            logger.warning(f"Cell {i} has fewer than {n} neighbors to sample from. Subsampling not performed.")
    # Clean up
    W_new.eliminate_zeros()

    return W_new

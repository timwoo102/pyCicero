import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.sparse as sp
from scipy import sparse
import scanpy as sc
from scipy.sparse import coo_matrix

device = "gpu"

if device == "cpu":
    from sklearn.neighbors import NearestNeighbors
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.decomposition import TruncatedSVD
    import umap as UMAP
if device == "gpu": 
    from sklearn.neighbors import NearestNeighbors
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.decomposition import TruncatedSVD #bottle neck with no good gpu implementation for sparse
    # import umap as UMAP
    from cuml.manifold import UMAP 
    
    #--------------------- offers minimal speedup on my V100 and dataset where n=35,000 -------------------------------
    # from cuml.neighbors import NearestNeighbors
    # from cuml.feature_extraction.text import TfidfTransformer
    # # from cuml.decomposition import TruncatedSVD # Doesn't work for sparse
    # from cuda_TruncatedSVD import TruncatedSVD 
    # from cuml.manifold import UMAP
    # import cupy as cp

import logging
logging.basicConfig(
    format='%(filename)s: %(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)
global LOGGER
LOGGER = logging.getLogger(__name__)

def estimate_sf(counts, round_exprs=True, method="mean-geometric-mean-total"):
    if sp.issparse(counts):
        CM = counts.copy()
        if round_exprs:
            CM.data = np.round(CM.data)
        cell_total = np.array(CM.sum(axis=0)).ravel()
    else:
        CM = counts.copy()
        if round_exprs:
            CM = np.round(CM)
        cell_total = np.sum(CM, axis=0)
    
    if method == "mean-geometric-mean-log-total":
        sfs = np.log(cell_total) / np.exp(np.mean(np.log(np.log(cell_total))))
    elif method == "mean-geometric-mean-total":
        sfs = cell_total / np.exp(np.mean(np.log(cell_total)))

    sfs[np.isnan(sfs)] = 1  
    return sfs

def normalize_expr_data(FM, size_factors, norm_method='log', pseudo_count=None):

    if norm_method not in ['log', 'size_only', 'none']:
        raise ValueError("norm_method must be one of 'log', 'size_only', or 'none'")
    

    if pseudo_count is None:
        pseudo_count = 1 if norm_method == 'log' else 0
    
    if norm_method == 'log':
        FM = (FM / size_factors)

        if pseudo_count != 1 or not sparse.isspmatrix(FM):
            FM = FM + pseudo_count
            FM = np.log2(FM)
        else:
            FM.data = np.log2(FM.data + 1)

    elif norm_method == 'size_only':

        FM = (FM / size_factors)
        FM = FM + pseudo_count

    return FM

def preprocess_cicero(adata: sc.AnnData, 
                      normalize = True,
               TruncatedSVD_params:  dict = {"n_components":50}, 
               tfidf_params: dict = {"norm":'l2'}, 
               umap_params:  dict = {"n_neighbors":15, "n_components":2, "min_dist":0.5, "n_epochs":1e3}):
    
    TFIDF_KEY = "X_tfidf"
    TSVD_KEY  = "X_tsvd"
    UMAP_KEY  = "X_umap"

    if "counts" not in adata.layers:
        LOGGER.warning("Counts layers not found. Coppying X to counts")
        adata.layers["counts"] = adata.X.copy()

    if normalize and "data" not in adata.layers:
        LOGGER.info("Estimaing Size Factors and Normalizing Data")
        size_factors = estimate_sf(adata.layers["counts"])
        data = normalize_expr_data(adata.layers["counts"], size_factors).tocsr()
        adata.layers["data"] = data
        LOGGER.info("Finished Size Factors and Normallization storing normalized sparse matrix into 'data'")
    elif "data" in adata.layers:
        LOGGER.info("'data' found in layers. Using it as normalized matrix for downstream processing")
        data = adata.layers["data"]
    else:
        LOGGER.info("Running Preprocssing without normalizing counts")
        data = adata.layers["counts"]
    
    # normalized_counts_row_sums = np.sum(adata.layers["data"], axis=1)
    # normalized_counts_row_sums = np.array(normalized_counts_row_sums)[:,0]
    # normalized_counts_filtered = adata.layers["data"][np.isfinite(normalized_counts_row_sums) & (normalized_counts_row_sums != 0), :]
    
    LOGGER.info("Running TF-IDF")
    tfidf = TfidfTransformer(**tfidf_params)
    X_tfidf = tfidf.fit_transform(data)
    adata.obsm[TFIDF_KEY] = X_tfidf
    LOGGER.info("Finsihed TF-IDF")

    LOGGER.info("Running TruncatedSVD")
    tsvd = TruncatedSVD(**TruncatedSVD_params)
    A_transformed = tsvd.fit_transform(X_tfidf)
    adata.obsm[TSVD_KEY] = A_transformed
    LOGGER.info("Finsihed TruncatedSVD")

    #TODO: consider adding harmony integration here
    
    LOGGER.info("Running UMAP")
    umap_model = UMAP(**umap_params)
    embedding = umap_model.fit_transform(A_transformed)
    adata.obsm[UMAP_KEY] = embedding
    LOGGER.info("Finsihed UMAP")

    return adata


#TODO Should make it such that we calculate optimal pseudo centers using modified k-means center approach
def calculate_overlap(embedding: pd.DataFrame, seed: int = 0, max_iterations: int = 5000, quiet = False):
    nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(embedding)
    distances, indices = nbrs.kneighbors(embedding)
    
    LOGGER.info(f"Generating Pseudobulk cicero observations with seed: {seed}")
    rng = np.random.default_rng(seed)
    nn_map = pd.DataFrame(indices)
    choices = nn_map.index.values
    rng.shuffle(choices)

    # Precompute sets for each index to avoid repeated DataFrame lookups.
    nn_sets = { idx: set(nn_map.loc[idx].values) for idx in nn_map.index }

    def overlap_set(idx1, idx2):
        set1 = nn_sets[idx1]
        set2 = nn_sets[idx2]
        # Compute the fraction of elements in idx1 that are also in idx2
        return len(set1.intersection(set2)) / len(set1)

    good_choices = []
    iterations = 0

    for choice in tqdm(choices, total = min(max_iterations, len(choices)), disable = quiet):
        if not good_choices:
            good_choices.append(choice)
        else:
            add = True
            for previous_choice in good_choices:
                if overlap_set(choice, previous_choice) > 0.90:
                    add = False
                    break
            if add:
                good_choices.append(choice)
        
        iterations += 1
        if iterations >= max_iterations:
            break
    if iterations == max_iterations:
        LOGGER.info(f"Reached Maximum itterations in pseudobulk observation generation. Consider increasing 'max_itterations'")
    LOGGER.info(f"Found {len(good_choices)} good choices")
    return {choice:nn_sets[choice] for choice in good_choices}

def aggregate_cells(counts, pseudobulk_cell_dict):

    # Get all group names and create a mapping to row indices
    group_names = list(pseudobulk_cell_dict.keys())
    group_index = {group: idx for idx, group in enumerate(group_names)}

    # all cell_ids lists have the same length
    L = len(pseudobulk_cell_dict[group_names[0]])

    # Preallocate the rows and cols using vectorized operations
    rows_arr = np.repeat([group_index[g] for g in group_names], L)
    cols_arr = np.concatenate([np.array(list(pseudobulk_cell_dict[g])) for g in group_names])
    data_arr = np.ones(rows_arr.shape[0], dtype=int)

    # Construct the grouping matrix in COO format and then convert to CSR for faster arithmetic
    M = coo_matrix((data_arr, (rows_arr, cols_arr)), shape=(len(group_names), counts.shape[0])).tocsr()
    aggregated_counts = M.dot(counts)
    return aggregated_counts

def make_cicero_adata(adata: sc.AnnData, aggregate_layer_key: str = "counts", embedding_key: str = "X_umap", embedding_df: pd.DataFrame = None, max_iterations: int = 5000, seed: int = 0, quiet: bool = True):
    """
        new_adata obs is set to cell center obs
    """
    
    if embedding_key is None and embedding is None:
        LOGGER.error("Must pass embedding_key or embedding_df to make_cicero_adata. preprocess_cicero function should be run first")

    if embedding_df is None:
        LOGGER.info(f"Using adata OBSM {embedding_key} to agregate cells")
        embedding_df = pd.DataFrame(adata.obsm[embedding_key])
    
    LOGGER.info("Calculating overlap")
    pseudobulk_cell_dict = calculate_overlap(embedding_df, seed = seed, max_iterations = max_iterations, quiet=quiet)
    LOGGER.info("Finished calculating overlap")
    
    LOGGER.info("Aggregating Cells")
    aggregated_counts = aggregate_cells(adata.layers[aggregate_layer_key], pseudobulk_cell_dict)
    LOGGER.info("Finished Aggregating Cells")

    new_adata = sc.AnnData(X=aggregated_counts, var=adata.var.copy(), obs = adata.obs.iloc[list(pseudobulk_cell_dict.keys()),:])
    
    return new_adata


def generate_umap(counts = None, data = None, 
                      normalize = True,
               TruncatedSVD_params:  dict = {"n_components":50}, 
               tfidf_params: dict = {"norm":'l2'}, 
               umap_params:  dict = {"n_neighbors":15, "n_components":2, "min_dist":0.5, "n_epochs":1e3}):

    if counts is not None and normalize:
       size_factors = estimate_sf(counts)
       data = normalize_expr_data(counts, size_factors).tocsr()

    tfidf = TfidfTransformer(**tfidf_params)
    X_tfidf = tfidf.fit_transform(data)

    tsvd = TruncatedSVD(**TruncatedSVD_params)
    A_transformed = tsvd.fit_transform(X_tfidf)
    #TODO: consider adding harmony integration here
    umap_model = UMAP(**umap_params)
    embedding = umap_model.fit_transform(A_transformed)

    return embedding
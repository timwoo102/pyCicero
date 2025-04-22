import pandas as pd
import numpy as np
import scanpy as sc
from tqdm import tqdm
from scipy import sparse
from scipy.sparse import csr_matrix, hstack, issparse

import logging
logging.basicConfig(
    format='%(filename)s: %(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)
global LOGGER
LOGGER = logging.getLogger(__name__)

def aggregate_adata(adata, cicero_adata, mapping_dict, aggregate_obs_column = "aggregate_obs_names", names_delim = ",", quiet = True):
    raw_index = pd.Series(np.arange(adata.n_obs), index=adata.obs_names)
    rows, cols = [], []

    for agg_i, names in enumerate(tqdm(cicero_adata.obs[aggregate_obs_column], total=cicero_adata.n_obs, desc="building aggregator", disable = quiet)):

        if isinstance(names, str):
            names = names.split(names_delim) #Names is sometimes stored as csv's in my case to be able to be serialized with adata
        for n in names:
            raw_name = mapping_dict.get(n) #Mapping from atac barcodes to rna barcodes (if its the same you should provide a mapping dict that is 1:1)
            if raw_name is None or raw_name not in adata.obs_names:   # skip missing names
                continue
            cols.append(raw_index[raw_name])       # raw‑cell (column) index
            rows.append(agg_i)                     # aggregate (row) index

    data = np.ones(len(rows), dtype=np.uint8)      # just "count 1"
    A = sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(cicero_adata.n_obs, adata.n_obs)
        )

    aggregated_counts = A @ adata.X            # (n_agg  ×  n_genes) CSR
    mapped_obs_names = list(map(mapping_dict.get, cicero_adata.obs_names))
    obs = adata.obs.loc[mapped_obs_names,:] #again were just going to put the obs of the cell center cluster please dont trust this obs
    var = adata.var
    aggregated_adata = sc.AnnData(aggregated_counts, obs = obs, var = var)
    
    return aggregated_adata

def subset_cicero_window(var, window_chromsome_name, window_start, window_end):
    return (var["Chromosome"] == window_chromsome_name) & (var["Start"] >= window_start) & (var["End"] <= window_end) 

def generate_matricies(cons, aggregated_adata, cicero_adata, quiet = True):

    # cicero_adata_column_indicies = np.zeros(cons.shape[0])
    # adata_column_indicies = np.zeros(cons.shape[0])

    # for index, (row_index, row) in tqdm(enumerate(cons.iterrows()), total = cons.shape[0], desc = "Generating matrix indicies", disable = quiet):
    #     cicero_adata_column_indicies[index] = np.where(subset_cicero_window(cicero_adata.var, row["Peak1_Chromosome"], row["Peak1_Start"], row["Peak1_End"]))[0][0]
    #     gene = row["Peak_1_Gene_Annotation"]
    #     if ";" in gene: #TODO FIX ANNOTATION SO THAT IT DUPES ROWS 
    #         gene = gene.split(";")[0]
    #     if gene not in aggregated_adata.var_names:
    #         adata_column_indicies[index] = aggregated_adata.n_vars
    #     else:
    #         adata_column_indicies[index] = np.where(aggregated_adata.var_names == gene)[0][0]

    ivals  = pd.IntervalIndex.from_arrays(cicero_adata.var.Start, cicero_adata.var.End, closed="both")
    cicero_adata.var["ival"] = ivals
    lookup = cicero_adata.var.set_index(["Chromosome", "ival"]).index
    cicero_adata_column_indicies = (pd.MultiIndex.from_arrays([cons.Peak1_Chromosome, pd.IntervalIndex.from_arrays(cons.Peak1_Start, cons.Peak1_End,closed="both")]).map(dict(zip(lookup, np.arange(len(lookup))))).to_numpy())

    genes = cons["Peak_1_Gene_Annotation"].str.split(";").str[0]
    gene_idx = aggregated_adata.var_names.get_indexer(genes)
    missing_mask = gene_idx == -1
    gene_idx[missing_mask] = aggregated_adata.n_vars
    adata_column_indicies = gene_idx.astype(np.int64)

    aggregated_counts = aggregated_adata.X
    zero_col = csr_matrix((aggregated_counts.shape[0], 1), dtype=aggregated_counts.dtype)
    rna_counts = hstack([aggregated_counts, zero_col], format="csr")
    atac_counts = cicero_adata.X

    atac_peak_counts = atac_counts[:,cicero_adata_column_indicies]
    rna_peak_counts  = rna_counts[:,adata_column_indicies]

    return rna_peak_counts, atac_peak_counts

def correlate_peaks(cons, aggregated_adata, cicero_adata, annotate = True, quiet = True):
    rna_peak_counts, atac_peak_counts = generate_matricies(cons, aggregated_adata, cicero_adata, quiet = quiet)
    correlation_coefficients = sparse_corr_cols(rna_peak_counts, atac_peak_counts)
    if annotate:
        cons["Correlation Coefficient"] = correlation_coefficients
        return cons
    else:
        return correlation_coefficients

def sparse_corr_cols(A, B):

    if not (issparse(A) and issparse(B)):
        raise TypeError("A and B must be SciPy sparse matrices")
    if A.shape != B.shape:
        raise ValueError("A and B must have the same shape")
    if A.ndim != 2:
        raise ValueError("input must be 2‑D")

    n_row = A.shape[0]                         # number of rows / observations

    sA  = np.asarray(A.sum(axis=0)).ravel()                 # Σ a
    sB  = np.asarray(B.sum(axis=0)).ravel()                 # Σ b
    sA2 = np.asarray(A.power(2).sum(axis=0)).ravel()        # Σ a²
    sB2 = np.asarray(B.power(2).sum(axis=0)).ravel()        # Σ b²
    sAB = np.asarray(A.multiply(B).sum(axis=0)).ravel()     # Σ a·b :contentReference[oaicite:1]{index=1}

    num   = n_row * sAB - sA * sB
    denom = np.sqrt((n_row * sA2 - sA**2) * (n_row * sB2 - sB**2))

    with np.errstate(divide="ignore", invalid="ignore"):
        r = num / denom

    return r

#my testcase this ran slower
def sparse_corr_cols_gpu(A, B):
    import cupy as cp
    from cupyx.scipy import sparse as sp

    if A.shape != B.shape:
        raise ValueError("shapes differ")

    n_row = A.shape[0]

    # column‑wise sums
    sA   = cp.asarray(A.sum(axis=0)).ravel()
    sB   = cp.asarray(B.sum(axis=0)).ravel()
    sA2  = cp.asarray(A.power(2).sum(axis=0)).ravel()
    sB2  = cp.asarray(B.power(2).sum(axis=0)).ravel()
    sAB  = cp.asarray(A.multiply(B).sum(axis=0)).ravel()

    num   = n_row * sAB  - sA * sB
    denom = cp.sqrt((n_row * sA2 - sA ** 2) * (n_row * sB2 - sB ** 2))
    return (num / denom).get()

#===================================UTIL===================================================================================
def normalize_peak_orientation(cons):
    peak1_cols = [
        "Peak1_Original", "Peak1_Chromosome", "Peak1_Start", "Peak1_End",
        "Peak1_Mean", "Peak1_Closest_TSS", "Peak1_Distal_Proximal_Annotation",
        "Peak_1_Gene_Annotation"
    ]
    peak2_cols = [
        "Peak2_Original", "Peak2_Chromosome", "Peak2_Start", "Peak2_End",
        "Peak2_Mean", "Peak2_Closest_TSS", "Peak2_Distal_Proximal_Annotation",
        "Peak_2_Gene_Annotation"
    ]

    mask = cons["Linkage_Type"] == "Proximal to Distal"
    if mask.any():
        cons.loc[mask, peak1_cols + peak2_cols] = (
            cons.loc[mask, peak2_cols + peak1_cols].values
        )
        cons.loc[mask, "Linkage_Type"] = "Distal to Proximal"

    return cons
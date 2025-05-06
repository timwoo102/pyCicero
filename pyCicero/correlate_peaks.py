import pandas as pd
import numpy as np
import scanpy as sc
from tqdm import tqdm
from scipy import sparse
from scipy.stats import t  
from scipy.sparse import csr_matrix, hstack, issparse
from inverse_covariance import QuicGraphicalLasso

import logging
logging.basicConfig(
    format='%(filename)s: %(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)
global LOGGER
LOGGER = logging.getLogger(__name__)

#=================================MAIN Beginning===================================================================

def correlate_peaks(cons, rna_adata, atac_adata, distal_cutoff = 1000, cons_subset_threshold = 0.2, method = "pearson", aggregate = False):

    if aggregate:
        LOGGER.info("Aggregating rna_adata to match cicero_adata")
        rna_adata = aggregate_rna_adata(rna_adata, atac_adata)
        LOGGER.info("Finished Aggregation")

    LOGGER.info("Generating Peak to Links")
    cons_subset, peak_to_gene = get_peak_to_gene_links(cons, atac_adata, distal_cutoff=distal_cutoff, cons_subset_threshold=cons_subset_threshold)
    LOGGER.info("Finished generating Peak to Links")
    atac_mtx, rna_mtx = prepare_matricies(peak_to_gene, rna_adata, atac_adata)

    LOGGER.info(f"Runnign Correlation with {method} correlation")
    if method == "pearson":
        correlations = sparse_corr_cols(atac_mtx, rna_mtx)
    if method == "partial":
        correlations = sparse_partial_corr_cols(atac_mtx, rna_mtx)
    LOGGER.info("Finsihed running Correlation")

    LOGGER.info("Adding Links and Correlation to Cons")
    links_mapping = pd.DataFrame(peak_to_gene, index = ["Gene Name"]).T
    links_mapping["correlation"] = correlations

    gene_map = links_mapping["Gene Name"].get
    corr_map = links_mapping["correlation"].get

    cons_subset["Gene"]             = (cons_subset["Peak1"].map(gene_map).fillna(cons_subset["Peak2"].map(gene_map)))
    cons_subset["RNA_Correlations"] = (cons_subset["Peak1"].map(corr_map).fillna(cons_subset["Peak2"].map(corr_map)))
    LOGGER.info("Finished")

    return cons_subset

#=================================MAIN END===================================================================

def aggregate_rna_adata(adata, cicero_adata, aggregate_obs_column = "aggregate_obs_names", names_delim = ",", quiet = True):
    raw_index = pd.Series(np.arange(adata.n_obs), index=adata.obs_names)
    rows, cols = [], []

    for agg_i, names in enumerate(tqdm(cicero_adata.obs[aggregate_obs_column], total=cicero_adata.n_obs, desc="building aggregator", disable = quiet)):

        if isinstance(names, str):
            names = names.split(names_delim) #Names is sometimes stored as csv's in my case to be able to be serialized with adata
        for name in names:

            if name not in adata.obs_names:   # skip missing names
                continue
            
            cols.append(raw_index[name])       # raw‑cell (column) index
            rows.append(agg_i)                     # aggregate (row) index

    data = np.ones(len(rows), dtype=np.uint8)      # just "count 1"
    A = sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(cicero_adata.n_obs, adata.n_obs)
        )

    aggregated_counts = A @ adata.X            # (n_agg  ×  n_genes) CSR
    obs = adata.obs.loc[cicero_adata.obs_names,:] #again were just going to put the obs of the cell center cluster please dont trust this obs
    var = adata.var
    aggregated_adata = sc.AnnData(aggregated_counts, obs = obs, var = var)
    
    return aggregated_adata

def get_peak_to_gene_links(cons, atac_adata, distal_cutoff = 1000, cons_subset_threshold = 0.2):

    cons_subset = cons[np.abs(cons["coaccess_score"]) >= cons_subset_threshold]

    #optional to add. Keeping for now
    atac_adata.var["Proximal|Distal"] = np.where(atac_adata.var["Distance to TSS"] > distal_cutoff,"Proximal","Distal")
    atac_adata.var["is_proximal"] = np.abs(atac_adata.var["Distance to TSS"]) <= distal_cutoff
    
    peak1_annotations = atac_adata.var.loc[cons_subset["Peak1"],:]
    peak2_annotations = atac_adata.var.loc[cons_subset["Peak2"],:]

    #we only analyze the proximal to distal (or distal to proxmial) connections 
    proximal_distal_mask = np.logical_xor(peak1_annotations["is_proximal"].values, peak2_annotations["is_proximal"].values)
    
    peak1_annotations = peak1_annotations[proximal_distal_mask & peak1_annotations["is_proximal"].values]
    peak2_annotations = peak2_annotations[proximal_distal_mask & peak2_annotations["is_proximal"].values]

    return cons_subset[proximal_distal_mask].copy(), peak1_annotations["Gene Name"].to_dict() | peak2_annotations["Gene Name"].to_dict() #merge two dicts

def prepare_matricies(peak_to_gene, rna_adata, atac_adata,):
    
    if np.any(~rna_adata.obs_names.isin(atac_adata.obs_names)) or np.any(~atac_adata.obs_names.isin(rna_adata.obs_names)):
        LOGGER.warning("rna and atac adata do not share the same obs_names. Trying to fix by computing union of cell names")
        total_obs = len(rna_adata.obs_names.union(atac_adata.obs_names))
        obs_names_intersection = rna_adata.obs_names.intersection(atac_adata.obs_names)
        rna_adata = rna_adata[obs_names_intersection,]
        atac_adata = atac_adata[obs_names_intersection,]
        LOGGER.warning(f"Subsetting to {len(obs_names_intersection)/total_obs*100:.2f}% of original observations")

    peaks = list(peak_to_gene.keys())
    genes = list(peak_to_gene.values())

    peaks_idx = atac_adata.var_names.get_indexer(peaks)
    genes_idx = rna_adata.var_names.get_indexer(genes)

    atac_mtx = atac_adata.X[:,peaks_idx]
    rna_mtx = rna_adata.X[:,genes_idx]

    return atac_mtx, rna_mtx


#=========================Correlation Functions==============================

def sparse_corr_cols(A, B):

    if not (issparse(A) and issparse(B)):
        raise TypeError("A and B must be SciPy sparse matrices")
    if A.shape != B.shape:
        raise ValueError("A and B must have the same shape")
    if A.ndim != 2:
        raise ValueError("input must be 2‑D")

    n = A.shape[0]

    sA  = np.asarray(A.sum(axis=0)).ravel()          
    sB  = np.asarray(B.sum(axis=0)).ravel()          
    sA2 = np.asarray(A.power(2).sum(axis=0)).ravel() 
    sB2 = np.asarray(B.power(2).sum(axis=0)).ravel() 
    sAB = np.asarray(A.multiply(B).sum(axis=0)).ravel()  

    num   = n * sAB - sA * sB
    denom = np.sqrt((n * sA2 - sA**2) * (n * sB2 - sB**2))

    with np.errstate(divide="ignore", invalid="ignore"):
        r = num / denom          # Pearson coefficients
        r[denom == 0] = np.nan

    return r
    df = n - 2
    # clip to avoid numerical issues
    r_clip = np.clip(r, -1 + 1e-15, 1 - 1e-15)
    t_stat = r_clip * np.sqrt(df / (1.0 - r_clip**2))
    p = 2.0 * t.sf(np.abs(t_stat), df)  # two-tailed

    p[np.isnan(r)] = np.nan
    return r, p

# partial covariance by Inverse Precision matrix from graphical lasso
# λ (ρ in the paper) – bigger → sparser precision matrix
def sparse_partial_corr_cols(A, B, lam=0.5, mode="default", max_iter=100, tol=1e-6, eps = 1e-4, init_method = "cov"):

    n_samples, p = A.shape
    X = hstack([A, B]).toarray()
    LOGGER.info("Starting GLASSO")
    glasso = QuicGraphicalLasso(lam=lam, mode=mode, max_iter=max_iter, tol=tol, init_method=init_method)
    glasso.fit(X)
    Theta = glasso.precision_
    LOGGER.info("Finished GLASSO")

    diag_sqrt = np.sqrt(np.diag(Theta))
    pc = -np.diagonal(Theta[:p, p:]) / (diag_sqrt[:p] * diag_sqrt[p:])
    return pc

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
# pyCicero

A Python implementation of [Cicero](https://cole-trapnell-lab.github.io/cicero-release/) (Pliner
*et al.*, *Mol. Cell* 2018) for single-cell chromatin accessibility. Main pieces:

- **Co-accessibility.** Cells are aggregated into overlapping kNN pseudobulks. Each 500 kb genomic
  window is then fitted with a distance-penalised graphical lasso, solved with the QUIC solver
  from [skggm](https://github.com/skggm/skggm), and windows run in parallel on all CPU cores.
- **Peak–gene links.** Co-accessible promoter–distal peak pairs are correlated with the
  expression of the promoter's gene in paired single-cell RNA, to call putative enhancers.
- **ROSE super-enhancers.** Nearby peaks are stitched and ranked by signal density, with
  automatic knee detection (after Whyte *et al.* 2013, Lovén *et al.* 2013).
- **Optional GPU.** TF-IDF/LSI/UMAP preprocessing can run on the GPU (CuPy / RAPIDS cuML).

## Pipeline

```
binarised peak x cell AnnData (+ HOMER peak annotation)
  -> preprocess.preprocess_cicero      size factors, log-normalisation, TF-IDF, LSI (drop LSI-1), UMAP
  -> preprocess.make_cicero_adata      overlapping kNN pseudobulks on the UMAP (k = 50; <= 90 % overlap)
  -> run_cicero.run_cicero             per-window distance parameter (bisection on up to 100 random
                                       windows), then graphical lasso with rho_ij = (1 - (1000/d_ij)^0.75) * mean distance
                                       parameter, partial correlations -> coaccess_score per peak pair
  -> correlate_peaks.correlate_peaks   keep promoter(|dist to TSS| <= 1 kb)-distal pairs,
                                       Pearson r between distal-peak accessibility and promoter-gene expression
  -> ROSE.run_ROSE                     stitch peaks within 12.5 kb, rank by signal / size, cut at the knee
```

## System requirements

| | |
|---|---|
| OS | Linux (tested: Ubuntu 22.04.5 LTS). Uses `multiprocessing` with fork; macOS/Windows untested. |
| Python | ≥ 3.10 (tested 3.11.9 and 3.12.2) |
| Hardware | Any multi-core CPU (one process per core by default). The demo needs ≈ 0.6–1.2 GB RAM. Optional: NVIDIA GPU + CUDA 12 for `device="gpu"` preprocessing. |

**Dependencies:**

- numpy < 2, scipy, pandas, scikit-learn < 1.8, scanpy, anndata, umap-learn, tqdm;
- [skggm](https://github.com/skggm/skggm) 0.2.8 (QUIC graphical lasso). The PyPI release needs
  two import fixes for current scikit-learn; `install/install_skggm.sh` applies them.
- Optional GPU path: `cupy-cuda12x` and RAPIDS `cuml`.

**Tested environments**

| environment | python | numpy | scipy | pandas | scikit-learn | scanpy | anndata | umap-learn | skggm |
|---|---|---|---|---|---|---|---|---|---|
| reference environment (development, full-scale runs; RAPIDS 24.08, CuPy 13.3.0) | 3.11.9 | 1.26.4 | 1.14.1 | 2.2.2 | 1.6.1 | 1.10.3 | 0.11.4 | 0.5.7 | 0.2.8 (patched) |
| clean `venv`, `install/install_skggm.sh` + `pip install .` (Nov 2026) | 3.12.2 | 1.26.4 | 1.17.1 | 2.3.3 | 1.7.2 | 1.11.5 | 0.12.19 | 0.5.12 | 0.2.8 (patched) |

## Installation

```bash
git clone https://github.com/timwoo102/pyCicero.git
cd pyCicero
bash install/install_skggm.sh     # numpy<2, scipy, scikit-learn<1.8, Cython; builds patched skggm 0.2.8
pip install .
```

or with conda, using the reference versions:

```bash
conda env create -f environment.yml && conda activate pycicero
bash install/install_skggm.sh && pip install .
```

Typical install time: about 1 minute in an empty virtual environment (measured: 56 s, including
compiling skggm). skggm needs a C++ compiler (`g++`).

## Demo

```bash
python examples/demo_synthetic.py --n-jobs 8 --out demo_output
```

The demo simulates paired ATAC + RNA data:

- 1,500 cells, 2 chromosomes, 300 peaks spaced 20 kb apart;
- 30 regulatory modules, each with one promoter peak and 9 distal peaks;
- each module opens, and its gene is expressed, in a random subset of 5 cell states.

It runs the full pipeline. **Expected output** (measured; exact counts vary slightly between
runs and library versions because UMAP is not seeded, see "Reproducibility"):

```
mean co-accessibility  within modules: ~0.09-0.10
mean co-accessibility  across modules: ~0.000-0.001
top-200 pairs within a module: 100%
~9-12 distal-promoter links with |coaccess| >= 0.2; 100% link the module's own gene
median RNA correlation of links: ~0.50-0.56
```

Files written: `demo_output/coaccessibility.csv` (Peak1, Peak2, coaccess_score) and
`peak_gene_links.csv`. **Expected runtime:** about 15–25 s wall time with 8 worker processes
(measured 13–20 s on a 96-core Linux server).

## Running on your own data

### Input

An `AnnData` of **cells × peaks** with binarised counts in `X`. `var` holds the HOMER
`annotatePeaks.pl` columns for each peak:

| column | used for |
|---|---|
| `Chr`, `Start`, `End` | genomic windows (peaks should be sorted by position) |
| `Mean` | peak midpoint `(Start + End) / 2`; used for peak–peak distances |
| `Distance to TSS` | promoter (≤ 1 kb) vs distal classification |
| `Gene Name` | gene assigned to a promoter peak |

For peak–gene correlation you also need an RNA `AnnData`, cells × genes, with the **same cell
barcodes** (e.g. 10x Multiome) and gene symbols as `var_names`.

```python
import numpy as np, pandas as pd, scanpy as sc
from scipy.io import mmread
from pyCicero import preprocess, run_cicero, correlate_peaks, ROSE

X = mmread("fmatrix.mtx").T.tocsr(); X.data[:] = 1                      # peaks x cells -> cells x peaks, binarised
atac = sc.AnnData(X, obs=pd.read_csv("metadata.csv", index_col=0))
ann = pd.read_csv("annotation.txt", sep="\t")                           # HOMER annotatePeaks.pl output
atac.var = ann.set_index(ann.columns[0]).loc[atac.var_names]            # peak ids must match HOMER PeakID
atac.var.insert(3, "Mean", (atac.var["Start"] + atac.var["End"]) / 2)
atac = atac[:, atac.var["Chr"].astype(str).str.startswith("chr")].copy()

atac = preprocess.preprocess_cicero(atac)                               # device="gpu" for RAPIDS
cicero_ad = preprocess.make_cicero_adata(atac, k=50, seed=0)
cons = run_cicero.run_cicero(cicero_ad)                                 # Peak1, Peak2, coaccess_score

links = correlate_peaks.correlate_peaks(cons, rna, atac, cons_subset_threshold=0.2)
# links: Peak1, Peak2, coaccess_score, Peak{1,2}_Proximal|Distal, Gene, RNA_Correlations
```

A typical enhancer-calling recipe: run the same steps on a copy of the ATAC matrix with peak
columns shuffled (a null). Then keep links with `|coaccess_score| ≥ 0.2` and `RNA_Correlations`
above mean + 2 SD of the shuffled links. The distal peak of each kept pair is the putative
enhancer.

Super-enhancers (ROSE) from an H3K27ac (or ATAC) cells × peaks AnnData with the same `var`
columns:

```python
ranked, knee = ROSE.run_ROSE(h3k27ac_adata, subset=False, max_window_size=12500,
                             TSS_exclusion_distance=2500, smooth_sigma=1.0)
super_enhancers = ranked.iloc[int(knee):]
```

### Key parameters

| function | parameter | default | meaning |
|---|---|---|---|
| `preprocess_cicero` | `TruncatedSVD_params`, `umap_params` | 50 LSI comps; UMAP 15 NN, min_dist 0.5 | embedding used to define pseudobulks; pass `umap_params={..., "random_state": 0}` for a deterministic embedding |
| `make_cicero_adata` | `k`, `seed`, `max_iterations` | 50, 0, 5000 | cells per pseudobulk (must be < number of cells), sampling seed |
| `run_cicero` | `n_jobs` | all CPUs | worker processes; peaks are split into `n_jobs` contiguous batches |
| `run_cicero` | `max_windows_per_batch` | None | optional cap on 500 kb windows per batch |
| `estimate_distance_parameter_parallel` | `max_sample_num`, `max_elements_per_window`, `seed` | 100, 200, 0 | windows sampled for the distance parameter |
| `find_distance_parameter` | `s`, `distance_constraint` | 0.75, 250 kb | Cicero penalty exponent and long-range distance |
| `correlate_peaks` | `distal_cutoff`, `cons_subset_threshold`, `method` | 1000 bp, 0.2, `"pearson"` | promoter definition, co-accessibility cut-off, `"pearson"` or `"partial"` |
| `run_ROSE` | `max_window_size`, `TSS_exclusion_distance`, `smooth_sigma` | 12.5 kb, 2.5 kb, 1.0 | stitching distance, TSS filter, knee smoothing |

### Reproducibility

Seeded:

- pseudobulk sampling (`make_cicero_adata(seed=0)`);
- distance-parameter windows (`seed=0`).

Not seeded by default:

- **UMAP** (`umap_params`), so pseudobulk membership and therefore co-accessibility scores vary
  slightly between runs. Add `"random_state": 0` to `umap_params` for determinism; UMAP then
  runs single-threaded.
- Results also depend on `n_jobs`, because windows that straddle batch boundaries are split.

### Notes and limitations

- Pseudobulking needs more than `k` cells (default 50). Smaller groups raise a scikit-learn
  `n_neighbors` error.
- `correlate_peaks` expects ATAC and RNA cells to share `obs_names`. If they don't, it falls back
  to the intersection and logs a warning.
- `run_ROSE` keeps peaks whose (signed) HOMER `Distance to TSS` is ≤ `TSS_exclusion_distance`.

## Repository layout

```
pyCicero/
  preprocess.py        size factors, TF-IDF / LSI / UMAP, kNN pseudobulks (make_cicero_adata)
  run_cicero.py        windows, distance parameter, QUIC graphical lasso, co-accessibility
  correlate_peaks.py   promoter-distal pairs, peak-gene (partial) correlation
  ROSE.py              peak stitching, signal ranking, knee detection
  gpu_truncated_svd.py CuPy truncated SVD for device="gpu"
examples/demo_synthetic.py
install/               patched skggm installer
notebooks/             worked examples on real data
```

## Citation

If you use pyCicero, please cite ⟨citation to be added⟩ and the methods it implements:

- Pliner H.A. *et al.* Cicero predicts cis-regulatory DNA interactions from single-cell chromatin
  accessibility data. *Mol. Cell* 71, 858–871 (2018).
- Hsieh C.-J. *et al.* QUIC: quadratic approximation for sparse inverse covariance estimation.
  *JMLR* 15, 2911–2947 (2014).
- Laska J. & Narayan M. skggm 0.2.7: A scikit-learn compatible package for Gaussian and related
  graphical models (2017). doi:10.5281/zenodo.830033
- Whyte W.A. *et al.* Master transcription factors and mediator establish super-enhancers at key
  cell identity genes. *Cell* 153, 307–319 (2013).
- Lovén J. *et al.* Selective inhibition of tumor oncogenes by disruption of super-enhancers.
  *Cell* 153, 320–334 (2013).

## License

Apache License 2.0 — see [LICENSE.md](LICENSE.md).

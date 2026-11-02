"""pyCicero demo on simulated single-cell ATAC + RNA (runs in about a minute on 8 CPU cores).

The genome has 2 chromosomes x 150 peaks spaced 20 kb apart. Every 10 consecutive peaks form a
regulatory module: one promoter peak (TSS of gene ``G<k>``) plus 9 distal peaks. Each module is open
in a random subset of 5 cell states and its gene is expressed in the same states. pyCicero should
link peaks within a module (high co-accessibility) and link distal peaks to the module's gene with
a positive RNA correlation.

    python examples/demo_synthetic.py [--n-jobs 8] [--out demo_output]
"""

import argparse
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from pyCicero import correlate_peaks, preprocess, run_cicero


def simulate(n_cells=1500, n_states=5, n_chr=2, peaks_per_chr=150, module_size=10, spacing=20_000, seed=0):
    rng = np.random.default_rng(seed)
    state = rng.integers(0, n_states, n_cells)
    rows = []
    for c in range(n_chr):
        for j in range(peaks_per_chr):
            start = 1_000_000 + j * spacing
            module = c * (peaks_per_chr // module_size) + j // module_size
            is_promoter = j % module_size == 0
            rows.append({"Chr": f"chr{c + 1}", "Start": start, "End": start + 500, "module": module,
                         "Distance to TSS": 0 if is_promoter else (j % module_size) * spacing,
                         "Gene Name": f"G{module}"})
    var = pd.DataFrame(rows, index=[f"peak{i}" for i in range(len(rows))])
    var.insert(3, "Mean", (var["Start"] + var["End"]) / 2)
    n_mod = var["module"].nunique()
    active = rng.random((n_mod, n_states)) < 0.4                      # module x state activity
    p_open = np.where(active[var["module"].to_numpy()][:, state].T, 0.6, 0.03)
    atac = sp.csr_matrix((rng.random(p_open.shape) < p_open).astype(np.float32))
    expr = np.where(active[:, state].T, 6.0, 0.3)
    rna = sp.csr_matrix(rng.poisson(expr).astype(np.float32))
    obs = pd.DataFrame({"state": state.astype(str)}, index=[f"cell{i}" for i in range(n_cells)])
    atac_ad = ad.AnnData(atac, obs=obs.copy(), var=var)
    rna_ad = ad.AnnData(rna, obs=obs.copy(), var=pd.DataFrame(index=[f"G{m}" for m in range(n_mod)]))
    return atac_ad, rna_ad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--out", default="demo_output")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    atac, rna = simulate()
    module = atac.var["module"]
    print(f"simulated {atac.n_obs} cells x {atac.n_vars} peaks, {rna.n_vars} genes")

    atac = preprocess.preprocess_cicero(atac)                         # TF-IDF -> LSI -> UMAP
    cicero_ad = preprocess.make_cicero_adata(atac, k=30, seed=0)      # aggregate kNN pseudobulks
    cons = run_cicero.run_cicero(cicero_ad, n_jobs=args.n_jobs)       # graphical-lasso co-accessibility
    cons = cons.dropna(subset=["coaccess_score"])
    same = module.loc[cons["Peak1"]].to_numpy() == module.loc[cons["Peak2"]].to_numpy()
    print(f"\n{len(cons):,} peak pairs scored")
    print(f"mean co-accessibility  within modules: {cons.loc[same, 'coaccess_score'].mean():.3f}")
    print(f"mean co-accessibility  across modules: {cons.loc[~same, 'coaccess_score'].mean():.3f}")
    top = cons.nlargest(200, "coaccess_score")
    top_same = (module.loc[top["Peak1"]].to_numpy() == module.loc[top["Peak2"]].to_numpy()).mean()
    print(f"top-200 pairs within a module: {top_same:.0%}")

    linked = correlate_peaks.correlate_peaks(cons, rna, atac, cons_subset_threshold=0.2)
    distal = np.where(linked["Peak1_Proximal|Distal"] == "Distal", linked["Peak1"], linked["Peak2"])
    correct_gene = (linked["Gene"].to_numpy() == atac.var.loc[distal, "Gene Name"].to_numpy())
    print(f"\n{len(linked):,} distal-promoter links with |coaccess| >= 0.2; "
          f"{correct_gene.mean():.0%} link the module's own gene")
    print(f"median RNA correlation of links: {linked['RNA_Correlations'].median():.3f}")

    cons.to_csv(out / "coaccessibility.csv", index=False)
    linked.to_csv(out / "peak_gene_links.csv", index=False)
    print(f"\nwrote {out}/ in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

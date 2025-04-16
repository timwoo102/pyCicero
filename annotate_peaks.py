import pandas as pd
import numpy as np
from tqdm import tqdm
import pybedtools

import logging
logging.basicConfig(
    format='%(filename)s: %(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)
global LOGGER
LOGGER = logging.getLogger(__name__)

def expand_cons(cons, columns_exapnd = ["Peak1", "Peak2"], coaccess_score_key = "coaccess_score", compute_mean = True):
    
    dfs = []
    for column in columns_exapnd:

        expanded_df_col_names = [column + "_" + x for x in ["Chromosome", "Start", "End"]]
        expanded_df = pd.DataFrame([x.split("-") for x in cons[column]], columns = expanded_df_col_names)
        expanded_df[expanded_df_col_names[1]] = expanded_df[expanded_df_col_names[1]].astype(np.int32)
        expanded_df[expanded_df_col_names[2]] = expanded_df[expanded_df_col_names[2]].astype(np.int32)
        
        if compute_mean:
            expanded_df[column + "_Mean"] = (expanded_df[expanded_df_col_names[1]] + expanded_df[expanded_df_col_names[2]]) // 2

        dfs.append(expanded_df)

    expanded_cons = pd.concat(dfs, axis = 1)
    expanded_cons[coaccess_score_key] = cons[coaccess_score_key]
    return expanded_cons

def calculate_closest_TSS(cons_chromosome_positions, tss_chromosome_positions):
    closest_tss_index = np.searchsorted(tss_chromosome_positions, cons_chromosome_positions, side="left")
    
    left_candidates = np.empty(closest_tss_index.shape, dtype=float)
    mask_left = closest_tss_index > 0
    left_candidates[mask_left] = tss_chromosome_positions[closest_tss_index[mask_left] - 1]
    left_candidates[~mask_left] = -np.inf

    right_candidates = np.empty(closest_tss_index.shape, dtype=float)
    mask_right = closest_tss_index < len(tss_chromosome_positions)
    right_candidates[mask_right] = tss_chromosome_positions[closest_tss_index[mask_right]]
    right_candidates[~mask_right] = np.inf

    left_diff = left_candidates - cons_chromosome_positions
    right_diff = right_candidates - cons_chromosome_positions

    closest_diff = np.where(np.abs(left_diff) <= np.abs(right_diff), left_diff, right_diff)
    
    return closest_diff

def annotate_distal_proximal(cons, tss, quiet = True):
    
    chromosomes = np.intersect1d(cons["Peak1_Chromosome"].unique(), tss["Chromosome"].unique())
    chromosomes_disregarded = np.setdiff1d(cons["Peak1_Chromosome"].unique(), tss["Chromosome"].unique())
    if len(chromosomes_disregarded) > 0:
        LOGGER.warning(f"GTF file did not contain all chromosomes or some chromosome names did not match disregarding closest TSS for the following chromosomes: {chromosomes_disregarded}")
    
    cons.insert(4, "Peak1_Closest_TSS", np.inf)
    cons.insert(9, "Peak2_Closest_TSS", np.inf)

    for chromosome in tqdm(chromosomes, disable = quiet):

        tss_chromosome_positions = tss[tss["Chromosome"] == chromosome].sort_values("TSS")["TSS"].values
        cons_chromosome_peak1_positions = cons.loc[cons["Peak1_Chromosome"] == chromosome, "Peak1_Mean"].values
        cons_chromosome_peak2_positions = cons.loc[cons["Peak1_Chromosome"] == chromosome, "Peak2_Mean"].values

        cons.loc[cons["Peak1_Chromosome"] == chromosome, "Peak1_Closest_TSS"] = calculate_closest_TSS(cons_chromosome_peak1_positions, tss_chromosome_positions)
        cons.loc[cons["Peak2_Chromosome"] == chromosome, "Peak2_Closest_TSS"] = calculate_closest_TSS(cons_chromosome_peak2_positions, tss_chromosome_positions)
    
    cons = cons[np.isfinite(cons["Peak1_Closest_TSS"]) | np.isfinite(cons["Peak2_Closest_TSS"])]

    cons.insert(5, "Peak1_Distal_Proximal_Annotation", "Distal")
    cons.insert(10, "Peak2_Distal_Proximal_Annotation", "Distal")
    cons.loc[np.abs(cons["Peak1_Closest_TSS"]) < 1000, "Peak1_Distal_Proximal_Annotation"] = "Proximal"
    cons.loc[np.abs(cons["Peak2_Closest_TSS"]) < 1000, "Peak2_Distal_Proximal_Annotation"] = "Proximal"
    cons = cons.copy() #?????????????? prevents some view warning 
    cons["Linkage_Type"] = cons["Peak1_Distal_Proximal_Annotation"] + "-to-" + cons["Peak2_Distal_Proximal_Annotation"]
    return cons

def annotate_peaks_with_genes(df_peaks, df_gene):

    df_peaks = df_peaks.copy()
    df_peaks['peak_id'] = df_peaks.index
    
    bed_peaks = pybedtools.BedTool.from_dataframe(df_peaks[['chr', 'start', 'end', 'peak_id']])
    bed_gene = pybedtools.BedTool.from_dataframe(df_gene)
    
    intersection = bed_peaks.intersect(b=bed_gene, wa=True, wb=True, loj=True)
    
    df_intersect = intersection.to_dataframe(names=[
        'chr_peak', 'start_peak', 'end_peak', 'peak_id',
        'chr_gene', 'start_gene', 'end_gene', 'gene_name'
    ])
    

    def aggregate_genes(series):

        genes = [g for g in series if g != '.' and pd.notnull(g)]
        if not genes:
            return None

        return ";".join(sorted(set(genes)))
    
    # Group by the unique peak identifier to aggregate gene names without collapsing duplicates in df_peaks.
    aggregated = df_intersect.groupby('peak_id').agg({
        'gene_name': aggregate_genes,
        'chr_peak': 'first',
        'start_peak': 'first',
        'end_peak': 'first'
    }).reset_index()

    return aggregated["gene_name"]

def annotate_genes(cons, genes_df):
    peak_1_df = cons.loc[:,("Peak1_Chromosome", "Peak1_Start", "Peak1_End")]
    peak_1_df.columns = ["chr", "start", "end"]

    peak_2_df = cons.loc[:,("Peak2_Chromosome", "Peak2_Start", "Peak2_End")]
    peak_2_df.columns = ["chr", "start", "end"]

    cons.insert(6, "Peak_1_Gene_Annotation", annotate_peaks_with_genes(peak_1_df, genes_df))
    cons.insert(13, "Peak_2_Gene_Annotation", annotate_peaks_with_genes(peak_2_df, genes_df))
    return cons
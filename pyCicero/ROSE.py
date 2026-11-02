from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
import pandas as pd
import numpy as np

def run_ROSE(adata, max_window_size = 12500, TSS_exclusion_distance = 2500, smooth_sigma = 1.0, subset = True, sort = True, quiet = True):

    stitched_peaks = stitch_peaks(adata.var, max_window_size= max_window_size, TSS_exclusion_distance = TSS_exclusion_distance, sort = sort, quiet = quiet)
    stitch_peaks_ranked = rank_peak_sizes(stitched_peaks, adata.X)
    optimal_knee = find_optimal_knee(stitch_peaks_ranked.index, stitch_peaks_ranked["Signal_to_Size"], smooth_sigma=smooth_sigma)
    
    if subset:
        return stitch_peaks_ranked.iloc[int(optimal_knee):,].copy()
    else:
        return stitch_peaks_ranked, optimal_knee

#generates non-overlapping peaks by stitching together peaks that are within the same window size
#Time: O(n)
def stitch_peaks(range_df, max_window_size = 12500, TSS_exclusion_distance = 2500, sort = True, quiet = True):
    
    range_df = range_df[range_df["Distance to TSS"] <= TSS_exclusion_distance]
    chrs = range_df["Chr"].values
    starts = range_df["Start"].values
    ends = range_df["End"].values

    stitched_ranges = []
    for chr in tqdm(np.unique(chrs), disable = quiet):

        chr_mask = chrs == chr
        curr_chr_starts = starts[chr_mask]
        curr_chr_ends = ends[chr_mask]

        if sort:
            sorted_idxs = np.argsort(curr_chr_starts)
            curr_chr_starts = curr_chr_starts[sorted_idxs]
            curr_chr_ends = curr_chr_ends[sorted_idxs]

        # compute res[i] = arr2[i+1] - arr1[i], 0th element is 0
        peak_dists = curr_chr_ends[1:] - curr_chr_starts[:-1]
        peak_dists = np.insert(peak_dists, 0, 0)

        #for every peak we determine which peaks are within the window size if the i-th peak is chosen as the "center"
        #peak_groupings keeps track of the number of peaks within the window for each peak
        peak_groupings = np.zeros(len(peak_dists), dtype=np.int16)
        idx_peaks_nearby = [[] for _ in range(len(peak_dists))]
        for i in range(len(peak_dists)):
            
            curr_dist = 0
            idx_peaks_nearby[i].append(i)
            peak_groupings[i] += 1
            
            #only need to forward check as if peak i is in range of peak j the oposite is true
            for j in range(i+1, len(peak_dists)): 
                
                #div 2 because we dont look backwards
                if curr_dist+peak_dists[j] > (max_window_size/2):
                    break

                else:
                    curr_dist += peak_dists[j]
                    peak_groupings[i] += 1
                    peak_groupings[j] += 1
                    idx_peaks_nearby[i].append(j)
                    idx_peaks_nearby[j].append(i)

        #peak_buckets is bucket sorted 2D list of [Number of Peaks in window][Indicies of peaks]
        peak_buckets = [[] for _ in range(max(peak_groupings) + 1)]
        for peak_idx, n_peaks in enumerate(peak_groupings):
            peak_buckets[n_peaks].append(peak_idx)

        stitched_range_start = []
        stitched_range_end = []
        stitched_nearby_peaks = []
        peak_idx_used = np.zeros(len(peak_dists), dtype = bool)
        #going from the largest buckets to the smallest until zero
        for i, bucket in enumerate(peak_buckets[::-1]):
            i = len(peak_buckets) - i

            for peak_idx in bucket:

                if peak_idx_used[peak_idx]:
                    continue

                nearby_peaks = np.array(idx_peaks_nearby[peak_idx]) #set to np array to allow bool mask subsetting
                nearby_peaks_used = peak_idx_used[nearby_peaks]
                
                if np.any(nearby_peaks_used):
                    #update idx_peaks_nearby by subseting to make sure it only contains unused peaks
                    idx_peaks_nearby[peak_idx] = nearby_peaks[~nearby_peaks_used]
                    new_n = np.sum(~nearby_peaks_used)          # how many neighbours are still free
                    if new_n: 
                        peak_buckets[new_n].append(peak_idx)
                    continue
                
                peak_idx_used[nearby_peaks] = True
                stitched_range_start.append(curr_chr_starts[nearby_peaks[0]])
                stitched_range_end.append(curr_chr_ends[nearby_peaks[-1]])
                stitched_nearby_peaks.append(nearby_peaks)

        stitched_ranges.append(pd.DataFrame({"Chr":chr, "Start":stitched_range_start, "End":stitched_range_end, "OriginalPeaksIDs":stitched_nearby_peaks}))
    
    stitched_ranges = pd.concat(stitched_ranges, axis = 0)
    
    return stitched_ranges

def rank_peak_sizes(stitched_peaks, matrix, weight = True):
    totals_per_peak = np.ravel(matrix.sum(axis=0)) 
    origin_lists = stitched_peaks["OriginalPeaksIDs"].to_numpy()   # object dtype
    lengths      = np.fromiter((len(v) for v in origin_lists), int, stitched_peaks.shape[0])
    owners       = np.repeat(np.arange(len(lengths)), lengths)     # owner for each id
    all_ids      = np.concatenate(origin_lists)                    # the ids themselves
    weights      = totals_per_peak[all_ids]                        # look-up in one shot
    signal = np.bincount(owners, weights=weights,
                        minlength=stitched_peaks.shape[0]).astype(float)

    stitched_peaks["Signal"] = signal
    stitched_peaks["Size"] = stitched_peaks["End"] - stitched_peaks["Start"]
    stitched_peaks["Signal_to_Size"] = stitched_peaks["Signal"]/stitched_peaks["Size"]
    stitched_peaks = stitched_peaks.sort_values("Signal_to_Size", ascending=True).reset_index(drop=True)

    return stitched_peaks

def find_optimal_knee(x, y, smooth_sigma = 2.0):
    
    if smooth_sigma > 0:
        y_s = gaussian_filter1d(y, sigma=smooth_sigma)
    else:
        y_s = y

    # First derivative
    dy_dx = np.gradient(y_s, x)

    # handle log(0)
    eps = np.finfo(y.dtype).eps
    dy_dx = np.maximum(dy_dx, eps)

    # log-derivative and its derivative
    dlog_dx = np.gradient(np.log(dy_dx), x)

    # Index of maximum
    i_star = np.argmax(dlog_dx)

    # Sub-grid refinement with quadratic fit to the local three points
    if 0 < i_star < len(x)-1:

        x_fit   = x[i_star-1:i_star+2]
        y_fit   = dlog_dx[i_star-1:i_star+2]
        coeffs  = np.polyfit(x_fit, y_fit, 2)
        a, b, _ = coeffs

        # parabola opens downward
        if a < 0:
            x_star = -b/(2*a)
        else:
            x_star = x[i_star]
    else:
        x_star = x[i_star]

    return x_star

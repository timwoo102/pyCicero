import numpy as np
import pandas as pd
from tqdm import tqdm
import scanpy as sc

from sklearn.metrics import pairwise_distances
from scipy.sparse import issparse
from inverse_covariance import QuicGraphicalLasso #skggm

import logging
logging.basicConfig(
    format='%(filename)s: %(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)
global LOGGER
LOGGER = logging.getLogger(__name__)

import multiprocessing
from multiprocessing import Pool
from functools import partial
from sklearn.metrics import pairwise_distances

#=====================MAIN====================================================

def run_cicero(cicero_adata):
    n_cpu = multiprocessing.cpu_count()
    LOGGER.info(f"Starting multiprocessing pool for estimate_distance_parameter_parallel with {n_cpu} processes")

    batches = n_cpu
    indices = np.linspace(0, cicero_adata.n_vars, batches + 1, dtype=int)
    cicero_adata_batch_subsets = [cicero_adata[:, indices[index]:indices[index + 1]].copy() for index in range(batches)]

    LOGGER.info("Generating Windows")
    genomic_windows_df = generate_genomic_windows(cicero_adata)
    LOGGER.info("Starting distance_parameter_estimation")
    distance_parameters = estimate_distance_parameter_parallel(cicero_adata, genomic_windows_df)
    LOGGER.info("Finished distance_parameter_estimation")
    LOGGER.info("Starting generate_cicero_models")
    process_subset_with_fixed_param = partial(process_subset, distance_mean=np.mean(distance_parameters))
    with multiprocessing.Pool(processes=n_cpu) as pool:
            LOGGER.info(f"Starting Cicero with {n_cpu} processes")
            cicero_results = pool.map(process_subset_with_fixed_param, cicero_adata_batch_subsets)
            LOGGER.info("Finished generate_cicero_models")

    LOGGER.info("Starting assemble_connections")
    cons_rec = assemble_connections(cicero_results)
    LOGGER.info("Finished assemble_connections")
    return cons_rec

#=====================MAIN-END====================================================

#generate directly from cicero_adata.var
def generate_genomic_windows(cicero_adata, window_size = 5e5):
    
    #assume sorted indicies #TODO:make a function that sorts 
    unique_chromosomes = cicero_adata.var["Chromosome"].unique()
    genomic_ranges = pd.DataFrame(index = range(len(unique_chromosomes)), columns = ["chromosome", "start", "end"])
    for index, chromsome in enumerate(cicero_adata.var["Chromosome"].unique()):
        var_df = cicero_adata.var[cicero_adata.var["Chromosome"] == chromsome]
        start = var_df.iloc[0]['Start']
        start -= start%window_size #shrink to nearest window size
        end = var_df.iloc[-1]['End']
        end = (end//window_size + 1) * window_size #extend to nearest window size
        genomic_ranges.iloc[index,] = [chromsome, start, end] 

    window_start_list = [np.arange(start, stop, window_size) for start, stop in zip(genomic_ranges["start"], genomic_ranges["end"])]
    window_end_list = [windows + window_size for windows in window_start_list] # Create a list of window arrays for each row.
    window_counts = [windows.size for windows in window_start_list] #number of windows per chromsome

    windows_df = pd.DataFrame({
    'window_chromsome_name': np.repeat(genomic_ranges['chromosome'].values, window_counts),
    'window_start': np.concatenate(window_start_list),
    'window_end': np.concatenate(window_end_list)
    })
    
    return windows_df        

#single core - can use cupy pairwise_dist if possible, offers some speedup but not much
#Helper for estimate_distance_parameter_parallel
def find_distance_parameter(data, distance_matrix,
                            s = 0.75, distance_constraint = 2.5e5, distance_parameter_convergence = 1e-22, max_itterations = 100,
                            distance_parameter_min = 0, distance_parameter_max = 2, initial_distance_parameter = 2,
                            QuicGraphicalLasso_parameters = {"init_method":"cov"},
                            quiet = True):
    
    curr_distance_parameter = initial_distance_parameter 
    
    if(np.sum(distance_matrix > distance_constraint)/2 < 1):
        return
    
    if issparse(data):
            data = data.toarray()

    for _ in tqdm(range(max_itterations), disable = quiet):

        rho_mat = get_rho_mat(distance_matrix, curr_distance_parameter, s)
        cov_mat = np.cov(data)
        np.fill_diagonal(cov_mat, cov_mat.diagonal() + 1e-4)
        gl_precision_matrix = QuicGraphicalLasso(lam = rho_mat, **QuicGraphicalLasso_parameters).fit(cov_mat).precision_
        
        n_big_entries = np.sum(distance_matrix > distance_constraint)
        if( (np.sum(gl_precision_matrix[distance_matrix > distance_constraint] != 0)/n_big_entries > 0.05) or # assures that the amount of long range connections are less than 5%
            (np.sum(gl_precision_matrix == 0)/(gl_precision_matrix.shape[0]**2) < 0.2)):                      # assures that the matrix has at least 20% zeros 
            longs_zero = False
        else:
            longs_zero = True
        
        if (longs_zero != True or curr_distance_parameter == 0):  # if the longs_zero condition is met or the distance_parameter is zero
            distance_parameter_min = curr_distance_parameter        # then we update the minimum parameter
        else:
            distance_parameter_max = curr_distance_parameter        # otherwise, update the maximum parameter
            

        new_distance_parameter = (distance_parameter_min + distance_parameter_max) / 2

        if new_distance_parameter == initial_distance_parameter:
            new_distance_parameter = 2 * initial_distance_parameter
            initial_distance_parameter = new_distance_parameter

        if distance_parameter_convergence > abs(curr_distance_parameter - new_distance_parameter):  # check convergence criteria
            return curr_distance_parameter
        else:
            curr_distance_parameter = new_distance_parameter
            
    return curr_distance_parameter

#Helper for estimate_distance_parameter_parallel
def _process_window_distance(window, cicero_adata,
                    max_elements_per_window, pairwise_distances_parameters,
                    find_distance_parameter_parameters = {}):

    # Subset the data based on window parameters
    cicero_subset = subset_cicero_adata_window(cicero_adata, window[0], window[1], window[2])
    cicero_subset.X = cicero_subset.X.astype(np.int32) #should disable COW which should be faster also disables 
    
    # Filter out windows with no variables or too many variables
    if cicero_subset.n_vars == 0 or cicero_subset.n_vars > max_elements_per_window:
        return None

    # Compute the distance matrix and extract the distance parameter
    mean_bp = cicero_subset.var["mean_bp"].values.reshape(-1,1)
    distance_matrix = pairwise_distances(mean_bp, 
                                         **pairwise_distances_parameters)
    distance_param = find_distance_parameter(cicero_subset.X.T, distance_matrix,
                                             **find_distance_parameter_parameters)
    return distance_param

def estimate_distance_parameter_parallel(cicero_adata, genomic_ranges,
                                         max_sample_num=100,
                                         max_elements_per_window=200,
                                         max_itterations=500, seed=0,
                                         quiet=True,
                                         find_distance_parameter_parameters={},
                                         pairwise_distances_parameters={},
                                         n_cpu = multiprocessing.cpu_count()):

    rng = np.random.default_rng(seed)
    selected_windows = genomic_ranges.index.values.copy()
    rng.shuffle(selected_windows)

    distance_parameters = []
    num_windows = min(len(selected_windows), max_itterations)
    window_indices = selected_windows[:num_windows]
    windows = genomic_ranges.iloc[window_indices,].values.tolist()
    
    func = partial(
        _process_window_distance,
        cicero_adata=cicero_adata,
        max_elements_per_window=max_elements_per_window,
        pairwise_distances_parameters=pairwise_distances_parameters,
        find_distance_parameter_parameters=find_distance_parameter_parameters
    )

    with multiprocessing.Pool(processes=n_cpu) as pool:
        results_iterator = pool.imap_unordered(func, windows, chunksize=10)
        for result in tqdm(results_iterator, total=num_windows, disable=quiet):
            if result is not None:
                distance_parameters.append(result)
                if len(distance_parameters) >= max_sample_num:
                    pool.terminate()
                    break

    if len(distance_parameters) == 0:
        LOGGER.error("No Distance Parameters were able to be calculated!!")
        return [1]
    
    return distance_parameters

def generate_cicero_models(cicero_adata, genomic_windows_df, distance_parameter,
                                        s = 0.75,
                                        window_size=5e5,
                                        max_elements_per_window = 200,
                                        quiet=True,
                                        pairwise_distances_parameters = {},
                                        QuicGraphicalLasso_parameters = {"init_method":"cov"}):
    
    qgl_objs = {}
    correlation_matricies = {}
    peak_names = {}
    itterations = 0
    for window_index in tqdm(genomic_windows_df.index.values, disable = quiet):
        
        window = genomic_windows_df.iloc[window_index,:]
        cicero_adata_window_subset = subset_cicero_adata_window(cicero_adata, **window.to_dict())
        cicero_adata_window_subset.X = cicero_adata_window_subset.X.astype(np.float32)
            
        if cicero_adata_window_subset.n_vars <= 1 or cicero_adata_window_subset.n_vars > max_elements_per_window:
            qgl_objs[window_index] = pd.NA
            correlation_matricies[window_index] = pd.NA
            continue
            
        mean_bp = np.array(cicero_adata_window_subset.var["mean_bp"].values).reshape(-1,1)
        distance_matrix = pairwise_distances(mean_bp, **pairwise_distances_parameters)

        data = cicero_adata_window_subset.X.T
        if issparse(data):
            data = data.toarray()

        rho_mat = get_rho_mat(distance_matrix, distance_parameter, s)
        cov_mat = np.cov(data)
        np.fill_diagonal(cov_mat, cov_mat.diagonal() + 1e-4)

        qgl_out = QuicGraphicalLasso(lam = rho_mat, **QuicGraphicalLasso_parameters).fit(cov_mat)
        # covariance_matricies[window_index] = qgl_out._covariance
        qgl_objs[window_index] = qgl_out
        correlation_matricies[window_index] = cov2cor(qgl_out.covariance_)
        peak_names[window_index] = cicero_adata_window_subset.var_names
        itterations += 1
        if itterations == 200:
            break
        
    genomic_windows_df["Fitted_QuicGraphicalLasso_obj"] = pd.Series(qgl_objs)
    genomic_windows_df["correlation_matrix"] = pd.Series(correlation_matricies)
    genomic_windows_df["peak_names"] = pd.Series(peak_names)
    return genomic_windows_df

def process_subset(curr_cicero_adata, distance_mean=0.3):
    genomic_windows_df = generate_genomic_windows(curr_cicero_adata)
    cicero_models = generate_cicero_models(curr_cicero_adata, genomic_windows_df, distance_mean)
    return cicero_models

def assemble_connections(cicero_results, quiet = True):

    good_cicero_results = pd.concat(cicero_results, axis = 0).dropna()
    correlation_dfs = []

    for index, row in tqdm(good_cicero_results.iterrows(), total = good_cicero_results.shape[0], disable = quiet):
        temp_correlation = row["correlation_matrix"]
        temp_correlation_names = row["peak_names"]
        temp_correlation_df = pd.DataFrame(temp_correlation, index = temp_correlation_names, columns = temp_correlation_names)
        temp_correlation_df = temp_correlation_df.reset_index().rename(columns={'x': 'row'})
        temp_correlation_df = temp_correlation_df.melt(id_vars='row', var_name='variable', value_name='value')
        correlation_dfs.append(temp_correlation_df)

    agg_df = pd.concat(correlation_dfs, axis=0)
    agg_df = agg_df.groupby(['row', 'variable'])['value'].agg(min_val='min', max_val='max', mean_val='mean').reset_index()
    agg_df['mean_coaccess'] = np.where(agg_df['min_val'] >= 0,
                                    agg_df['mean_val'],
                                    np.where(agg_df['max_val'] <= 0, agg_df['mean_val'], np.nan))
    agg_df = agg_df[agg_df['row'] < agg_df['variable']].copy() #keep only one copy/diagonal elements
    
    return agg_df

#====================================HELPER FUNCTIONS============================================================================
def subset_cicero_adata_window(cicero_adata, window_chromsome_name, window_start, window_end):
    cicero_adata_window_subset = cicero_adata[:,(cicero_adata.var["Start"] >= window_start) & 
                                              (cicero_adata.var["End"] <= window_end) &
                                              (cicero_adata.var["Chromosome"] == window_chromsome_name)].copy()
    sc.pp.filter_cells(cicero_adata_window_subset, min_counts=1)
    cicero_adata_window_subset.var["mean_bp"] = (cicero_adata_window_subset.var["Start"] + cicero_adata_window_subset.var["End"])/2
    return cicero_adata_window_subset

def cov2cor(V):
    
    D = np.diag(V)
    Is = D.copy()
    pos = (~np.isnan(D)) & (D > 0)
    Is[pos] = np.sqrt(1.0 / D[pos])
    Is[~pos] = np.nan
    
    if not np.all(pos) or not np.all(np.isfinite(Is)):
        LOGGER.warning("diag(V) had non-positive or NA entries; the non-finite result may be dubious")

    r = (Is[:, np.newaxis] * V) * Is[np.newaxis, :]
    np.fill_diagonal(r, 1)
    
    return r

def get_rho_mat(dist_matrix, distance_parameter, s, xmin = 1000):
    np.seterr(divide='ignore')
    out = (1 - (xmin / dist_matrix)**s) * distance_parameter
    out[~np.isfinite(out)] = 0
    out[out < 0] = 0
    return out

#=================================================OLD THINGS THAT ARE SLOW BUT STILL COULD BE USEFUL =====================================================================================
def generate_genomic_windows_from_df(genomic_ranges, window_size = 500000):
    #returns a pd.DataFrame two columns chrosome, window_start, window_end for all possible genomic windows of window size

    adjusted_sizes = ((genomic_ranges['size'] + window_size - 1) // window_size) * window_size #make sure to capture the full chromsome range

    window_start_list = [np.arange(0, s, window_size) for s in adjusted_sizes]
    window_end_list = [windows + window_size for windows in window_start_list] # Create a list of window arrays for each row.
    window_counts = [windows.size for windows in window_start_list] #number of windows per chromsome

    windows_df = pd.DataFrame({
    'window_chromsome_name': np.repeat(genomic_ranges['chromosome'].values, window_counts),
    'window_start': np.concatenate(window_start_list),
    'window_end': np.concatenate(window_end_list)
    })

    return windows_df

# ===============================slower method for some reason COW issue?==================
def estimate_distance_parameter(cicero_adata, genomic_windows_df,
                                window = 5e5, max_sample_num = 100, max_elements_per_window = 200,
                                max_itterations = 500, seed = 0, quiet = True,
                                find_distance_parameter_parameters = {},
                                pairwise_distances_parameters = {},
                                device = "cpu",
                                ):
    
    if device == "gpu": 
        from cuml.metrics import pairwise_distances
    else:
        from sklearn.metrics import pairwise_distances
      
    rng = np.random.default_rng(seed)
    selected_windows = genomic_windows_df.index.values.copy()
    rng.shuffle(selected_windows)

    itterations = 0
    distance_parameters = []

    for window_index in tqdm(selected_windows, disable = quiet, total = min(len(selected_windows), max_itterations)):
        
        if len(distance_parameters) == max_sample_num:
            break

        if itterations == max_itterations:
            break
        
        itterations += 1

        window = genomic_windows_df.iloc[window_index,:]
        cicero_adata_window_subset = subset_cicero_adata_window(cicero_adata, **window.to_dict())
        cicero_adata_window_subset.X = cicero_adata_window_subset.X.astype(np.float32)
        
        if cicero_adata_window_subset.n_vars <= 1 or cicero_adata_window_subset.n_vars > max_elements_per_window:
            continue
        
        mean_bp = np.array(cicero_adata_window_subset.var["mean_bp"].values).reshape(-1,1)
        distance_matrix = pairwise_distances(mean_bp, **pairwise_distances_parameters)
        distance_parameter = find_distance_parameter(cicero_adata_window_subset.X.T, distance_matrix, **find_distance_parameter_parameters)
        
        if distance_parameter is not None:
            distance_parameters.append(distance_parameter)

    if len(distance_parameters) == 0:
        LOGGER.error("No Distance Parameters were able to be calcualted!!")
        return [1]
    
    return distance_parameters

"""
    # could run something like this but its slower for some reason (?)
    # and also distance_param is calcualted on a per batch basis which is
    # different from what cicero originally does but arguably is better
"""
def run_cicero_by_subset(cicero_adata):
    
    n_cpu = n_cpu = multiprocessing.cpu_count()
    batches = n_cpu

    indices = np.linspace(0, cicero_adata.n_vars, batches + 1, dtype=int)
    cicero_adata_batch_subsets = [cicero_adata[:, indices[index]:indices[index + 1]].copy() for index in range(batches)]
    with multiprocessing.Pool(processes=n_cpu) as pool:
        LOGGER.info(f"Starting Cicero with {n_cpu} processes")
        cicero_results = pool.map(process_subset_complete, cicero_adata_batch_subsets)
        LOGGER.info("Finished generate_cicero_models")
    cons_rec = assemble_connections(cicero_results)
    return cons_rec

def process_subset_complete(curr_cicero_adata):
    genomic_windows_df = generate_genomic_windows(curr_cicero_adata)
    distance_parameters = estimate_distance_parameter(curr_cicero_adata, genomic_windows_df)
    distance_mean = np.mean(distance_parameters)
    cicero_results = generate_cicero_models(curr_cicero_adata, genomic_windows_df, distance_mean)
    return cicero_results

#=====================even older and less useful==============================================
# equiavlent methods needed for generate_cicero_models 
# due to processes hanging to wait to read memory too slow
def _process_window_precision_matrix(window_index, genomic_windows_df, cicero_adata, max_elements_per_window,
                    pairwise_distances_parameters, distance_parameter, s,
                    QuicGraphicalLasso_parameters):

    window = genomic_windows_df.iloc[window_index, :]
    window_params = window.to_dict()

    cicero_adata_window_subset = subset_cicero_adata_window(cicero_adata, **window_params)
    cicero_adata_window_subset.X = cicero_adata_window_subset.X.astype(np.float32)

    if cicero_adata_window_subset.n_vars <= 1 or cicero_adata_window_subset.n_vars > max_elements_per_window:
        return window_index, pd.NA


    mean_bp = np.array(cicero_adata_window_subset.var["mean_bp"].values).reshape(-1, 1)
    distance_matrix = pairwise_distances(mean_bp, **pairwise_distances_parameters)

    data = cicero_adata_window_subset.X.T
    if issparse(data):
        data = data.toarray()

    rho_mat = get_rho_mat(distance_matrix, distance_parameter, s)
    cov_mat = np.cov(data)
    np.fill_diagonal(cov_mat, cov_mat.diagonal() + 1e-4)

    gl_precision_matrix = QuicGraphicalLasso(lam=rho_mat, **QuicGraphicalLasso_parameters)\
                            .fit(cov_mat).precision_

    return window_index, gl_precision_matrix

def compute_precision_matrices_parallel(cicero_adata, genomic_ranges, distance_parameter,
                                        s = 0.75,
                                        window_size=5e5,
                                        max_elements_per_window = 200,
                                        quiet=True,
                                        pairwise_distances_parameters = {},
                                        QuicGraphicalLasso_parameters = {"init_method":"cov"}):

    # Generate the genomic windows.
    genomic_windows_df = generate_genomic_windows(genomic_ranges, window_size=window_size)

    # Get the window indices and limit the number if desired.
    window_indices = genomic_windows_df.index.values

    precision_matricies = {}

    LOGGER.info("Starting multiprocessing pool for precision matrix computation")
    pool = Pool(processes=multiprocessing.cpu_count())
    
    # Create a partial function with extra parameters fixed.
    func = partial(_process_window_precision_matrix,
                   genomic_windows_df=genomic_windows_df,
                   cicero_adata=cicero_adata,
                   max_elements_per_window=max_elements_per_window,
                   pairwise_distances_parameters=pairwise_distances_parameters,
                   distance_parameter=distance_parameter,
                   s=s,
                   QuicGraphicalLasso_parameters=QuicGraphicalLasso_parameters)
    
    # Use imap_unordered with a chosen chunksize to reduce dispatch overhead.
    results_iterator = pool.imap_unordered(func, window_indices, chunksize=10)

    for window_idx, precision_matrix in tqdm(results_iterator, total=len(window_indices), disable=quiet):
        precision_matricies[window_idx] = precision_matrix

    pool.close()
    pool.join()

    # Assign the precision matrices into the genomic_windows_df.
    genomic_windows_df["precision_matricies"] = pd.Series(precision_matricies)
    return genomic_windows_df

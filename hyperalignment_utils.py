import os
import neuroboros as nb
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import zscore

def load_saved_transform(sid,transform_dir,roi,target_flag,align_method,whichrois):
    if whichrois == 'cortical':
        transform_fn = os.path.join(transform_dir,
            f"{sid}_targets_{target_flag}_{roi}h_{align_method}.npz")
    else:
        transform_fn = os.path.join(transform_dir,
            f"{sid}_targets_{target_flag}_{roi}_{align_method}.npy")
    if os.path.exists(transform_fn):
        xfm = nb.load(transform_fn)
    else:
        raise FileNotFoundError(f"Transform file not found: {transform_fn}")
    return xfm

def calc_slmeans(ts, sls):
    N_SLs = len(sls)
    sl_means = np.zeros((ts.shape[0], N_SLs))
    for sl in range(N_SLs):
        sl_means[:,sl] = np.mean(ts[:,sls[sl]], axis=1)
    return sl_means

def calculate_connectivity(ts_target,ts_seed,zscore_axis=0):
    conn = 1 - cdist(ts_target.T, ts_seed.T, "correlation")
    conn = np.nan_to_num(zscore(conn, axis=zscore_axis, nan_policy='omit'))
    return conn

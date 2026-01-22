import os
import numpy as np
import pandas as pd
import neuroboros as nb
from scipy.spatial.distance import pdist
from shared_utils.hyperalignment_utils import calculate_connectivity

def pairwise_corrsim_dot(X,dtype=np.float32):
    """
    X: (N, F) or (N, ...)
    dtype: data precision
    Returns condensed pairwise Pearson correlation similarity.
    """
    X = np.asarray(X)
    if X.ndim < 2:
        raise ValueError("Expected X to have shape (N, ...).")

    F = X.reshape(X.shape[0], -1)   # (N, F)
    F = F.astype(dtype, copy=False)

    # z-score per subject
    F -= F.mean(axis=1, keepdims=True)
    std = F.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    F /= std

    # correlation matrix
    C = (F @ F.T) / F.shape[1]      # (N, N)

    # extract condensed upper triangle
    iu = np.triu_indices(C.shape[0], k=1)
    return C[iu].astype(dtype, copy=False)

def loo_corrsim_dot(X,dtype=np.float32):
    """
    X: (N, F) or (N, ...)
    dtype: data precision
    Returns:
        r: (N,) correlation of each row with the mean of all others
    """
    X = np.asarray(X)
    if X.ndim < 2:
        raise ValueError("Expected X to have shape (N, ...).")

    N = X.shape[0]
    F = X.reshape(N, -1).astype(dtype, copy=False)

    # z-score per subject
    F -= F.mean(axis=1, keepdims=True)
    std = F.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    F /= std

    # sum across subjects
    F_sum = F.sum(axis=0, keepdims=True)   # (1, F)

    # leave-one-out mean
    F_loo = (F_sum - F) / (N - 1)           # (N, F)

    # z-score LOO mean
    F_loo -= F_loo.mean(axis=1, keepdims=True)
    std = F_loo.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    F_loo /= std

    # correlation
    r = np.sum(F * F_loo, axis=1) / F.shape[1]
    return r.astype(dtype, copy=False)

def parse_data_coarse_fine(X):
    """
    X: (N, T, Vsl)
    Returns:
      coarse_X:     (N, T, 1)     # mean across vertices
      fine_X:       (N, T, Vsl)   # residuals
    """
    coarse_X = X.mean(axis=2, keepdims=True)   # (N,T,1)
    fine_X = X - coarse_X                      # (N,T,Vsl)
    return coarse_X, fine_X

def conn_features_all_subjects(targets_ts, seed_ts, zscore_axis=0, dtype=np.float32,flatten=True):
    """
    targets_ts: (N, T, nTargets)
    seed_ts:    (N, T, nSeeds)
    dtype: data precision
    returns:    (N, nTargets*nSeeds) connectivity features (flattened)
    """
    N = seed_ts.shape[0]
    nTargets = targets_ts.shape[2]
    nSeeds = seed_ts.shape[2]
    feats = np.empty((N, nTargets * nSeeds), dtype=dtype)
    for s in range(N):
        conn = calculate_connectivity(
            targets_ts[s], seed_ts[s], zscore_axis=zscore_axis, z=True, metric="correlation"
        )
        if flatten:
            feats[s] = conn.reshape(-1).astype(dtype)
        else:
            feats[s] = conn.astype(dtype)
    return feats

def rdm_features_all_subjects(X,dtype=np.float32,metric="correlation"):
    """
    X: (N, T, Vsl)
    dtype: data precision
    metric: distance metric, input to scipy pdist function
    returns (N, T*(T-1)//2) with each row = subject's pdist(X_s, 'correlation')
    """
    N, T, _ = X.shape
    L = T * (T - 1) // 2
    feats = np.empty((N, L), dtype=dtype)
    for s in range(N):
        feats[s] = pdist(X[s], metric=metric)
    return np.nan_to_num(feats)

def compute_targets_ts_from_mapping(dms_lr, source_space, target_space, mask=True, lr = "lr", dtype=np.float32):
    mapping = nb.mapping(lr, source_space, target_space, mask=mask)
    N, T, Vsrc = dms_lr.shape
    X2 = dms_lr.reshape(N*T, Vsrc)      # memmap view
    Y2 = X2 @ mapping                   # returns dense
    Y2 = np.asarray(Y2, dtype=dtype)
    return Y2.reshape(N, T, mapping.shape[1])  # (N,T,Vtgt)

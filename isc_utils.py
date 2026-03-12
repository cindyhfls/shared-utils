import os
import numpy as np
import pandas as pd
import neuroboros as nb
from scipy.spatial.distance import pdist,squareform,cdist
from shared_utils.hyperalignment_utils import calculate_connectivity

# I did some tests and it looked like numpy.corrcoef and scipy.pearsonr and neuroboros.pearsonr are consistent with zscoring with ddof=0 (population standard deviation instead of sample standard deviation), which is what we have here

def FisherAvg(r):
    return np.tanh(np.mean(np.atanh(r)))

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
    F = F.astype(dtype, copy=True) # The original code with copy=False seemed to not have produced any problem, but ChatGPT suggest making an explicit copy

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

def pairwise_corrsim_dot_xy(X, Y, dtype=np.float32, nan=None):
    """
    Pairwise Pearson correlation similarity between rows of X and rows of Y.

    Parameters
    ----------
    X : array, shape (N, F) or (N, ...)
    Y : array, shape (M, F) or (M, ...)  # must match feature size after reshape
    dtype : np dtype
        Working dtype.
    nan : float or None
        If not None, replace NaNs in output with this value.

    Returns
    -------
    C : array, shape (N, M)
        C[i, j] = corr( X[i], Y[j] ) computed across flattened features.
        Equivalent to: 1 - cdist(X, Y, metric="correlation") (up to fp precision).
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    if X.ndim < 2 or Y.ndim < 2:
        raise ValueError("Expected X and Y to have shape (N, ...) and (M, ...).")

    Fx = X.reshape(X.shape[0], -1).astype(dtype, copy=True)  # (N, F)
    Fy = Y.reshape(Y.shape[0], -1).astype(dtype, copy=True)  # (M, F)

    if Fx.shape[1] != Fy.shape[1]:
        raise ValueError(f"Feature mismatch: X has {Fx.shape[1]} features, Y has {Fy.shape[1]}.")

    # z-score each row (ddof=0 matches np.std default; aligns with many cdist impls)
    Fx -= Fx.mean(axis=1, keepdims=True)
    Fy -= Fy.mean(axis=1, keepdims=True)

    sx = Fx.std(axis=1, keepdims=True)
    sy = Fy.std(axis=1, keepdims=True)
    sx[sx == 0] = 1.0
    sy[sy == 0] = 1.0

    Fx /= sx
    Fy /= sy

    # correlation similarity
    C = (Fx @ Fy.T) / Fx.shape[1]   # (N, M)

    if nan is not None:
        C = np.nan_to_num(C, copy=False, nan=nan)

    return C.astype(dtype, copy=False)

def pairwise_corrsim_dot_avg(X,dtype=np.float32):
    """
    X: (N, F) or (N, ...)
    dtype: data precision
    Returns condensed pairwise Pearson correlation similarity, averaged across each subject.
    """
    r = pairwise_corrsim_dot(X,dtype=dtype)
    R = squareform(r)
    ravg = np.empty(R.shape[0])
    for curr_row in range(len(R)):
        lst = [i for i in range(len(R)) if i != curr_row]
        ravg[curr_row] = FisherAvg(R[curr_row,lst])
    return ravg.astype(dtype,copy=False)

def pairwise_corrsim_dot_xy_avg(X,Y,dtype=np.float32):
    """
    X: (N, F) or (N, ...)
    Y: (N, F) or (N, ...)
    dtype: data precision
    Returns condensed pairwise Pearson correlation similarity, averaged across each subject and the within-subject (diagonals).
    """
    R = pairwise_corrsim_dot_xy(X, Y, dtype)
    ravg = np.empty(len(R))
    for curr_row in range(len(R)):
        lst = [i for i in range(len(R)) if i != curr_row]
        ravg[curr_row] = FisherAvg(R[curr_row,lst])
    return ravg,np.diagonal(R)

def loo_corrsim_dot(X, dtype=np.float32):
    """
    X: (N, F) or (N, ...)
    dtype: output / working precision

    Returns
    -------
    r : (N,)
        Correlation-like similarity of each row with the leave-one-out
        mean of the other rows.

    Notes
    -----
    - Each row is z-scored using its own valid (non-NaN) entries.
    - The LOO mean is computed columnwise, omitting NaNs.
    - The final dot product uses only columns where both the row and
      its LOO mean are finite.
    """
    X = np.asarray(X)
    N = X.shape[0]
    F = X.reshape(N, -1).astype(dtype, copy=True)

    # Row-wise z-score, ignoring NaNs
    F -= np.nanmean(F, axis=1, keepdims=True)
    std = np.nanstd(F, axis=1, keepdims=True)
    std[std == 0] = 1.0
    F /= std

    # Valid-entry mask and zero-filled copy for sums
    valid = np.isfinite(F)
    F0 = np.nan_to_num(F, nan=0.0)

    # Columnwise sums / counts across all rows
    F_sum = F0.sum(axis=0, keepdims=True)        # (1, P)
    F_count = valid.sum(axis=0, keepdims=True)   # (1, P)

    # Leave-one-out sums / counts
    loo_sum = F_sum - F0                         # (N, P)
    loo_count = F_count - valid                  # (N, P)

    # Leave-one-out mean, undefined where no valid "other" rows
    F_loo = np.full_like(F, np.nan)
    mask_loo = loo_count > 0
    F_loo[mask_loo] = loo_sum[mask_loo] / loo_count[mask_loo]

    # Row-wise z-score of the LOO mean, ignoring NaNs
    F_loo -= np.nanmean(F_loo, axis=1, keepdims=True)
    std_loo = np.nanstd(F_loo, axis=1, keepdims=True)
    std_loo[std_loo == 0] = 1.0
    F_loo /= std_loo

    # Final correlation-like dot product on overlapping valid columns only
    overlap = np.isfinite(F) & np.isfinite(F_loo)
    num = np.nansum(F * F_loo, axis=1)
    den = overlap.sum(axis=1)

    r = np.full(N, np.nan, dtype=dtype)
    good = den > 1
    r[good] = num[good] / den[good]

    return r.astype(dtype, copy=False)

def loo_corrsim_dot_xy(X,Y,dtype=np.float32):
    """
    X: (N, F) or (N, ...)
    Y: (N, F) or (N, ...)
    dtype: data precision
    Returns:
        r: (N,) correlation of each row in X with the mean of all others in Y
    """
    X = np.asarray(X)
    if X.ndim < 2 or Y.ndim <2:
        raise ValueError("Expected X to have shape (N, ...).")
    if (X.shape[0] !=Y.shape[0]) | (X.shape[1] !=Y.shape[1]):
        raise ValueError(f"X and Y does not have the same shape.")

    N = X.shape[0]
    Fx = X.reshape(X.shape[0], -1).astype(dtype, copy=True)  # (N, F)
    Fy = Y.reshape(Y.shape[0], -1).astype(dtype, copy=True)  # (N, F)

    # z-score each row (ddof=0 matches np.std default; aligns with many cdist impls)
    Fx -= Fx.mean(axis=1, keepdims=True)
    Fy -= Fy.mean(axis=1, keepdims=True)

    sx = Fx.std(axis=1, keepdims=True)
    sy = Fy.std(axis=1, keepdims=True)
    sx[sx == 0] = 1.0
    sy[sy == 0] = 1.0

    Fx /= sx
    Fy /= sy

    # sum across subjects
    Fy_sum = Fy.sum(axis=0, keepdims=True)   # (1, F)

    # leave-one-out mean
    Fy_loo = (Fy_sum - Fy) / (N - 1)           # (N, F)

    # z-score LOO mean
    Fy_loo -= Fy_loo.mean(axis=1, keepdims=True)
    std = Fy_loo.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    Fy_loo /= std

    # correlation
    r = np.sum(Fx * Fy_loo, axis=1) / Fx.shape[1]
    return r.astype(dtype, copy=False)

def psf_corrsim(X,dist,roi,voxel_pairs):
    all_r = np.empty(X.shape[0])
    for isubj in range(X.shape[0]):
        X_others_mean = np.mean(np.delete(X, isubj, axis=0), axis=0)
        if dist==0:
            r = nb.pearsonr(X[isubj,:,:],X_others_mean)
            all_r[isubj] = FisherAvg(r)
        elif voxel_pairs[roi][dist].size == 0:
            all_r[isubj] = np.nan
        else:
            r = nb.pearsonr(X[isubj,:,voxel_pairs[roi][dist][0]].T,X_others_mean[:,voxel_pairs[roi][dist][1]])
            all_r[isubj] = FisherAvg(r)
    return all_r

def psf_corrsim_withinsubj(X,dist,roi,voxel_pairs):
    all_r = np.empty(X.shape[0])
    for isubj in range(X.shape[0]):
        if dist==0:
            r = nb.pearsonr(X[isubj,:,:],X[isubj,:,:])
            all_r[isubj] = FisherAvg(r)
        elif voxel_pairs[roi][dist].size == 0:
            all_r[isubj] = np.nan
        else:
            r = nb.pearsonr(X[isubj,:,voxel_pairs[roi][dist][0]].T,X[isubj,:,voxel_pairs[roi][dist][1]].T)
            all_r[isubj] = FisherAvg(r)
    return all_r

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

def conn_features_all_subjects(targets_ts, seed_ts, zscore_axis=0, z=True, dtype=np.float32,flatten=True):
    """
    targets_ts: (N, T, nTargets)
    seed_ts:    (N, T, nSeeds)
    dtype: data precision
    returns:    (N, nTargets*nSeeds) connectivity features (flattened)
    """
    N = seed_ts.shape[0]
    nTargets = targets_ts.shape[2]
    nSeeds = seed_ts.shape[2]
    if flatten:
        feats = np.empty((N, nTargets * nSeeds), dtype=dtype)
    else:
        feats = np.empty((N,nTargets,nSeeds),dtype=dtype)
    for s in range(N):
        conn = calculate_connectivity(
            targets_ts[s], seed_ts[s], zscore_axis=zscore_axis, z=z, metric="correlation"
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

import os
import numpy as np
import pandas as pd
import neuroboros as nb
from scipy.spatial.distance import pdist,squareform,cdist
from shared_utils.hyperalignment_utils import calculate_connectivity

# I did some tests and it looked like numpy.corrcoef and scipy.pearsonr and neuroboros.pearsonr are consistent with zscoring with ddof=0 (population standard deviation instead of sample standard deviation), which is what we have here

def FisherAvg(r,axis=0,has_nan=False):
    if has_nan:
        return np.tanh(np.nanmean(np.atanh(r),axis=axis))
    else:
        return np.tanh(np.mean(np.atanh(r),axis=axis))

def pairwise_corrsim_dot(X, dtype=np.float32,skip_zscore=False, has_nan=False):
    X = np.asarray(X)
    N = X.shape[0]
    F = X.reshape(N, -1).astype(dtype, copy=False)
    if not skip_zscore:
        if has_nan:
            F -= np.nanmean(F,axis=1, keepdims=True)
            std = np.nanstd(F,axis=1, keepdims=True)
            std[std==0] = 1.0
            F /= std
            F = np.nan_to_num(F)
        else:
            F -= np.mean(F,axis=1, keepdims=True)
            std = np.std(F,axis=1, keepdims=True)
            std[std==0] = 1.0
            F /= std

    num = F @ F.T
    if has_nan:
        valid = np.isfinite(F)  # (N, T*V)
        counts_mat = valid.astype(np.float32) @ valid.astype(np.float32).T  # (N, N)
    else:
        counts_mat = F.shape[1]
    with np.errstate(invalid="ignore", divide="ignore"):
        C = num / counts_mat

    iu = np.triu_indices(N, k=1)
    return C[iu].astype(dtype, copy=False)

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

def loo_corrsim_dot(X,dtype=np.float32):
    """
    This does not take into account the NaNs, we will think about it.
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

def conn_features_all_subjects(targets_ts, seed_ts, zscore_axis=0, z=True, dtype=np.float32,flatten=True,counts=None,**kwargs):
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
        c = counts[s] if counts is not None else None
        conn = calculate_connectivity(
            targets_ts[s], seed_ts[s],
            zscore_axis=zscore_axis, z=z, counts=c,**kwargs
        )
        if flatten:
            feats[s] = conn.reshape(-1).astype(dtype)
        else:
            feats[s] = conn.astype(dtype)
    return feats

def rdm_features_all_subjects(X, dtype=np.float32,z=False,has_nan=True):
    N, T, V = X.shape
    L_full = T * (T - 1) // 2
    iu_full = np.triu_indices(T, k=1)
    feats = np.zeros((N, L_full), dtype=dtype)

    for s in range(N):
        rdm_s = pdist(X[s], metric='correlation')
        if z:
            if has_nan:
                # zscore on valid pairs only
                rdm_mean = np.nanmean(rdm_s)
                rdm_std  = np.nanstd(rdm_s)
                rdm_std[rdm_std==0] = 1.0
                rdm_s    = (rdm_s - rdm_mean) / rdm_std
                feats[s] = np.nan_to_num(rdm_s)
            else:
                rdm_mean = np.mean(rdm_s)
                rdm_std = np.std(rdm_s)
                rdm_std[rdm_std==0] = 1.0
                feats[s] = (rdm_s - rdm_mean) / rdm_std
        else:
            feats[s] = rdm_s # to match with previous results
    return feats

def compute_targets_ts_from_mapping(dms_lr, source_space, target_space, mask=True, lr = "lr", dtype=np.float32):
    mapping = nb.mapping(lr, source_space, target_space, mask=mask)
    N, T, Vsrc = dms_lr.shape
    X2 = dms_lr.reshape(N*T, Vsrc)      # memmap view
    Y2 = X2 @ mapping                   # returns dense
    Y2 = np.asarray(Y2, dtype=dtype)
    return Y2.reshape(N, T, mapping.shape[1])  # (N,T,Vtgt)

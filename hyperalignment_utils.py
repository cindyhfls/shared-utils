import os,functools
import neuroboros as nb
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import zscore
from .data_io_utils import get_dm,get_rois_and_space
from hyperalignment.ridge import ridge
from hyperalignment.procrustes import procrustes

def downsample_template(tpl_fn,roi,template_input_space='onavg-ico64',data_space='onavg-ico32'):
    '''
    Assumed the input to be the gradient descent templates by Feilong (no mask, ico-8 by ico-64 shape)
    Downsample to ico8 by ico32 (masked)
    '''
    # tpl_dir = 'template/sparta-gd_nll8.0_50000' # symlink to Feilong's folder
    mask_lr_ico8 = nb.mask('lr',space='onavg-ico8')
    mappings = {'l':nb.mapping('l',template_input_space,data_space,target_mask=True),
        'r':nb.mapping('r',template_input_space,data_space,target_mask=True)}
    if roi == 'l':
        tpl_l = np.load(tpl_fn) [mask_lr_ico8]@mappings['l']
        return tpl_l
    elif roi == 'r':
        tpl_r = np.load(tpl_fn) [mask_lr_ico8]@mappings['r']
        return tpl_r


def get_hyperalignment_func(alignfunc,alpha=None):
    # which transformation method?
    if alignfunc == "procr" or alignfunc == "procrustes":
        func = functools.partial(procrustes, reflection=True, scaling=False)
    elif alignfunc == "ridge" or alignfunc == "ridgeCV":
        func = functools.partial(ridge, alpha=alpha)
    else:
        ValueError("Unsupported alignment method provided! Choose from 'procr','ridge'")
    return func

def load_saved_transform(sid,transform_dir,roi,target_flag,align_method,whichrois): # -> not generic enough, consider removing
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

def get_target_ts(dm, target_name,target_item=None):
    if isinstance(dm,np.ndarray): # I'll add the dictionary case later
        nS,nT,_ = dm.shape
        if target_name == 'mapping':
            mapping = target_item
            all_target_ts = np.zeros((nS,nT,mapping.shape[1]))
            for isubj in np.arange(nS):
                all_target_ts[isubj] = dm[isubj] @ mapping
        elif target_name == 'searchlight_mean':
            sls = target_item
            all_target_ts = np.zeros((nS,nT,len(sls)))
            for isubj in np.arange(nS):
                all_target_ts[isubj] = calc_slmeans(dm[isubj],sls)
    elif isinstance(dm,dict):
        all_target_ts = []
        for subj_ts in dm.values():
            target_ts = []
            for target_roi in subj_ts.keys():
                if target_name == 'mapping':
                    mapping = target_item
                    target_ts.append(subj_ts[target_roi] @ mapping[target_roi])
                elif target_name == 'searchlight_mean':
                    target_ts.append(np.mean(subj_ts[target_roi],axis=1)[:,np.newaxis])
            all_target_ts.append(np.concatenate(target_ts,axis=1))
    if len(all_target_ts)==1:
        return all_target_ts[0]
    else:
        return all_target_ts

def calc_target_ts_single_subj(dset,sid,target,task,run,**kwargs):
    targetrois,targetspace = get_rois_and_space(target,dset.surface_space,dset.volume_space)
    target_ts = get_dm(dset, sid, targetrois, task, run, targetspace,
            atlas=target,array_output=False,**kwargs)
    if target.startswith('onavg'):
        MAPPINGS = {roi:nb.mapping(roi, dset.surface_space, target, mask=True) for roi in targetrois}
        target_ts = get_target_ts(target_ts,'mapping',target_item=MAPPINGS)
    else:
        target_ts = get_target_ts(target_ts,'searchlight_mean')
    return target_ts

def calculate_connectivity(ts_target,ts_seed,zscore_axis=0,z=True, metric='correlation'):
    conn = 1 - cdist(ts_target.T, ts_seed.T, metric)
    if z:
        conn = np.nan_to_num(zscore(conn, axis=zscore_axis, nan_policy='omit'))
    return conn

def calc_dm_across_runs(dset,sid,roi,seedspace,targets,seeds,task,run,separate_zscore=True,zscore_axis=0,**kwargs):
    if not isinstance(targets,list):
        targets = [targets]
    if not isinstance(sid,str):
        ValueError("Expect single sid not an array or list")
    if type(run) is int:
        run = [run]
    dm_allruns = []
    for curr_run in run:
        seed_ts = get_dm(dset,sid,roi,task,curr_run, seedspace,atlas=seeds,array_output=True,**kwargs)
        dm = []
        for target in targets:
            if target == "None":
                raise ValueError("Transformations not needed for anatomical alignment. Exiting")
            else:
                if target == 'response':
                    dm_allruns.append(seed_ts)
                else:
                    target_ts = calc_target_ts_single_subj(dset,sid, target, task, curr_run,**kwargs)
                    if separate_zscore:
                        dm.append(calculate_connectivity(target_ts,seed_ts,zscore_axis=zscore_axis,z=True))
                    else:
                        dm.append(calculate_connectivity(target_ts,seed_ts,zscore_axis=zscore_axis,z=False))
                    dm = np.concatenate(dm,axis=0)
                    if not separate_zscore:
                        dm = np.nan_to_num(zscore(dm, axis=zscore_axis, nan_policy='omit'))
                    dm_allruns.append(dm)
    try:
        return np.stack(dm_allruns,axis=0)
    except:
        print("Not able to concatenate across runs, different lengths?")
        return dm_allruns

def reliability_weighting_hyperalignment(X, Y, func, voxel_reliability_scores=None, threshold=None):
    """
    Reliability-weighted hyperalignment.

    X, Y: (N, P)  # N timepoints/samples, P voxels/features
    func: callable (X, Y) -> transform
          - Procrustes: returns orthogonal Q (P,P) such that X @ Q ~ Y
          - Ridge: returns linear T_w (P,P) such that X_w @ T_w ~ Y_w

    voxel_reliability_scores: (P,) in [0,1] ideally (or will be clipped)
    threshold: if not None, binarize weights: w=1 if score>=threshold else 0

    Returns:
      - If func is procrustes-like: Q (P,P), orthogonal (up to numeric error)
      - If func is ridge-like:     T (P,P) mapped back to original space
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    if X.shape != Y.shape:
        raise ValueError(f"X and Y must have same shape, got {X.shape} vs {Y.shape}")
    if X.ndim != 2:
        raise ValueError(f"Expected 2D arrays (N,P); got X.ndim={X.ndim}")

    P = X.shape[1]

    # No weighting
    if voxel_reliability_scores is None:
        return func(X, Y)

    scores = np.asarray(voxel_reliability_scores).reshape(-1)
    if scores.shape[0] != P:
        raise ValueError(f"voxel_reliability_scores must have length P={P}, got {scores.shape[0]}")

    if threshold is None:
        w = np.clip(scores.astype(float), 0.0, 1.0)
    else:
        w = np.ones(P, dtype=float)
        w[scores < threshold] = 0.0

    # Construct W = diag(sqrt(w))
    wsqrt = np.sqrt(w)
    # Efficient column scaling instead of making a dense diag
    X_scaled = X * wsqrt[None, :]
    Y_scaled = Y * wsqrt[None, :]

    T_scaled = func(X_scaled, Y_scaled)

    # Decide whether we should map back (ridge) or not (procrustes).
    # Procrustes returns an orthogonal matrix in voxel space; keep it orthogonal.
    # Ridge returns an unconstrained linear map in the scaled space; map back.
    name = getattr(func, "__name__", None)
    is_partial = hasattr(func, "keywords")  # functools.partial
    if is_partial:
        base = getattr(func, "func", None)
        base_name = getattr(base, "__name__", "")
    else:
        base_name = name or ""

    procrustes_like = "procr" in base_name.lower()

    if procrustes_like:
        # This is already the orthogonal Q that best aligns X_scaled -> Y_scaled.
        # Applying it to unweighted X is what you want, and it preserves orthogonality.
        return T_scaled

    # Ridge-like: map back to original space: T = W * T_scaled * W^{-1}
    # Use pseudo-inverse on W if any zeros
    inv_wsqrt = np.zeros_like(wsqrt)
    nz = wsqrt > 0
    inv_wsqrt[nz] = 1.0 / wsqrt[nz]

    # Left-multiply scales rows, right-multiply scales columns
    T = (wsqrt[:, None] * T_scaled) * inv_wsqrt[None, :]
    return T

def evaluate_transformation(X_val, Y_val, xfm):
    """
    Evaluate the transformation on validation set
    """
    # Predict on validation set
    Y_pred = X_val @ xfm  # (16,285) @ (285,285) = (16,285)

    # Calculate metrics
    mse = np.mean((Y_val - Y_pred) ** 2)
    correlation = np.mean([np.corrcoef(Y_val[:, i], Y_pred[:, i])[0, 1]
                          for i in range(Y_val.shape[1])])
    return {
        'mse': mse,
        'correlation': correlation
    }

def loo_cv(X, Y, func,voxel_reliability_scores=None):
    """
    Leave one run out cross-validation
    X : array (nruns, nT, nV)
    Y : array to align to (nT,nV)
    func: the function, ridge or procrustes used to align the data
    reliability: reliability [0-1] to be used for weighting
    """
    nruns = X.shape[0]
    mse_train_all = []
    mse_val_all   = []

    for i in range(nruns):
        X_val = X[i]
        idx = np.arange(nruns) != i
        X_train = np.mean(X[idx], axis=0)
        R = reliability_weighting_hyperalignment(X_train, Y, func, voxel_reliability_scores=voxel_reliability_scores,threshold=None)

        mt = evaluate_transformation(X_train, Y, R)
        mv = evaluate_transformation(X_val, Y, R)

        mse_train_all.append(mt['mse'])
        mse_val_all.append(mv['mse'])

    # average across LOO folds
    return np.mean(mse_train_all), np.mean(mse_val_all)

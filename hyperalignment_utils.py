import os,functools
import neuroboros as nb
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import zscore
from .data_io_utils import get_dm,get_rois_and_space
from hyperalignment.ridge import ridge
from hyperalignment.procrustes import procrustes

def get_hyperalignment_func(alignfunc,alpha=None):
    # which transformation method?
    if alignfunc == "procr" or alignfunc == "procrustes":
        func = functools.partial(procrustes, reflection=True, scaling=False)
    elif alignfunc == "ridge" or alignfunc == "ridgeCV":
        func = functools.partial(ridge, alpha=alpha)
    else:
        ValueError("Unsupported alignment method provided! Choose from 'procr','ridge'")
    return func

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

def calc_target_ts_single_subj(dset,sid,target,task,run,saved_beta_path=None):
    targetrois,targetspace = get_rois_and_space(target,dset.surface_space,dset.volume_space)
    target_ts = get_dm(dset, sid, targetrois, task, run, targetspace,
            atlas=target,saved_beta_path=saved_beta_path,array_output=False)
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

def calc_dm_across_runs(dset,sid,roi,seedspace,targets,seeds,task,run,saved_beta_path=None,zscore_axis=0,separate_zscore=True):
    dm_allruns = []
    for curr_run in run:
        seed_ts = get_dm(dset,sid,roi,task,curr_run, seedspace,atlas=seeds,saved_beta_path=saved_beta_path,array_output=True)
        dm = []
        for target in targets:
            if target == "None":
                raise ValueError("Transformations not needed for anatomical alignment. Exiting")
            else:
                if target == 'response':
                    dm_allruns.append(seed_ts)
                else:
                    target_ts = calc_target_ts_single_subj(dset,sid, target, task, curr_run,saved_beta_path=saved_beta_path)
                    if separate_zscore:
                        dm.append(calculate_connectivity(target_ts,seed_ts,zscore_axis=zscore_axis,z=True))
                    else:
                        dm.append(calculate_connectivity(target_ts,seed_ts,zscore_axis=zscore_axis,z=False))
                    dm = np.concatenate(dm,axis=0)
                    if not separate_zscore:
                        dm = np.nan_to_num(zscore(dm, axis=zscore_axis, nan_policy='omit'))
                    dm_allruns.append(dm)
    return np.stack(dm_allruns,axis=0)

# def calc_reliability_single_subj(dset,sid,targets,seeds,task,run,outdir=None,saved_beta_path=None,zscore_ax=0):
#     seedrois,seedspace = get_rois_and_space(seeds,dset.surface_space,dset.volume_space)
#     if seeds.startswith('Tian'):
#         seedrois = ['Tian_Subcortex']
#     seed_ts_allruns = []
#     for curr_run in run:
#         seed_ts = get_dm(dset, sid, seedrois, task, curr_run, seedspace,
#                         atlas=seeds,saved_beta_path=saved_beta_path,array_output=True)
#         seed_ts_allruns.append(seed_ts[0])
#     for target in targets:
#         targetrois,targetspace = get_rois_and_space(target,dset.surface_space,dset.volume_space)
#         conn_allruns = []
#         for irun,curr_run in enumerate(run):
#             target_ts = get_dm(dset, sid, targetrois, task, curr_run, targetspace,
#                     atlas=target,saved_beta_path=saved_beta_path,array_output=False)
#             if target.startswith('onavg'):
#                 MAPPINGS = {roi:nb.mapping(roi, dset.surface_space, target, mask=True) for roi in targetrois}
#                 target_ts = get_target_ts(target_ts,'mapping',target_item=MAPPINGS)
#             else:
#                 target_ts = get_target_ts(target_ts,'searchlight_mean')
#             conn = calculate_connectivity(target_ts,seed_ts_allruns[irun],zscore_axis = zscore_ax)
#             conn_allruns.append(conn)
#         conn_allruns = np.stack(conn_allruns,axis=0)
#         calpha = cronbach_alpha(conn_allruns,rep_axis=0,var_axis=1)
#         if outdir:
#             save_cfn = f"{outdir}/{sid}_CronbachAlpha_run{''.join(map(str,run))}_reliability_seeds-{seeds}_targets-{target}.npy"
#             np.save(save_cfn,calpha)
#         return calpha

def reliability_weighting_hyperalignment(X, Y, func, voxel_reliability_scores=None,threshold=None):
    """
    Hyperalignment with reliability weighting
    X, Y: (16, 285) - source and target data
    reliability_weights: (285,) - per-voxel reliability
    alpha: regularization strength
    """

    if voxel_reliability_scores is None:
       xfm = func(X, Y)
    else:
        if threshold is None:
            weights = np.clip(voxel_reliability_scores, 0.0, 1.0)
        else:
            weights = np.ones_like(voxel_reliability_scores)
            weights[voxel_reliability_scores < threshold] = 0.0

        # Apply reliability weighting to features (voxels)
        W_reliability = np.diag(np.sqrt(weights))  # (285, 285)

        # Transform X: X_scaled = X @ W_reliability
        X_scaled = X @ W_reliability  # (16, 285)

        # Standard ridge on scaled data
        R_scaled = func(X_scaled, Y)  # (285, 285)

        # Transform back to original voxel space
        xfm = W_reliability @ R_scaled  # (285, 285)
    return xfm

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
        # R, reliability_weights = reliability_weighting_ridge(
        #     X_train, Y, reliability, alpha=ridge_alpha
        # )

        mt = evaluate_transformation(X_train, Y, R)
        mv = evaluate_transformation(X_val, Y, R)

        mse_train_all.append(mt['mse'])
        mse_val_all.append(mv['mse'])

    # average across LOO folds
    return np.mean(mse_train_all), np.mean(mse_val_all)

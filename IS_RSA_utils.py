'''
April 27, 2026:
For the implementation of two-sided permutation p-values, we used the scipy definition where it multiplies the smaller of the one-sided p-value,
but I realized in many packages permutation p-values were implemented with the absolute value sum(abs(self._correlations) >= abs(self.r)) / self.perms,
which seems to be more principled. I am not going to change it back but just so you are aware.
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html
'''
import os
import neuroboros as nb
import numpy as np
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr,pearsonr,rankdata
from collections import defaultdict
from joblib import Parallel, delayed
def normalize_to_unit(data):
    datamin = np.min(data)
    datamax = np.max(data)
    return (data-datamin)/(datamax-datamin)

def normalize_zscore(K_vec):
    return (K_vec - K_vec.mean()) / K_vec.std()

def normalize_frobenius(K): # Standard in kernel literature, preserves origin
    return K / np.linalg.norm(K, 'fro')

def get_IS_RSA_models(data1,data2=None,metric='NN',normalize='',diag_only=False):
    to_vector = False
    if data2 is None:
        data2=data1.copy()
        if normalize!='frobenius':
            to_vector = True

    data1 = data1.flatten()
    data2 = data2.flatten()

    if metric == 'NN': #'nearest neighbor model'
        sim_func = lambda x1,x2: abs(x1-x2)*(-1)
    elif metric == 'AnnaK_mean':
        sim_func = lambda x1,x2: (x1+x2)/2
    elif metric == 'AnnaK_max':
        sim_func = lambda x1,x2: max(x1,x2)
    elif metric == 'AnnaK_min':
        sim_func = lambda x1,x2: min(x1,x2)
    elif metric == 'mean_minus_absdiff': # proposed in Chamberlain et al. 2024 Imaging Neurosci as a mixture, mentioned in Finn et al. 2020
        sim_func = lambda x1,x2: (x1+x2)/2-abs(x1-x2)
    elif metric == 'negative_mean_minus_absdiff':
        sim_func = lambda x1,x2: -(x1+x2)/2-abs(x1-x2)
    elif metric == 'product': # linear kernel
        sim_func = lambda x1,x2: x1*x2 # highly correlated with the annaK
    elif metric == 'same_diff':  # categorical similarity e.g. sex, site
        sim_func = lambda x1, x2: float(x1 == x2)  # 1 if same, 0 if different

    if diag_only:
        assert len(data1)==len(data2),"requires the two inputs to be the same for diag_only"
        similarity = np.zeros(len(data1))
        for ii in range(len(data1)):
            similarity[ii] = sim_func(data1[ii],data2[ii])
    else:
        similarity = np.zeros((len(data1),len(data2)))
        for ii in range(len(data1)):
            if to_vector:
                for jj in range(ii+1,len(data2)):
                    similarity[ii,jj] = sim_func(data1[ii],data2[jj])
                    similarity[jj,ii] = similarity[ii,jj]
            else:
                for jj in range(len(data2)):
                    similarity[ii,jj] = sim_func(data1[ii],data2[jj])
        if  to_vector:
            similarity = squareform(similarity,checks=False)
    if normalize=='unit': # not recommended because sensitivity to outlier and removes range difference
        return normalize_to_unit(similarity)
    elif normalize=='frobenius': # normalization on the matrix, recommended when using as a precomputed kernel
        return normalize_frobenius(similarity)
    elif normalize=='zscore': # normalization on the vector, recommended for OLS
        return normalize_zscore(similarity)
    else:
        return similarity # no normalization is required if using rank based methods

def get_valid_pair_indices(n_total, missing):
    """
    Get valid pair in condensed distance matrix
    """
    i, j = np.triu_indices(n_total, k=1)

    # Create boolean mask: True if pair involves a missing
    missing_mask = np.zeros(len(i), dtype=bool)
    for m in missing:
        missing_mask = missing_mask | (i == m) | (j == m)
    valid_pair_indices = np.where(~missing_mask)[0]

    return valid_pair_indices

def generate_permutations(n, nperm, seed=0): # this function can give repeats, don't use it
    np.random.seed(seed)
    perm = []
    for _ in range(nperm):
        perm.append(np.random.permutation(n))
    return perm

def permute_matrix(v,perm):
    M = squareform(v)
    M_perm = M[np.ix_(perm,perm)]
    return squareform(M_perm)

def calc_correlation(br_similarity,bh_similarity,method='spearman',n_jobs=-1):
    assert(~np.any(np.isnan(bh_similarity)))
    if method=='spearman':
        corr_func = spearmanr
    elif method == 'pearson':
        corr_func = pearsonr
    jobs = []
    for isl in range(br_similarity.shape[0]):
        jobs.append(
            delayed(corr_func)(
            br_similarity[isl],bh_similarity
            )
        )
    all_results = Parallel(n_jobs=n_jobs, prefer="threads")(jobs)
    r_vals = np.array([r for r, p in all_results])
    return r_vals

def IS_RSA_custom(neural_similarity,bx_similarity, perm, valid_idx=None):
    '''
    # the IS_RSA_variance_partitioning meant to replace this since this is single variable and uses slow scipy implmenetation
    # but if we want to use this for some reason,
    # let's remove the valid_idx part and remove the non valid pairs before the input
    neural_similarity: ndarray shaped (nsearchlights, npairs)
    bx_similarity: dictionaries of ndarrays shaped (nsearchlights, npairs)
    perm: list of ndarrays shaped (npairs*,) -> npairs* removed NaNs already
    valid_idx: specify which of the npairs are valid
    # The code I found online <https://github.com/jwcarr/mantel/tree/master/mantel>, which was used in Chamberlain et al. had very inefficient handling of NaN values. And more detrimentally, an incorrect way of calculating p-value for two-tails.
    '''
    if valid_idx is None:
        valid_idx = np.ones(neural_similarity.shape[1], dtype=bool)
    results = defaultdict(dict)
    for metric in bx_similarity.keys():
        bh_similarity = bx_similarity[metric][valid_idx]
        observed = calc_correlation(neural_similarity[:,valid_idx],bh_similarity,method='spearman')
        permuted = []
        nperm = len(perm)
        for iperm in range(nperm):
            M_perm = permute_matrix(np.arange(len(valid_idx)),perm[iperm])
            permuted.append(calc_correlation(neural_similarity[:,valid_idx],bh_similarity[M_perm],method='spearman'))
        permuted = np.stack(permuted,axis=1)
        results[metric]['r'] = observed
        results[metric]['p-upper'] = np.array([np.mean(np.array(permuted[isl]) >= observed[isl]) for isl in range(len(observed))])
        results[metric]['p-lower'] = np.array([np.mean(np.array(permuted[isl]) <= observed[isl]) for isl in range(len(observed))])
        results[metric]['p-twotail'] = np.clip(2*np.min(np.stack((results[metric]['p-lower'],results[metric]['p-upper']),axis=0),axis=0),a_min=0,a_max = 1)
    return results

def _condensed_perm_idx(n_subjects, perm):
    """
    Given a subject-level permutation array, return the permuted condensed pair
    indices without going through squareform. Equivalent to:
        permute_matrix(np.arange(npairs), perm)
    but avoids constructing the full n×n matrix.
    """
    rows, cols = np.triu_indices(n_subjects, k=1)
    new_rows = perm[rows]
    new_cols = perm[cols]
    swap = new_rows > new_cols
    new_rows[swap], new_cols[swap] = new_cols[swap], new_rows[swap]
    return (new_rows * n_subjects
            - new_rows * (new_rows + 1) // 2
            + new_cols - new_rows - 1)

def _ols_r2_batch(Y, X):
    """Y: (nvalid, nsearchlights), X: (nvalid, k). Returns R² per searchlight."""
    nvalid = Y.shape[0]
    Xc = np.column_stack([X, np.ones(nvalid)])
    coef, _, _, _ = np.linalg.lstsq(Xc, Y, rcond=None)
    ss_res = np.sum((Y - Xc @ coef) ** 2, axis=0)
    ss_tot = np.sum((Y - Y.mean(axis=0)) ** 2, axis=0)
    return np.maximum(0.0, 1 - ss_res / ss_tot)

def _pvals(obs, null):
    p_upper = np.mean(null >= obs, axis=0)
    p_lower = np.mean(null <= obs, axis=0)
    p_two   = np.clip(2 * np.minimum(p_upper, p_lower), 0, 1)
    return p_upper, p_lower, p_two

def _compute_metrics_batch(Y_ranked, X_ranked, names):
    nvalid, nsearchlights = Y_ranked.shape
    nkernels = len(names)

    total_r2 = _ols_r2_batch(Y_ranked, X_ranked)
    partial_r = {}
    unique_r2 = {}

    for i, name in enumerate(names):
        other_idx = [j for j in range(nkernels) if j != i]
        X_other   = X_ranked[:, other_idx]
        X_target  = X_ranked[:, i]

        unique_r2[name] = np.maximum(0.0, total_r2 - _ols_r2_batch(Y_ranked, X_other))

        X_other_c = np.column_stack([X_other, np.ones(nvalid)])
        coef_y, _, _, _ = np.linalg.lstsq(X_other_c, Y_ranked, rcond=None)
        resid_y = Y_ranked - X_other_c @ coef_y

        coef_x, _, _, _ = np.linalg.lstsq(X_other_c, X_target, rcond=None)
        resid_x = X_target - X_other_c @ coef_x

        resid_y_c = resid_y - resid_y.mean(axis=0)
        resid_x_c = resid_x - resid_x.mean()
        num   = resid_x_c @ resid_y_c
        denom = (np.sqrt(np.sum(resid_x_c ** 2))
                 * np.sqrt(np.sum(resid_y_c ** 2, axis=0)))
        partial_r[name] = np.where(denom > 0, num / denom, 0.0)

    return partial_r, unique_r2, total_r2

def IS_RSA_variance_partitioning(neural_similarity, bx_similarity, perm=[],
                                  groups=None,n_jobs=-1):
    """
    groups : dict mapping group name to list of predictor names, e.g.
             {'age':    ['age-NN', 'age-AnnaK_mean'],
              'motion': ['meanFD-NN', 'meanFD-AnnaK_mean'],
              'SRS':    ['SRS_SCI_T-NN', 'SRS_SCI_T-AnnaK_mean']}
    """
    names         = list(bx_similarity.keys())
    nsearchlights = neural_similarity.shape[0]

    X_ranked = np.column_stack([
        rankdata(bx_similarity[k]).astype(float)
        for k in names
    ])
    Y_ranked = rankdata(neural_similarity, axis=1).astype(float).T

    # ── helper: group unique R² from a given X matrix ─────────────
    def _group_unique_r2(X, total_r2):
        out = {}
        if groups is None:
            return out
        for grp_name, grp_members in groups.items():
            keep_idx = [i for i, n in enumerate(names) if n not in grp_members]
            if len(keep_idx) == 0:
                # all predictors are in this group
                out[grp_name] = total_r2.copy()
            else:
                X_reduced = X[:, keep_idx]
                out[grp_name] = np.maximum(
                    0.0, total_r2 - _ols_r2_batch(Y_ranked, X_reduced)
                )
        return out

    # ── observed ──────────────────────────────────────────────────
    obs_partial_r, obs_unique_r2, obs_total_r2 = _compute_metrics_batch(
        Y_ranked, X_ranked, names
    )
    obs_group_unique_r2 = _group_unique_r2(X_ranked, obs_total_r2)

    results = defaultdict(dict)
    results['full_model']['r2']        = obs_total_r2
    for name in names:
        for key, obs in [('partial_r', obs_partial_r[name]),
                                ('unique_r2', obs_unique_r2[name])]:

            results[name][key]= obs
    for grp_name in (groups or {}):
        results[grp_name]['unique_r2']  = obs_group_unique_r2[grp_name]

    # ── permutations ──────────────────────────────────────────────────
    if perm: # skip if empty
        nperm         = len(perm)
        n_subjects    = len(perm[0])
        perm_condensed_list = [_condensed_perm_idx(n_subjects, p) for p in perm]

        # ── permutations ──────────────────────────────────────────────
        def _perm_worker(perm_condensed):
            X_perm = X_ranked[perm_condensed]
            partial_r, unique_r2, total_r2 = _compute_metrics_batch(Y_ranked, X_perm, names)
            grp_unique_r2 = _group_unique_r2(X_perm, total_r2)
            return partial_r, unique_r2, grp_unique_r2, total_r2

        perm_results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_perm_worker)(pc) for pc in perm_condensed_list
        )

        null_partial_r      = {n: np.stack([r[0][n] for r in perm_results]) for n in names}
        null_unique_r2      = {n: np.stack([r[1][n] for r in perm_results]) for n in names}
        null_total_r2       = np.stack([r[3] for r in perm_results])
        null_group_unique_r2 = {}
        if groups is not None:
            null_group_unique_r2 = {
                g: np.stack([r[2][g] for r in perm_results])
                for g in groups
            }

        # get p-values from permutations
        pu, pl, pt = _pvals(obs_total_r2, null_total_r2)
        results['full_model']['p-upper']   = pu
        results['full_model']['p-lower']   = pl
        results['full_model']['p-twotail'] = pt

        for name in names:
            for key, obs, null in [('partial_r', obs_partial_r[name], null_partial_r[name]),
                                    ('unique_r2', obs_unique_r2[name], null_unique_r2[name])]:
                pu, pl, pt = _pvals(obs, null)
                results[name][key]                = obs
                results[name][f'p-upper-{key}']   = pu
                results[name][f'p-lower-{key}']   = pl
                results[name][f'p-twotail-{key}'] = pt

        for grp_name in (groups or {}):
            obs  = obs_group_unique_r2[grp_name]
            null = null_group_unique_r2[grp_name]
            pu, pl, pt = _pvals(obs, null)
            results[grp_name]['p-upper-unique_r2']      = pu
            results[grp_name]['p-lower-unique_r2']      = pl
            results[grp_name]['p-twotail-unique_r2']    = pt

    return results

def IS_RSA_variance_partitioning_FL(neural_similarity, bx_similarity, perm=[],
                                     groups=None, n_jobs=-1, checkpoint_dir=None):
    """
    Freedman-Lane version of IS_RSA_variance_partitioning.

    Instead of permuting X (behavioral similarity), this permutes Y residuals
    after removing all predictors except the target. This gives a tighter null
    distribution for group tests when predictors are orthogonal, because the
    null only captures variance unexplained by other predictors.

    Efficiency: all lstsq calls are precomputed once per target. The permutation
    loop uses only matrix multiplications via a precomputed orthonormal basis U
    for the target predictors:
        unique_R2_perm = ||U.T @ R_p||^2 / SS_tot(Y_hat + R_p)
    where R_p = residuals[perm_condensed].

    Trade-off: O(n_targets x n_perm) passes vs O(n_perm) for the original, but
    each pass is much cheaper (no lstsq per permutation).

    Note: partial_r p-values are not computed (unique_r2 only).
    """
    names    = list(bx_similarity.keys())
    X_ranked = np.column_stack([
        rankdata(bx_similarity[k]).astype(float) for k in names
    ])
    Y_ranked = rankdata(neural_similarity, axis=1).astype(float).T
    nvalid   = Y_ranked.shape[0]

    # ── observed stats (identical to original) ────────────────────
    obs_partial_r, obs_unique_r2, obs_total_r2 = _compute_metrics_batch(
        Y_ranked, X_ranked, names
    )
    obs_group_unique_r2 = {}
    if groups is not None:
        for grp_name, grp_members in groups.items():
            keep_idx = [i for i, n in enumerate(names) if n not in grp_members]
            if len(keep_idx) == 0:
                obs_group_unique_r2[grp_name] = obs_total_r2.copy()
            else:
                obs_group_unique_r2[grp_name] = np.maximum(
                    0.0, obs_total_r2 - _ols_r2_batch(Y_ranked, X_ranked[:, keep_idx])
                )

    results = defaultdict(dict)
    results['full_model']['r2'] = obs_total_r2
    for name in names:
        results[name]['partial_r'] = obs_partial_r[name]
        results[name]['unique_r2'] = obs_unique_r2[name]
    for grp_name in (groups or {}):
        results[grp_name]['unique_r2'] = obs_group_unique_r2[grp_name]

    if not perm:
        return results

    n_subjects = len(perm[0])
    perm_condensed_list = [_condensed_perm_idx(n_subjects, p) for p in perm]

    # ── precompute quantities for FL null (no lstsq in permutation loop) ──
    def _fl_precompute(keep_idx, target_idx):
        """
        Returns (Y_hat, residuals, U, Y_hat_c, SS_yhat):
          U       : (nvalid, k_target) orthonormal basis for target predictors
                    after projecting out the reduced model — numerator = ||U.T @ R_p||^2
          Y_hat_c : Y_hat - mean(Y_hat), for SS_tot computation
          SS_yhat : ||Y_hat_c||^2 per searchlight
        """
        if len(keep_idx) == 0:
            X_other_c = np.ones((nvalid, 1))
        else:
            X_other_c = np.column_stack([X_ranked[:, keep_idx], np.ones(nvalid)])

        # fit reduced model → Y_hat and residuals
        coef, _, _, _ = np.linalg.lstsq(X_other_c, Y_ranked, rcond=None)
        Y_hat     = X_other_c @ coef
        residuals = Y_ranked - Y_hat

        # orthonormal basis U for target predictors after projecting out X_reduced
        X_target = X_ranked[:, target_idx]  # (nvalid, k_target)
        coef_x, _, _, _ = np.linalg.lstsq(X_other_c, X_target, rcond=None)
        X_target_orth = X_target - X_other_c @ coef_x   # residualize target on reduced
        U, _ = np.linalg.qr(X_target_orth)              # (nvalid, k_target)
        U = U[:, :X_target_orth.shape[1]]

        Y_hat_c = Y_hat - Y_hat.mean(axis=0)
        SS_yhat = np.sum(Y_hat_c ** 2, axis=0)
        return residuals, U, Y_hat_c, SS_yhat

    def _fl_null(keep_idx, target_idx, obs):
        residuals, U, Y_hat_c, SS_yhat = _fl_precompute(keep_idx, target_idx)

        def _worker(pc):
            R_p   = residuals[pc]                              # (nvalid, nsearchlights)
            R_p_c = R_p - R_p.mean(axis=0)
            numer = np.sum((U.T @ R_p) ** 2, axis=0)          # ||U.T R_p||^2
            cross = np.sum(Y_hat_c * R_p_c, axis=0)
            denom = SS_yhat + 2 * cross + np.sum(R_p_c ** 2, axis=0)
            return np.where(denom > 0, np.maximum(0.0, numer / denom), 0.0)

        null = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_worker)(pc) for pc in perm_condensed_list
        )
        return np.stack(null)   # (nperm, nsearchlights)

    # ── individual predictors ──────────────────────────────────────
    for i, name in enumerate(names):
        ckpt_file = os.path.join(checkpoint_dir, f'null_{name}.npy') if checkpoint_dir else None
        if ckpt_file and os.path.exists(ckpt_file):
            print(f'  [ckpt] loading null for {name}')
            null = np.load(ckpt_file)
        else:
            other_idx = [j for j in range(len(names)) if j != i]
            null = _fl_null(other_idx, [i], obs_unique_r2[name])
            if ckpt_file:
                os.makedirs(checkpoint_dir, exist_ok=True)
                np.save(ckpt_file, null)
                print(f'  [ckpt] saved null for {name}')
        pu, pl, pt = _pvals(obs_unique_r2[name], null)
        results[name]['p-upper-unique_r2']   = pu
        results[name]['p-lower-unique_r2']   = pl
        results[name]['p-twotail-unique_r2'] = pt

    # ── groups ────────────────────────────────────────────────────
    for grp_name, grp_members in (groups or {}).items():
        safe_name = grp_name.replace('/', '_')
        ckpt_file = os.path.join(checkpoint_dir, f'null_grp_{safe_name}.npy') if checkpoint_dir else None
        if ckpt_file and os.path.exists(ckpt_file):
            print(f'  [ckpt] loading null for group {grp_name}')
            null = np.load(ckpt_file)
        else:
            keep_idx   = [i for i, n in enumerate(names) if n not in grp_members]
            target_idx = [i for i, n in enumerate(names) if n in grp_members]
            null = _fl_null(keep_idx, target_idx, obs_group_unique_r2[grp_name])
            if ckpt_file:
                os.makedirs(checkpoint_dir, exist_ok=True)
                np.save(ckpt_file, null)
                print(f'  [ckpt] saved null for group {grp_name}')
        pu, pl, pt = _pvals(obs_group_unique_r2[grp_name], null)
        results[grp_name]['p-upper-unique_r2']   = pu
        results[grp_name]['p-lower-unique_r2']   = pl
        results[grp_name]['p-twotail-unique_r2'] = pt

    return results

def check_kernel_multicollinearity(bx_similarity, rank=True):
    """
    Check multicollinearity among model kernels by computing pairwise
    correlations and Variance Inflation Factors (VIF).

    Parameters
    ----------
    bx_similarity : dict of str -> (npairs,) ndarray
        Model kernels as condensed distance vectors (same format as used in
        IS_RSA_variance_partitioning).
    rank : bool, optional
        If True (default), rank-transform each kernel before computing
        correlations/VIF, consistent with the Spearman-based analyses
        in this module.

    Returns
    -------
    corr_matrix : (nkernels, nkernels) ndarray
        Pairwise Pearson correlations of the (optionally ranked) kernels.
    vif : dict of str -> float
        VIF for each kernel. VIF = 1 / (1 - R²), where R² comes from
        regressing that kernel on all others. VIF > 5 suggests moderate
        multicollinearity; VIF > 10 is often considered severe.
    names : list of str
        Kernel names in the same order as rows/columns of corr_matrix.
    """
    names = list(bx_similarity.keys())
    nkernels = len(names)

    X = np.column_stack([bx_similarity[k] for k in names]).astype(float)

    if rank:
        for j in range(nkernels):
            X[:, j] = rankdata(X[:, j])

    # Pairwise correlation matrix
    corr_matrix = np.corrcoef(X, rowvar=False)

    # VIF for each kernel: regress kernel i on all others, get R²
    vif = {}
    for i, name in enumerate(names):
        y = X[:, i]
        X_other = np.column_stack([X[:, j] for j in range(nkernels) if j != i])
        X_other_c = np.column_stack([X_other, np.ones(len(y))])
        coef, _, _, _ = np.linalg.lstsq(X_other_c, y, rcond=None)
        ss_res = np.sum((y - X_other_c @ coef) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = max(0.0, 1 - ss_res / ss_tot)
        vif[name] = 1.0 / (1.0 - r2) if r2 < 1.0 else np.inf

    return corr_matrix, vif, names

def IS_RSA_neural_reliability(neural_similarity1, neural_similarity2,perm=[],n_jobs=-1):
    Y_ranked = rankdata(neural_similarity1, axis=1).astype(float).T # variable to predict, to be consistent with the behavioral models
    X_ranked = rankdata(neural_similarity2, axis=1).astype(float).T

    results = defaultdict(dict)
    results['all_sl']['r2']=_ols_r2_batch(Y_ranked,X_ranked)
    Yc = Y_ranked - Y_ranked.mean(axis=0)
    Xc = X_ranked - X_ranked.mean(axis=0)
    results['same_sl']['r2'] = ((Xc * Yc).sum(axis=0) /
        np.sqrt((Xc**2).sum(axis=0) * (Yc**2).sum(axis=0))) ** 2
    if perm: # get p-values from permutations
        nperm         = len(perm)
        n_subjects    = len(perm[0])
        perm_condensed_list = [_condensed_perm_idx(n_subjects, p) for p in perm]
        def _perm_worker(perm_condensed):
            all_sl_r2 = _ols_r2_batch(Y_ranked, X_ranked[perm_condensed])
            Xp = X_ranked[perm_condensed]
            Xpc = Xp - Xp.mean(axis=0)
            same_sl_r2 = ((Xpc * Yc).sum(axis=0) /
                np.sqrt((Xpc**2).sum(axis=0) * (Yc**2).sum(axis=0))) ** 2
            return same_sl_r2,all_sl_r2

        perm_results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_perm_worker)(pc) for pc in perm_condensed_list
        )

        null_same_sl_r2       = np.stack([r[0] for r in perm_results])
        null_all_sl_r2       = np.stack([r[1] for r in perm_results])

        pu, pl, pt = _pvals(results['same_sl']['r2'], null_same_sl_r2)
        results['same_sl']['p-upper']   = pu
        results['same_sl']['p-lower']   = pl
        results['same_sl']['p-twotail'] = pt

        pu, pl, pt = _pvals(results['all_sl']['r2'], null_all_sl_r2)
        results['all_sl']['p-upper']   = pu
        results['all_sl']['p-lower']   = pl
        results['all_sl']['p-twotail'] = pt
    return results

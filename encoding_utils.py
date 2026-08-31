print(
    "shared_utils/encoding_utils.py is retired -- moved to litcoder_core's "
    "encoding.models.reference (same content, verified identical). Update your import to: "
    "from encoding.models.reference import <name> (with litcoder_core on sys.path)."
)

# import numpy as np
# import scipy.stats
#
#
# def explainable_variance(data, bias_correction=True, do_zscore=True):
#     """Compute explainable variance for a set of voxels. Copied directly from https://github.com/gallantlab/voxelwise_tutorials/blob/main/voxelwise_tutorials/utils.py
#     Citation: Tom Dupré la Tour et al. 2025
#     Parameters
#     ----------
#     data : array of shape (n_repeats, n_times, n_voxels)
#         fMRI responses of the repeated test set.
#     bias_correction: bool
#         Perform bias correction based on the number of repetitions.
#     do_zscore: bool
#         z-score the data in time. Only set to False if your data time courses
#         are already z-scored.
#
#     Returns
#     -------
#     ev : array of shape (n_voxels, )
#         Explainable variance per voxel.
#     """
#     if do_zscore:
#         data = scipy.stats.zscore(data, axis=1)
#
#     mean_var = data.var(axis=1, dtype=np.float64, ddof=1).mean(axis=0)
#     var_mean = data.mean(axis=0).var(axis=0, dtype=np.float64, ddof=1)
#     ev = var_mean / mean_var
#
#     if bias_correction:
#         n_repeats = data.shape[0]
#         ev = ev - (1 - ev) / (n_repeats - 1)
#     return ev
#
#
# def noise_ceiling_rmax(data, ddof=1):
#     """Schoppe et al. (2016, Frontiers in Computational Neuroscience) noise ceiling, per the
#     "Box 4"-style definition reproduced in review papers on voxelwise encoding models. This
#     targets a DIFFERENT quantity than explainable_variance() above: r_max is the ceiling for
#     predicting the AVERAGE of the M repeats (ytilde), not the ceiling for predicting a single
#     repeat -- so r_max and sqrt(explainable_variance(...)) are not interchangeable (r_max is
#     systematically higher; it grows toward 1 as M grows, since averaging more repeats makes
#     ytilde itself a less noisy target). Use this specifically when comparing a model's
#     prediction against the repeat-averaged response (see absolute_correlation /
#     normalized_correlation below), not as a single-repeat reliability estimate.
#
#         ytilde   = (1/M) sum_k y_k
#         signal_var = (1/M) sum_k Var_t(y_k)  -  (M/(M-1)) * (1/M) sum_k Var_t(y_k - ytilde)
#         r_max    = sqrt(signal_var / Var_t(ytilde))
#
#     (Both terms are MEANS across the M repeats -- the M/(M-1) factor bias-corrects the second
#     mean, it does not multiply a raw, un-averaged sum. Named "signal_var" rather than "ev" in
#     the code below, deliberately not reusing explainable_variance()'s "ev" name: this is an
#     absolute-variance-units estimate of the true signal variance itself, not the dimensionless
#     var_mean/mean_var ratio that explainable_variance() computes -- see the note there.)
#
#     Parameters
#     ----------
#     data : array of shape (n_repeats, n_times, n_voxels)
#         fMRI responses of the repeated test set (M = n_repeats).
#
#     Returns
#     -------
#     r_max : array of shape (n_voxels,)
#         Clipped at 0 before the sqrt (EV can go slightly negative for near-zero-signal
#         voxels, same convention as explainable_variance()).
#     """
#     data = np.asarray(data, dtype=np.float64)
#     M = data.shape[0]
#     y_tilde = data.mean(axis=0)  # (n_times, n_voxels)
#     var_y_tilde = y_tilde.var(axis=0, ddof=ddof)
#     per_repeat_var = data.var(axis=1, ddof=ddof).mean(axis=0)
#     residual_mean_var = (data - y_tilde[None]).var(axis=1, ddof=ddof).mean(axis=0)
#     signal_var = per_repeat_var - (M / (M - 1)) * residual_mean_var
#     return np.sqrt(np.clip(signal_var / var_y_tilde, 0, None))
#
#
# def absolute_correlation(data, y_hat, ddof=1):
#     """r_abs = Corr_t(ytilde, yhat): correlation between the model prediction and the
#     trial-averaged measured response (Box 4). Bounded above by noise_ceiling_rmax(data), not
#     by 1, since ytilde is itself a noisy estimate of the true signal.
#
#     Parameters
#     ----------
#     data : array of shape (n_repeats, n_times, n_voxels)
#     y_hat : array of shape (n_times, n_voxels)
#         Model-predicted response for the same timepoints.
#
#     Returns
#     -------
#     r_abs : array of shape (n_voxels,)
#     """
#     data = np.asarray(data, dtype=np.float64)
#     y_tilde = data.mean(axis=0)  # (n_times, n_voxels)
#     yt_c = y_tilde - y_tilde.mean(axis=0)
#     yh_c = y_hat - y_hat.mean(axis=0)
#     cov = (yt_c * yh_c).sum(axis=0) / (data.shape[1] - ddof)
#     var_yt = y_tilde.var(axis=0, ddof=ddof)
#     var_yh = y_hat.var(axis=0, ddof=ddof)
#     return cov / np.sqrt(var_yt * var_yh)
#
#
# def normalized_correlation(data, y_hat, ddof=1):
#     """r_norm = r_abs / r_max ~= Corr_t(yhat, mu) (Box 4): an estimate of model performance
#     against the unobservable true experiment-dependent signal mu, correcting r_abs for the
#     fact that it's bounded above by r_max < 1 rather than by 1.
#
#     Parameters
#     ----------
#     data : array of shape (n_repeats, n_times, n_voxels)
#     y_hat : array of shape (n_times, n_voxels)
#
#     Returns
#     -------
#     r_norm : array of shape (n_voxels,)
#     """
#     r_abs = absolute_correlation(data, y_hat, ddof=ddof)
#     r_max = noise_ceiling_rmax(data, ddof=ddof)
#     return r_abs / r_max
#
#
# def noise_floor(y_true, y_pred, block_size, n_permutations=1000, alpha=0.05, seed=None):
#     """Permutation-based noise floor (block-permutation null, following the review's Section
#     6.3.1 "Estimating the noise floor"): the model PREDICTION is shuffled by permuting whole
#     blocks of temporally-contiguous samples (preserving within-block autocorrelation, unlike
#     shuffling individual timepoints), correlated against the real measured response, and this
#     is repeated many times to build a per-voxel null distribution of prediction accuracy
#     expected by chance. The floor is the (1-alpha)-percentile of that null distribution --
#     voxels whose actual model r exceeds their own floor are candidates for significance
#     (correct for multiple comparisons across voxels afterward, e.g. via bh_fdr_threshold on
#     permutation p-values from noise_floor_pvalues).
#
#     Choose block_size so that block_size * TR spans about 30-60 seconds (per the review), long
#     enough to preserve the measured signal's own temporal autocorrelation structure.
#
#     Parameters
#     ----------
#     y_true : array of shape (n_times, n_voxels)
#         Measured (held-out) brain activity.
#     y_pred : array of shape (n_times, n_voxels)
#         Model-predicted activity for the same timepoints.
#     block_size : int
#         Number of timepoints (TRs) per contiguous block, e.g. round(45 / TR) for ~45s blocks.
#     n_permutations : int
#     alpha : float
#         Floor is set at the 100*(1-alpha) percentile of the null distribution.
#     seed : int or None
#
#     Returns
#     -------
#     floor : array of shape (n_voxels,)
#         The (1-alpha)-percentile null correlation per voxel.
#     null_r : array of shape (n_permutations, n_voxels)
#         The full null distribution (kept around so noise_floor_pvalues can use it directly
#         instead of rebuilding it).
#     """
#     rng = np.random.default_rng(seed)
#     y_true = np.asarray(y_true, dtype=np.float64)
#     y_pred = np.asarray(y_pred, dtype=np.float64)
#     n_times, n_voxels = y_true.shape
#     n_blocks = int(np.ceil(n_times / block_size))
#     block_ids = np.repeat(np.arange(n_blocks), block_size)[:n_times]
#
#     yt_c = y_true - y_true.mean(axis=0)
#     yt_std = y_true.std(axis=0, ddof=1)
#
#     null_r = np.empty((n_permutations, n_voxels), dtype=np.float64)
#     for p in range(n_permutations):
#         perm_block_order = rng.permutation(n_blocks)
#         perm_idx = np.concatenate([np.where(block_ids == b)[0] for b in perm_block_order])
#         yp_perm = y_pred[perm_idx]
#         yp_c = yp_perm - yp_perm.mean(axis=0)
#         yp_std = yp_perm.std(axis=0, ddof=1)
#         cov = (yt_c * yp_c).mean(axis=0)
#         null_r[p] = cov / (yt_std * yp_std)
#
#     floor = np.percentile(null_r, 100 * (1 - alpha), axis=0)
#     return floor, null_r
#
#
# def noise_floor_pvalues(observed_r, null_r):
#     """Permutation p-value per voxel from noise_floor()'s null distribution: the fraction of
#     null-distribution values at least as extreme as the observed r, with the standard +1
#     correction (so p is never exactly 0 regardless of n_permutations).
#
#     Parameters
#     ----------
#     observed_r : array of shape (n_voxels,)
#         The model's actual (non-permuted) prediction accuracy per voxel.
#     null_r : array of shape (n_permutations, n_voxels)
#         From noise_floor().
#
#     Returns
#     -------
#     pvalues : array of shape (n_voxels,)
#     """
#     n_permutations = null_r.shape[0]
#     exceed_count = (null_r >= observed_r[None, :]).sum(axis=0)
#     return (1 + exceed_count) / (1 + n_permutations)
#
#
# def bh_fdr_threshold(pvalues, alpha=0.05):
#     """Benjamini-Hochberg FDR correction (Benjamini & Hochberg, 1995), as referenced in the
#     review's Section 6.3.1 for correcting the per-voxel noise-floor significance tests across
#     all voxels.
#
#     Parameters
#     ----------
#     pvalues : array of shape (n_voxels,)
#     alpha : float
#         Target false discovery rate.
#
#     Returns
#     -------
#     significant : array of shape (n_voxels,), bool
#         Which voxels are significant at FDR level alpha.
#     threshold : float
#         The largest p-value declared significant (0.0 if none are).
#     """
#     pvalues = np.asarray(pvalues)
#     n = pvalues.size
#     order = np.argsort(pvalues)
#     ranked = pvalues[order]
#     critical_values = alpha * (np.arange(1, n + 1) / n)
#     below = ranked <= critical_values
#     if not below.any():
#         return np.zeros(n, dtype=bool), 0.0
#     threshold = ranked[np.max(np.where(below)[0])]
#     return pvalues <= threshold, threshold

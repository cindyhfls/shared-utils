from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score
from scipy.spatial.distance import squareform
from typing import Literal, Any
from sklearn.model_selection import KFold,StratifiedGroupKFold
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import os
import numpy as np
import neuroboros as nb
import pandas as pd
from shared_utils.isc_utils import FisherAvg
from scipy.linalg import eigh, svd, LinAlgError

def check_PSD(K, eps = 1e-8):
    eigvals = np.linalg.eigvalsh(K)  # eigvalsh assumes symmetric, faster
    is_psd = np.all(eigvals >= -eps)  # small tolerance for float errors
    if ~is_psd:
        print(eigvals[eigvals<0])
    return is_psd

def get_connectivity_PCs(triu, mask=None,eps=1e-7):
    """
    Adapt from https://github.com/feilong/IDM_pred/blob/main/package/IDM_pred/io/__init__.py
    This function outputs a 2-D NumPy array which is the input `X` of the prediction pipeline. The shape of the 2-D array is (n_subjects, n_PCs).
    Each row is a sample, i.e., the PCs of a subject's connectivity profile. Each column is a feature, i.e., a PC's score across all subjects.
    The function loads npy files which contain upper triangles of individual differences matrices (IDMs). See https://github.com/feilong/IDM_pred#data for details.

    Parameters
    triu : vectorized upper triangle of the IDM (1-D array of length n_subjects*(n_subjects-1)/2)
    mask : boolean array of length n_subjects, optional
    eps : float, default=1e-7
        The tolerance parameter for eigenvalue decomposition.

    Returns
    -------
    X : ndarray of shape (n_subjects, n_PCs)
    """
    ns = int((1 + np.sqrt(1 + 8*len(triu))) / 2)
    mat = np.zeros((ns, ns))
    triu_idx = np.triu_indices(ns, k=1)
    mat[triu_idx] = v
    np.fill_diagonal(mat, 1.0)
    w, v = eigh(mat, lower=False)
    if mask is not None:
        mat = mat[mask][:,mask]
    w, v = eigh(mat, lower=False)
    assert np.all(w > -eps)
    w[w < 0] = 0
    U, s = v[:, ::-1][:, :-1], np.sqrt(w[::-1][:-1])
    X = (U * s[np.newaxis])
    return X

def make_folds_family_optional(
    t: pd.DataFrame,
    n_splits: int = 5,
    yvar_col: str = "age",
    sex_col: str = "sex",
    site_col: str = "study_site",
    family_col: str = "family_id",
    n_yvar_bins: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    '''
    1. Keep families together — if participants share a family_id, they must all land in the same fold. Mixing related individuals across train/test would cause data leakage (a model trained on a sibling could "cheat" on the held-out sibling).
    2. Balance folds by demographics — each fold should have a roughly similar distribution of age, sex, and study site.
    '''
    df = t.copy()

    # groups: real family_id when available; otherwise unique per row
    fam = df[family_col]
    df["_group"] = np.where(
        fam.notna(),
        fam.astype(str),
        "singleton_" + df.index.astype(str)
    )

    # strat columns
    df["_sex"]  = df[sex_col].astype("object").fillna("UNK").astype(str)
    df["_site"] = df[site_col].astype("object").fillna("UNK").astype(str)

    # exclude rows with NaN in yvar_col from fold assignment
    yvar_notna = df[yvar_col].notna()
    df_valid = df.loc[yvar_notna].copy()

    # yvar bins: use quantile binning if n_yvar_bins is set, otherwise use raw values
    if n_yvar_bins is None:
        df_valid["_yvarbin"] = df_valid[yvar_col].astype(str)
    else:
        df_valid["_yvarbin"] = pd.qcut(
            df_valid[yvar_col], q=n_yvar_bins, duplicates="drop"
        ).astype(str)

    df_valid["_strata"] = df_valid["_yvarbin"] + "|" + df_valid["_sex"] + "|" + df_valid["_site"]

    sgkf = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )

    folds_valid = np.full(len(df_valid), -1, dtype=int)
    X_dummy = np.zeros((len(df_valid), 1))
    y = df_valid["_strata"].to_numpy()
    groups = df_valid["_group"].to_numpy()

    for fold, (_, test_idx) in enumerate(sgkf.split(X_dummy, y, groups=groups)):
        folds_valid[test_idx] = fold

    out = t.copy()
    out["fold"] = np.nan
    out.loc[yvar_notna, "fold"] = folds_valid.astype(float)
    return out

def _nan_report(M, name):
    n = np.isnan(M).sum()
    if n == 0:
        return None
    return {
        f"{name}_nan_count": int(n),
        f"{name}_nan_rows": np.where(np.isnan(M).any(axis=1))[0].astype(int).tolist()[:20],
        f"{name}_nan_cols": np.where(np.isnan(M).any(axis=0))[0].astype(int).tolist()[:20],
    }

# NOTE: _normalize_kernel is a no-op when called after np.fill_diagonal(S, 1.0),
# because all diagonal entries are 1 → denom = sqrt(1*1) = 1 for every entry.
# The normalize parameter in fit_similarity_model is therefore kept but disabled.
# def _normalize_kernel(K: np.ndarray, eps: float = 1e-12) -> np.ndarray:
#     """
#     Normalize a square kernel so that K_ii becomes 1 (when diag > 0), via:
#         K_ij <- K_ij / sqrt(K_ii * K_jj)
#
#     NOTE: This function requires K to be square because it uses diag(K).
#     """
#     K = np.asarray(K, dtype=np.float64)
#     if K.ndim != 2 or K.shape[0] != K.shape[1]:
#         raise ValueError(f"_normalize_kernel expects a square matrix. Got {K.shape}.")
#     d = np.diag(K).copy()
#     d = np.maximum(d, eps)
#     denom = np.sqrt(np.outer(d, d))
#     return K / denom

def _corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = y_true - np.mean(y_true)
    yp = y_pred - np.mean(y_pred)
    denom = (np.linalg.norm(yt) * np.linalg.norm(yp))
    return float((yt @ yp) / denom) if denom > 0 else 0.0

# Scoring
def fold_score(y_true, y_pred,scoring):
    if scoring == "r2":
        return float(r2_score(y_true, y_pred))
    elif scoring == "corr":
        return _corr(y_true, y_pred)
    elif scoring == "mse":
        return float(np.mean((y_true-y_pred)**2))
    else:
        raise ValueError("scoring must be 'corr','mse', or 'r2'.")

def _ridge_svd(X_train, X_test, y_train, alpha, npc, fit_intercept=True):
    """
    SVD-based ridge regression in PC space.
    Adapted from ABCD_hyperalignment_JNeurosci/ridge.py.

    X_train : (n_train, n_features)
    X_test  : (n_test,  n_features)
    npc     : number of leading PCs to retain (None = all)
    """
    if fit_intercept:
        X_offset = np.mean(X_train, axis=0, keepdims=True)
        y_offset = np.mean(y_train)
        X_train  = X_train - X_offset
        y_train  = y_train - y_offset
        X_test   = X_test  - X_offset
    try:
        U, s, Vt = svd(X_train, full_matrices=False)
    except LinAlgError:
        U, s, Vt = svd(X_train, full_matrices=False, lapack_driver='gesvd')
    X_test_V = X_test @ Vt.T[:, :npc]
    UT_y     = U.T[:npc] @ y_train
    d        = s[:npc] / (s[:npc]**2 + alpha)
    yhat     = X_test_V @ (d * UT_y)
    if fit_intercept:
        yhat += y_offset
    return yhat


def fit_pc_ridge(S_train, S_test, y_train, fold_labels_train,
                 scoring='r2', alphas=None, npc_fracs=None, fit_intercept=True):
    """
    RETIRED: this is ridge on the similarity rows themselves, not IDM-style
    PC ridge.  It is not equivalent to precomputed KRR or to predict_g.py's
    eigendecomposition-based PC features.  Keep only for historical code;
    use fit_similarity_model(..., method='krr') for kernel ridge instead.

    Inner-CV grid search over (alpha, npc) then predict on S_test.

    S_train          : (n_train, n_train)  — similarity matrix used as feature matrix
    S_test           : (n_test,  n_train)
    fold_labels_train: 1-D int array of inner CV fold assignments for training subjects
    npc_fracs        : fractions of n_train to use as npc candidates (default auto)

    Returns y_hat (n_test,), info dict.
    """
    n_train = S_train.shape[0]
    if alphas is None:
        alphas = np.logspace(-9, 6, 16)
    if npc_fracs is None:
        npc_fracs = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
    npcs = np.unique(np.maximum(1, np.round(np.array(npc_fracs) * n_train).astype(int)))

    unique_folds = np.unique(fold_labels_train)
    score_sums   = np.zeros((len(alphas), len(npcs)))

    for val_fold in unique_folds:
        tr  = fold_labels_train != val_fold
        val = fold_labels_train == val_fold
        Xtr  = S_train[np.ix_(tr,  tr)]
        Xval = S_train[np.ix_(val, tr)]
        ytr  = y_train[tr]
        yval = y_train[val]

        if fit_intercept:
            X_off = np.mean(Xtr, axis=0, keepdims=True)
            y_off = np.mean(ytr)
            Xtr_c  = Xtr  - X_off
            Xval_c = Xval - X_off
            ytr_c  = ytr  - y_off
        else:
            Xtr_c, Xval_c, ytr_c, y_off = Xtr, Xval, ytr, 0.0

        try:
            U, s, Vt = svd(Xtr_c, full_matrices=False)
        except LinAlgError:
            U, s, Vt = svd(Xtr_c, full_matrices=False, lapack_driver='gesvd')

        UT_y  = U.T @ ytr_c          # (k,)
        Xval_V = Xval_c @ Vt.T       # (n_val, k)

        for ia, alpha in enumerate(alphas):
            d      = s / (s**2 + alpha)     # (k,) — standard ridge weights
            d_UT_y = d * UT_y               # (k,)
            for in_, npc in enumerate(npcs):
                yhat  = Xval_V[:, :npc] @ d_UT_y[:npc] + y_off
                score_sums[ia, in_] += fold_score(yval, yhat, scoring)

    score_grid = score_sums / len(unique_folds)
    best_ia, best_in = np.unravel_index(np.argmax(score_grid), score_grid.shape)
    best_alpha = float(alphas[best_ia])
    best_npc   = int(npcs[best_in])

    y_hat = _ridge_svd(S_train, S_test, y_train, best_alpha, best_npc, fit_intercept)
    info  = {"best_params": {"alpha": best_alpha, "npc": best_npc,
                              "npc_frac": best_npc / n_train}}
    return y_hat, info


def make_family_groups(family_ids):
    """Return local index arrays for leave-one-family-out CV.

    Missing family IDs are treated as singleton families.
    """
    family_ids = np.asarray(family_ids, dtype=object)
    groups = []
    seen = set()
    for i, family_id in enumerate(family_ids):
        if pd.isna(family_id):
            groups.append(np.array([i], dtype=int))
        elif family_id not in seen:
            seen.add(family_id)
            groups.append(np.where(family_ids == family_id)[0].astype(int))
    return groups


def fit_krr_family_hat(S_train, S_test, y_train, family_groups,
                       scoring='r2', alphas=None):
    """Tune precomputed KRR by block leave-one-family-out hat-matrix CV.

    This is exact for the fixed-kernel, no-intercept KRR model used by
    ``KernelRidge(kernel='precomputed')``.  It is intended for hyperparameter
    selection inside an outer training set; ``S_test`` is never used during
    tuning.  The returned prediction is a refit on all of ``S_train``.

    Similarity matrices such as pairwise ISC are not guaranteed to be positive
    semidefinite.  Instead of clipping negative eigenvalues (which changes the
    training kernel but not the raw ``S_test`` rows), the training kernel is
    shifted on its diagonal: K' = S_train + c I with c = max(0, -lambda_min).
    K' is PSD, every leave-family-out sub-kernel of it is PSD (eigenvalue
    interlacing), and the shift leaves all between-subject entries unchanged.
    ``alphas`` is the ridge penalty on K'; the effective penalty on S_train is
    c + alpha.  When S_train is PSD, c = 0 and nothing changes.

    WARNING: the shift does NOT make alpha selection safe for non-PSD
    similarities.  The hat-matrix LOFO shortcut only sees S_train, whose rows
    are consistent with K', but real test rows need not be: with scrubbed
    pairwise ISC, train+test lambda_min was ~10x more negative than train-only
    (HBN/scratch_analysis/2026-10-05_krr_alpha_selection_check/).  Test rows then
    load on K''s near-null directions, which tiny alphas amplify by ~1/alpha,
    and the shortcut cannot detect it.  For such similarities choose alpha by
    refit-based CV that scores held-out subjects from raw rows (e.g.
    fit_similarity_model with fold labels).
    """
    S_train = np.asarray(S_train, dtype=np.float64)
    S_test = np.asarray(S_test, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64).ravel()
    if alphas is None:
        alphas = np.logspace(-9, 3, 13)
    if S_train.ndim != 2 or S_train.shape[0] != S_train.shape[1]:
        raise ValueError("S_train must be square for precomputed KRR.")
    if S_test.ndim != 2 or S_test.shape[1] != S_train.shape[0]:
        raise ValueError("S_test must have one column per training subject.")
    if len(y_train) != S_train.shape[0]:
        raise ValueError("y_train must match S_train.")
    if not family_groups:
        raise ValueError("family_groups must contain at least one group.")

    # KRR uses (K + alpha I)^-1.  The eigendecomposition makes the full
    # smoother H_alpha = K(K + alpha I)^-1 cheap to evaluate for each alpha.
    S_train = 0.5 * (S_train + S_train.T)
    eigenvalues, eigenvectors = eigh(S_train, lower=False)
    min_eigenvalue = float(eigenvalues[0])  # eigh returns ascending eigenvalues
    diagonal_shift = max(0.0, -min_eigenvalue)
    eigenvalues = eigenvalues + diagonal_shift  # eigenvalues of S_train + c I (all >= 0)
    qTy = eigenvectors.T @ y_train

    best_score = -np.inf
    best_alpha = None
    for alpha in np.asarray(alphas, dtype=float):
        h_eigenvalues = eigenvalues / (eigenvalues + alpha)
        y_fit = eigenvectors @ (h_eigenvalues * qTy)
        residual = y_train - y_fit
        y_hat_family = np.full_like(y_train, np.nan)

        for family in family_groups:
            family = np.asarray(family, dtype=int)
            Q_family = eigenvectors[family, :]
            H_ff = (Q_family * h_eigenvalues) @ Q_family.T
            correction = np.linalg.solve(
                np.eye(len(family)) - H_ff,
                residual[family],
            )
            y_hat_family[family] = y_train[family] - correction

        score = fold_score(y_train, y_hat_family, scoring)
        if score > best_score:
            best_score = score
            best_alpha = float(alpha)

    # Refit on all outer-training subjects and predict the untouched outer test.
    coefficients = eigenvectors @ (qTy / (eigenvalues + best_alpha))
    y_hat = S_test @ coefficients
    info = {
        "best_params": {"alpha": best_alpha},
        "inner_loofamily_score": float(best_score),
        "min_eigenvalue": min_eigenvalue,
        "diagonal_shift": diagonal_shift,
    }
    return y_hat, info


def pc_ridge_loocv_outer(S_full, y_full,
                         scoring='r2', alphas=None, npc_fracs=None, fit_intercept=True):
    """
    Hat-matrix LOOCV used as the **outer** prediction loop (IDM_pred style).

    Every subject is left out once; their prediction ŷ_loocv[i] uses only the
    other N-1 subjects.  Hyperparameters (alpha, npc) are selected globally by
    maximising the LOOCV score over all N subjects.

    Returns the LOOCV predictions directly (not a refit on the full data),
    so R²(y_full, y_loocv) is directly comparable to a 5-fold outer-CV R².

    O(N³) — one SVD, then O(N²) per (alpha, npc) combo.
    """
    n = S_full.shape[0]
    if alphas is None:
        alphas = np.logspace(-9, 6, 16)
    if npc_fracs is None:
        npc_fracs = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
    npcs = np.unique(np.maximum(1, np.round(np.array(npc_fracs) * n).astype(int)))

    if fit_intercept:
        X_off = np.mean(S_full, axis=0, keepdims=True)
        y_off = float(np.mean(y_full))
        S_c   = S_full - X_off
        y_c   = y_full - y_off
    else:
        S_c, y_c, y_off = S_full, y_full, 0.0

    try:
        U, s, _ = svd(S_c, full_matrices=False)
    except LinAlgError:
        U, s, _ = svd(S_c, full_matrices=False, lapack_driver='gesvd')

    UT_y = U.T @ y_c
    U2   = U ** 2

    best_score   = -np.inf
    best_alpha   = float(alphas[0])
    best_npc     = int(npcs[-1])
    best_y_loocv = np.full(n, np.nan)

    for alpha in alphas:
        # omd = α/(s²+α).  Both e and denom scale as alpha, so their ratio is
        # well-conditioned for all alpha > 0 — no threshold or fallback needed.
        omd = alpha / (s**2 + alpha)
        for npc in npcs:
            e     = U[:, :npc] @ (omd[:npc] * UT_y[:npc])   # (I−H)y_c, stable
            denom = U2[:, :npc] @ omd[:npc]                   # 1−H_diag, always > 0 for alpha > 0
            # alpha cancels in e/denom → ratio is independent of alpha magnitude
            y_loocv = y_full - e / np.maximum(denom, np.finfo(float).tiny)

            sc = fold_score(y_full, y_loocv, scoring)
            if sc > best_score:
                best_score   = sc
                best_alpha   = float(alpha)
                best_npc     = int(npc)
                best_y_loocv = y_loocv.copy()

    info = {"best_params": {"alpha": best_alpha, "npc": best_npc,
                            "npc_frac": best_npc / n},
            "loocv_score": best_score}
    return best_y_loocv, info


def fit_pc_ridge_loocv_hat(S_train, S_test, y_train,
                           scoring='r2', alphas=None, npc_fracs=None, fit_intercept=True):
    """
    pc_ridge with hyperparameter selection via analytical LOOCV (Hat-matrix shortcut).

    Uses the identity ŷ_loocv[i] = y[i] - e[i] / (1 - H_ii), where
      e[i]  = y[i] - ŷ[i]  (full-data residual, centered)
      H_ii  = Σ_{k=1}^{npc} U[i,k]² · s_k² / (s_k² + α)  (hat-matrix diagonal)

    One SVD of S_train is reused during tuning; the final _ridge_svd call
    performs another decomposition. Grid cost is O(N²) per (alpha, npc).

    CAUTION: this implementation has not been validated against explicit
    leave-one-out refits of the intended model. Known concerns to check:
    - With truncated components, e below includes only shrinkage residuals
      within the retained subspace, omitting target variation outside it.
      Likewise, the retained-component denominator is not generally 1-H_ii.
      Intercept leverage must also be accounted for when fit_intercept=True;
      the H_ii expression above describes only the component contribution.
    - The component basis is estimated once from all supplied training
      subjects. Truncating this fixed basis is not equivalent to recomputing
      the basis after excluding each inner-validation subject. An exact
      fixed-kernel ridge LOOCV shortcut is not inherently leaky, but that
      equivalence has not been established for this implementation.
    - If y_train was first adjusted for age/sex/motion using all supplied
      training subjects, inner-validation targets contributed to that fit.
      This helper does not refit that questionnaire-score adjustment within
      inner folds (this is unrelated to fMRI time-series denoising).
    These issues can change tuning scores and selected hyperparameters; the
    direction of any performance change is not established. They do not by
    themselves expose outer-test targets, provided all fitting stays within
    outer training. Validate predictions against explicit refits, including
    truncation, intercept, and covariate handling, before relying on this
    helper. It implements individual LOO, not leave-one-family-out.
    """
    n_train = S_train.shape[0]
    if alphas is None:
        alphas = np.logspace(-9, 6, 16)
    if npc_fracs is None:
        npc_fracs = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
    npcs = np.unique(np.maximum(1, np.round(np.array(npc_fracs) * n_train).astype(int)))

    if fit_intercept:
        X_off = np.mean(S_train, axis=0, keepdims=True)
        y_off = float(np.mean(y_train))
        S_c   = S_train - X_off
        y_c   = y_train - y_off
    else:
        S_c, y_c, y_off = S_train, y_train, 0.0

    try:
        U, s, Vt = svd(S_c, full_matrices=False)
    except LinAlgError:
        U, s, Vt = svd(S_c, full_matrices=False, lapack_driver='gesvd')

    UT_y = U.T @ y_c   # (N,)
    U2   = U ** 2       # (N, N) — precomputed for H_diag sweep

    score_grid = np.full((len(alphas), len(npcs)), -np.inf)

    for ia, alpha in enumerate(alphas):
        omd = alpha / (s**2 + alpha)  # α/(s²+α), stable; ratio e/denom independent of alpha

        for in_, npc in enumerate(npcs):
            e     = U[:, :npc] @ (omd[:npc] * UT_y[:npc])
            denom = U2[:, :npc] @ omd[:npc]
            y_loocv = y_train - e / np.maximum(denom, np.finfo(float).tiny)

            score_grid[ia, in_] = fold_score(y_train, y_loocv, scoring)

    best_ia, best_in = np.unravel_index(np.argmax(score_grid), score_grid.shape)
    best_alpha = float(alphas[best_ia])
    best_npc   = int(npcs[best_in])

    y_hat = _ridge_svd(S_train, S_test, y_train, best_alpha, best_npc, fit_intercept)
    info  = {"best_params": {"alpha": best_alpha, "npc": best_npc, "npc_frac": best_npc / n_train}}
    return y_hat, info


def pc_ridge_loocv_precomp_outer(S_full, y_full,
                                  scoring='r2', alphas=None, npc_fracs=None, fit_intercept=True):
    """
    IDM_pred-style outer LOOCV using the eigh precomputation trick.

    Precomputes X = V·sqrt(λ) once from eigh(S_full).  Per-fold: leave subject i
    out, fit ridge on X[mask] via SVD, predict X[i].  Accumulates y_loocv_all
    over all i, selects best (alpha, npc) globally, returns LOOCV predictions.

    Directly comparable to pc_ridge_loocv_outer (hat-matrix shortcut).
    O(N⁴) vs O(N³) for the hat-matrix version.
    """
    n = S_full.shape[0]
    if alphas is None:
        alphas = np.logspace(-9, 6, 16)
    if npc_fracs is None:
        npc_fracs = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
    npcs = np.unique(np.maximum(1, np.round(np.array(npc_fracs) * n).astype(int)))

    w, v = eigh(S_full, lower=False)
    w    = np.maximum(w, 0.0)
    w    = w[::-1];  v = v[:, ::-1]
    X_full = v[:, :-1] * np.sqrt(w[:-1])   # (N, N-1)

    y_loocv_all = np.zeros((len(alphas), len(npcs), n))

    for i in range(n):
        mask   = np.ones(n, bool);  mask[i] = False
        X_tr   = X_full[mask]       # (N-1, N-1)
        X_val  = X_full[i:i+1]     # (1,   N-1)
        y_tr   = y_full[mask]

        if fit_intercept:
            X_off    = np.mean(X_tr, axis=0, keepdims=True)
            y_off_i  = float(np.mean(y_tr))
            X_tr_c   = X_tr  - X_off
            X_val_c  = X_val - X_off
            y_tr_c   = y_tr  - y_off_i
        else:
            X_tr_c, X_val_c, y_tr_c, y_off_i = X_tr, X_val, y_tr, 0.0

        try:
            U_f, s_f, Vt_f = svd(X_tr_c, full_matrices=False)
        except LinAlgError:
            U_f, s_f, Vt_f = svd(X_tr_c, full_matrices=False, lapack_driver='gesvd')

        UT_y_f = U_f.T @ y_tr_c
        Xval_V = X_val_c @ Vt_f.T   # (1, N-1)

        for ia, alpha in enumerate(alphas):
            d_f    = 1.0 / (s_f + alpha)
            d_UT_y = d_f * UT_y_f
            for in_, npc in enumerate(npcs):
                y_loocv_all[ia, in_, i] = float(Xval_V[0, :npc] @ d_UT_y[:npc]) + y_off_i

    score_grid = np.array([
        [fold_score(y_full, y_loocv_all[ia, in_], scoring)
         for in_ in range(len(npcs))]
        for ia in range(len(alphas))
    ])
    best_ia, best_in = np.unravel_index(np.argmax(score_grid), score_grid.shape)
    best_alpha = float(alphas[best_ia])
    best_npc   = int(npcs[best_in])

    info = {"best_params": {"alpha": best_alpha, "npc": best_npc, "npc_frac": best_npc / n},
            "loocv_score": float(score_grid[best_ia, best_in])}
    return y_loocv_all[best_ia, best_in], info


def fit_pc_ridge_loocv_precomp(S_train, S_test, y_train,
                                scoring='r2', alphas=None, npc_fracs=None, fit_intercept=True):
    """
    pc_ridge LOOCV using the get_connectivity_PCs precomputation trick (IDM_pred style).

    Precomputes X = V · sqrt(λ) once from eigh(S_train) so that per-fold indexing
    replaces per-fold eigendecomposition of the (N-1)×(N-1) submatrix.  A per-fold
    SVD of the (N-1)×(N-1) X_train slice is still required for ridge, so complexity
    remains O(N⁴) but with ~2× fewer big operations per fold than naive LOOCV.

    For large N_train the wall-clock time is dominated by N × SVD(N-1, N-1).
    Pass a smaller S_train (subsampled subjects) to estimate timing, then
    extrapolate with t_full ≈ t_sub × (N_full / N_sub)⁴.
    """
    n_train = S_train.shape[0]
    if alphas is None:
        alphas = np.logspace(-9, 6, 16)
    if npc_fracs is None:
        npc_fracs = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
    npcs = np.unique(np.maximum(1, np.round(np.array(npc_fracs) * n_train).astype(int)))

    # ── precompute X = V·sqrt(λ) from full eigendecomposition (IDM_pred trick) ──
    w, v = eigh(S_train, lower=False)  # ascending eigenvalues
    w    = np.maximum(w, 0.0)
    w    = w[::-1];  v = v[:, ::-1]   # descending
    X_full = v[:, :-1] * np.sqrt(w[:-1])  # (N, N-1); drops near-zero last eigenvalue

    # Accumulate all LOO predictions; score globally (per-sample R² is undefined)
    y_loocv_all = np.zeros((len(alphas), len(npcs), n_train))

    for i in range(n_train):
        mask   = np.ones(n_train, bool); mask[i] = False
        X_tr   = X_full[mask]      # (N-1, N-1) — no re-eigendecomposition
        X_val  = X_full[i:i+1]    # (1,   N-1)
        y_tr   = y_train[mask]

        if fit_intercept:
            X_off   = np.mean(X_tr, axis=0, keepdims=True)
            y_off   = float(np.mean(y_tr))
            X_tr_c  = X_tr  - X_off
            X_val_c = X_val - X_off
            y_tr_c  = y_tr  - y_off
        else:
            X_tr_c, X_val_c, y_tr_c, y_off = X_tr, X_val, y_tr, 0.0

        try:
            U_f, s_f, Vt_f = svd(X_tr_c, full_matrices=False)
        except LinAlgError:
            U_f, s_f, Vt_f = svd(X_tr_c, full_matrices=False, lapack_driver='gesvd')

        UT_y_f = U_f.T @ y_tr_c
        Xval_V = X_val_c @ Vt_f.T   # (1, N-1)

        for ia, alpha in enumerate(alphas):
            d_f    = 1.0 / (s_f + alpha)
            d_UT_y = d_f * UT_y_f
            for in_, npc in enumerate(npcs):
                y_loocv_all[ia, in_, i] = float(Xval_V[0, :npc] @ d_UT_y[:npc]) + y_off

    score_grid = np.array([
        [fold_score(y_train, y_loocv_all[ia, in_], scoring)
         for in_ in range(len(npcs))]
        for ia in range(len(alphas))
    ])
    best_ia, best_in = np.unravel_index(np.argmax(score_grid), score_grid.shape)
    best_alpha = float(alphas[best_ia])
    best_npc   = int(npcs[best_in])

    y_hat = _ridge_svd(S_train, S_test, y_train, best_alpha, best_npc, fit_intercept)
    info  = {"best_params": {"alpha": best_alpha, "npc": best_npc, "npc_frac": best_npc / n_train}}
    return y_hat, info


def fit_similarity_model(
    S: np.ndarray,
    y: np.ndarray,
    *,
    method: Literal["krr", "knn", "svr"] = "krr",
    precomputed: bool = True,
    kernel: str = "rbf",               # used only when precomputed=False
    gammas: np.ndarray | None = None,  # used only when precomputed=False and kernel='rbf'
    scoring: Literal["corr", "r2","mse"] = "r2",  # r2 (== MSE ranking); corr ignores prediction scale
    n_splits: int = 5,
    n_repeats: int = 1,
    random_state: int = 0,
    eps: float = 1e-5,
    # hyperparameter grids (optional; sensible defaults below)
    alphas: np.ndarray | None = None,          # KRR
    knn_ks: np.ndarray | None = None,          # KNN
    svr_Cs: np.ndarray | None = None,          # SVR
    svr_epsilons: np.ndarray | None = None,    # SVR, absolute outcome units; default scales with SD(y)
) -> tuple[Any, dict]:
    """
    Fit a regression model using either a precomputed N×N similarity matrix or raw features.

    precomputed=True (default):
      S is an N×N similarity matrix (must be PSD for krr/svr).
      - "krr": KernelRidge(alpha=..., kernel="precomputed")
      - "svr": SVR(C=..., epsilon=..., kernel="precomputed")
      - "knn": KNeighborsRegressor(n_neighbors=..., metric="precomputed") on D = 1 - K

    precomputed=False:
      S is an N×F raw feature matrix (or length-N vector for a single feature).
      Sklearn computes kernels/distances internally.
      - "krr": KernelRidge(alpha=..., kernel=kernel, gamma=gamma)
      - "svr": SVR(C=..., epsilon=..., kernel=kernel, gamma=gamma)
      - "knn": KNeighborsRegressor(n_neighbors=..., metric='euclidean')

    Returns:
      final_model: refit on full data with chosen hyperparams
      info: dict with chosen params and CV diagnostics
    """
    S = np.asarray(S, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()

    if precomputed:
        if S.ndim != 2 or S.shape[0] != S.shape[1]:
            raise ValueError(f"S must be square (N×N) when precomputed=True. Got shape {S.shape}.")
        if y.shape[0] != S.shape[0]:
            raise ValueError(f"y must have length N={S.shape[0]}. Got {y.shape[0]}.")
        S = S.copy()
        np.fill_diagonal(S, 1.0)
        K = S
        assert check_PSD(K, eps=eps),f"kernel matrix is not PSD at tolerence level {eps}. "
    else:
        # S is raw feature matrix
        if S.ndim == 1:
            S = S.reshape(-1, 1)
        if y.shape[0] != S.shape[0]:
            raise ValueError(f"y must have length N={S.shape[0]}. Got {y.shape[0]}.")
        K = S  # reuse K variable; sliced by row index in CV loop
        if kernel == "rbf" and gammas is None:
            from sklearn.metrics.pairwise import pairwise_distances
            dists = pairwise_distances(S, metric="euclidean")
            med = np.median(dists[dists > 0])
            gammas = np.array([0.25, 0.5, 1.0, 2.0, 4.0]) / (med ** 2 + 1e-12)

    # Defaults for hyperparameter grids
    if alphas is None:
        alphas = np.logspace(-9, 3, 13)
    if knn_ks is None:
            knn_ks = np.array([5, 10, 20, 50, 75, 100], dtype=int)
        # knn_ks = np.array([1, 3, 5, 10, 20, 50], dtype=int)
    if svr_Cs is None:
        svr_Cs = np.logspace(-3, 3, 13)
        # svr_Cs = np.logspace(-2, 3, 10)
    if svr_epsilons is None:
        # epsilon is in outcome units, so scale the default grid by SD of the training outcome:
        # the old absolute grid [1e-3 ... 1, 10, 100, 1000] was meaningless above ~SD (e.g. age in
        # years, SRS residual SD ~10) and wasted a third of the SVR fits. The fractions below are
        # a judgment call, not validated.
        svr_epsilons = np.array([0.0, 0.1, 0.25, 0.5, 1.0]) * float(np.std(y))

    rng = np.random.RandomState(random_state)

    # Store per-repeat best params/scores
    repeat_best_params = []
    repeat_best_scores = []

    # Optional: keep full CV scores for diagnostics
    cv_scores = []
    for rep in range(n_repeats):
        train_idx_list = []
        val_idx_list = []
        seed = int(rng.randint(0, 2**31 - 1))
        if not isinstance(n_splits, int):
            assert n_repeats==1, ("if using the outer CV folds, it does not make sense to re-run inner CV loop here")
            fold_labels = np.asarray(n_splits)
            assert fold_labels.shape[0] == K.shape[0], (
                f"fold label vector must have length {K.shape[0]} (same as K). "
                f"Got {fold_labels.shape[0]}."
            )

            unique_folds = np.unique(fold_labels)
            assert unique_folds.size >= 2, "Need at least 2 folds for CV."

            for val_fold in unique_folds:
                val_idx = np.where(fold_labels == val_fold)[0]
                if val_idx.size == 0:
                    continue
                train_idx = np.where(fold_labels != val_fold)[0]
                if train_idx.size == 0:
                    continue
                train_idx_list.append(train_idx)
                val_idx_list.append(val_idx)
        else:
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
            for train_idx, val_idx in kf.split(np.arange(K.shape[0])):
                train_idx_list.append(train_idx)
                val_idx_list.append(val_idx)

        # Build param grid
        if method == "krr":
            if not precomputed and kernel == "rbf":
                grid = [{"alpha": float(a), "gamma": float(g)} for a in alphas for g in gammas]
            else:
                grid = [{"alpha": float(a)} for a in alphas]
        elif method == "knn":
            grid = [{"n_neighbors": int(k)} for k in knn_ks]
        elif method == "svr":
            if not precomputed and kernel == "rbf":
                grid = [{"C": float(C), "epsilon": float(eps), "gamma": float(g)} for C in svr_Cs for eps in svr_epsilons for g in gammas]
            else:
                grid = [{"C": float(C), "epsilon": float(eps)} for C in svr_Cs for eps in svr_epsilons]
        else:
            raise ValueError("method must be one of: 'krr', 'knn', 'svr'.")

        # Accumulate scores per param
        param_to_scores: dict[tuple, list[float]] = {tuple(sorted(p.items())): [] for p in grid}

        for train_idx, val_idx in zip(train_idx_list,val_idx_list):#kf.split(np.arange(K.shape[0])):
            y_train, y_val = y[train_idx], y[val_idx]

            if precomputed:
                K_train = K[np.ix_(train_idx, train_idx)]
                K_val   = K[np.ix_(val_idx, train_idx)]  # (n_val, n_train) for precomputed kernels
            else:
                K_train = K[train_idx]   # raw feature rows
                K_val   = K[val_idx]

            for p in grid:
                key = tuple(sorted(p.items()))
                if method == "krr":
                    if precomputed:
                        model = KernelRidge(alpha=p["alpha"], kernel="precomputed")
                    else:
                        model = KernelRidge(alpha=p["alpha"], kernel=kernel, gamma=p.get("gamma"))
                    model.fit(K_train, y_train)
                    y_pred = model.predict(K_val)

                elif method == "svr":
                    if precomputed:
                        model = SVR(C=p["C"], epsilon=p["epsilon"], kernel="precomputed")
                    else:
                        model = SVR(C=p["C"], epsilon=p["epsilon"], kernel=kernel, gamma=p.get("gamma"))
                    model.fit(K_train, y_train)
                    y_pred = model.predict(K_val)

                else:  # knn
                    if precomputed:
                        D_train = 1.0 - K_train
                        D_val   = 1.0 - K_val
                        model = KNeighborsRegressor(
                            n_neighbors=p["n_neighbors"],
                            metric="precomputed",
                            weights="distance",
                        )
                        model.fit(D_train, y_train)
                        y_pred = model.predict(D_val)
                    else:
                        model = KNeighborsRegressor(
                            n_neighbors=p["n_neighbors"],
                            metric="euclidean",
                            weights="distance",
                        )
                        model.fit(K_train, y_train)
                        y_pred = model.predict(K_val)

                s = fold_score(y_val, y_pred, scoring)
                param_to_scores[key].append(s)

        # Pick best param this repeat
        if scoring =="corr":
            mean_scores = {k: float(FisherAvg(v)) for k, v in param_to_scores.items()}
        else:
            mean_scores = {k: float(np.mean(v)) for k, v in param_to_scores.items()}
        best_key = max(mean_scores, key=mean_scores.get)
        best_score = mean_scores[best_key]
        best_params = dict(best_key)

        repeat_best_params.append(best_params)
        repeat_best_scores.append(best_score)
        cv_scores.append(mean_scores)

    # Choose final params: pick the repeat with highest best_score (simple),
    # OR you could aggregate across repeats more formally.
    best_rep = int(np.argmax(repeat_best_scores))
    best_params_final = repeat_best_params[best_rep]

    # Fit final model on full data
    if method == "krr":
        if precomputed:
            final_model = KernelRidge(alpha=best_params_final["alpha"], kernel="precomputed")
        else:
            final_model = KernelRidge(alpha=best_params_final["alpha"], kernel=kernel, gamma=best_params_final.get("gamma"))
        final_model.fit(K, y)

    elif method == "svr":
        if precomputed:
            final_model = SVR(C=best_params_final["C"], epsilon=best_params_final["epsilon"], kernel="precomputed")
        else:
            final_model = SVR(C=best_params_final["C"], epsilon=best_params_final["epsilon"], kernel=kernel, gamma=best_params_final.get("gamma"))
        final_model.fit(K, y)

    elif method == "knn":
        if precomputed:
            D = 1.0 - K
            final_model = KNeighborsRegressor(
                n_neighbors=best_params_final["n_neighbors"],
                metric="precomputed",
                weights="distance",
            )
            final_model.fit(D, y)
        else:
            final_model = KNeighborsRegressor(
                n_neighbors=best_params_final["n_neighbors"],
                metric="euclidean",
                weights="distance",
            )
            final_model.fit(K, y)

    info = {
        "method": method,
        "precomputed": precomputed,
        "kernel": kernel if not precomputed else "precomputed",
        "scoring": scoring,
        "svr_epsilons": np.asarray(svr_epsilons, dtype=float),
        "n_splits": n_splits,
        "n_repeats": n_repeats,
        "best_params": best_params_final,
        "repeat_best_params": repeat_best_params,
        "repeat_best_scores": np.array(repeat_best_scores, dtype=float),
        # "mean_repeat_best_score": float(np.mean(repeat_best_scores)),
        # "std_repeat_best_score": float(np.std(repeat_best_scores)),
        "cv_mean_scores_per_repeat": cv_scores,  # dicts: param->mean score
    }
    return final_model, info

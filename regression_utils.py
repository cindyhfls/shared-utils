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

# PCA linear ridge regression (did I write this? maybe use Feilong's existing code from the INT repo)
def PCA_ridge(X,y,pca_n_comps=[50, 100, 150],ridge_alphas=np.logspace(-2, 4, 8)):
    pipe = Pipeline([
        ("pca", PCA()),
        ("ridge", Ridge())
    ])

    param_grid = {
        "pca__n_components": pca_n_comps,
        "ridge__alpha": ridge_alphas
    }

    gs = GridSearchCV(
        pipe,
        param_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1
    )

    gs.fit(X, y)
    # print("Best CV R2:", gs.best_score_)
    return gs

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

def _normalize_kernel(K: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Normalize a square kernel so that K_ii becomes 1 (when diag > 0), via:
        K_ij <- K_ij / sqrt(K_ii * K_jj)

    NOTE: This function requires K to be square because it uses diag(K).
    """
    K = np.asarray(K, dtype=np.float64)
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError(f"_normalize_kernel expects a square matrix. Got {K.shape}.")
    d = np.diag(K).copy()
    d = np.maximum(d, eps)
    denom = np.sqrt(np.outer(d, d))
    return K / denom

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

def fit_similarity_model(
    S: np.ndarray,
    y: np.ndarray,
    *,
    method: Literal["krr", "knn", "svr"] = "krr",
    precomputed: bool = True,
    kernel: str = "rbf",               # used only when precomputed=False
    gammas: np.ndarray | None = None,  # used only when precomputed=False and kernel='rbf'
    normalize: bool = True,
    scoring: Literal["corr", "r2","mse"] = "corr",
    n_splits: int = 5,
    n_repeats: int = 1,
    random_state: int = 0,
    # hyperparameter grids (optional; sensible defaults below)
    alphas: np.ndarray | None = None,          # KRR
    knn_ks: np.ndarray | None = None,          # KNN
    svr_Cs: np.ndarray | None = None,          # SVR
    svr_epsilons: np.ndarray | None = None,    # SVR
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
        K = _normalize_kernel(S) if normalize else S
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
        svr_epsilons = np.array([1e-3, 1e-2, 1e-1, 0.2, 0.5, 1.0, 10, 100, 1000], dtype=float)
        # svr_epsilons = np.array([1e-3, 1e-2, 1e-1, 0.2, 0.5, 1.0], dtype=float)

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

                s = fold_score(y_val, y_pred,scoring)
                param_to_scores[key].append(s)

        # Pick best param this repeat
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
        "normalize": normalize,
        "scoring": scoring,
        "n_splits": n_splits,
        "n_repeats": n_repeats,
        "best_params": best_params_final,
        "repeat_best_params": repeat_best_params,
        "repeat_best_scores": np.array(repeat_best_scores, dtype=float),
        "mean_repeat_best_score": float(np.mean(repeat_best_scores)),
        "std_repeat_best_score": float(np.std(repeat_best_scores)),
        "cv_mean_scores_per_repeat": cv_scores,  # dicts: param->mean score
    }
    return final_model, info

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import numpy as np

# PCA linear ridge regression
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

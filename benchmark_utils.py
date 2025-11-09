import os
import neuroboros as nb
import numpy as np

def calculate_isc(dms, pairwise = True, metric='correlation', out_fn=None, whichrois='cortical'):
    if out_fn is not None and os.path.exists(out_fn):
        return nb.load(out_fn)
    if isinstance(dms,dict):
        dms = np.stack(list(dms.values()), axis=0)
    elif isinstance(dms,list):
        dms = np.stack(dms, axis = 0)

    isc_each = nb.isc(dms, pairwise=pairwise, metric=metric)
    isc_mean = np.tanh(np.mean(np.arctanh(isc_each), axis=1))
    if out_fn is not None:
        nb.save(out_fn, isc_mean)
        if whichrois=="cortical":
            nb.save(out_fn.replace(".npy", "_eachvertex.npy"), isc_each)
        elif whichrois=="subcortical":
            nb.save(out_fn.replace(".npy", "_eachvoxel.npy"), isc_each)
    return isc_mean

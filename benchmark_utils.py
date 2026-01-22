import os
import neuroboros as nb
import numpy as np
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr,pearsonr
from collections import defaultdict
from joblib import Parallel, delayed

def normalize_to_unit(data):
    datamin = np.min(data)
    datamax = np.max(data)
    return (data-datamin)/(datamax-datamin)

def get_IS_RSA_models(data1,data2=None,metric='NN',normalize=True,diag_only=False):
    if data2 is None:
        data2=data1.copy()
        to_vector = True
    else:
        to_vector = False
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
    elif metric == 'product':
        sim_func = lambda x1,x2: x1*x2 # highly correlated with the annaK

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
    if normalize:
        return normalize_to_unit(similarity)
    else:
        return similarity
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

def generate_permutations(n, nperm, seed=0):
    np.random.seed(seed)
    perm = []
    for _ in range(nperm):
        perm.append(np.random.permutation(n))
    return perm

def permute_matrix(v,perm):
    M = squareform(v)
    M_perm = M[np.ix_(perm,perm)]
    return squareform(M_perm)

def calc_correlation(br_similarity,bh_similarity,method='spearman'):
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
    all_results = Parallel(n_jobs=-1, prefer="threads")(jobs)
    r_vals = np.array([r for r, p in all_results])
    return r_vals
    # To-do: calculate correlations with covariate controls

def IS_RSA_custom(neural_similarity,bx_similarity, perm, valid_idx=None):
    '''
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

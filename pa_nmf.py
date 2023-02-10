## NMF 

import numpy as np
import os
import pickle as pkl

from sklearn.decomposition import NMF

def obj(k_NMF=2, seeded=False, max_iters=200, verbose=False):
    if not seeded:
        return NMF(n_components=k_NMF, init="random", max_iter=max_iters, verbose=verbose)
    else:
        return NMF(n_components=k_NMF, init="custom", max_iter=max_iters, verbose=verbose)

def train(nmf_obj, data, E0=None, C0=None):

    k_NMF = nmf_obj.n_components

    if E0 is not None or C0 is not None:
        assert E0 is not None and C0 is not None, "Both E0 and C0 must be specified"
        W0 = np.copy(C0.T, order='C')
        H0 = np.copy(E0.T, order='C')
        if k_NMF > W0.shape[1]:
            W0 = np.concatenate((W0, np.ones((W0.shape[0], k_NMF - W0.shape[1]))), axis=1)
            H0 = np.concatenate((H0, np.ones((k_NMF - H0.shape[0], H0.shape[1]))), axis=0)
        return nmf_obj.fit(data.T, W=W0, H=H0)            
    else:
        return nmf_obj.fit(data.T)

def get_spectra(nmf_obj):
    return nmf_obj.components_.T

def unmix(nmf_obj, data):
    return nmf_obj.transform(data.T).T

def save_model(nmf_obj, fname):
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    with open(fname, 'wb') as fout:
        pkl.dump(nmf_obj, fname)

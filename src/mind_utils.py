# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 10:53:42 2018

@author: joram
"""

#Import standard packages
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_palette("colorblind")
import pandas as pd
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.decomposition import PCA
import argparse
# custom code
import sys
sys.path.append('/mnt/data/joram/')
import os
import csv
from utils import load_data, remove_outlier, split_data
from scipy.ndimage.filters import gaussian_filter1d
from scipy.interpolate import interp1d

def mapping(s, t, s_new, k,c):
    """ Map s_new to t_new based on known mapping of
        s (source) to t (target),
        with s original/intrinsic coordinates
        and t intrinsic/original coordinates """
    n, s_dim = s.shape
    t_dim = t.shape[1]
    n_new = s_new.shape[0]
    # 1. determine nearest neighbors
    dist = np.sum((s[np.newaxis] - s_new[:,np.newaxis])**2,-1)
    nn_ids = np.argsort(dist)[:,:k] # change to [:,:k]
    nns = np.row_stack([s[nn_ids[:,ki]] for ki in range(k)])
    nns = nns.reshape((n_new, k, s_dim), order='F')
    # 2 determine gram matris; 
    dif = s_new[:,np.newaxis] - nns
    G = np.tensordot(dif,dif,axes=([2],[2]))
    G = G[np.arange(n_new),:,np.arange(n_new)]
    # 3. determine weights not worth vectorizing this 
    weights = np.zeros((n_new, k))
    for i_n in range(n_new): 
        weights[i_n] = np.linalg.inv(G[i_n]+c*np.eye(k)).dot(np.ones((k,)))
    weights /= np.sum(weights, -1, keepdims=True)
    # 4. compute coordinates
    t_nns = np.row_stack([t[nn_ids[:,ki]] for ki in range(k)])
    t_nns = t_nns.reshape((n_new,k, t_dim), order='F')
    t_new = np.dot(weights, t_nns)
    t_new = t_new[np.arange(n_new), np.arange(n_new)]
    return t_new

def compute_activity(location, avg_activity, spatial_bins, noise_sigma=0):
    """ Activity of neurons for sequence of locations"""
    interpolator = interp1d(spatial_bins, avg_activity,axis=0,bounds_error=False, fill_value='extrapolate')
    mean_activity = interpolator(location)
    if noise_sigma > 0:
        activity = np.random.normal(loc=mean_activity, scale=noise_sigma)# np.random.normal(scale=noise_sigma, size=activity.shape)
    else:
        activity = mean_activity
    return activity
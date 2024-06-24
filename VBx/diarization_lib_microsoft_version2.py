#!/usr/bin/env python

# Copyright 2013-2019 Lukas Burget, Mireia Diez (burget@fit.vutbr.cz, mireia@fit.vutbr.cz)
# Licensed under the Apache License, Version 2.0 (the "License")

import numpy as np
import scipy.linalg as spl
import errno, os
from scipy.special import softmax
from scipy.spatial.distance import cosine


def twoGMMcalib_lin(s, niters=20):
    """
    Train two-Gaussian GMM with shared variance for calibration of scores 's'
    Returns threshold for original scores 's' that "separates" the two gaussians
    and array of linearly callibrated log odds ratio scores.
    """
    weights = np.array([0.5, 0.5])
    means = np.mean(s) + np.std(s) * np.array([-1, 1])
    var = np.var(s)
    threshold = np.inf
    for _ in range(niters):
        lls = np.log(weights)-0.5*np.log(var) - 0.5*(s[:,np.newaxis]-means)**2/var
        gammas = softmax(lls, axis=1)
        cnts = np.sum(gammas, axis=0)
        weights = cnts / cnts.sum()
        means = s.dot(gammas) / cnts
        var = ((s**2).dot(gammas) / cnts - means**2).dot(weights)
        threshold =  -0.5*(np.log(weights**2/var)-means**2/var).dot([1,-1])/(means/var).dot([1,-1])
    return threshold, lls[:,means.argmax()]-lls[:,means.argmin()]


def AHC(sim_mx, threshold=0):
    """ Performs UPGMA variant (wikipedia.org/wiki/UPGMA) of Agglomerative
    Hierarchical Clustering using the input pairwise similarity matrix.
    Input:
        sim_mx    - NxN pairwise similarity matrix
        threshold - threshold for stopping the clustering algorithm
                    (see function twoGMMcalib_lin for its estimation)
    Output:
        cluster labels stored in an array of length N containing (integers in
        the range from 0 to C-1, where C is the number of dicovered clusters)
    """
    dist = -sim_mx
    dist[np.diag_indices_from(dist)] = np.inf
    clsts = [[i] for i in range(len(dist))]
#    print('AHC') 
#    print(dist.argmin())
#    print(dist)
    while True:
        mi, mj = np.sort(np.unravel_index(dist.argmin(), dist.shape))
        print(mi,mj,dist[mi, mj])
        if dist[mi, mj] > -threshold:
            break
#        print(dist[mi,:], 'row')
#        print((dist[mi,:]*len(clsts[mi])+dist[mj,:]*len(clsts[mj]))/(len(clsts[mi])+len(clsts[mj])))
        dist[:, mi] = dist[mi,:] = (dist[mi,:]*len(clsts[mi])+dist[mj,:]*len(clsts[mj]))/(len(clsts[mi])+len(clsts[mj]))
        dist[:, mj] = dist[mj,:] = np.inf
        clsts[mi].extend(clsts[mj])
        clsts[mj] = None
    labs= np.empty(len(dist), dtype=int)
    for i, c in enumerate([e for e in clsts if e]):
        labs[c] = i
    return labs

def find_neighbor_clusters(cluster_list, idx, backward=True):
    if cluster_list[idx] is not None:
        return idx
    else:
        # search backwards until reach a cluster
        new_idx = idx
        while cluster_list[new_idx] is None:
            if backward:
                new_idx -= 1
                # no cluster left to merge with
                if new_idx < 0:
                    return None
            # search forward
            else:
                new_idx += 1
                if new_idx > len(cluster_list):
                    return None
        return new_idx


def AHC_neighbor(pair_similarity, threshold):
    dist = -pair_similarity
    print('inital_distances', dist)
    # initialize as pairs?
    clsts = [[i] for i in range(len(dist)+1)]
    while True:
        mi = np.argmin(dist)
        if dist[mi] > -threshold:
            break
        # update neighboring similarity
        print(dist)
        dist[mi] = (dist[mi]*len(clsts[mi]) + (dist[mi] + dist[mi+1])*len(clsts[mi+1]))/(len(clsts[mi]) + len(clsts[mi+1]))
        if mi != 0:
            dist[mi-1] = (dist[mi-1]*len(clsts[mi-1]) + (dist[mi-1] + dist[mi])*len(clsts[mi]))/(len(clsts[mi-1]) + len(clsts[mi]))
    
        dist[mi+1:mi+1+len(clsts[mi])] = np.inf    
        print(dist)
#        dist[mi] = np.inf
        clsts[mi].extend(clsts[mi+1])
        clsts[mi+1] = None
    labs= np.empty(len(dist)+1, dtype=int)
    for i, c in enumerate([e for e in clsts if e]):
        labs[c] = i
    print(clsts, labs)
    return labs

def PLDA_scoring_in_LDA_space(Fe, Ft, diagAC):
    """ Produces matrix of pairwise log likelihood ratio scores evaluated using
    PLDA model for enrollment (i-/x-)vectors Fe and test vectors Ft, which are
    mean normalized (or centered) and transformed by LDA-transformation
    (i.e. within-class covariance matrix is identity and across-class covariance
    matrix is diagonal).
    Input:
        Fe     - NxD matrix of enrollment vectors (in rows)
        Ft     - MxD matrix of test vectors (in rows)
        diagAC - D-dimensional diagonal of across class covariance matrix
    Output:
        out    - NxM matrix of log likelihood ratio scores
    """
    # See (7-8) in L. Burget et al.: "Discriminatively trained probabilistic
    # linear discriminant analysis for speaker verification", in ICASSP 2011.
    iTC      = 1.0 / (1 +   diagAC)
    iWC2AC   = 1.0 / (1 + 2*diagAC)
    ldTC    = np.sum(np.log(1 +   diagAC))
    ldWC2AC = np.sum(np.log(1 + 2*diagAC))
    Gamma = -0.25*(iWC2AC + 1 - 2*iTC)
    Lambda= -0.5 *(iWC2AC - 1)
    k = - 0.5*(ldWC2AC - 2*ldTC)
    return  np.dot(Fe * Lambda, Ft.T) + (Fe**2).dot(Gamma)[:,np.newaxis] + (Ft**2).dot(Gamma) + k


def kaldi_ivector_plda_scoring_dense(kaldi_plda, x, target_energy=0.1, pca_dim=None):
    """ Given input array of N x-vectors and pretrained PLDA model, this function
    calculates NxN matrix of pairwise similarity scores for the following AHC
    clustering. This function produces exactly the same similarity scores as the
    standard kaldi diarization recipe.
    Input:
        kaldi_plda    - PLDA model using the kaldi parametrization (mu, tr, psi)
                        as loaded by 'read_plda' function.
        x             - matrix of x-vectors (NxR)
        target_energy - Before calculating the similarity matrix, PCA is estimated
                        on the input x-vextors. The x-vectors (and PLDA model) are
                        then projected into low-dimensional space preservin at
                        least 'target_energy' variability in the x-vectors.
        pca_dim       - This parameter overwrites 'target_energy' and directly
                        specifies the PCA target dimensionality.
    Output:
        matrix of pairwise similarities between the input x-vectors
    """
    plda_mu, plda_tr, plda_psi = kaldi_plda
    [energy,PCA]=spl.eigh(np.cov(x.T, bias=True))
    if pca_dim is None:
      energy=np.cumsum(energy[::-1])
      pca_dim=np.sum(energy/energy[-1]<=target_energy) + 2
      # we need at least 2 dimensions, so 2 more dimensions are always added

    PCA=PCA[:,:-pca_dim-1:-1]
    print("pca_dim:", pca_dim)

    plda_tr_inv_pca=PCA.T.dot(np.linalg.inv(plda_tr))
    W = plda_tr_inv_pca.dot(plda_tr_inv_pca.T)
    B = (plda_tr_inv_pca*plda_psi).dot(plda_tr_inv_pca.T)
    acvar, wccn = spl.eigh(B,  W)
    x = np.dot(x-plda_mu,PCA).dot(wccn)
    x *= np.sqrt(x.shape[1] / np.dot(x**2, 1.0 / (acvar + 1.0)))[:,np.newaxis] # kaldi style length-norm
    #Lambda, Gamma, c, k = PLDA_params_to_bilinear_form(np.eye(pca_dim), np.diag(acvar), np.zeros((pca_dim,)))
    #return bilinear_scoring(Lambda, Gamma, c, k, x, x)
    return PLDA_scoring_in_LDA_space(x, x, acvar)


def read_xvector_timing_dict(kaldi_segments):
    """ Loads kaldi 'segments' file with the timing information for individual x-vectors.
    Each line of the 'segments' file is expected to contain the following fields:
    - x-vector name (which needs to much the names of x-vectors loaded from kaldi archive)
    - file name of the recording from which the xvector is extracted
    - start time
    - end time
    Input:
        kaldi_segments - path (including file name) to the Kaldi 'segments' file
    Outputs:
         segs_dict[recording_file_name] = (array_of_xvector_names, array_of_start_and_end_times)
    """
    segs = np.loadtxt(kaldi_segments, dtype=object)
    split_by_filename = np.nonzero(segs[1:,1]!=segs[:-1,1])[0]+1
    return {s[0,1]: (s[:,0], s[:,2:].astype(float)) for s in np.split(segs, split_by_filename)}


def merge_adjacent_labels(starts, ends, labels):
    """ Labeled segments defined as start and end times are compacted in such a way that
    adjacent or overlapping segments with the same label are merged. Overlapping
    segments with different labels are further adjusted not to overlap (the boundary
    is set in the middle of the original overlap).
    Input:
         starts - array of segment start times in seconds
         ends   - array of segment end times in seconds
         labels - array of segment labels (of any type)
    Outputs:
          starts, ends, labels - compacted and ajusted version of the input arrays
    """
    # Merge neighbouring (or overlaping) segments with the same label
    adjacent_or_overlap = np.logical_or(np.isclose(ends[:-1], starts[1:]), ends[:-1] > starts[1:])
    to_split = np.nonzero(np.logical_or(~adjacent_or_overlap, labels[1:] != labels[:-1]))[0]
    starts  = starts[np.r_[0, to_split+1]]
    ends    = ends[np.r_[to_split, -1]]
    labels  = labels[np.r_[0, to_split+1]]
  
    # Fix starts and ends times for overlapping segments
    overlaping = np.nonzero(starts[1:]<ends[:-1])[0]
    ends[overlaping] = starts[overlaping+1] = (ends[overlaping]+starts[overlaping+1]) / 2.0
    return starts, ends, labels


def segment_to_frame_labels(starts, ends, labels, length=0, frame_rate=100., empty_label=None):
    """ Obtain frame-by-frame labels from labeled segments defined by start and end times
    Input:
        starts - array of segment start times in seconds
        ends   - array of segment end times in seconds
        labels - array of segment labels (of any type)
        length:  Output array is truncted or augmented (with 'empty_label' values) to have this length.
                 For negative 'length', it will be only augmented if shorter than '-length'.
                 By default (length=0), the last element of 'ends' array determine the lenght of the output.
        frame_rate: frame rate of the output array (in frames per second)
    Outputs:
        frms  - array of frame-by-frame labels
    """
    min_len, max_len = (length, length) if length > 0 else (-length, None)
    starts = np.rint(frame_rate*starts).astype(int)
    ends   = np.rint(frame_rate*ends  ).astype(int)
    if not ends.size:
      return np.full(min_len, empty_label)

    frms = np.repeat(np.r_[np.c_[[empty_label]*len(labels),    labels     ].flat, empty_label],
                     np.r_[np.c_[starts - np.r_[0, ends[:-1]], ends-starts].flat, max(0, min_len-ends[-1])])
    return frms[:max_len]


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise


def l2_norm(vec_or_matrix):
    """ L2 normalization of vector array.

    Args:
        vec_or_matrix (np.array): one vector or array of vectors

    Returns:
        np.array: normalized vector or array of normalized vectors
    """
    if len(vec_or_matrix.shape) == 1:
        # linear vector
        return vec_or_matrix / np.linalg.norm(vec_or_matrix)
    elif len(vec_or_matrix.shape) == 2:
        return vec_or_matrix / np.linalg.norm(vec_or_matrix, axis=1, ord=2)[:, np.newaxis]
    else:
        raise ValueError('Wrong number of dimensions, 1 or 2 is supported, not %i.' % len(vec_or_matrix.shape))


def cos_similarity(x):
    """Compute cosine similarity matrix in CPU & memory sensitive way

    Args:
        x (np.ndarray): embeddings, 2D array, embeddings are in rows

    Returns:
        np.ndarray: cosine similarity matrix

    """
    assert x.ndim == 2, f'x has {x.ndim} dimensions, it must be matrix'
    x = x / (np.sqrt(np.sum(np.square(x), axis=1, keepdims=True)) + 1.0e-32)
    assert np.allclose(np.ones_like(x[:, 0]), np.sum(np.square(x), axis=1))
    max_n_elm = 200000000
    step = max(max_n_elm // (x.shape[0] * x.shape[0]), 1)
    retval = np.zeros(shape=(x.shape[0], x.shape[0]), dtype=np.float64)
    x0 = np.expand_dims(x, 0)
    x1 = np.expand_dims(x, 1)
    for i in range(0, x.shape[1], step):
        product = x0[:, :, i:i+step] * x1[:, :, i:i+step]
        retval += np.sum(product, axis=2, keepdims=False)
    assert np.all(retval >= -1.0001), retval
    assert np.all(retval <= 1.0001), retval
    return retval

# AHC optimziation functions
def get_group_clusters(xvector_group, threshold):
    """
    Given a smaller group of x-vectors (of size group_size) and a threshold

    Returns an array of clustered x-vectors defined by a new centroid, an array of x-vector cluster labels,
    and the number of clusters made in the group.
    """
    # similarity
    scr_mx_group = cos_similarity(xvector_group)
    # adjust threshold
    thr_group, junk_group = twoGMMcalib_lin(scr_mx_group.ravel())
    #cluster
    labels_group = AHC(scr_mx_group, thr_group + threshold)
    # get clusters
    group_cluster_number = np.max(labels_group) + 1
    # get new cluster centroid
    group_processed_xvectors= np.empty((group_cluster_number,xvector_group.shape[1]))
    for label in range(group_cluster_number):
        cluster_centroid = np.mean(xvector_group[labels_group==label,:], axis=0)
        group_processed_xvectors[label] = cluster_centroid
    return group_processed_xvectors, labels_group, group_cluster_number

def get_all_group_clusters(x, group_size_xvectors, total_xvectors, threshold):
    """
    Given an array of x-vectors, a size to group the x-vectors by (int), the total number of x-vectors, and a threshold value, cluster the x-vectors in groups.

    Returns an array of labels for all x-vectors and an array of clustered x-vectors
    """
    # initalize labels and x-vectors
    all_group_cluster_labels = np.empty((total_xvectors,),dtype=np.int8)
    all_xvectors_group_clustered = np.empty_like(x)
    index_offset = 0
    # clustser all x-vectors in groups
    for group_start in range(0,total_xvectors,group_size_xvectors):
        # slice x-vector group
        if group_start + group_size_xvectors > total_xvectors:
            group_end = total_xvectors
        else:
            group_end = group_start + group_size_xvectors
        group_xvectors, group_labels, group_cluster_total = get_group_clusters(x[group_start:group_end,:], threshold)
        # update labels
        all_group_cluster_labels[group_start:group_start+group_labels.size] = group_labels + index_offset
        # update the array of clustered x-vectors
        all_xvectors_group_clustered[index_offset:index_offset+group_cluster_total,:] = group_xvectors
        index_offset += group_cluster_total
    # get the clustered x-vectors
    all_xvectors_group_clustered = all_xvectors_group_clustered[0:index_offset,:]
    return all_group_cluster_labels, all_xvectors_group_clustered



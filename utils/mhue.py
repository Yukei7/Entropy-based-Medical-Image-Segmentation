#!/usr/bin/env python

import numpy as np
import math
from scipy.stats import gaussian_kde
from tqdm.notebook import tqdm
import cv2 as cv

def get_uncertainty(image, kde=False, off_center=False, info_w=False, stabs=None, bdts_o=None, bdts_b=None, kernel_size=13, sigma=1):

    img = image.copy()
    
    if info_w:
        fil = gauss_kernel(kernel_size, sigma)
        _,weights = get_weights(img,bdts_o,bdts_b,stabs,fil)
    
    inten_min = np.min(img)
    inten_max = np.max(img)
    n_pixels = np.size(img)
    t1 = inten_min + 2
    t2 = inten_max - 2
    uncertainty = []

    for i in range(t1, t2):
        # get index of pixels assigned to object/background
        index_o = np.where((img >= i) & (img != 0))
        index_b = np.where((img < i) & (img != 0))
        # calculate theta(prob of the pixels belong to object/background)
        nums = np.size(index_o[0])
        active_rate = nums / n_pixels
        p_o = np.zeros(img.shape)
        p_b = np.zeros(img.shape)
        value_o = img[index_o]
        value_b = img[index_b]
        if kde:
            if (value_o.size < 2) or (value_b.size < 2):
                continue
            X = [i for i in range(256)]
            kde_o = gaussian_kde(value_o)
            kde_b = gaussian_kde(value_b)
            o_prob = kde_o(X)
            b_prob = kde_b(X)
            for j in range(len(img)):
                p_o[j] = o_prob[img[j]]
                p_b[j] = b_prob[img[j]]

        else:
            # mean, variance of object pixels
            mean_o = np.mean(value_o)
            var_o = np.var(value_o)
            # mean, variance of background pixels
            mean_b = np.mean(value_b)
            var_b = np.var(value_b)
            # Gaussian dist
            p_o = np.exp(-np.power(img - mean_o, 2) / (2 * var_o)) / (math.sqrt(2 * math.pi * var_o))
            p_b = np.exp(-np.power(img - mean_b, 2) / (2 * var_b)) / (math.sqrt(2 * math.pi * var_b))
                
        # conditional prob obtained by Bayes formula
        p_to = active_rate * p_o
        p_tb = (1 - active_rate) * p_b
        p_t = p_to + p_tb
        mhue_i = p_to / p_t
        
        if off_center:
            # Off-centered Entropy
            obj_rate = np.array(np.where(img>=i)[0].size / img.size)
            mhue_i[np.where(mhue_i < obj_rate)] = mhue_i[np.where(mhue_i < obj_rate)] / (2*obj_rate)
            mhue_i[np.where(mhue_i >= obj_rate)] = (mhue_i[np.where(mhue_i >= obj_rate)]+1-2*obj_rate) / (2-2*obj_rate)

        u = (-mhue_i) * np.log(mhue_i) - (1-mhue_i) * np.log(1-mhue_i)
        if info_w:
            # Weighted Information Entropy
            # weight the impurity with context info(distance, stability)
            u = u.flatten() * weights[i-t1].flatten()
        uncertainty.append(u)

    for i in range(len(uncertainty)):
        uncertainty[i][np.isnan(uncertainty[i])] = 0

    return uncertainty


def min_max(lst):
    epsilon = 1e-9
    return (lst-np.min(lst)) / (np.max(lst)-np.min(lst) + epsilon)


def get_HU(uncertainty, homogeneity):
    HU = []
    for i in range(len(uncertainty)):
        u = uncertainty[i].flatten()
        h = homogeneity.flatten()
        HU_i = u*(1-h)+(1-u)*h
        HU.append(HU_i) 
    return HU


def get_weights(image,bdts_o,bdts_b,stabs,fil):
    weights = []
    denominator = 0
    corres_stabs = []
    for idx in range(len(stabs)):
        corres_stab = cv.filter2D(stabs[idx],-1,fil)
        corres_stabs.append(corres_stab)
        weight = (1-np.exp(-corres_stab)+0.1) / 1.1
        weights.append(weight)
    return corres_stabs,weights


def get_minomajo_ratio(image):
    ratio = np.array([])
    for i in range(np.min(image)+2,np.max(image)-2):
        ratio = np.append(ratio,np.where(image<=i)[0].size / np.where(image>i)[0].size)
    ratio[np.where(ratio>1)] = 1 / ratio[np.where(ratio>1)]
    return ratio


def gauss_kernel(kernel_size, sigma):
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size//2
    if sigma<=0:
        sigma = ((kernel_size-1)*0.5-1)*0.3+0.8
    s = sigma**2
    sum_val =  0
    for i in range(kernel_size):
        for j in range(kernel_size):
            x, y = i-center, j-center
            kernel[i, j] = np.exp(-(x**2+y**2)/2*s)
            sum_val += kernel[i, j]
    kernel = kernel/sum_val
    return kernel
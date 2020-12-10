import numpy as np
import pandas as pd
import talib
import numba
from numba import jit,njit
import cv2 as cv

# numba < 0.54

def get_stability(image, bdt, width=8):
    tmin = np.min(image) + 2
    tmax = np.max(image) - 2
    npix = image.size
    stab_lst = []

    for t in range(tmin, tmax):
        idx = np.where(bdt[t - tmin].reshape(image.shape) == 1)
        windows = np.zeros((width,npix))
        mid_width = width // 2
        for w in range(1, mid_width+1):
            if t - tmin + w >= len(bdt) or t - tmin + w < 0:
                continue
                
            res = np.zeros(image.shape)
            # current edge
            edge1 = np.zeros(image.shape, dtype=np.uint8)
            edge1[bdt[t - tmin].reshape(image.shape) == 1] = 255
            # next edge
            edge2 = np.zeros(image.shape, dtype=np.uint8)
            edge2[bdt[t - tmin + w].reshape(image.shape) == 1] = 255

            edge2not = cv.bitwise_not(edge2)
            dist = cv.distanceTransform(edge2not, distanceType=cv.DIST_L2, maskSize=3, dstType=cv.CV_32F)
            windows[mid_width+w-1,] = dist.flatten()

            w2 = -w
            # next edge
            edge2 = np.zeros(image.shape, dtype=np.uint8)
            edge2[bdt[t - tmin + w2].reshape(image.shape) == 1] = 255
            edge2not = cv.bitwise_not(edge2)
            dist = cv.distanceTransform(edge2not, distanceType=cv.DIST_L2, maskSize=3, dstType=cv.CV_32F)
            windows[w-1,] = dist.flatten()


        # Absolute sum of second order differences
        # The larger the difference, the more unstable.
        window_diff = np.sum(np.abs(np.diff(windows.T,1)),axis=1).reshape(image.shape)
        
        # unstab -> stab?
        # window_diff = np.exp(-window_diff)
        
        window_diff = 1-np.exp(-window_diff)
        
        # take only the edge
        stab = np.zeros(image.shape)
        stab[idx] = window_diff[idx]
        stab_lst.append(stab)
    return stab_lst


def min_max(lst):
    epsilon = 1e-9
    return (lst-np.min(lst)) / (np.max(lst)-np.min(lst) + epsilon)

    
def pix_scale(stab,bdts):
    output = []
    epsilon = 1e-9
    for i in range(len(stab)):
        tmp = np.sum(stab[i]) / (np.where(bdts[i]==1)[0].size+epsilon)
        output.append(tmp)
    return output


def get_emastab(stab,period=14):
    testema = talib.EMA(np.array(stab),period)
    return pd.DataFrame(testema).fillna(method='bfill').values
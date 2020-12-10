import numpy as np
import pandas as pd
import talib
import numba
from numba import jit,njit

# numba < 0.54

def get_stability(image, bdt, width=3, r_max=2, neighbours_min=3):
    tmin = np.min(image) + 2
    tmax = np.max(image) - 2
    npix = image.size
    stab_lst = []
    for t in range(tmin, tmax):
        # current edge
        idx = np.where(bdt[t - tmin].reshape(image.shape) == 1)
        tmp1, tmp2 = idx
        idx_lst = list(zip(tmp1, tmp2))
        windows = __windowing(image, bdt, t, width, r_max, neighbours_min, idx_lst)
        # Absolute sum of second order differences
        window_diff = np.sum(np.abs(np.diff(windows.T,1)),axis=1)
        # unstab -> stab?
        window_diff = (1-np.exp(-window_diff)).reshape(image.shape)
        
        stab = np.zeros(image.shape)
#         print(stab.shape, window_diff.shape, idx.shape)
        stab[idx] = window_diff[idx]
        stab_lst.append(stab)
    return stab_lst


@jit(nopython=True)
def __windowing(image, bdt, t, width, r_max, neighbours_min, idx):
    # assume width is odd
    mid_width = width // 2
    
    tmin = np.min(image) + 2
    tmax = np.max(image) - 2
    npix = image.size
    windows = np.zeros((width,npix))
    window_counter = 0
    for w in range(-mid_width,mid_width+1):
        if w == 0:
            continue
        dist = np.zeros(image.shape)
        # for each point in the current edge, find the neighbours in the next edge.
        for (x, y) in idx:
            if (x == 0) or (y == 0) or (x == image.shape[0] - 1) or (y == image.shape[1] - 1):
                continue
            r = 1
            cord = []
            while len(cord) < neighbours_min and r < r_max:
                for m in range(x - r, x + r + 1):
                    for n in range(y - r, y + r + 1):
                        if (m<=0) or (n<=0) or (m>=image.shape[0]-1) or (n>=image.shape[1]-1) or (t-tmin+w>=len(bdt)):
                            continue
                        if bdt[t - tmin + w][m, n] == 1 and [m, n] not in cord:
                            cord.append([m, n])
                r += 1
            if len(cord) < neighbours_min:
                # something large enough
                dist[x, y] = r_max
                continue
            # Calculate the mean value of the position
            cordt = np.array(cord)
            xo, yo = cordt[:, 0], cordt[:, 1]
            xc, yc = np.mean(xo), np.mean(yo)
            dist[x, y] = ((xc-x)**2 + (yc-y)**2)**0.5
        windows[window_counter,] = dist.flatten()
        window_counter += 1
    return windows


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
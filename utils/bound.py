import numpy as np
from tqdm.notebook import tqdm
import cv2 as cv
import numba
from numba import jit,njit
from newutils.mhue import gauss_kernel
import pandas as pd


def get_bound(image, bdts_o, bdts_b):
    edgeness = []
    for t in range(len(bdts_o)):
        # TODO: filter or not?
        fil = gauss_kernel(kernel_size=1, sigma=1)
        # near edge
        bdt = bdts_o[t] + bdts_b[t]
#         bdt = bdts_o[t]
        edge = np.zeros(shape=bdt.shape)
        # TODO: BDT setting
        edge[np.where(bdt==1)] = 1
        edge[np.where(image==t+np.min(image)+2)] = 1
        edge = cv.filter2D(edge,-1,fil)
        edge[np.where(edge>=0.1)] = 1
        edge[np.where(edge<0.1)] = 0
        edgeness.append(edge.flatten())
    
    # edgeness: (t, flatten_image_pixel)
    edgeness = np.array(edgeness)
    lower, upper = [0] * edgeness.shape[1], [0] * edgeness.shape[1]
    for i in range(edgeness.shape[1]):
        p = edgeness[:,i]
        if np.sum(p) == 0:
            continue
        lower[i] = np.min(np.where(p == 1))
        upper[i] = np.max(np.where(p == 1))
    lower, upper = np.array(lower), np.array(upper)
    return lower, upper


def get_scharr_bounding(image, scharr, bdts_o, bdts_b, percentile=80):
    lower, upper = get_bound(image, bdts_o, bdts_b)
    
#     # adaptive delta
#     tmp = np.array([upper-lower, image.flatten()])
#     lst = pd.DataFrame(tmp.T,columns=['delta', 'gray'])
#     tmp = pd.DataFrame(index=np.arange(0,256))
#     tmp['mean'] = lst.groupby('gray').agg(mean=pd.NamedAgg(column='delta', aggfunc='mean'))
#     delta_lst = np.round(tmp.interpolate()).to_numpy(dtype=np.int16)
#     # assure delta is odd
#     delta_lst = (delta_lst + ((delta_lst+1)%2))[:,0]
    
    # delta by percentile
    delta = np.percentile(upper-lower,percentile)
    delta = int(delta + (delta+1)%2)
    
    scharr_b = []
    for t in range(np.min(image)+2, np.max(image)-2):
        fil = (t > lower) & (t < upper)
        sch = scharr * fil.reshape(image.shape)
        # rescale?
        # sch = 1 - np.exp(-(sch**2/(np.var(sch)/gradImg.size)))
        scharr_b.append(sch)
        
    scharr_b_cum = []
    half_delta = delta // 2
    scharr_b_pad = [scharr_b[0] for _ in range(half_delta)]
    scharr_b_pad.extend(scharr_b)
    scharr_b_pad.extend([scharr_b[-1] for _ in range(half_delta)])

    for t in range(len(scharr_b)):
        tmp = np.sum(scharr_b[t:t+delta],axis=0)
        if np.max(tmp) > 0:
            tmp = tmp / np.max(tmp)
        scharr_b_cum.append(tmp)
    
    return scharr_b_cum, delta
import numpy as np
from tqdm.notebook import tqdm
import cv2 as cv
import numba
from numba import jit,njit
from vmdpy import VMD

def get_score(image, uncertainty, homogeneity, kernel_size=7, bounding=False):
    conv = []
    mid_kernel = int(kernel_size/2)
    for i in range(len(uncertainty)):
        u = uncertainty[i].reshape(image.shape)
        if bounding:
            h = homogeneity[i].reshape(image.shape)
        else:
            h = homogeneity.copy()
        u_pad = cv.copyMakeBorder(u, mid_kernel, mid_kernel, mid_kernel, mid_kernel, cv.BORDER_REPLICATE)
        scharr_pad = cv.copyMakeBorder(h, mid_kernel, mid_kernel, mid_kernel, mid_kernel, cv.BORDER_REPLICATE)
        u_rank_all = np.argsort(u_pad, axis=None).reshape(u_pad.shape) + 1
        
        convoluted = __conv(u,u_rank_all,u_pad,scharr_pad,mid_kernel)
        conv.append(convoluted)
    return conv

@jit(nopython=True)
def __conv(u,u_rank_all,u_pad,scharr_pad,mid_kernel):
    kernel_size = mid_kernel * 2 + 1
    convoluted = np.zeros(u.shape)
    for i in range(u.shape[0]):
        for j in range(u.shape[1]):
            conv_u = u_pad[i:i+2*mid_kernel+1,j:j+2*mid_kernel+1].flatten()
            conv_scharr = scharr_pad[i:i+2*mid_kernel+1,j:j+2*mid_kernel+1].flatten()
            # rank
            u_rank = conv_u.argsort()
            u_rank = u_rank - np.min(u_rank) + 1
            kernel_weight = 2*(u_rank/(kernel_size**2))-1
            convoluted[i,j] = np.sum(kernel_weight*conv_scharr)
    return convoluted


def vmd_extract(region_i, var_min=0.01, alpha=2000, tau=0.0, K=5, DC=0, init=1, tol=1e-7):
    r_sum = list(map(lambda x:np.sum(x),region_i))
    r_recon_all, _, _ = VMD(r_sum, alpha, tau, K, DC, init, tol)
    r_recon = list(r_recon_all[0,:])
    for mode in range(1,r_recon_all.shape[0]):
        if np.var(r_recon_all[mode,:])/np.sum(np.var(r_recon_all,axis=1)) > var_min:
            r_recon += r_recon_all[mode,:]
    return r_recon
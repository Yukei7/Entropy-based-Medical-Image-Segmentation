import numpy as np
from tqdm.notebook import tqdm
import cv2 as cv
import numba
from numba import jit,njit

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

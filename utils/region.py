import numpy as np
from tqdm.notebook import tqdm
import cv2 as cv
import numba
from numba import jit,njit
from vmdpy import VMD
from numpy.lib.stride_tricks import as_strided


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
        convoluted = __conv(u,u_pad,scharr_pad,mid_kernel)
        conv.append(convoluted)
    return conv

@jit(nopython=True)
def __conv(u,u_pad,scharr_pad,mid_kernel):
    kernel_size = mid_kernel * 2 + 1
    convoluted = np.zeros(u.shape)
    max_scharr = np.max(scharr_pad)
    
    # gaussian
    fil = np.zeros((kernel_size,kernel_size))
    center = kernel_size//2
    sum_val = 0
    s = 1
    for i in range(kernel_size):
        for j in range(kernel_size):
            x, y = i-center, j-center
            fil[i, j] = np.exp(-(x**2+y**2)/2*s)
            sum_val += fil[i, j]
    fil = fil/sum_val
    
    for i in range(u.shape[0]):
        for j in range(u.shape[1]):
            conv_u = u_pad[i:i+2*mid_kernel+1,j:j+2*mid_kernel+1].copy().flatten()
            conv_scharr = scharr_pad[i:i+2*mid_kernel+1,j:j+2*mid_kernel+1].copy().flatten()
#             # rank
#             u_rank = conv_u.argsort()
#             u_rank = u_rank - np.min(u_rank) + 1
#             # TODO: penalize?
#             kernel_weight = (2*(u_rank/(kernel_size**2))-1)

            # origin
            kernel_weight = conv_u.copy()
            lmbda = np.max(kernel_weight)
            # lmbda = np.median(kernel_weight)
            convoluted[i,j] = np.sum(fil.flatten()*kernel_weight*conv_scharr) - lmbda*np.sum(fil.flatten()*kernel_weight)
            # convoluted[i,j] -= lmbda*kernel_weight
            
#             kernel_weight = conv_u.copy()
#             tp = np.sum(kernel_weight*conv_scharr)
#             fp = np.sum(kernel_weight*(1-conv_scharr))
#             fn = np.sum((1-kernel_weight)*conv_scharr)
#             # when beta<0, precision is more important than recall
#             beta = 2
#             factor = (1+beta**2)/beta**2
#             convoluted[i,j] = factor * tp / (2*tp + fn + fp + 1e-8)
            
    return convoluted


def __softmax(x):
    exp_x = np.exp(x)
    return exp_x/np.sum(exp_x)



def gauss_kernel(kernel_size, sigma):
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size//2
    if sigma<=0:
        raise ValueError('sigma should be non-negative')
    s = sigma**2
    sum_val = 0
    for i in range(kernel_size):
        for j in range(kernel_size):
            x, y = i-center, j-center
            kernel[i, j] = np.exp(-(x**2+y**2)/2*s)
            sum_val += kernel[i, j]
    kernel = kernel/sum_val
    return kernel


def vmd_extract(region_i, var_min=0.01, alpha=2000, tau=0.0, K=5, DC=0, init=1, tol=1e-7):
    r_sum = list(map(lambda x:np.sum(x),region_i))
    r_recon_all, _, _ = VMD(r_sum, alpha, tau, K, DC, init, tol)
    r_recon = list(r_recon_all[0,:])
    for mode in range(1,r_recon_all.shape[0]):
        if np.var(r_recon_all[mode,:])/np.sum(np.var(r_recon_all,axis=1)) > var_min:
            r_recon += r_recon_all[mode,:]
    return r_recon
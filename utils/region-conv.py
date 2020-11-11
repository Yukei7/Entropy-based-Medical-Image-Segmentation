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
        # u_pad = cv.copyMakeBorder(u, mid_kernel, mid_kernel, mid_kernel, mid_kernel, cv.BORDER_REPLICATE)
        # scharr_pad = cv.copyMakeBorder(h, mid_kernel, mid_kernel, mid_kernel, mid_kernel, cv.BORDER_REPLICATE)
        # convoluted = __conv(u,u_pad,scharr_pad,mid_kernel)
        
        convoluted = np.multiply(u,h)
        # fil = gauss_kernel(3,1)
        fil = np.ones((3,3))
        convoluted = __conv2d(convoluted, fil, stride=1, padding='same')
        # unpad
        penalize = np.multiply(__pool2d(convoluted,kernel_size,pool_mode='median'),__pool2d(convoluted,kernel_size,pool_mode='sum'))
        convoluted = convoluted - penalize
        conv.append(convoluted)
    return conv

@jit(nopython=True)
def __conv(u,u_pad,scharr_pad,mid_kernel):
    kernel_size = mid_kernel * 2 + 1
    convoluted = np.zeros(u.shape)
    max_scharr = np.max(scharr_pad)
    
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
            lmbda = np.median(kernel_weight)
            convoluted[i,j] = np.sum(kernel_weight*conv_scharr) - lmbda*np.sum(kernel_weight)
            
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


def __pool2d(A, kernel_size, stride=1, padding='same', pool_mode='max'):
    '''
    2D Pooling

    Parameters:
        A: input 2D array
        kernel_size: int, the size of the window
        stride: int, the stride of the window
        padding: int, implicit zero paddings on both sides of the input
        pool_mode: string, 'max', 'avg' or 'sum'
    '''
    if padding == 'same':
        padding = int((kernel_size - 1) // 2)
    
    # Padding
    A = np.pad(A, padding, mode='edge')

    # Window view of A
    output_shape = ((A.shape[0] - kernel_size)//stride + 1, (A.shape[1] - kernel_size)//stride + 1)
    kernel_size = (kernel_size, kernel_size)
    A_w = as_strided(A, shape=output_shape+kernel_size, strides=(stride*A.strides[0],stride*A.strides[1])+A.strides)
    A_w = A_w.reshape(-1, *kernel_size)

    # Return the result of pooling
    if pool_mode == 'max':
        return A_w.max(axis=(1,2)).reshape(output_shape)
    elif pool_mode == 'avg':
        return A_w.mean(axis=(1,2)).reshape(output_shape)
    elif pool_mode == 'sum':
        return A_w.sum(axis=(1,2)).reshape(output_shape)
    elif pool_mode == 'median':
        return np.median(A_w,axis=(1,2)).reshape(output_shape)

    
def __conv2d(A, kernel, stride=1, padding='same'):
    '''
    2D Pooling

    Parameters:
        A: input 2D array
        stride: int, the stride of the window
    '''
    kernel_size = (kernel.shape[0], kernel.shape[1])
    if padding == 'same':
        padding = int((kernel_size[0] - 1) // 2)
    # Padding
    A = np.pad(A, padding, mode='edge')
    
    # Window view of A
    output_shape = ((A.shape[0] - kernel_size[0])//stride + 1, (A.shape[1] - kernel_size[1])//stride + 1)
    A_w = as_strided(A, shape=output_shape+kernel_size, strides=(stride*A.strides[0],stride*A.strides[1])+A.strides)
    A_w = A_w.reshape(-1, *kernel_size)

    f = lambda x,y:x*kernel
    return np.sum(np.apply_over_axes(f,A_w,(1,2)),axis=(1,2)).reshape(output_shape)
    

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
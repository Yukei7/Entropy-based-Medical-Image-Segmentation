import numpy as np
from tqdm.notebook import tqdm
import cv2 as cv


def get_score(image, uncertainty, homogeneity, kernel_size=7):
    conv = []
    mid_kernel = int(kernel_size/2)
    for i in tqdm(range(len(uncertainty))):
        u = uncertainty[i].reshape(image.shape)
        convoluted = np.zeros(image.shape)
        u_pad = cv.copyMakeBorder(u, mid_kernel, mid_kernel, mid_kernel, mid_kernel, cv.BORDER_REPLICATE)
        scharr_pad = cv.copyMakeBorder(homogeneity, mid_kernel, mid_kernel, mid_kernel, mid_kernel, cv.BORDER_REPLICATE)
        u_rank_all = np.argsort(u_pad, axis=None).reshape(u_pad.shape) + 1
        for m in range(u.shape[0]):
            for n in range(u.shape[1]):
                u_rank = u_rank_all[m:m+2*mid_kernel+1,n:n+2*mid_kernel+1].copy()
                u_rank = u_rank - np.min(u_rank) + 1
                conv_u = u_pad[m:m+2*mid_kernel+1,n:n+2*mid_kernel+1]
#                 u_rank = np.argsort(conv_u, axis=None).reshape(conv_u.shape) + 1
                std = np.std(conv_u)
                kernel_weight = 2*(np.random.normal(loc=0, scale=std**2, size=(kernel_size, kernel_size)) +
                                   u_rank/(kernel_size**2))-1
                convoluted[m, n] = np.sum(kernel_weight*scharr_pad[m:m+2*mid_kernel+1, n:n+2*mid_kernel+1])
        conv.append(convoluted)
    return conv

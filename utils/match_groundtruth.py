import cv2 as cv
import numpy as np


def get_error_rate(img1,img2):
    if img1.size != img2.size:
        raise ValueError('The size of image is different.')
    n_pix = img1.size
    match_pix = np.sum(img1!=img2)
    return match_pix/n_pix

def get_best_t(img_lst,img_idx,truth_img_lst):
    best_t = []
    print('='*50)
    print('Finding best threshold by error rate')
    print('='*50)
    for idx,img in enumerate(img_lst):
        truth = truth_img_lst[idx]
        inten_max = np.max(img)
        inten_min = np.min(img)
        err_lst = []
        for t in range(inten_min,inten_max):
            _,est_obj = cv.threshold(img,t,255,0)
            err = get_error_rate(est_obj,truth)
            err_lst.append(err)
        least_err = np.argmin(err_lst)
        best_t.append(least_err+inten_min)
        print('Index: {}, done. Best threshold at: {}'.format(img_idx[idx],least_err+inten_min))
    print('='*50)
    return best_t
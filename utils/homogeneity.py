#!/usr/bin/env python

import numpy as np
import cv2 as cv
import numba
from numba import jit,njit


@jit(nopython=True)
def __get_gradvalue(gradImg, gradcount):
    # grad: gradImg[x,y]
    max_gradvalue = np.sum(gradcount)
    gradvalue = np.zeros(gradImg.shape)
    for i in range(gradImg.shape[0]):
        for j in range(gradImg.shape[1]):
            gradvalue[i, j] = np.sum(gradcount[0:gradImg[i, j] + 1]) / max_gradvalue
    return gradvalue
         
    
def rank_normalize(gradImg):
    # Rank-normalization
    _, gradcount = np.unique(gradImg.flatten(), return_counts=True)
    gradvalue = __get_gradvalue(gradImg, gradcount)
    return gradvalue


def get_scharr(image):
    gray_img = image.copy()
#     gray_img = cv.fastNlMeansDenoising(gray_img,templateWindowSize=7,searchWindowSize=27)
    scharrx = cv.Scharr(gray_img, ddepth=cv.CV_16S, dx=1, dy=0, scale=1.0, delta=0.0, borderType=cv.BORDER_DEFAULT)
    scharry = cv.Scharr(gray_img, ddepth=cv.CV_16S, dx=0, dy=1, scale=1.0, delta=0.0, borderType=cv.BORDER_DEFAULT)
    gradImg = abs(scharrx) + abs(scharry)
    return gradImg


@jit(nopython=True)
def __get_gradvalue_mhue(gradImg, indices, gradcount):
    max_gradvalue = np.sum(gradcount)
    gradvalue = np.zeros(gradImg.shape)
    for i in range(gradImg.shape[0]):
        for j in range(gradImg.shape[1]):
            gradvalue[i, j] = np.sum(gradcount[0:np.argwhere(indices==gradImg[i, j])[0,0] + 1]) / max_gradvalue
    return gradvalue
            
    
def rank_normalize_mhue(gradImg):
    # Rank-normalization
    indices, gradcount = np.unique(gradImg.flatten(), return_counts=True)
    gradvalue = __get_gradvalue_mhue(gradImg, indices, gradcount)
    return gradvalue

def get_scharr_mhue(image):
    gray_img = image.copy()
#     gray_img = cv.fastNlMeansDenoising(gray_img,templateWindowSize=7,searchWindowSize=27)
    scharrx = cv.Scharr(gray_img, ddepth=cv.CV_16S, dx=1, dy=0, scale=1.0, delta=0.0, borderType=cv.BORDER_DEFAULT)
    scharry = cv.Scharr(gray_img, ddepth=cv.CV_16S, dx=0, dy=1, scale=1.0, delta=0.0, borderType=cv.BORDER_DEFAULT)
#     gradImg = abs(scharrx) + abs(scharry)
    gradImg = (scharrx.astype(np.float)**2 + scharry.astype(np.float)**2)**0.5
    return gradImg


def get_sobel(image):
    sobelx = cv.Sobel(image, ddepth=cv.CV_16S, dx=1, dy=0, scale=1.0, delta=0.0, borderType=cv.BORDER_DEFAULT)
    sobely = cv.Sobel(image, ddepth=cv.CV_16S, dx=0, dy=1, scale=1.0, delta=0.0, borderType=cv.BORDER_DEFAULT)
    gradImg = abs(sobelx) + abs(sobely)
    return gradImg

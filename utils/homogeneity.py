#!/usr/bin/env python

import numpy as np
import cv2 as cv


def rank_normalize(gradImg):
    # Rank-normalization
    _, gradcount = np.unique(gradImg.flatten(), return_counts=True)
    gradvalue = np.zeros(gradImg.shape)
    max_gradvalue = sum(gradcount)

    def get_gradvalue(x, y):
        return sum(gradcount[0:gradImg[x, y] + 1]) / max_gradvalue

    # grad: gradImg[x,y]
    for i in range(gradImg.shape[0]):
        for j in range(gradImg.shape[1]):
            gradvalue[i, j] = get_gradvalue(i, j)
    return gradvalue


def get_scharr(image):
    gray_img = image.copy()
    gray_img = cv.fastNlMeansDenoising(gray_img,templateWindowSize=7,searchWindowSize=27)
    scharrx = cv.Scharr(gray_img, ddepth=cv.CV_16S, dx=1, dy=0, scale=1.0, delta=0.0, borderType=cv.BORDER_DEFAULT)
    scharry = cv.Scharr(gray_img, ddepth=cv.CV_16S, dx=0, dy=1, scale=1.0, delta=0.0, borderType=cv.BORDER_DEFAULT)
    gradImg = abs(scharrx) + abs(scharry)

    print('Get Scharr gradient value, done.')
    return gradImg


def get_sobel(image):
    '''
    input:
        image: [ndarray]
    output:
        UN_n: sobel boundary uncertainty
    '''
    sobelx = cv.Sobel(image, ddepth=cv.CV_16S, dx=1, dy=0, scale=1.0, delta=0.0, borderType=cv.BORDER_DEFAULT)
    sobely = cv.Sobel(image, ddepth=cv.CV_16S, dx=0, dy=1, scale=1.0, delta=0.0, borderType=cv.BORDER_DEFAULT)
    gradImg = abs(sobelx) + abs(sobely)

    print('Get Sobel gradient value, done.')
    return gradImg

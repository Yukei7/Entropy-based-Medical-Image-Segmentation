import cv2 as cv
from scipy.ndimage import binary_fill_holes
import numpy as np

def crop_image(image):
    x1 = np.min(np.where(image>1)[0])
    x2 = np.max(np.where(image>1)[0])
    y1 = np.min(np.where(image>1)[1])
    y2 = np.max(np.where(image>1)[1])
    return image[x1:x2, y1:y2]


def mask_image(image, beta=10):
    _,mask = cv.threshold(image,beta,255,0)
    mask = binary_fill_holes(mask)
    image = image*mask
    image = crop_image(image)
    return image
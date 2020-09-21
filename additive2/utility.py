import os
from typing import List, Union, Iterable

import numpy as np
import cv2
from sklearn.cluster import  KMeans

def dfe(file):
    d, f = os.path.split(file)
    f, e = os.path.splitext(f)
    return d, f, e

def get_largest_contour(mask):
    contours, hierarchy = cv2.findContours(mask.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return max(contours, key=lambda x: len(x))

def draw_largest_contour(mask, fill=-1):
    max_contour = get_largest_contour(mask)
    imt = np.zeros_like(mask)
    cv2.drawContours(imt, [max_contour], -1, 1, fill)
    return imt.astype('uint8')

def crop_rect(img, rect):
    # get the parameter of the small rectangle
    center, size, angle = rect[0], rect[1], rect[2]
    center, size = tuple(map(int, center)), tuple(map(int, size))

    # get row and col num in img
    height, width = img.shape[0], img.shape[1]

    # calculate the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1)
    # rotate the original image
    img_rot = cv2.warpAffine(img, M, (width, height))

    # now rotated rectangle becomes vertical and we crop it
    img_crop = cv2.getRectSubPix(img_rot, size, center)

    return img_crop, img_rot


def get_rotated_rect(I, isImage=True):
    if isImage:
        if isinstance(I, str):
            img = cv2.imread(I, 0)
        else:
            img = I
        contours, hierarchy = cv2.findContours(img.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        max_contour = max(contours, key=lambda x: len(x))
        return cv2.minAreaRect(max_contour)
    return cv2.minAreaRect(I)


def remove_outliers(x, n_classes=2):
    #x = np.array(x).reshape(-1, 1)
    kmeans = KMeans(n_classes)
    classes = kmeans.fit_predict(x)
    c = max(list(range(n_classes)), key=lambda x : np.sum(classes==x))
    return x[classes==c], classes, c


def min_max_scale(x, mn=0, mx=255, dtype='uint8'):
    xmin, xmax = np.min(x), np.max(x)
    out = (x-xmin)/(xmax-xmin) * (mx-mn) + mn
    return out.astype(dtype)

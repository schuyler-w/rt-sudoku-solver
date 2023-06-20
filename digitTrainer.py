from __future__ import print_function
import keras
import os
import numpy as np
import cv2
import random
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras import backend

from scipy import ndimage

## helper function that returns an optimized theoretical shift 
## used to centralize an image according to center of mass
def get_center_shift(image):
    cy, cx = ndimage.measurements.center_of_mass(image)
    rows, cols = image.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)
    return shiftx, shifty

## helper function for center of mass shift 
def shift(image, sx, sy): ## params: (image, shiftx, shifty)
    rows, cols = image.shape ## get rows and columns from image
    M = np.float32([[1, 0, sx], [0, 1, sy]]) ## convert to array scalar
    shifted = cv2.warpAffine(image, M, (cols, rows))
    return shifted

## function to shift according to center of mass
def shift_com(image): 
    image = cv2.bitwise_not(image) ## convert image to using masking
    
    shiftx, shifty = get_center_shift(image)
    shifted = shift(image, shiftx, shifty)
    image = shifted
    
    image = cv2.bitwise_not(image)
    return image


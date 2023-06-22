from __future__ import print_function
import numpy as np
import cv2
import os
from scipy import ndimage

## helper function that returns an optimized theoretical shift 
## used to centralize an image according to center of mass
def get_shift(image):
    cy, cx = ndimage.center_of_mass(image)
    rows, cols = image.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)
    return shiftx, shifty

## helper function for center of mass shift 
def shift(image, sx, sy): ## params: (image, shiftx, shifty)
    rows, cols = image.shape ## get rows and columns from image
    S = np.float32([[1, 0, sx], [0, 1, sy]]) ## convert to array scalar
    shifted = cv2.warpAffine(image, S, (cols, rows))
    return shifted

## function to shift according to center of mass
def shift_com(image): 
    image = cv2.bitwise_not(image) ## convert image to using masking
    
    shiftx, shifty = get_shift(image)
    shifted = shift(image, shiftx, shifty)
    image = shifted
    image = cv2.bitwise_not(image)
    return image

# Reads training data
def create_data(CATEGORIES, DATADIR, rows, cols, data):
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category) ## joins path to find model directory
        class_num = CATEGORIES.index(category)
        
        for image in os.listdir(path):
            img = cv2.imread(os.path.join(path, image), cv2.IMREAD_GRAYSCALE)
            new_img = cv2.resize(img, (rows, cols)) 
            new_img = shift_com(new_img)
            data.append([new_img, class_num])
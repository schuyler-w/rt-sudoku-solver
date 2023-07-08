#Neural Network modules
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

import cv2                         #Standard library for solving computer vision
import copy                        #Required for deep copy
import import_ipynb                #For importing other ipynb files
import math                        #Standard library for mathematical tasks
from scipy import ndimage          #Library for multidimensional image processing

%run RealTimeSudokuSolver.ipynb    #Run RealTimeSudokuSolver
%run sudokuSolver.ipynb            #Run sudokuSolver

import solver                		#To call detectAndSolveSudoku

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 1280)    # HD Camera
cap.set(4, 720)

input_shape = (28, 28, 1)
num_classes = 9
# Initialising the CNN
model = Sequential()
# Add First Convolution Layer
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
# Add Second Convolution Layer
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
# MaxPooling
model.add(MaxPooling2D(pool_size=(2, 2)))
# 25% Dropout
model.add(Dropout(0.25))
#Flatenning
model.add(Flatten())
#Full Connection
model.add(Dense(128, activation='relu'))
# 50% Dropout
model.add(Dropout(0.5))
# Output Layer
model.add(Dense(num_classes, activation='softmax'))
# Load weights from pre-trained model. This model is trained using digitRecognition.ipynb in directory CNN Digit Training
model.load_weights("digitRecognition.h5")   
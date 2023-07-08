from __future__ import print_function
import numpy as np
import random
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import preprocessor as pp

DATADIR = "model"
CATEGORIES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
training_data = []

batch_size = 128
num_classes = 10
epochs = 5

## input image dimensions
image_rows = 28 ## hardcoded rows and column params
image_cols = 28
            
pp.create(CATEGORIES, DATADIR, image_rows, image_cols, training_data) ## runs function to create classification data

## randomly mix up data
random.shuffle(training_data)

## spliting train-test in 80-20 ratio
x_train = [] ## initialize training and test arrays
y_train = []
x_test = []
y_test = []

for i in range(len(training_data)*8//10):
    x_train.append(training_data[i][0])
    y_train.append(training_data[i][1])

for j in range(len(training_data)*8//10, len(training_data)):
    x_test.append(training_data[j][0])
    y_test.append(training_data[j][1])
    
## reshape data
x_train = np.array(x_train)
x_train = x_train.reshape(x_train.shape[0], image_rows, image_cols, 1)

x_test = np.array(x_test)
x_test = x_test.reshape(x_test.shape[0], image_rows, image_cols, 1)

input_shape = (image_rows, image_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

## normalize data
x_train /= 255
x_test /= 255

## convert class vectors(ints) to binary class matrices
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

model = Sequential() # initialize model object as sequential class
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=input_shape)) ## ReLU activation: max(x, 0)
model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.50))
model.add(Dense(num_classes, activation='softmax')) ## softmax = generalization of logisitic function to multiple dimensions

model.compile(loss=tf.keras.losses.categorical_crossentropy, 
              optimizer="Adam", 
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=batch_size,epochs=epochs, verbose=1, validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss: ', score[0])
print('Test accuracy: ', score[1])

model_json = model.to_json()
with open('data.json', 'w') as json_file:
    json_file.write(model_json)
model.save('data.meta', 
            overwrite=True,
            include_optimizer=True,
            save_format='tf')

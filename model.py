#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 17:59:45 2017

@author: alex
"""

import csv
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Cropping2D, MaxPooling2D ,Flatten, Dense, Lambda, Conv2D, Activation, Dropout
from keras.backend import tf as ktf
import matplotlib.pyplot as plt
import cv2

def random_augmentation(image, angle):
    
    choice = np.random.randint(3)
    #Random brightness
    if choice == 0:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        V = hsv[:,:,2] #ex. V = 0.7
        rmin = -V # -0.7
        rmax = 1-V # 0.3
        error = np.random.uniform(rmin,rmax) # -.7 ... .3
        hsv[:,:,2] =  hsv[:,:,2] + error # 0.7 - 0.7 ... 0.7 + 0.3
        image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

    #Random flip
    elif choice == 1:
        if np.random.rand() < 0.5:
            image = cv2.flip(image, 1)
            angle = -angle
    else:
        # do nothing
        image = image
        angle = angle
    
    return image, angle


# randomly choose, generate and return an array of images and an array of steering angles with each a size of nbatches
def batch_generator(files, angles,nbatches):
    X_ = []
    y_ = []

    while True:
        for i in range(nbatches):
            random_choice = np.random.randint(len(files))
            image = plt.imread(files[random_choice])
            
            image, angle = random_augmentation(image, angles[random_choice])
            
            X_.append(image)
            y_.append(angle)
            
        yield np.array(X_, dtype=np.float32), np.array(y_, dtype=np.float32)
        
        X_ = []
        y_ = []

# randomly choose and return an array of images and an array of steering angles with each a size of nbatches
def validation_batch_generator(files, angles, nbatches):
    X_ = []
    y_ = []

    while True:
        for i in range(nbatches):
            random_choice = np.random.randint(len(files))
            image = plt.imread(files[random_choice])
            angle = angles[random_choice]
            X_.append(image)
            y_.append(angle)
        
        yield np.array(X_, dtype=np.float32), np.array(y_, dtype=np.float32)
        X_ = []
        y_ = []

np.random.seed(888)
images = []
angles = []
correction = [0.,.15,-.15]
nepochs = 100
batch_size = 64
lines = []

# load the filenames of the images (center, left, right)
with open('./data/first/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

with open('./data/second/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

with open('./data/third/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

with open('./data/careful/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

#with open('./data/fourth/driving_log.csv') as csvfile:
#    reader = csv.reader(csvfile)
#    for line in reader:
#        lines.append(line)

#with open('./data/fifth/driving_log.csv') as csvfile:
#    reader = csv.reader(csvfile)
#    for line in reader:
#        lines.append(line)

# fill arrays with imagenames and fitting angles as well as correct the steering angles
for line in lines:
    for i in range(3):
        images.append(line[i])
        angle = float(line[3])
        angle += correction[i]
        angles.append(angle)

# split and shufffle into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(images, angles, test_size=0.2 , random_state=888)


# create the model
model = Sequential()

model.add(Cropping2D(cropping=((60,25), (0,0)), input_shape = (160,320,3)))

model.add(Lambda(lambda x: x / 255.0 - 0.5))

model.add(Conv2D(24,(5,5), activation = "relu"))
model.add(MaxPooling2D(2, padding="same"))

model.add(Conv2D(36,(5,5), activation = "relu"))
model.add(MaxPooling2D(2, padding="same"))

model.add(Conv2D(48,(5,5), activation = "relu"))
model.add(MaxPooling2D(2, padding="same"))

model.add(Conv2D(64,(3,3), activation = "relu"))
#model.add(MaxPooling2D())

model.add(Conv2D(64,(3,3), activation = "relu"))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(1164, activation = "relu"))
model.add(Dense(100, activation = "relu"))
model.add(Dense(50, activation = "relu"))
model.add(Dense(10, activation = "relu"))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')

# train the model and save the history for later processing
history = model.fit_generator(batch_generator(X_train, y_train, batch_size),steps_per_epoch = (len(X_train)/batch_size), epochs = nepochs, validation_data = validation_batch_generator(X_valid, y_valid, batch_size),validation_steps = int(len(X_valid)/batch_size), max_q_size = 50)

# save the model for thesimulator
model.save('model.h5')

# utilities to plot the model architecture
from keras.utils import plot_model
plot_model(model,show_shapes = True,  to_file='model.png')

# plot and save the loss over epochs for the training and validation provided by the "fit" function
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.savefig('loss.png')
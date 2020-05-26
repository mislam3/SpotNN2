# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 19:39:20 2020

@author: siamm
"""

from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt



# resize all images 
IMAGE_SIZE = [224, 224]

train_path = 'content/train_data/train'
valid_path = 'content/test'

# add preprocessing layer to the front of VGG
vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# DO NOT train existing weights
for layer in vgg.layers:
    layer.trainable = False

# useful for getting number if classes
# categories taken according to the folders inside
folders = glob('content/train_data/train*')

# Our layers -more can be added if required
x = Flatten()(vgg.output)
# x = Dense(1000, activation='relu)(x)
prediction = Dense(len(folders), activation='softmax')(x)

# create a model object
model = Model(inputs=vgg.input, outputs=prediction)

# view the structure of the model
model.summary()

# tell the model what cost optimization method to use
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   class_mode = 'categorical')

'''r=model.fit_generator(training_set,
                          samples_per_epoch = 8000,
                          nb_epoch = 5,
                          validation_data = train_set,
                          nb_val_samples = 2000)'''

#fit the model
r = model.fit_generator(
    training_set,
    validation_data=test_set,
    epochs=5,
    steps_per_epoch=len(training_set),
    validation_steps=len(test_set)
)

# loss
plt.plot(r.history['loss'], label='train_loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')


import tensorflow as tf
from keras.models import load_model
# model.save('facefeatures_new_model.h5')
model.save('vehiclefeatures_new_model.h5')


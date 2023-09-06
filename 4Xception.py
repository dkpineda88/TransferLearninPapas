# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 11:46:07 2023

@author: Lenovo
"""

# -*- coding: utf-8 -*-

"""
Created on Tue May 30 14:40:34 2023

@author: Lenovo

"""

import matplotlib.pyplot as plt
# Tensorflow
import tensorflow as tf
#import tensorflow_hub as hub
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model , model_from_json , Model
from tensorflow.keras.preprocessing import image 
from tensorflow import keras 
from keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Model
from tensorflow.keras.applications import imagenet_utils

import pandas as pd
#import cv2

NUM_CLASSES = 3

# Fixed for Cats & Dogs color images
CHANNELS = 3

IMAGE_RESIZE = 229
RESNET50_POOLING_AVERAGE = 'avg'
DENSE_LAYER_ACTIVATION = 'softmax'
OBJECTIVE_FUNCTION = 'categorical_crossentropy'

# Common accuracy metric for all outputs, but can use different metrics for different output
LOSS_METRICS = ['accuracy']

# EARLY_STOP_PATIENCE must be < NUM_EPOCHS
NUM_EPOCHS = 50
EARLY_STOP_PATIENCE = 3
# These steps value should be proper FACTOR of no.-of-images in train & valid folders respectively
# Training images processed in each step would be no.-of-train-images / STEPS_PER_EPOCH_TRAINING
STEPS_PER_EPOCH_TRAINING = 10
STEPS_PER_EPOCH_VALIDATION = 10

# These steps value should be proper FACTOR of no.-of-images in train & valid folders respectively
# NOTE that these BATCH* are for Keras ImageDataGenerator batching to fill epoch step input
BATCH_SIZE_TRAINING = 32
BATCH_SIZE_VALIDATION = 32

# Using 1 to easily manage mapping between test_generator & prediction for submission preparation
BATCH_SIZE_TESTING = 1

from keras.applications import Xception

IMG_SHAPE = (229, 229, 3)
Xception_MODEL=tf.keras.applications.Xception(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

Xception_MODEL.trainable=False
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(3,activation='softmax')
model = tf.keras.Sequential([
  Xception_MODEL,
  global_average_layer,
  prediction_layer
])

model.compile(optimizer='adam', 
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=["accuracy"])



from keras.preprocessing.image import ImageDataGenerator

image_size = IMAGE_RESIZE


train_data_dir = 'PlagaPapaTL/train'
validation_data_dir = 'PlagaPapaTL/validate'
test_data_dir = 'PlagaPapaTL/test'
nb_train_samples =32
nb_validation_samples = 32
IMAGE_SIZE = (229, 229)


img_width, img_height = 224,224
input_shape = (img_width, img_height, 3)

train_datagen = ImageDataGenerator(
	rescale=1. / 255,
	shear_range=0.2,
	zoom_range=0.2,
	horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_directory(
	train_data_dir,
	target_size=(img_width, img_height),
	batch_size=BATCH_SIZE_TRAINING,
	class_mode='binary')
validation_generator = test_datagen.flow_from_directory(
	validation_data_dir,
	target_size=(img_width, img_height),
	batch_size=BATCH_SIZE_TRAINING,
	class_mode='binary')

test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
	test_data_dir,
	target_size=(img_width, img_height),
	batch_size=BATCH_SIZE_TRAINING,
	class_mode='binary')

model.summary()
print(validation_data_dir)

fit_history = model.fit(        
        train_generator,
        steps_per_epoch= 10,
        epochs = 50,
        validation_data=validation_generator,
        validation_steps= 10,
        
)

#exacitud
fit_history.history['accuracy']

test_score=model.evaluate(test_generator)

#precision
fit_history.history['precision']
#sensibilidad
fit_history.history['recall']

acc = fit_history.history['accuracy']
val_acc = fit_history.history['val_accuracy']
loss = fit_history.history['loss']
val_loss = fit_history.history['val_loss']
plt.figure(figsize = (16,10))
plt.subplot(1,2,1)
plt.plot(range(50), acc, label = 'Entrenamiento de precisión')
plt.plot(range(50), val_acc, label = 'Validación de precisión')
plt.legend(loc='lower right')
plt.title('Entrenamiento y Validación de precisión')


plt.subplot(1,2,2)
plt.plot(range(50), loss, label = 'Entrenamiento de perdida')
plt.plot(range(50), val_loss, label = 'Validación de pérdidas')
plt.legend(loc = 'upper right')
plt.title('Entrenamiento y Validación de Pérdidas')








converter=tf.lite.TFLiteConverter.from_keras_model(ResNET_MODEL)
tflite_model = converter.convert()

open("models/3/papas.tflite", "wb").write(tflite_model)


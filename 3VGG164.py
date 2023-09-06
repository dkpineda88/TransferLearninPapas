# -*- coding: utf-8 -*-

"""
Created on Tue May 30 14:40:34 2023

@author: Lenovo
"""
import os
import numpy as np
import matplotlib.pyplot as plt
# Tensorflow
import tensorflow as tf
import tensorflow_hub as hub
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

IMAGE_RESIZE = 224
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


from keras.applications import ResNet50,VGG16
from keras.models import Sequential
from keras.layers import Dense,Dropout


IMG_SHAPE = (224, 224, 3)
VGG16_MODEL=tf.keras.applications.VGG16(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

VGG16_MODEL.trainable=False
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(3,activation='softmax')
model = tf.keras.Sequential([
  VGG16_MODEL,
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(64, activation = 'relu'),
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
IMAGE_SIZE = (224, 224)


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

fit_history = model.fit(        
        train_generator,
        steps_per_epoch= 10,
        epochs = 50,
        validation_data=validation_generator,
        validation_steps= 10,
        
)

#exacitud
fit_history.history['accuracy']
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

test_score=model.evaluate(test_generator)

def representative_datasetp():
  for image_batch, labels_batch in train_ds:
    yield [image_batch]


converter = tf.lite.TFLiteConverter.from_keras_model(neural_net)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.representative_dataset=representative_datasetp
tflite_quant_model = converter.convert()
open("models/VGG16/papasOptVGG.tflite", "wb").write(tflite_quant_model)

import numpy as np
for images_batch, labels_batch in test.take(1):
    
    first_image = images_batch[0].numpy().astype('uint8')
    first_label = labels_batch[0].numpy()
    
    print("first image to predict")
    plt.imshow(first_image)
    print("actual label:",class_names[first_label])
    
    batch_prediction = neural_net.predict(images_batch)
    print("predicted label:",class_names[np.argmax(batch_prediction[0])])
    
    
    
def predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(images[i].numpy())
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence

for images, labels in test.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        
        predicted_class, confidence = predict(neural_net, images[i].numpy())
        actual_class = class_names[labels[i]] 
        
        plt.title(f"Actual: {actual_class},\n Predicted: {predicted_class}.\n Confidence: {confidence}%")
        
        plt.axis("off")
        
from IPython.display import Image
img1=Image(filename='PlagasPapas/Sano/image (1).jpg', width=224,height=224) 
        
        
predicted_class, confidence = predict(neural_net, img1)

predicted_class

confidence

converter=tf.lite.TFLiteConverter.from_keras_model(neural_net)
tflite_model = converter.convert()

open("models/3/papas.tflite", "wb").write(tflite_model)


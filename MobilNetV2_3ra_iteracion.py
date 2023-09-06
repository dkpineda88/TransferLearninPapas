# -*- coding: utf-8 -*-

"""
Created on Tue May 30 14:40:34 2023

@author: Lenovo
"""

import numpy as np
import matplotlib.pyplot as plt
# Tensorflow
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_model_optimization as tfmot
from keras.preprocessing.image import ImageDataGenerator


dataset = tf.keras.preprocessing.image_dataset_from_directory("PlagasPapas",shuffle = True,image_size = IMAGE_SIZE, batch_size = BATCH_SIZE)
class_names = dataset.class_names
class_names

plt.figure(figsize=(10,10))
for image_batch, label_batch in dataset.take(1):
    for i in range(15):
        ax=plt.subplot(3,5,i+1)
        plt.imshow(image_batch[i].numpy().astype("uint8"))
        plt.title(class_names[label_batch[i]])
        plt.axis("off")
        
train_size = 0.8
len(dataset)*train_size

train = dataset.take(75)
len(train)

rest = dataset.skip(75)
len(rest)


val_size = 0.1
len(dataset)*val_size

test = rest.skip(9)
len(test)


# Define function to split data
def get_dataset_partitions_tf(ds, train_split = 0.8, val_split = 0.1, test_split = 0.1,
                             shuffle=True, shuffle_size = 10000):
    ds_size = len(ds)
    
    if shuffle:
        ds = ds.shuffle(shuffle_size, seed = 12)
    
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    
    train = ds.take(train_size)
    validate = ds.skip(train_size).take(val_size)
    test = ds.skip(train_size).skip(val_size)
    return train, validate, test

# Calling function to split data
train, validate, test = get_dataset_partitions_tf(dataset)


# Cache data and optimize speed
train = train.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)
validate = validate.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)
test = test.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)


# Resize and rescale image
resize_and_rescale = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.Resizing(224,224),
    tf.keras.layers.experimental.preprocessing.Rescaling(1.0/255)
])

# Data augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2)
])


# Define neural network architecture
n_classes = 3
# input shape = image size, channel
input_shape = (BATCH_SIZE, 224,224,CHANNELS)
train_data_dir = 'PlagaPapaTL/train'
validation_data_dir = 'PlagaPapaTL/validate'
test_data_dir = 'PlagaPapaTL/test'
nb_train_samples =10
nb_validation_samples = 10
epochs = 50
batch_size = 28
IMAGE_SIZE = (224, 224)
BATCH_SIZE_TRAINING = 28
BATCH_SIZE_VALIDATION = 28



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




mobilenet_v2 ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"
inception_v3 = "https://tfhub.dev/google/imagenet/inception_v3/classification/5"

#mobilenet_v2 = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"

mobile_net_layers = hub.KerasLayer(mobilenet_v2, input_shape=input_shape)
mobile_net_layers.trainable = False


neural_net = tf.keras.Sequential([
  mobile_net_layers,
  tf.keras.layers.Dropout(0.3),
  tf.keras.layers.Dense(64, activation = 'relu'),
  # softmax normalized probability of classes  
  tf.keras.layers.Dense(3,activation = 'softmax'),
])



neural_net.summary()

neural_net.compile(
    optimizer = 'adam',
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics = ['accuracy']
)

model_fit = neural_net.fit(train_generator,
steps_per_epoch= 10,
epochs = 50,
validation_data=validation_generator,
validation_steps= 10,)

model_fit.history['accuracy']

scores = neural_net.evaluate(test_generator)

acc = model_fit.history['accuracy']
val_acc = model_fit.history['val_accuracy']
loss = model_fit.history['loss']
val_loss = model_fit.history['val_loss']
plt.figure(figsize = (16,10))
plt.subplot(1,2,1)
plt.plot(range(epochs), acc, label = 'Entrenamiento de precisión')
plt.plot(range(epochs), val_acc, label = 'Validación de precisión')
plt.legend(loc='lower right')
plt.title('Entrenamiento y Validación de precisión')

plt.subplot(1,2,2)
plt.plot(range(epochs), loss, label = 'Entrenamiento de perdida')
plt.plot(range(epochs), val_loss, label = 'Validación de pérdidas')
plt.legend(loc = 'upper right')
plt.title('Entrenamiento y Validación de Pérdidas')


##MATRIZ DE CONFUSION
from sklearn.metrics import classification_report, confusion_matrix

Y_pred = neural_net.predict_generator(validation_generator, 200 // batch_size+1)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(validation_generator.classes, y_pred))
print('Classification Report')



class_names=dataset_test.class_names

# Calling function to split data
import numpy as np
for images_batch, labels_batch in test_generator:
    
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


#Cuantificacion del modelo
         
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  train_data_dir,
  validation_split=0.1,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
          
def representative_datasetp():
  for image_batch, labels_batch in train_ds:
    yield [image_batch]


converter = tf.lite.TFLiteConverter.from_keras_model(neural_net)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.representative_dataset=representative_datasetp
tflite_quant_model = converter.convert()
open("models/3/papasOpt4.tflite", "wb").write(tflite_quant_model)
#tflite_filepath.write_bytes(tflite_model)


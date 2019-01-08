#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : test_basic_keras.py
# @Author: MoonKuma
# @Date  : 2019/1/7
# @Desc  : https://www.tensorflow.org/tutorials/keras/basic_classification


# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

# down loading data set (auto saved in: C:\Users\7q\.keras\datasets)
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# show details
train_images.shape
# show pics
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show() # show the exact pics
plt.close()
# [0~255] -> [0~1]
train_images = train_images / 255.0
test_images = test_images / 255.0
# show first 25 plots
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show() # show the exact pics
plt.close()
# model
# compile model layers
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)), # Flatten use to reshape 28*28 data set into 724*1 data set
    keras.layers.Dense(128, activation=tf.nn.relu), # 128 nodes full connection, nn:neural network, type = relu (724*1 --relu--> 128*1)
    keras.layers.Dense(10, activation=tf.nn.softmax) # 10 nodes full connection, type = softmax (128*1 --softmax--> 10*1)
])
# compile other parameters
model.compile(optimizer=tf.train.AdamOptimizer(), # A optimizer to minimize cost(sum of lost) functions
              loss='sparse_categorical_crossentropy', # A loss func to find the proper w/b
              metrics=['accuracy']) # Output when training/testing
# try fit
model.fit(train_images, train_labels, epochs=5, batch_size=100)
# try evaluate
test_loss, test_acc = model.evaluate(test_images, test_labels)
# try predict
predictions = model.predict(test_images)
# argmax of soft max
class_names[np.argmax(predictions[0])]
class_names[test_labels[0]]

# some other plot

def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue') # blue will cover red up if the prediction is correct

#pic
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
plt.show()

i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
plt.show()

# Plot the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions, test_labels)
plt.show()
# Predict single pics
img = test_images[0]
print(img.shape)
img = (np.expand_dims(img,0))
print(img.shape)
predictions_single = model.predict(img) # this require 3-dimensions
print(predictions_single)
class_names[np.argmax(predictions_single)]
plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)

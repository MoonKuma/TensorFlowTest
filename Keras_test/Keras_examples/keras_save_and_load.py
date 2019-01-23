#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : keras_save_and_load.py
# @Author: MoonKuma
# @Date  : 2019/1/18
# @Desc  : https://www.tensorflow.org/tutorials/keras/save_and_restore_models



from __future__ import absolute_import, division, print_function

import os

import tensorflow as tf
from tensorflow import keras

tf.__version__


# MNIST

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0


# Returns a short sequential model
def create_model():
    model = tf.keras.models.Sequential([
    keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                loss=tf.keras.losses.sparse_categorical_crossentropy,
                metrics=['accuracy'])

    return model

# Create a basic model instance
model = create_model()
model.summary()
# check points is a way to save the model after every epoch
checkpoint_path = "C:/Users/7q/PycharmProjects/TensorFlowTest/Keras_test/Keras_examples/training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
# Create checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
model = create_model()
model.fit(train_images, train_labels,  epochs = 10,
          validation_data = (test_images,test_labels),
          callbacks = [cp_callback])  # pass callback to training


# Here we load weights from checkpoint data
model = create_model()  # reinitialize the model
loss, acc = model.evaluate(test_images, test_labels)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))

model.load_weights(checkpoint_path)
loss, acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

# Manually saved
model.save_weights('C:/Users/7q/PycharmProjects/TensorFlowTest/Keras_test/Keras_examples/training_1/my_checkpoint')
# Restore the weights
model = create_model()
model.load_weights('./checkpoints/my_checkpoint')

# Besides saving the check point file, one can also save the entire model
model = create_model()
model.fit(train_images, train_labels, epochs=5)

# Save entire model to a HDF5 file
model.save('C:/Users/7q/PycharmProjects/TensorFlowTest/Keras_test/Keras_examples/training_1/my_model.h5')

# Recreate the exact same model, including weights and optimizer.
new_model = keras.models.load_model('C:/Users/7q/PycharmProjects/TensorFlowTest/Keras_test/Keras_examples/training_1/my_model.h5')
new_model.summary()

loss, acc = new_model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))





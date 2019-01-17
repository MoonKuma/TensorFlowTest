#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : keras_regression.py
# @Author: MoonKuma
# @Date  : 2019/1/14
# @Desc  : https://www.tensorflow.org/tutorials/keras/basic_regression

from __future__ import absolute_import, division, print_function

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
# seaborn is python's way to plot statistic data (as an enhancement of plt)
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)

# download data
dataset_path = keras.utils.get_file("auto-mpg.data", "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
dataset_path

column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                      na_values = "?", comment='\t',
                      sep=" ", skipinitialspace=True)

dataset = raw_dataset.copy()
dataset.tail()

# clean nan
dataset.isna().sum()
dataset = dataset.dropna()

# convert column 'Origin' into one-hot
origin = dataset.pop('Origin')
dataset['USA'] = (origin == 1)*1.0
dataset['Europe'] = (origin == 2)*1.0
dataset['Japan'] = (origin == 3)*1.0
dataset.tail()

# split out train and test data set
train_dataset = dataset.sample(frac=0.8,random_state=0) # random state is the random seed
test_dataset = dataset.drop(train_dataset.index)

# plot with sns
sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")
plt.show()

# describe
train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()

# get label
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')

# normalize
def norm(x):
  return (x - train_stats['mean']) / train_stats['std']

normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

# Caution: The statistics used to normalize the inputs here are as important as the model weights.
# Let's try and see the difference later

def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]), # layers of A0
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(1) # no activation func(linea activation func in the final layer), to predict continuous results
  ])
  optimizer = tf.train.RMSPropOptimizer(0.001)
  model.compile(loss='mse', # mse is the mean square
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model


model = build_model()

model.summary()


# test prediction
example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
example_result

# train
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 1000

history = model.fit(
  normed_train_data, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0, # here use the same train data as validation
  callbacks=[PrintDot()])

# Visualize the model's training progress using the stats stored in the history object.
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

# plot out history
def plot_history(history):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mean_absolute_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
             label='Val Error')
    plt.legend()
    plt.ylim([0, 5])
    plt.show()

    plt.clf()
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mean_squared_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],
             label='Val Error')
    plt.legend()
    plt.ylim([0, 20])
    plt.show()

# plot out
plot_history(history)



# patience: number of epochs with no improvement after which training will be stopped.
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)

# here implement early stop
model1 = build_model()
history = model1.fit(normed_train_data, train_labels, epochs=EPOCHS,
                    validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])

plot_history(history)

# try test data
loss, mae, mse = model1.evaluate(normed_test_data, test_labels, verbose=0)

print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))

# predict
test_predictions = model1.predict(normed_test_data).flatten()

def plot_prediction(test_labels, test_predictions, head_name):
    plt.clf()
    plt.scatter(test_labels, test_predictions)
    plt.title(head_name)
    plt.xlabel('True Values [MPG]')
    plt.ylabel('Predictions [MPG]')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([0,plt.xlim()[1]])
    plt.ylim([0,plt.ylim()[1]])
    _ = plt.plot([-100, 100], [-100, 100])
    plt.show()

# some notes from the website

'''
This notebook introduced a few techniques to handle a regression problem.
1. Mean Squared Error (MSE) is a common loss function used for regression problems (different than classification problems).
2. Similarly, evaluation metrics used for regression differ from classification. A common regression metric is Mean Absolute Error (MAE).
3. When input data features have values with different ranges, each feature should be scaled independently.
4. If there is not much training data, prefer a small network with few hidden layers to avoid overfitting.
5. Early stopping is a useful technique to prevent over-fitting.
'''

# now lets try train without normalize

model2 = build_model()
history_without_normal = model2.fit(train_dataset, train_labels, epochs=EPOCHS,
                    validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])
plot_history(history_without_normal)
test_predictions2 = model2.predict(test_dataset).flatten()

# compare these two to see the importance of implementing normalize
plot_prediction(test_labels, test_predictions, 'After_Normalize')
plot_prediction(test_labels, test_predictions2, 'Before_Normalize')


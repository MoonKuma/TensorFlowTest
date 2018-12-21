#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : test_eager_api.py
# @Author: MoonKuma
# @Date  : 2018/12/18
# @Desc  : Test Eager Execution, this could be applied to tf.contrib.data.make_csv_dataset(read data inside a pipeline)
'''
导入所需的 Python 模块（包括 TensorFlow），然后针对此程序启用 Eager Execution。借助 Eager Execution，
TensorFlow 会立即评估各项操作，并返回具体的值，而不是创建稍后执行的计算图。如果您习惯使用 REPL 或 python 交互控制台，
对于 Eager Execution 您会用起来得心应手。TensorFlow 1.8 及以上版本中提供了 Eager Execution。

Refer : https://www.tensorflow.org/tutorials/eager/custom_training_walkthrough
'''

from __future__ import absolute_import, division, print_function

import os
import matplotlib.pyplot as plt
from modenship_data.get_local import local_path
import tensorflow as tf
import tensorflow.contrib.eager as tfe

tf.enable_eager_execution()

# train_dataset_url = "http://download.tensorflow.org/data/iris_training.csv"
#
# train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url),
#                                            origin=train_dataset_url)

train_dataset_fp = 'test_data/iris_training.csv'
test_dataset_fp = 'test_data/iris_test.csv'

print("Local copy of the dataset file: {}".format(train_dataset_fp))

# column order in CSV file
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

feature_names = column_names[:-1]
label_name = column_names[-1]

# print("Features: {}".format(feature_names))
# print("Label: {}".format(label_name))



batch_size = 32

train_dataset = tf.contrib.data.make_csv_dataset(
    train_dataset_fp,
    batch_size,
    column_names=column_names,
    label_name=label_name,
    num_epochs=1)

test_dataset = tf.contrib.data.make_csv_dataset(
    test_dataset_fp,
    batch_size,
    column_names=column_names,
    label_name=label_name,
    num_epochs=1)



def pack_features_vector(features, labels):
  """Pack the features into a single array."""
  features = tf.stack(list(features.values()), axis=1)
  return features, labels

train_dataset = train_dataset.map(pack_features_vector)
test_dataset = test_dataset.map(pack_features_vector)

features, labels = next(iter(train_dataset))

print(features[:5])


model = tf.keras.Sequential([
  tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)),  # input shape required
  tf.keras.layers.Dense(10, activation=tf.nn.relu),
  tf.keras.layers.Dense(3)
])

predictions = model(features)
predictions[:5]

tf.nn.softmax(predictions[:5])

def loss(model, x, y):
  y_ = model(x)
  return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)

l = loss(model, features, labels)
print("Loss test: {}".format(l))

def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

global_step = tf.train.get_or_create_global_step()

loss_value, grads = grad(model, features, labels)
print("Step: {}, Initial Loss: {}".format(global_step.numpy(),
                                          loss_value.numpy()))

optimizer.apply_gradients(zip(grads, model.variables), global_step)

print("Step: {},         Loss: {}".format(global_step.numpy(),
                                          loss(model, features, labels).numpy()))

## Note: Rerunning this cell uses the same model variables

# keep results for plotting
train_loss_results = []
train_accuracy_results = []

num_epochs = 201

for epoch in range(num_epochs):
  epoch_loss_avg = tfe.metrics.Mean()
  epoch_accuracy = tfe.metrics.Accuracy()

  # Training loop - using batches of 32
  for x, y in train_dataset:
    # Optimize the model
    loss_value, grads = grad(model, x, y)
    optimizer.apply_gradients(zip(grads, model.variables),
                              global_step)

    # Track progress
    epoch_loss_avg(loss_value)  # add current batch loss
    # compare predicted label to actual label
    epoch_accuracy(tf.argmax(model(x), axis=1, output_type=tf.int32), y)

  # end epoch
  train_loss_results.append(epoch_loss_avg.result())
  train_accuracy_results.append(epoch_accuracy.result())

  if epoch % 50 == 0:
    print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                epoch_loss_avg.result(),
                                                                epoch_accuracy.result()))


print('FINISH TRAINING')

test_accuracy = tfe.metrics.Accuracy()

for (x, y) in test_dataset:
    print(x,y)
    logits = model(x)
    prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
    test_accuracy(prediction, y)
print("Test set accuracy: {:.3%}".format(test_accuracy.result()))



#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : 现代海战数据预测(EAGER).py
# @Author: MoonKuma
# @Date  : 2018/12/18
# @Desc  : Predict user login

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

# start eager execution
tf.enable_eager_execution()


def pack_features_vector(features, labels):
  """Pack the features into a single array."""
  features = tf.stack(list(features.values()), axis=1)
  return features, labels



# test data set
# filenames = ['modenship_data/tf_data_modernshiplogin_newuser.txt']
train_filenames = ['tf_data_modernshiplogin_newuser_train_small.txt']
test_filenames = ['tf_data_modernshiplogin_newuser_test_small.txt']
# 训练样本数据
train_num_examples = 4000000
# 测试样本数据
test_num_examples = 889288
# 训练次数
train_num = 20000
column_names = ['datetime', 'uid', 'level', 'vip_level', 'pay_money', 'diamond_remain', 'online_time', 'login']
select_columns = [ 'level', 'vip_level', 'diamond_remain', 'online_time', 'login']
column_defaults = [ tf.float32, tf.float32, tf.float32, tf.float32, tf.int32]
train_dataset = tf.data.experimental.make_csv_dataset(file_pattern=local_path(train_filenames[0]), batch_size=100,
                                                shuffle_buffer_size=20000,
                                                shuffle=True,
                                                column_defaults=column_defaults,
                                                column_names=column_names,
                                                num_parallel_reads=2,
                                                label_name='login',
                                                num_epochs=1,
                                                select_columns=select_columns,
                                                field_delim='\t').map(pack_features_vector)

test_dataset = tf.data.experimental.make_csv_dataset(file_pattern=local_path(test_filenames[0]), batch_size=100,
                                                shuffle_buffer_size=20000,
                                                shuffle=True,
                                                column_defaults=column_defaults,
                                                column_names=column_names,
                                                num_parallel_reads=2,
                                                num_epochs=1,
                                                label_name='login',
                                                select_columns=select_columns,
                                                field_delim='\t').map(pack_features_vector)

# map data

features, labels = next(iter(train_dataset))


model = tf.keras.Sequential([
  tf.keras.layers.Dense(18, activation=tf.nn.relu, input_shape=(len(select_columns)-1,)),  # input shape required
  tf.keras.layers.Dense(18, activation=tf.nn.relu),
  tf.keras.layers.Dense(2)
])

def loss(model, x, y):
  y_ = model(x)
  # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=prediction))
  return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)
  # return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=y_))




def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)


# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
optimizer = tf.train.AdamOptimizer(0.001)


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
    line_num = 1
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
    logits = model(x)
    prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
    test_accuracy(prediction, y)
print("Test set accuracy: {:.3%}".format(test_accuracy.result()))

# model = tf.keras.Sequential([
#     # Adds a densely-connected layer with 64 units to the model:
#     tf.keras.layers.Dense(32, activation='relu', input_shape=(6,)),
#     # Add another:
#     tf.keras.layers.Dense(32, activation='relu'),
#     # Add a softmax layer with 10 output units:
#     tf.keras.layers.Dense(2, activation='softmax')])
#
# model.compile(optimizer=tf.train.AdamOptimizer(0.001),
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
#
# model.fit(train_dataset, epochs=10, steps_per_epoch=300)

'''
# As a result, this tf classifier works only a little better than its competitors(SVM or Linear) on full data scale
# Accuracy:
# - TF(full data): 77.3%
# - TF(small data): 70.2%
# - SVM(small data): 75.7%
# - Linear/Logistic(full/small data): 75.0%
# BTW, svm is incapable in running with full data scale
'''
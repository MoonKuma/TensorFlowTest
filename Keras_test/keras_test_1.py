#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : keras_test_1.py
# @Author: MoonKuma
# @Date  : 2018/12/17
# @Desc  : test code result from "https://www.tensorflow.org/guide/keras"

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

print(tf.VERSION)
print(tf.keras.__version__)


# 序列模型
model = tf.keras.Sequential()
# Adds a densely-connected layer with 64 units to the model:
model.add(layers.Dense(64, activation='relu'))
# Add another:
model.add(layers.Dense(64, activation='relu'))
# Add a softmax layer with 10 output units:
model.add(layers.Dense(10, activation='softmax'))


# 配置层
# Create a sigmoid layer:
# layers.Dense(64, activation='sigmoid')
# # Or:
# layers.Dense(64, activation=tf.sigmoid)
#
# # A linear layer with L1 regularization of factor 0.01 applied to the kernel matrix:
# layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l1(0.01))
#
# # A linear layer with L2 regularization of factor 0.01 applied to the bias vector:
# layers.Dense(64, bias_regularizer=tf.keras.regularizers.l2(0.01))
#
# # A linear layer with a kernel initialized to a random orthogonal matrix:
# layers.Dense(64, kernel_initializer='orthogonal')
#
# # A linear layer with a bias vector initialized to 2.0s:
# layers.Dense(64, bias_initializer=tf.keras.initializers.constant(2.0))

# 设置训练流程
model = tf.keras.Sequential([
# Adds a densely-connected layer with 64 units to the model:
layers.Dense(64, activation='relu'),
# Add another:
layers.Dense(64, activation='relu'),
# Add a softmax layer with 10 output units:
layers.Dense(10, activation='softmax')])

model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

'''
tf.keras.Model.compile 采用三个重要参数：
optimizer：此对象会指定训练过程。从 tf.train 模块向其传递优化器实例，例如 tf.train.AdamOptimizer、tf.train.RMSPropOptimizer 或 tf.train.GradientDescentOptimizer。
loss：要在优化期间最小化的函数。常见选择包括均方误差 (mse)、categorical_crossentropy 和 binary_crossentropy。损失函数由名称或通过从 tf.keras.losses 模块传递可调用对象来指定。
metrics：用于监控训练。它们是 tf.keras.metrics 模块中的字符串名称或可调用对象。
'''

data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))
val_data = np.random.random((100, 32))
val_labels = np.random.random((100, 10))

model.fit(data, labels, epochs=10, batch_size=32,
          validation_data=(val_data, val_labels))

'''
tf.keras.Model.fit 采用三个重要参数：
epochs：以周期为单位进行训练。一个周期是对整个输入数据的一次迭代（以较小的批次完成迭代）。
batch_size：当传递 NumPy 数据时，模型将数据分成较小的批次，并在训练期间迭代这些批次。此整数指定每个批次的大小。请注意，如果样本总数不能被批次大小整除，则最后一个批次可能更小。
validation_data：在对模型进行原型设计时，您需要轻松监控该模型在某些验证数据上达到的效果。传递此参数（输入和标签元组）可以让该模型在每个周期结束时以推理模式显示所传递数据的损失和指标。
'''
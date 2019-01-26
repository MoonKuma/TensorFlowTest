#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : Eager_tf_basic.py
# @Author: MoonKuma
# @Date  : 2019/1/25
# @Desc  : basic api of tf and eager
# Including:
#   https://www.tensorflow.org/tutorials/eager/eager_basics
#   https://www.tensorflow.org/tutorials/eager/automatic_differentiation


import tensorflow as tf
import numpy as np

tf.enable_eager_execution()

# BASIC TEST
# test tf
print(tf.add(1, 2))
print(tf.add([1, 2], [3, 4]))
print(tf.square(5))
print(tf.reduce_sum([1, 2, 3]))
print(tf.encode_base64("hello world"))
print(tf.square(2) + tf.square(3))
# Each Tensor has a shape and a datatype
x = tf.matmul([[1]], [[2, 3]])
print(x.shape)
print(x.dtype)

# The most obvious differences between NumPy arrays and TensorFlow Tensors are:
# [!]Tensors are immutable (like tuple)
# [!]Tensor may be hosted in GPU memory while NumPy arrays are always backed by host memory

# Conversion
# TensorFlow operations automatically convert NumPy ndarrays to Tensors.
# NumPy operations automatically convert Tensors to NumPy ndarrays.
ndarray = np.ones([3, 3])

print("TensorFlow operations convert numpy arrays to Tensors automatically")
tensor = tf.multiply(ndarray, 42)
print(tensor)

print("And NumPy operations convert Tensors to numpy arrays automatically")
print(np.add(tensor, 1))

print("The .numpy() method explicitly converts a Tensor to a numpy array")
print(tensor.numpy())

# DATASETS API
ds_tensors = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6])

# Create a CSV file
import tempfile
_, filename = tempfile.mkstemp()

with open(filename, 'w') as f:
  f.write("""Line 1
Line 2
Line 3
  """)

ds_file = tf.data.TextLineDataset(filename)

ds_tensors = ds_tensors.map(tf.square).shuffle(2).batch(2)

ds_file = ds_file.batch(2)

'''
    When eager execution is enabled Dataset objects support iteration. If you're familiar with the use of Datasets
in TensorFlow graphs, note that there is no need for calls to Dataset.make_one_shot_iterator() or get_next() calls.
'''
print('Elements of ds_tensors:')
for x in ds_tensors:
    print(x)

print('\nElements in ds_file:')
for x in ds_file:
    print(x)


# GradientTape - as a tf way of computing derivatives(back propagation)
'''
TensorFlow provides the tf.GradientTape API for automatic differentiation - computing the gradient of a computation
with respect to its input variables. TensorFlow "records" all operations executed inside the context of a 
tf.GradientTape onto a "tape". TensorFlow then uses that tape and the gradients associated with each recorded 
operation to compute the gradients of a "recorded" computation using reverse mode differentiation.
'''
x = tf.ones((2, 2))

with tf.GradientTape() as t:
    t.watch(x)
    y = tf.reduce_sum(x)
    z = tf.multiply(y, y)

# Derivative of z with respect to the original input tensor x
# [!]Tensor record the reduce_sum and multiply operations, and compute the gradients
dz_dx = t.gradient(z, x)
for i in [0, 1]:
    for j in [0, 1]:
        assert dz_dx[i][j].numpy() == 8.0

# You can also request gradients of the output with respect to intermediate values
#  computed during a "recorded" tf.GradientTape context.
x = tf.ones((2, 2))

with tf.GradientTape() as t:
    t.watch(x)
    y = tf.reduce_sum(x)
    z = tf.multiply(y, y)
# Use the tape to compute the derivative of z with respect to the
# intermediate value y.
dz_dy = t.gradient(z, y)
assert dz_dy.numpy() == 8.0
# dz_dx = t.gradient(z, x) # this will raise a runtime error
# [!]GradientTape.gradient can only be called once on non-persistent tapes.
# To compute multiple gradients over the same computation, create a persistent gradient tape
x = tf.constant(3.0)
with tf.GradientTape(persistent=True) as t:
    t.watch(x)
    y = x * x
    z = y * y
dz_dx = t.gradient(z, x)  # 108.0 (4*x^3 at x = 3)
dy_dx = t.gradient(y, x)  # 6.0
del t  # Drop the reference to the tape

# Because tapes record operations as they are executed, Python control flow
# (using ifs and whiles for example) is naturally handled:
def f(x, y):
    output = 1.0
    for i in range(y):
        if i > 1 and i < 5:
            output = tf.multiply(output, x)
    return output

def grad(x, y):
    with tf.GradientTape() as t:
        t.watch(x)
        out = f(x, y)
    return t.gradient(out, x)

x = tf.convert_to_tensor(2.0)

assert grad(x, 6).numpy() == 12.0
assert grad(x, 5).numpy() == 12.0
assert grad(x, 4).numpy() == 4.0


# Operations inside of the GradientTape context manager are recorded for automatic differentiation
x = tf.Variable(1.0)  # Create a Tensorflow variable initialized to 1.0

with tf.GradientTape() as t:
    with tf.GradientTape() as t2:
        y = x * x * x
    # Compute the gradient inside the 't' context manager
    # which means the gradient computation is differentiable as well.
    dy_dx = t2.gradient(y, x)  # gradient manager watched this func(although the func itself is a gradient computation)
d2y_dx2 = t.gradient(dy_dx, x)
assert dy_dx.numpy() == 3.0
assert d2y_dx2.numpy() == 6.0
'''
To compare, the following will return none since computations happens out of gradient manager won't be watched
'''
with tf.GradientTape() as t:
    with tf.GradientTape() as t2:
        y = x * x * x
    # Compute the gradient inside the 't' context manager
    # which means the gradient computation is differentiable as well.
dy_dx = t2.gradient(y, x)  # gradient manager watched this func(although the func itself is a gradient computation)
d2y_dx2 = t.gradient(dy_dx, x)
assert dy_dx.numpy() == 3.0
assert d2y_dx2 == None






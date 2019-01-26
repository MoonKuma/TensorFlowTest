#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : eager_custom_training.py
# @Author: MoonKuma
# @Date  : 2019/1/26
# @Desc  : https://www.tensorflow.org/tutorials/eager/custom_training
# Using the GradientTape (as introduced in Eager_tf_basic), one only need to finish the forward propagation
import tensorflow as tf

tf.enable_eager_execution()

x = tf.zeros([10, 10])
x += 2
print(x)

# A Variable is an object which stores a value and, when used in a
#  TensorFlow computation, will implicitly read from this stored value

v = tf.Variable(1.0)
assert v.numpy() == 1.0

# Re-assign the value
v.assign(3.0)
assert v.numpy() == 3.0

# Use `v` in a TensorFlow operation like tf.square() and reassign
v.assign(tf.square(v))
assert v.numpy() == 9.0

'''
Computations using Variables are automatically traced when computing gradients. For Variables representing embeddings 
TensorFlow will do sparse updates by default, which are more computation and memory efficient.
Using Variables is also a way to quickly let a reader of your code know that this piece of state is mutable.
'''


class Model(object):
  def __init__(self):
    # Initialize variable to (5.0, 0.0)
    # In practice, these should be initialized to random values.
    self.W = tf.Variable(5.0)
    self.b = tf.Variable(0.0)

  def __call__(self, x):
    return self.W * x + self.b

model = Model()

assert model(3.0).numpy() == 15.0
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

# 使用numpy生成100个随机点
x_data = np.random.rand(100)
# 定义y_data 模型样本(真实值)，使用x_data 斜率0.1 偏置截距0.2
y_data = x_data * 0.1 + 0.2

# 构造一个线性模型
# 定义变量k,b 初始值是0
k = tf.Variable(0.)
#k = tf.Variable(0.5)
b = tf.Variable(0.)
#b = tf.Variable(1.1)
# 线性模型 y是预测值 斜率k 偏置截距b
y = x_data * k + b

# 定义一个二次代价函数,reduce_mean求平均值，square求平方
# y_data-y ： 真实值减去预测值（误差值）
loss = tf.reduce_mean(tf.square(y_data - y))

# 定义一个使用梯度下降法来进行训练的优化器GradientDescentOptimizer,学习率：0.2
optimizer = tf.train.GradientDescentOptimizer(0.2)

# 指定最小化代价函数，loss越小，预测值越接近真实值
train = optimizer.minimize(loss)

# 全局变量初始化
init = tf.global_variables_initializer()

with tf.Session() as sess:
    # 在sess中执行全局变量初始化
    sess.run(init)
    for step in range(401):
        sess.run(train)
        # 每运行20次，打印k和b的值
        if step % 20 == 0:
            print(step, sess.run([k, b]))

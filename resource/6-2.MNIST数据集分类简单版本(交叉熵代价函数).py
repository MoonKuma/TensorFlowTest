# -*- coding: utf-8 -*-
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

'''
mnist(手写数字)数据集:http://yann.lecun.com/exdb/mnist/
图片是采集的不同人手写从0到9的数字
6万(55000)张训练图片和1万张测试图片构成的，每张图片都是28*28=（784个像素）大小的黑白色构成，0到1的浮点数，白色：0 黑色：1
'''

# 载入mnist数据集
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# 总循环次数,把所有的图片训练steps_num次
steps_num = 21

# 每个批次的大小
batch_size = 100
# 训练样本总数量（55000）
print("mnist.train.num_examples : %s" % mnist.train.num_examples)
# 测试样本总数量（10000）
print("mnist.test.images length: %s" % len(mnist.test.images))
# 计算一共有多少批次，mnist.train.num_examples：训练样本总数量
n_batch = mnist.train.num_examples // batch_size
print("n_batch : %s" % n_batch)

# 定义真实样本输入x, 行：批次 列：28*28=784
x = tf.placeholder(tf.float32, [None, 784])
# 定义真实样本标签y 使用one-hot方式存储 列：0到9,10列
y = tf.placeholder(tf.float32, [None, 10])

# 创建简单的神经网络
# 权值，行：784 列：10
W = tf.Variable(tf.zeros([784, 10]))
# 偏置值 10
b = tf.Variable(tf.zeros([10]))
# 预测值 使用softmax激活函数
prediction = tf.nn.softmax(tf.matmul(x, W) + b)

'''
# 使用二次代价函数
loss = tf.reduce_mean(tf.square(y - prediction))
'''
# 使用交叉熵代价函数
#loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=prediction))

# 使用梯度下降法训练，学习率：0.2
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

# 结果存放在布尔型列表中 correct_prediction
# tf.argmax(y, 1): 获取y数组（一维张量）最大值所在位置
# tf.argmax(prediction, 1)：获取预测值prediction数组（一维张量）最大值所在位置
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
# 求准确率，tf.cast(correct_prediction, tf.float32)：将correct_prediction转换成float32类型，之后求平均值
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    # 在sess中执行全局变量初始化
    sess.run(init)
    # 把所有的图片训练steps_num次
    for step in range(steps_num):
        for _ in range(n_batch):
            # 获得batch_size个图片，图片数据保存在batch_xs，图片标签保存在batch_ys
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
        # 用测试集图片测试准确率, mnist.test.images:测试图片，mnist.test.labels：测试图片标签
        test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("step: %s, test accuracy: %s" % (step, test_acc))

'''
交叉熵代价函数							                  二次代价函数
step: 0, test accuracy: 0.8242							step: 0, test accuracy: 0.831
step: 1, test accuracy: 0.8946							step: 1, test accuracy: 0.8704
step: 2, test accuracy: 0.9026							step: 2, test accuracy: 0.8817
step: 3, test accuracy: 0.9049							step: 3, test accuracy: 0.8883
step: 4, test accuracy: 0.9076							step: 4, test accuracy: 0.8938
step: 5, test accuracy: 0.9108							step: 5, test accuracy: 0.8981
step: 6, test accuracy: 0.9125							step: 6, test accuracy: 0.9001
step: 7, test accuracy: 0.9133							step: 7, test accuracy: 0.9019
step: 8, test accuracy: 0.9152							step: 8, test accuracy: 0.9035
step: 9, test accuracy: 0.9165							step: 9, test accuracy: 0.9051
step: 10, test accuracy: 0.9176							step: 10, test accuracy: 0.9064
step: 11, test accuracy: 0.9184							step: 11, test accuracy: 0.9072
step: 12, test accuracy: 0.9191							step: 12, test accuracy: 0.9078
step: 13, test accuracy: 0.92							step: 13, test accuracy: 0.9095
step: 14, test accuracy: 0.9193							step: 14, test accuracy: 0.9093
step: 15, test accuracy: 0.9201							step: 15, test accuracy: 0.9103
step: 16, test accuracy: 0.9196							step: 16, test accuracy: 0.9115
step: 17, test accuracy: 0.9214							step: 17, test accuracy: 0.9126
step: 18, test accuracy: 0.9213							step: 18, test accuracy: 0.9133
step: 19, test accuracy: 0.9214							step: 19, test accuracy: 0.9132
step: 20, test accuracy: 0.9209							step: 20, test accuracy: 0.9138
'''
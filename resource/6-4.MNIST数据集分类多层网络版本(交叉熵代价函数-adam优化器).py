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

# 总循环次数,把所有的图片训练steps_num次(可以调整整体训练次数，使模型收敛)
steps_num = 51

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
# 定义神经元工作百分比(dropout值)
keep_prod = tf.placeholder(tf.float32)
# 定义学习率变量，初始值0.001，在训练过程中动态改变学习率
lr = tf.Variable(0.001, dtype=tf.float32)

# 创建多层神经网络
# 输入层1 接2000个神经元
# 权值，行：784 列：500(500个神经元)，调整初始化值:tf.truncated_normal(截断的正态分布值，stddev=0.1 标准差0.1)
W1 = tf.Variable(tf.truncated_normal([784, 500], stddev=0.1))
# 偏置值, 500, 调整初始化值为:0.1
b1 = tf.Variable(tf.zeros([500]) + 0.1)
# 定义L1的输出，激活函数使用tanh(双曲正切激活函数)
L1 = tf.nn.tanh(tf.matmul(x, W1) + b1)
# 定义神经元dropout
L1_drop = tf.nn.dropout(L1, keep_prod)

# 定义隐藏层2
W2 = tf.Variable(tf.truncated_normal([500, 300], stddev=0.1))
# 偏置值, 300, 调整初始化值为:0.1
b2 = tf.Variable(tf.zeros([300]) + 0.1)
# 定义L1的输出，激活函数使用tanh(双曲正切激活函数)
L2 = tf.nn.tanh(tf.matmul(L1_drop, W2) + b2)
# 定义神经元dropout
L2_drop = tf.nn.dropout(L2, keep_prod)

# 定义输出层3
W3 = tf.Variable(tf.truncated_normal([300, 10], stddev=0.1))
# 偏置值, 10, 调整初始化值为:0.1
b3 = tf.Variable(tf.zeros([10]) + 0.1)
# 预测值 使用softmax激活函数
prediction = tf.nn.softmax(tf.matmul(L2_drop, W3) + b3)

# 使用交叉熵（cross_entropy）代价函数, softmax_cross_entropy:softmax配合交叉熵
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=prediction))

# 使用Adam优化器训练,学习率：1e-4 10的负4次方：0.001，用lr变量在训练过程中动态改变  (一般使用Adam优化器，学习率都设置的比较小)
train_step = tf.train.AdamOptimizer(lr).minimize(loss)

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
        # 每迭代一个周期重新赋值lr(学习率): 0.001乘以0.95的step次方。（如果接近准确率最高时（loss值最低），学习率太大可能产生反复震荡的情况）
        sess.run(tf.assign(lr, 0.001 * (0.95 ** step)))
        for _ in range(n_batch):
            # 获得batch_size个图片，图片数据保存在batch_xs，图片标签保存在batch_ys
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # 执行训练 (调整dropout参数，1.0代表100%的神经元工作，0.85代表85%的神经元工作)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prod: 0.85})
        # 用测试集图片测试的准确率, mnist.test.images:测试图片，mnist.test.labels：测试图片标签
        test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prod: 1.0})
        # 用训练集图片测试的准确率, mnist.train.images:训练图片，mnist.train.labels：训练图片标签
        train_acc = sess.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels, keep_prod: 1.0})
        # 当前周期的准确率
        leaning_rate = sess.run(lr)
        print("step: %s, test accuracy: %s, train accuracy: %s, leaning rate: %s" % (
        step, test_acc, train_acc, leaning_rate))

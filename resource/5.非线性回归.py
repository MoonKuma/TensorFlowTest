# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

'''
matplotlib是python中强大的画图模块
python -m pip install -U pip setuptools 进行升级
安装: python -m pip install matplotlib
查看安装的所有模块: python -m pip list
'''

# 生成样本数据
# 使用numpy生成200个随机点
# np.linspace(-0.5, 0.5, 200):从-0.5到0.5产生200个均匀分布的点
# np.newaxis: 增加一个维度, x_data 就得到一个200行一列的数据
x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
# 生成干扰项: 随机值 概率分布的均值:0, 概率分布的标准差:0.02，形状和x_data一样
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = np.square(x_data) + noise

# 定义两个placeholder,
#  [None, 1]: 行不确定，1列, 是根据样本定义的
x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

# 输入x，得到y，和真实模型接近，代表模型构建成功
# 定义神经网络中间层（10个中间层）,
# 中间层权值 随机赋值，1行10列，1代表输入，10代表输出。
Weights_L1 = tf.Variable(tf.random_normal([1, 10]))
# 中间层偏置值 初始化为0
biases_L1 = tf.Variable(tf.zeros([1, 10]))
# 中间层信号总合 x矩阵输入值 和 权值做矩阵乘法 加上 偏置值 得到信号总合
Wx_plus_b_L1 = tf.matmul(x, Weights_L1) + biases_L1
# 定义中间层激活函数L1 中间层的输出，使用双曲正切函数作用于信号总合Wx_plus_b_L1
L1 = tf.nn.tanh(Wx_plus_b_L1)

# 定义神经网络输出层
# 输出层权值 随机赋值，10行1列(中间层10个神经元，输出1个神经元)
Weights_L2 = tf.Variable(tf.random_normal([10, 1]))
# 输出层偏置值 输出1个神经元，所以1个偏置值
biases_L2 = tf.Variable(tf.zeros([1, 1]))
# 输出层信号总合 L1中间层的输出，进入输出层
Wx_plus_b_L2 = tf.matmul(L1, Weights_L2) + biases_L2
# 定义输出层激活函数(预测结果)
prediction = tf.nn.tanh(Wx_plus_b_L2)

# 定义二次代价函数,reduce_mean求平均值，square求平方 y-prediction ： 真实值减去预测值（误差值）
loss = tf.reduce_mean(tf.square(y - prediction))
# 使用梯度下降法训练
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    # 在sess中执行全局变量初始化
    sess.run(tf.global_variables_initializer())
    # 训练2000次
    for _ in range(2000):
        sess.run(train_step, feed_dict={x: x_data, y: y_data})

    # 获得预测值
    prediction_value = sess.run(prediction, feed_dict={x: x_data})
    # 查看预测结果(画图表示)
    plt.figure();
    # 散点图
    plt.scatter(x_data, y_data);
    plt.plot(x_data, prediction_value, 'r-', lw=5)
    plt.show()

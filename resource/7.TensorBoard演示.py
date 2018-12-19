# -*- coding: utf-8 -*-
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
'''
启动TensorBorad命令:
1.移动到当前盘符下
2.输入启动命令: tensorboard --host=192.168.8.174 --port=6006 --logdir=E:\PycharmProjects\TensorFlow\Test\logs
   默认端口:6006
'''


# 参数概要方法
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        # 平均值
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        # 标准差
        tf.summary.scalar('stddev', stddev)
        # 最大值
        tf.summary.scalar('max', tf.reduce_max(var))
        # 最小值
        tf.summary.scalar('min', tf.reduce_min(var))
        # 直方图
        tf.summary.histogram('histogram', var)


# 载入mnist数据集
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# 总循环次周期数,把所有的图片训练steps_num次(可以调整整体训练次数，使模型收敛)
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

# 定义TensorBorad命名空间
with tf.name_scope('input'):
    # 定义真实样本输入x, 行：批次 列：28*28=784
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    # 定义真实样本标签y 使用one-hot方式存储 列：0到9,10列
    y = tf.placeholder(tf.float32, [None, 10], name='y-input')
    # 定义神经元工作百分比(dropout值)
    keep_prod = tf.placeholder(tf.float32, name='keep_prod')
    # 定义学习率变量，初始值0.001(1e-3)，在训练过程中动态改变学习率
    lr = tf.Variable(1e-3, dtype=tf.float32, name='lr')

with tf.name_scope('Layer1'):
    with tf.name_scope('Weight1'):
        # 创建多层神经网络
        # 输入层1 接2000个神经元
        # 权值，行：784 列：500(500个神经元)，调整初始化值:tf.truncated_normal(截断的正态分布值，stddev=0.1 标准差0.1)
        W1 = tf.Variable(tf.truncated_normal([784, 500], stddev=0.1), name='W1')
        variable_summaries(W1)
    with tf.name_scope('biases1'):
        # 偏置值, 500, 调整初始化值为:0.1
        b1 = tf.Variable(tf.zeros([500]) + 0.1, name='b1')
        variable_summaries(b1)
    with tf.name_scope('wx_plus_b1'):
        wx_plus_b1 = tf.matmul(x, W1) + b1
    with tf.name_scope('L1_out'):
        # 定义L1的输出，激活函数(Activation function)使用tanh(双曲正切激活函数)
        L1_out = tf.nn.tanh(wx_plus_b1)
    with tf.name_scope('L1_drop'):
        # 定义神经元dropout
        L1_drop = tf.nn.dropout(L1_out, keep_prod)

with tf.name_scope('Layer2'):
    with tf.name_scope('Weight2'):
        # 定义隐藏层2
        W2 = tf.Variable(tf.truncated_normal([500, 300], stddev=0.1), name='W2')
        variable_summaries(W2)
    with tf.name_scope('biases2'):
        # 偏置值, 300, 调整初始化值为:0.1
        b2 = tf.Variable(tf.zeros([300]) + 0.1, name='b2')
        variable_summaries(b2)
    with tf.name_scope('wx_plus_b2'):
        wx_plus_b2 = tf.matmul(L1_drop, W2) + b2
    with tf.name_scope('L2_out'):
        # 定义L2的输出，激活函数使用tanh(双曲正切激活函数)
        L2_out = tf.nn.tanh(wx_plus_b2)
    with tf.name_scope('L2_drop'):
        # 定义神经元dropout
        L2_drop = tf.nn.dropout(L2_out, keep_prod)

with tf.name_scope('Layer3'):
    with tf.name_scope('Weight3'):
        # 定义输出层3
        W3 = tf.Variable(tf.truncated_normal([300, 10], stddev=0.1), name='W3')
        variable_summaries(W3)
    with tf.name_scope('biases3'):
        # 偏置值, 10, 调整初始化值为:0.1
        b3 = tf.Variable(tf.zeros([10]) + 0.1, name='b3')
        variable_summaries(b3)
    with tf.name_scope('wx_plus_b3'):
        wx_plus_b3 = tf.matmul(L2_drop, W3) + b3
    with tf.name_scope('prediction'):
        # 预测值 使用softmax激活函数
        prediction = tf.nn.softmax(wx_plus_b3)

with tf.name_scope('loss'):
    # 使用交叉熵（cross_entropy）代价函数, softmax_cross_entropy:softmax配合交叉熵
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=prediction))
    tf.summary.scalar('loss', loss)

with tf.name_scope('train'):
    # 使用Adam优化器训练,学习率：1e-3：10的负3次方：0.001，用lr变量在训练过程中动态改变学习率(一般使用Adam优化器，学习率都设置的比较小)
    # Adam:自适应时刻估计算法（Adaptive Moment Estimation）能计算每个参数的自适应学习率。收敛速度更快，学习效果更为有效
    train_step = tf.train.AdamOptimizer(lr).minimize(loss)

with tf.name_scope('init'):
    # 初始化变量
    init = tf.global_variables_initializer()

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        # 结果存放在布尔型列表中 correct_prediction
        # tf.argmax(y, 1): 获取y数组（一维张量）最大值所在位置
        # tf.argmax(prediction, 1)：获取预测值prediction数组（一维张量）最大值所在位置
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
    with tf.name_scope('accuracy_result'):
        # 求准确率，tf.cast(correct_prediction, tf.float32)：将correct_prediction转换成float32类型，之后求平均值
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

# 合并所有的指标
merged = tf.summary.merge_all()

with tf.Session() as sess:
    # 在sess中执行全局变量初始化
    sess.run(init)
    # 指定summary日志存放文件夹:logs(这里使用相对路径，也可以用就绝对路径),写入图信息
    writer = tf.summary.FileWriter('logs', sess.graph)
    # 把所有的图片训练steps_num次
    for step in range(steps_num):
        # 每迭代一个周期重新赋值lr(学习率): 0.001乘以0.95的step次方。（如果接近准确率最高时（loss值最低），学习率太大可能产生反复震荡的情况）
        sess.run(tf.assign(lr, 0.001 * (0.95 ** step)))
        for batch in range(n_batch):
            # 获得batch_size个图片，图片数据保存在batch_xs，图片标签保存在batch_ys
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # 执行训练 (调整dropout参数，1.0代表100%的神经元工作，0.85代表85%的神经元工作)
            summary, _ = sess.run([merged, train_step], feed_dict={x: batch_xs, y: batch_ys, keep_prod: 0.85})

        # 写入summary信息和周期step
        writer.add_summary(summary, step)
        # 用测试集图片测试的准确率, mnist.test.images:测试图片，mnist.test.labels：测试图片标签
        test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prod: 1.0})
        # 用训练集图片测试的准确率, mnist.train.images:训练图片，mnist.train.labels：训练图片标签
        train_acc = sess.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels, keep_prod: 1.0})
        # 当前周期的准确率
        leaning_rate = sess.run(lr)
        print("step: %s, test accuracy: %s, train accuracy: %s, leaning rate: %s" % (
            step, test_acc, train_acc, leaning_rate))

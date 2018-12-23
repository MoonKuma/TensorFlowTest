# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 载入数据集
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 输入图片是28*28（784），对于一张图片，每次输入一行，一行有28个数据
n_inputs = 28

# 把一张图片看成一个序列，这个序列一共有28行，相当于有28次输入
max_time = 28

# 隐层神经元个数，在LSTM网络中叫block
lstm_size = 128

# 10个分类
n_classes = 10

# 每批次50个样本
batch_size = 50

# 计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size

# 这里的None表示第一个维度可以是任意的长度
x = tf.placeholder(tf.float32, [None, 784])
# 正确的标签
y = tf.placeholder(tf.float32, [None, 10])

batch_size_p = tf.placeholder(tf.int32, [], name='batch_size')

# 定义神经元工作百分比(dropout值)
keep_prod = tf.placeholder(tf.float32)

# 初始化权值，lstm_size:行是100,相当于中间隐藏层block的个数，n_classes：列是10（10个分类），只定义了一个隐藏层
weights = tf.Variable(tf.truncated_normal([lstm_size, n_classes], stddev=0.1))
# 初始化偏置值，因为只有10个分类，所以偏置值也是n_classes：10个
biases = tf.Variable(tf.constant(0.1, shape=[n_classes]))


# 定义RNN网络
def RNN(X):
    # 做数据转换，X:50行（batch_size），784列 --》 [50个样本（-1), 每个样本28行（max_time）, 每行28列（n_inputs）]   inputs=[batch_size, max_time, n_inputs]
    inputs = tf.reshape(X, [-1, max_time, n_inputs])

    # 定义单层LSTM基本CELL，中间隐藏层block数：lstm_size
    lstm_cell = tf.contrib.rnn.LSTMBlockCell(num_units=lstm_size)
    # rnn DropoutWrapper
    lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, output_keep_prob=keep_prod)
    # An initial state for the RNN and call lstm_cell.zero_state Return zero-filled state tensor(s).
    init_state = lstm_cell.zero_state(batch_size_p, dtype=tf.float32)
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, inputs, initial_state=init_state, dtype=tf.float32)
    final_hidden_state = final_state[1]

    # FusedRNNCell 针对mnist数据如何定义输入shape？ 目前方式不对 还需研究！！
    # FusedRNNCell instance should be time-major
    # inputs = tf.reshape(X, [max_time, -1, n_inputs])
    # fused_rnn_cell = tf.contrib.rnn.LSTMBlockFusedCell(num_units=lstm_size)
    # outputs, final_state = fused_rnn_cell(inputs, dtype=tf.float32)
    # final_hidden_state = final_state[1]

    # 创建多层（堆叠两层RNNCell)的LSTMCells，第一层 （lstm_size/2） 个 cell，第二层 lstm_size 个 cell。
    # rnn_layers = [tf.nn.rnn_cell.LSTMCell(num_units=size) for size in [lstm_size / 2, lstm_size]]
    # multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
    # outputs, final_state = tf.nn.dynamic_rnn(cell=multi_rnn_cell, inputs=inputs, dtype=tf.float32)
    # final_hidden_state = final_state[1][1]

    """
    tf.nn.dynamic_rnn 运行LSTM网络，得到输出结果
    经过LSTM的运行,返回两个输出结果，output 和 final_state；
    output：里面包含了所有时刻的输出 H；
    final_state：里面包含了最后一个时刻的输出 C（cell state） 和 H（hidden state）
      final_state[0]:是最后一个时刻的 cell state 输出
      final_state[1]:是最后一个时刻的 hidden state 输出
    所以如果想用dynamic_rnn得到最终状态的输出，只需要最后一个时刻的状态输出，直接使用final_state[1]就可以了
    (其实final_state[1] 和 output最后一个时刻的值一样)。
    """
    results = tf.nn.softmax(tf.matmul(final_hidden_state, weights) + biases)
    return results


# 调用RNN函数,计算返回结果
prediction = RNN(x)
# 损失函数
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
# 使用AdamOptimizer进行优化
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# 结果存放在一个布尔型列表中
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))  # argmax返回一维张量中最大的值所在的位置
# 求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # 把correct_prediction变为float32类型
# 初始化
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(100):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, batch_size_p: batch_size, keep_prod: 0.75})
        train_acc = sess.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels, batch_size_p: mnist.train.num_examples, keep_prod: 1.0})
        test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, batch_size_p: mnist.test.num_examples, keep_prod: 1.0})
        print("Iter " + str(epoch) + ", Train Accuracy= " + str(train_acc * 100) + ", Test Accuracy= " + str(test_acc * 100))

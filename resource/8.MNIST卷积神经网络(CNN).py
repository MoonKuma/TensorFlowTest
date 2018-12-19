# -*- coding: utf-8 -*-
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.python import debug as tf_debug

'''
启动TensorBorad命令:
1.移动到当前盘符下
2.输入启动命令: tensorboard --host=192.168.8.174 --port=6006 --debugger_port=6007 --logdir=E:\PycharmProjects\TensorFlow\Test\projector_cnn
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


# 初始化权值
def weight_variable(shape, name):
    # 生成一个截断的正态分布
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


# 初始化偏置
def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


# 卷积层
def conv2d(x, W):
    # x input tensor of shape `[batch, in_height, in_width, in_channels]`
    # W filter / kernel tensor of shape [filter_height, filter_width, in_channels, out_channels]
    # `strides[0] = strides[3] = 1 必须填：1, strides[1]代表x方向的步长，strides[2]代表y方向的步长
    # padding: A `string` from: `"SAME", "VALID"`
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# 池化层:2x2的池化层
def max_pool_2x2(x):
    # ksize [1,x,y,1]
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 文件路径
DIR = "E:/PycharmProjects/TensorFlow/Test/"
# 载入mnist数据集
mnist = input_data.read_data_sets(DIR + "MNIST_data", one_hot=True)

# 每个批次的大小
batch_size = 100

# 总训练steps_num次
# steps_num = 1001
steps_num = mnist.train.num_examples // batch_size * 51
print("total train steps: %s" % steps_num)

# 训练样本总数量（55000）
print("mnist.train.num_examples : %s" % mnist.train.num_examples)
# 测试样本总数量（10000）
print("mnist.test.images length: %s" % len(mnist.test.images))

# 图片数量
# 全量载入(10000个测试样本数量)
image_num = len(mnist.test.images)
# image_num = 10

# 载入图片
# tf.stack:打包，实例：x = tf.constant([1, 4]) y = tf.constant([2, 5]) z = tf.constant([3, 6]) tf.stack([x, y, z])  # [[1, 4], [2, 5], [3, 6]]
embedding = tf.Variable(tf.stack(mnist.test.images[:image_num]), trainable=False, name='embedding')

# 定义TensorBorad命名空间,定义输入参数的placeholder
with tf.name_scope('input'):
    # 定义真实样本输入x, 行：批次 列：28*28=784
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    # 定义真实样本标签y 使用one-hot方式存储 列：0到9,10列
    y = tf.placeholder(tf.float32, [None, 10], name='y-input')
    # 定义神经元工作百分比(dropout值)
    keep_prob = tf.placeholder(tf.float32, name='keep_prod')
    with tf.name_scope('x_image'):
        # 改变x的格式转为4D的向量[batch, in_height, in_width, in_channels]`
        x_image = tf.reshape(x, [-1, 28, 28, 1], name='x_image')
    # TensorBoard中显示图片
    with tf.name_scope('input_reshape'):
        # 转换图片数据，[-1, 28, 28, 1] ： 图片数量（-1代表不确定值）,28 行,28 列，1 维度（黑白的维度1，彩色的维度3）
        image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
        # 放入10张图片
        tf.summary.image('input_image', image_shaped_input, 10)

# 定义卷积层1
with tf.name_scope('Conv1'):
    # 初始化第一个卷积层的权值和偏置
    with tf.name_scope('W_conv1'):
        # 定义卷积核是5*5的采样窗口，32个卷积核从1个平面抽取特征
        W_conv1 = weight_variable([5, 5, 1, 32], name='W_conv1')
        variable_summaries(W_conv1)
    with tf.name_scope('b_conv1'):
        # 每一个卷积核一个偏置值,32个卷积核
        b_conv1 = bias_variable([32], name='b_conv1')
        variable_summaries(b_conv1)

    # 把x_image和权值向量进行卷积，再加上偏置值，然后应用于relu激励函数
    with tf.name_scope('conv2d_1'):
        conv2d_1 = conv2d(x_image, W_conv1) + b_conv1
        variable_summaries(conv2d_1)
    with tf.name_scope('relu'):
        h_conv1 = tf.nn.relu(conv2d_1)
    with tf.name_scope('h_pool1'):
        # 进行max-pooling
        h_pool1 = max_pool_2x2(h_conv1)

# 定义卷积层2
with tf.name_scope('Conv2'):
    # 初始化第二个卷积层的权值和偏置
    with tf.name_scope('W_conv2'):
        # 5*5的采样窗口，64个卷积核从32个平面(上一层的输出32个卷积核)抽取特征
        W_conv2 = weight_variable([5, 5, 32, 64], name='W_conv2')
        variable_summaries(W_conv2)
    with tf.name_scope('b_conv2'):
        # 每一个卷积核一个偏置值,64个卷积核
        b_conv2 = bias_variable([64], name='b_conv2')
        variable_summaries(b_conv2)

    # 把h_pool1和权值向量进行卷积，再加上偏置值，然后应用于relu激活函数
    with tf.name_scope('conv2d_2'):
        conv2d_2 = conv2d(h_pool1, W_conv2) + b_conv2
        variable_summaries(conv2d_2)
    with tf.name_scope('relu'):
        h_conv2 = tf.nn.relu(conv2d_2)
    with tf.name_scope('h_pool2'):
        # 进行max-pooling
        h_pool2 = max_pool_2x2(h_conv2)

'''
28*28的图片第一次卷积后还是28*28，第一次池化后变为14*14
第二次卷积后为14*14，第二次池化后变为了7*7
经过上面操作后得到64张7*7的平面
'''

# 定义第一个全连接层
with tf.name_scope('fc1'):
    # 初始化第一个全连接层的权值，偏置值
    with tf.name_scope('W_fc1'):
        # 上一层有7*7*64个神经元，全连接层设置1024个神经元
        W_fc1 = weight_variable([7 * 7 * 64, 1024], name='W_fc1')
        variable_summaries(W_fc1)
    with tf.name_scope('b_fc1'):
        # 偏置值1024个神经元
        b_fc1 = bias_variable([1024], name='b_fc1')
        variable_summaries(b_fc1)
    with tf.name_scope('h_pool2_flat'):
        # 把池化层2的输出扁平化为1维
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64], name='h_pool2_flat')
        variable_summaries(h_pool2_flat)
    with tf.name_scope('Wx_plus_b1'):
        # 计算第一个全连接层的输出
        Wx_plus_b1 = tf.matmul(h_pool2_flat, W_fc1) + b_fc1
        variable_summaries(Wx_plus_b1)
    with tf.name_scope('relu'):
        '''
       # relu激励函数
       h_fc1 = tf.nn.relu(Wx_plus_b1)
       '''
        # 使用softsign激励函数
        h_fc1 = tf.nn.softsign(Wx_plus_b1)
    with tf.name_scope('h_fc1_drop'):
        # dropout
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, name='h_fc1_drop')

# 定义第二个全连接层
with tf.name_scope('fc2'):
    # 初始化第二个全连接层的权值，偏置值
    with tf.name_scope('W_fc2'):
        # 上一层有1024个神经元，输出10个神经元
        W_fc2 = weight_variable([1024, 10], name='W_fc2')
        variable_summaries(W_fc2)
    with tf.name_scope('b_fc2'):
        b_fc2 = bias_variable([10], name='b_fc2')
        variable_summaries(b_fc2)
    with tf.name_scope('Wx_plus_b2'):
        # 计算第二个全连接层的输出
        Wx_plus_b2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        variable_summaries(Wx_plus_b2)
    with tf.name_scope('softmax'):
        # 预测值 使用softmax激励函数
        prediction = tf.nn.softmax(Wx_plus_b2)

# 定义交叉熵代价函数
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=prediction), name='cross_entropy')
    tf.summary.scalar('loss', loss)

# 使用AdamOptimizer进行优化
with tf.name_scope('train'):
    # 使用Adam优化器训练,学习率：1e-4：10的负4次方：0.0001，用lr变量在训练过程中动态改变学习率(一般使用Adam优化器，学习率都设置的比较小)
    # Adam:自适应时刻估计算法（Adaptive Moment Estimation）能计算每个参数的自适应学习率。收敛速度更快，学习效果更为有效
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

# 求准确率
with tf.name_scope('accuracy'):
    # 结果存放在布尔型列表中 correct_prediction
    # tf.argmax(y, 1): 获取y数组（一维张量）最大值所在位置
    # tf.argmax(prediction, 1)：获取预测值prediction数组（一维张量）最大值所在位置
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
    # 求准确率，tf.cast(correct_prediction, tf.float32)：将correct_prediction转换成float32类型，之后求平均值
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

# 合并所有的summary指标
merged = tf.summary.merge_all()

# 开启会话
with tf.Session() as sess:
    # 初始化变量
    sess.run(tf.global_variables_initializer())

    # 生成metadata文件
    if tf.gfile.Exists(DIR + 'projector_cnn/projector/'):
        #  tf.gfile.DeleteRecursively:删除目录下的所有文件
        tf.gfile.DeleteRecursively(DIR + 'projector_cnn/projector')
    tf.gfile.MakeDirs(DIR + 'projector_cnn/projector/')
    with open(DIR + 'projector_cnn/projector/metadata.tsv', 'w') as f:
        labels = sess.run(tf.argmax(mnist.test.labels[:], 1))
        for i_image_num in range(image_num):
            f.write(str(labels[i_image_num]) + '\n')

    if tf.gfile.Exists(DIR + 'projector_cnn/test/'):
        #  tf.gfile.DeleteRecursively:删除目录下的所有文件
        tf.gfile.DeleteRecursively(DIR + 'projector_cnn/test')
    tf.gfile.MakeDirs(DIR + 'projector_cnn/test/')

    # 定义saver
    saver = tf.train.Saver()
    # 指定训练summary日志存放文件夹
    projector_writer = tf.summary.FileWriter(DIR + 'projector_cnn/projector', sess.graph)

    config = projector.ProjectorConfig()
    embed = config.embeddings.add()
    embed.tensor_name = embedding.name
    embed.metadata_path = DIR + 'projector_cnn/projector/metadata.tsv'
    embed.sprite.image_path = DIR + 'projector_cnn/data/mnist_10k_sprite.png'
    embed.sprite.single_image_dim.extend([28, 28])
    projector.visualize_embeddings(projector_writer, config)

    # 记录测试集运行数据log的writer
    test_writer = tf.summary.FileWriter(DIR + 'projector_cnn/test', sess.graph)

    for step in range(steps_num):
        # 获取训练样本
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)

        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        # 执行训练 (调整dropout参数，1.0代表100%的神经元工作，0.75代表75%的神经元工作)
        summary, _ = sess.run([merged, train_step], feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.75}, options=run_options, run_metadata=run_metadata)

        # 写入run_metadata信息和周期step
        projector_writer.add_run_metadata(run_metadata, 'step%03d' % step)
        # 写入训练summary信息和周期step
        projector_writer.add_summary(summary, step)

        # 记录测试集计算的参数
        test_batch_xs, test_batch_ys = mnist.test.next_batch(batch_size)
        test_summary = sess.run(merged, feed_dict={x: test_batch_xs, y: test_batch_ys, keep_prob: 1.0})
        # 写入测试summary信息和周期step
        test_writer.add_summary(test_summary, step)

        if step % 100 == 0:
            test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
            train_acc = sess.run(accuracy, feed_dict={x: mnist.train.images[:10000], y: mnist.train.labels[:10000],
                                                      keep_prob: 1.0})
            print("step: %s, test accuracy: %s, train accuracy: %s" % (step, test_acc, train_acc))

    # 保存训练模型
    saver.save(sess, DIR + 'projector_cnn/projector/model.ckpt', global_step=steps_num)
    projector_writer.close()
    test_writer.close()

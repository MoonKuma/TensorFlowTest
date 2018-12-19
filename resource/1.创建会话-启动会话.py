# -*- coding: utf-8 -*-
import tensorflow as tf
import os

# tensorflow 版本 1.11.0(tf.__version__)，python 版本 3.6.5(python -V)，anaconda(conda) 版本 4.5.11 (conda -V)
# 解决Tensorflow 使用时cpu编译不支持警告
# 使用TensorFlow模块时，弹出错误Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX AVX2
# 1.忽略这个警告
# 这是默认的显示等级，显示所有信息
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'
# 只显示 warning 和 Error   
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
# 只显示 Error
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
# 2.彻底解决，换成支持cpu用AVX2编译的TensorFlow版本。
# https://github.com/lakshayg/tensorflow-build
# https://github.com/fo40225/tensorflow-windows-wheel
# https://github.com/fo40225/tensorflow-windows-wheel/tree/master/1.11.0
# https://github.com/fo40225/tensorflow-windows-wheel/tree/master/1.11.0/py36/CPU/avx2
# 安装：
# pip install --ignore-installed --upgrade C:\Users\7q\PycharmProjects\TensorFlowTest\resource\tensorflow-1.11.0-cp36-cp36m-win_amd64.whl

# 查看tensorflow版本
print(tf.__version__)

# 查看tensorflow安装路径
print(tf.__path__)

# 创建常量op(operation操作),矩阵
m1 = tf.constant([[3, 3]])
m2 = tf.constant([[2], [3]])
# 创建一个计算的op，矩阵乘法，传入m1，m2 (3*2+3*3)
product = tf.matmul(m1, m2)
# 这里的结果并不是一个真正计算后的结果 15，而是一个 Tensor(张量),因为这些op 需要在一个会话的图 (graph)中进行
print(product)

# 定义一个会话后,会创建一个默认的图(graph)
sess = tf.Session()
# 调用sess(会话)的run方法来执行矩阵乘法op(product)
# run(product)触发了图中3个op
result = sess.run(product)
print(result)
# 关闭会话
sess.close()

# 定义会话的简便写法(这样定义的会话，执行完成后就不用显示的调用会话关闭方法sess.close()，会自动关闭会话)
with tf.Session() as sess:
    # sess = tf.Session()
    # 调用sess(会话)的run方法来执行矩阵乘法op(product)
    # run(product)触发了图中3个op
    result = sess.run(product)
    print(result)

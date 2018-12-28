# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

'''
对数几率回归是回答是或否的问题，也既二分类问题,只有发生“y=1”（正例）与不发生“y=0”（反例）两种结果（分类问题）.
sigmiod函数：给定一个计算输出，这个函数返回的是一个'yes'的概率(从0到1的值)
数据集：使用泰坦尼克数据集(http://www.kaggle.com/c/titanic) Titanic_data,预测泰坦尼克号乘客的生存概率

数据列说明：
PassengerId：乘客id（没有实际含义，去掉）
Survived：生存情况：存活（1），死亡（0）
Pclass：客舱等级(1=1st；2=2nd；3=3rd)
Name：乘客姓名(没有实际含义，去掉)
Sex：乘客性别
Age：乘客年龄
SibSp：在船兄弟姐妹数/配偶数
Parch：在船父母/子女数
Ticket：船票编号(没有实际含义，去掉)
Fare:船票价格
Cabin：客舱号
Embarked：登船港口
'''

training_epochs = 25000

batch_size = 500

# 读取训练集数据
data = pd.read_csv(r'Titanic_data/train.csv')
print(data.columns)

# 提取有用特征
data = data[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']]

# 相关特殊字段值的预处理数值化
# age列的nan值填充成所有乘客的均值
data['Age'] = data['Age'].fillna(np.uint16(data['Age'].mean()))

# Cabin客舱号,pd.factorize函数可以将Series中的标称型数据映射称为一组数字，相同的标称型映射为相同的数字。
# factorize函数的返回值是一个tuple（元组），元组中包含两个元素。
# 第一个元素是一个array，其中的元素是标称型元素映射为的数字；
# 第二个元素是Index类型，其中的元素是所有标称型元素，没有重复。
data['Cabin'] = pd.factorize(data.Cabin)[0]

# 对data数据集中所有nan值填充为0
data.fillna(0, inplace=True)

# sex 如果是'male'填充1 否则填充0
data['Sex'] = [1 if x == 'male' else 0 for x in data.Sex]

# Pclass客舱等级(1=1st；2=2nd；3=3rd),使用独热编码，防止线性关系,astype 转化数据类型
data['p1'] = np.array(data['Pclass'] == 1).astype(np.int8)
data['p2'] = np.array(data['Pclass'] == 2).astype(np.int8)
data['p3'] = np.array(data['Pclass'] == 3).astype(np.int8)
del data['Pclass']

# Embarked登船港口,unique方法查看Embarked的值 ['S' 'C' 'Q' 0],同样使用独热编码
print(data.Embarked.unique())
data['e1'] = np.array(data['Embarked'] == 'S').astype(np.int8)
data['e2'] = np.array(data['Embarked'] == 'C').astype(np.int8)
data['e3'] = np.array(data['Embarked'] == 'Q').astype(np.int8)
del data['Embarked']

# 获得训练数据
data_train = data[['Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'p1', 'p2', 'p3', 'e1', 'e2', 'e3']]
# 获得训练数据的标签,reshape 转换成len(data)行 ,1列
data_target = data['Survived'].values.reshape(len(data), 1)
print(np.shape(data_train), np.shape(data_target))

# 读取测试集数据并预处理
data_test = pd.read_csv(r'Titanic_data/test.csv')
data_test = data_test[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']]
data_test['Age'] = data_test['Age'].fillna(np.uint16(data_test['Age'].mean()))
data_test['Cabin'] = pd.factorize(data_test.Cabin)[0]
data_test.fillna(0, inplace=True)
data_test['Sex'] = [1 if x == 'male' else 0 for x in data_test.Sex]
data_test['p1'] = np.array(data_test['Pclass'] == 1).astype(np.int8)
data_test['p2'] = np.array(data_test['Pclass'] == 2).astype(np.int8)
data_test['p3'] = np.array(data_test['Pclass'] == 3).astype(np.int8)
del data_test['Pclass']
data_test['e1'] = np.array(data_test['Embarked'] == 'S').astype(np.int8)
data_test['e2'] = np.array(data_test['Embarked'] == 'C').astype(np.int8)
data_test['e3'] = np.array(data_test['Embarked'] == 'Q').astype(np.int8)
del data_test['Embarked']

# 读取测试集结果数据并预处理
test_lable = pd.read_csv(r'Titanic_data/gender_submission.csv')
# 只取出测试集的Survived信息，并排成1列
test_lable = np.reshape(test_lable.Survived.values.astype(np.float32), (test_lable.shape[0], 1))
print(test_lable.shape)


x = tf.placeholder(tf.float32, shape=[None, data_train.shape[1]])
y = tf.placeholder(tf.float32, shape=[None, 1])

weight = tf.Variable(tf.random_normal([data_train.shape[1], 1]))
biases = tf.Variable(tf.random_normal([1]))
output = tf.matmul(x, weight) + biases
# 将sigmoid作为激励函数处理output(sigmoid适合用于二分类问题)，将大于0.5的视作一类，小于的视作一类。cast相当于转化把值为0或1
prediction = tf.cast(tf.sigmoid(output) > 0.5, tf.float32)
# sigmoid_cross_entropy_with_logits是专用于为sigmoid计算损失值的函数。reduce_mean计算平均值，这里把向量值转化为标量值。label是目标值
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=output))
train_step = tf.train.AdamOptimizer().minimize(loss)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, y), tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())
loss_train = []
acc_train = []
acc_test = []

for i in range(training_epochs):
    # np.random.permutation: 随机打乱原来的元素顺序
    # 函数shuffle与permutation都是对原来的数组进行重新洗牌（即随机打乱原来的元素顺序）；
    # 区别在于shuffle直接在原来的数组上进行操作，改变原来数组的顺序，无返回值。
    # 而permutation不直接在原来的数组上进行操作，而是返回一个新的打乱顺序的数组，并不改变原来的数组。
    index = np.random.permutation(len(data_target))
    # 将训练集打成乱序，防止过拟合
    data_target = data_target.take(index)
    data_train = data_train.take(index)

    for n in range(len(data_target) // batch_size + 1):
        batch_xs = data_train[n * batch_size:(n * batch_size + batch_size)]
        batch_ys = data_target[n * batch_size:(n * batch_size + batch_size)]
        batch_ys = batch_ys.reshape(len(batch_ys), 1)
        sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
    if i % 1000 == 0:
        train_loss_temp = sess.run(loss, feed_dict={x: batch_xs, y: batch_ys})
        loss_train.append(train_loss_temp)
        train_acc_tmp = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys})
        acc_train.append(train_acc_tmp)
        test_acc_tmp = sess.run(accuracy, feed_dict={x: data_test, y: test_lable})
        acc_test.append(test_acc_tmp)
        print(i, train_loss_temp, train_acc_tmp, test_acc_tmp)

# plt.plot:根据loss_train画黑色(-k)的折线图
plt.plot(loss_train, 'k-')
plt.title('loss_train')
plt.show()  # 显示图表

plt.plot(acc_train, 'b-', label='train_acc')
plt.plot(acc_test, 'r--', label='test_acc')
# 画标题
plt.title('train and test accuracy')
# 画图例
plt.legend()
# 显示图表
plt.show()

# 将最终的预测结果输出成csv文件
# 得到最终预测结果
res = sess.run(prediction, feed_dict={x: data_test})
# 把结果转换层DataFrame形式，方便输出
res_in_df = pd.DataFrame(res)
res_in_df.columns = ['res']
test = pd.read_csv(r'Titanic_data/test.csv')
res_in_df['PassengerId'] = test['PassengerId']
res_in_df['Survived'] = res_in_df['res'].astype(int)
del res_in_df['res']
# 输出成csv文件
res_in_df.to_csv(r'Titanic_data/Result.csv', index=0)

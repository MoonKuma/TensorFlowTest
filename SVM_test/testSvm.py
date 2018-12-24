#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : testSvm.py
# @Author: MoonKuma
# @Date  : 2018/9/7
# @Desc  : test Svm using an internet example

import numpy as npy
from sklearn import svm
from sklearn.model_selection import train_test_split


def iris_type(s):
    it = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    return it[s]


def show_accuracy(y_predict, y_real, note):
    if len(y_predict) != len(y_real) or len(y_predict) == 0:
        msg = 'require equal and non-zero length from both list, while len(y_predict)', len(y_predict), ' and len(y_real)', len(y_real)
        print(msg)
        return
    count_total = 0
    count_correct = 0
    for i in range(0, len(y_predict)):
        count_total += 1
        if y_predict[i] == y_real[i]:
            count_correct += 1
    report = note + ':' + str(count_correct*100/count_total) + '%'
    return report

file_train = 'test_data/iris_training.csv'
data = npy.loadtxt(file_train, dtype=float, delimiter=',', skiprows=1)
x_train, y_train = npy.split(data, (4,), axis=1)

file_test = 'test_data/iris_test.csv'
data = npy.loadtxt(file_test, dtype=float, delimiter=',', skiprows=1)
x_test, y_test = npy.split(data, (4,), axis=1)


# x = x[:, :4] # as dimension increased, the accuracy for prediction (both train and test) adds up, which means adding dimension could still benefit the prediction
# x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.6)
clf = svm.SVC()
clf.fit(x_train, y_train.ravel())
'''


'''

y_hat_train = clf.predict(x_train)
y_hat_test = clf.predict(x_test)
print("***")
print(show_accuracy(y_hat_train, y_train, 'train-predict_auto'))
print("***")
print(show_accuracy(y_hat_test, y_test, 'test-predict_auto'))


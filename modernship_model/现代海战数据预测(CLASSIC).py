#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : 现代海战数据预测(CLASSIC).py
# @Author: MoonKuma
# @Date  : 2018/12/19
# @Desc  : Linear regression method

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from modenship_data.get_local import local_path


def show_accuracy(y_predict, y_real, note):
    if len(y_predict) != len(y_real) or len(y_predict) == 0:
        msg = 'require equal and non-zero length from both list, while len(y_predict)', len(y_predict), ' and len(y_real)', len(y_real)
        print(msg)
        return
    count_total = 0
    count_correct = 0
    for i in range(0, len(y_predict)):
        count_total += 1
        if round(y_predict[i]) == y_real[i]:
            count_correct += 1
    report = note + ':' + str(count_correct*100/count_total) + '%'
    return report


file_train = local_path('tf_data_modernshiplogin_newuser_train.txt')[0]
data = np.loadtxt(file_train, dtype=np.float64, delimiter='\t', skiprows=0)
no_use, x_train, y_train = np.split(data, [2,7,], axis=1)

file_test = local_path('tf_data_modernshiplogin_newuser_test.txt')[0]
data = np.loadtxt(file_test, dtype=np.float64, delimiter='\t', skiprows=1)
no_use, x_test, y_test = np.split(data, [2,7,], axis=1)

x_train_scale = preprocessing.scale(x_train)
x_test_scale = preprocessing.scale(x_test)
linear_reg = LinearRegression().fit(x_train_scale, y_train.ravel())

logistic_reg = LogisticRegression().fit(x_train_scale, y_train.ravel())


linear_reg_y_hat_test = linear_reg.predict(x_test_scale).ravel()
logistic_reg_y_hat_test = logistic_reg.predict(x_test_scale).ravel()


print("***")
print(show_accuracy(linear_reg_y_hat_test, y_test, 'linear_reg_test'))
print("***")
print(show_accuracy(logistic_reg_y_hat_test, y_test, 'logistic_reg_test'))



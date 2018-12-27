#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : 现代海战数据预测(SVM).py
# @Author: MoonKuma
# @Date  : 2018/12/19
# @Desc  :

import numpy as np
from sklearn import svm
from modenship_data.get_local import local_path
from sklearn import preprocessing

file_train = local_path('tf_data_modernshiplogin_newuser_train_small.txt')[0]
data = np.loadtxt(file_train, dtype=np.float64, delimiter='\t', skiprows=0)
no_use, x_train, y_train = np.split(data, [2,7,], axis=1)

file_test = local_path('tf_data_modernshiplogin_newuser_test_small.txt')[0]
data = np.loadtxt(file_test, dtype=np.float64, delimiter='\t', skiprows=1)
no_use, x_test, y_test = np.split(data, [2,7,], axis=1)

x_train_scale = preprocessing.scale(x_train)
x_test_scale = preprocessing.scale(x_test)

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

clf = svm.SVC()
clf.fit(x_train_scale, y_train.ravel())
'''
SVM struggles in handling large data sample, here use the small sample version instead.
SVM perform far better than TF with this small samples

(quote from  sklearn.svm.SVC(BaseSVC))
"""C-Support Vector Classification.

    The implementation is based on libsvm. The fit time complexity
    is more than quadratic with the number of samples which makes it hard
    to scale to dataset with more than a couple of 10000 samples.
...    
"""
'''

y_hat_test = clf.predict(x_test_scale)
print("***")
print(show_accuracy(y_hat_test, y_test, 'test-predict_auto'))
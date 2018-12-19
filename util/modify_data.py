#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : modify_data.py
# @Author: MoonKuma
# @Date  : 2018/12/18
# @Desc  :

filenames = ['C:\\Users\\7q\\PycharmProjects\\TensorFlowTest\\modenship_data\\tf_data_modernshiplogin_newuser.txt']
file_train = open('C:\\Users\\7q\\PycharmProjects\\TensorFlowTest\\modenship_data\\tf_data_modernshiplogin_newuser_train.txt','w')
file_test = open('C:\\Users\\7q\\PycharmProjects\\TensorFlowTest\\modenship_data\\tf_data_modernshiplogin_newuser_test.txt','w')

import random

def dissect():
    global filenames
    global file_train
    global file_test
    start = 1
    with open(filenames[0]) as file_r:
        index = 0
        for line in file_r.readlines():
            if index==0:
                index = 1
                continue
            line = line.strip()
            line = line.split('\t')
            new_line = line[0] + '\t' + line[1] + '\t' + line[2] + '\t' + line[3] + '\t' + line[4] + '\t' + line[5] + '\t' +line[6] + '\t'
            new_value = str(int(bool(int(line[7]))))
            new_line = new_line + new_value + '\n'
            if start == int(new_value):
                start = 1- int(new_value)
                if random.random()>0.3:
                    file_train.write(new_line)
                else:
                    file_test.write(new_line)


dissect()








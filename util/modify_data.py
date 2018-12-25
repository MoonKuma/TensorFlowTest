#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : modify_data.py
# @Author: MoonKuma
# @Date  : 2018/12/18
# @Desc  :

import random
from modenship_data.get_local import local_path

filenames = local_path('tf_data_modernshiplogin_newuser.txt')
file_train_name = local_path('tf_data_modernshiplogin_newuser_train.txt')[0]
file_test_name = local_path('tf_data_modernshiplogin_newuser_test.txt')[0]
file_train_small_name= local_path('tf_data_modernshiplogin_newuser_train_small.txt')[0]
file_test_small_name = local_path('tf_data_modernshiplogin_newuser_test_small.txt')[0]



def dissect():
    global filenames
    file_train = open(file_train_name, 'w')
    file_test = open(file_test_name, 'w')
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

def get_small_sample(file_in_name,file_out_name,line_number):
    with open(file_in_name, 'r') as file_in:
        count = 0
        for index, line in enumerate(file_in):
            count += 1
        if count == 0:
            return "Not data in read files"
        ratio = float(line_number) / count
        print("RATIO", ratio)
        print("COUNT", count)
    # if use enumerate to avoid loading in at one times, than need to refresh the enumerate after each use
    with open(file_in_name, 'r') as file_in:
        with open(file_out_name, 'w') as file_out:
            lines = 0
            for index, line in enumerate(file_in):
                if random.random() < ratio:
                    file_out.write(line)
                    lines = lines+1
                    if lines >= line_number:
                        break
            print("LINES", lines)



# execute
get_small_sample(file_train_name, file_train_small_name, 10000)
get_small_sample(file_test_name, file_test_small_name, 3000)










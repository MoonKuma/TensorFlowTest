#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : 现代海战数据预测(SVM).py
# @Author: MoonKuma
# @Date  : 2018/12/19
# @Desc  :

from modenship_data.get_local import local_path

file_name = 'tf_data_modernshiplogin_newuser_train.txt'
data_file = local_path(file_name)
print(data_file)
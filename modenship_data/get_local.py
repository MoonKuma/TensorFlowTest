#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : get_local.py
# @Author: MoonKuma
# @Date  : 2018/12/19
# @Desc  : get absolute location of current file
import os

current_path = os.path.realpath(os.path.abspath(os.path.split(__file__)[0]))

def local_path(file_name):
    global current_path
    file_full_name = os.path.join(current_path, file_name)
    if os.path.exists(file_full_name):
        return [file_full_name]
    else:
        msg = '[Caution]File Not Exist: ' + file_full_name
        print(msg)
        return [file_full_name]

# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 15:40:32 2020

@author: 11134423
"""
import pandas as pd

def read_1_data(filename):
    input_file = './data/'+filename
    df = pd.read_csv(input_file,index_col = 0)
    df.index = pd.to_datetime(df.index, format= '%Y%m')
    # df.columns = df.columns.str.replace(' ', '') NOT required in our code

    return df #, columns, date_vec

def read_2_data(filename):
    input_file = './data/'+filename
    df = pd.read_csv(input_file,index_col = 0)
    df.index = pd.to_datetime(df.index, format= '%Y')
    # df.columns = df.columns.str.replace(' ', '') NOT required in our code

    return df #, columns, date_vec
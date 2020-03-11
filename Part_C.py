# Import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools as it
from scipy.optimize import minimize
import random

## ----------- Import data -------------------------------------------------

input_file_path = './data/48_Industry_Portfolios_Returns.CSV'
df_Returns = pd.read_csv(input_file_path,
                   index_col = 0)
input_file_path = './data/48_Industry_Portfolios_Firm-Size.CSV'
df_Firm_Size = pd.read_csv(input_file_path,
                   index_col = 0)
input_file_path = './data/48_Industry_Portfolios_NB-Firms.CSV'
df_NB_Firms = pd.read_csv(input_file_path,
                   index_col = 0)
input_file_path = './data/48_Industry_Portfolios_MktBook.CSV'
df_MktBook = pd.read_csv(input_file_path,
                   index_col = 0)

## ----------- Data cleansing -------------------------------------------------
df_Returns.index = pd.to_datetime(df_Returns.index, format= '%Y%m')
df_Firm_Size.index = pd.to_datetime(df_Firm_Size.index, format= '%Y%m')
df_NB_Firms.index = pd.to_datetime(df_NB_Firms.index, format= '%Y%m')
df_MktBook.index = pd.to_datetime(df_MktBook.index, format= '%Y')

df_Returns.columns = df_Returns.columns.str.replace(' ', '')
df_Firm_Size.columns = df_Firm_Size.columns.str.replace(' ', '')
df_NB_Firms.columns = df_NB_Firms.columns.str.replace(' ', '')
df_MktBook.columns = df_MktBook.columns.str.replace(' ', '')

industries = df_Returns.columns


##Q1:
Market_cap_C = df_Firm_Size*df_NB_Firms
Momentum_C = df_Returns.rolling(12).mean()
Momentum_C = pd.DataFrame.dropna(Momentum_C)
Book_to_Mkt_C = np.ones((len(df_Returns),len(industries)))
Book_to_Mkt_C= pd.DataFrame(Book_to_Mkt_C)
Book_to_Mkt_C.index = df_Returns.index
Book_to_Mkt_C.columns = industries
for i in range(0,len(Book_to_Mkt_C)):
    for j in range(0,len(industries)):
        if Book_to_Mkt_C.index[i].year == df_MktBook.index[0].year:
            B_M_1 = df_MktBook[df_MktBook.index.year == Book_to_Mkt_C.index[i].year].iloc[0,j]
            Book_to_Mkt_C.iloc[i,j] = B_M_1
        elif Book_to_Mkt_C.index[i].month <= 6:
            B_M_2 = df_MktBook[df_MktBook.index.year == Book_to_Mkt_C.index[i].year-1].iloc[0,j]
            Book_to_Mkt_C.iloc[i,j] = B_M_2
        else:
            B_M_3 = df_MktBook[df_MktBook.index.year == Book_to_Mkt_C.index[i].year].iloc[0,j]
            Book_to_Mkt_C.iloc[i,j] = B_M_3


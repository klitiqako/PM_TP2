# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 14:38:37 2020

@author: 11134423
"""

# Import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools as it
from scipy.optimize import minimize
import random

# Import function
from function  import myf
from function  import read_data
from variables import myvar

# Initialize some parameters
start_date="1926-07-01"
end_date="2020-01-01"

# Importing Data
all_monthly_returns=read_data.read_1_data("TP2_monthly_returns.csv")
industries = all_monthly_returns.columns
date_vec = all_monthly_returns.index
date_list = date_vec.tolist()
all_risk_free_rates =read_data.read_1_data("TP2_monthly_risk_free.csv")
all_avg_firm_size=read_data.read_1_data("TP2_avg_firm_size.csv")
all_num_firms=read_data.read_1_data("TP2_numb_of_firms.csv")


## ---------- Part A ----------------------------------------------------------

# Cleaning and Preping the Data
idx = list(date_vec.strftime("%Y-%m-%d")).index(start_date)

working_monthly_returns = all_monthly_returns.loc[date_vec[idx]:date_vec[idx+59],:]
mu = working_monthly_returns.mean()
covariance_matrix = working_monthly_returns.cov()
volatilities = np.diagonal(np.sqrt(covariance_matrix))
n_industries = mu.size
working_risk_free_rates = all_risk_free_rates.loc[date_vec[idx]:date_vec[idx+59],['RF']]
rf = working_risk_free_rates.iloc[-1]
working_avg_firm_size=all_avg_firm_size.loc[date_vec[idx]:date_vec[idx+59],:]
avg_firm_size = working_avg_firm_size.iloc[-1]
working_num_firms=all_num_firms.loc[date_vec[idx]:date_vec[idx+59],:]
num_firms = working_num_firms.iloc[-1]

#1) the portfolio that maximizes the Sharpe ratio without short-sale constraints
(tmp1, tmp2, tmp3) = myf.minvarpf(working_monthly_returns, [], rf, risk_free_allowed = True, tangency = True)
myvar.P1_weights = tmp3

myvar.P1_return.append((myvar.P1_weights @ (1+(working_monthly_returns.iloc[-1])/100) - 1 ) * 100)

#2) the portfolio that maximizes the Sharpe ratio with short-sale constraints;
(tmp1, tmp2, tmp3) = myf.minvarpf_noshortsale(working_monthly_returns, [], rf, risk_free_allowed = True, tangency = True)
myvar.P2_weights = tmp3

myvar.P2_return.append((myvar.P2_weights @ (1+(working_monthly_returns.iloc[-1])/100) - 1 ) * 100)

#3) the portfolio where the weight of each asset is inversely related to its variance;
Inv_variance =  sum(1/np.diag(covariance_matrix))
myvar.P3_weights = [
                    1/covariance_matrix.iloc[0,0], 1/covariance_matrix.iloc[1,1], 1/covariance_matrix.iloc[2,2],
                    1/covariance_matrix.iloc[3,3], 1/covariance_matrix.iloc[4,4], 1/covariance_matrix.iloc[5,5],
                    1/covariance_matrix.iloc[6,6], 1/covariance_matrix.iloc[7,7], 1/covariance_matrix.iloc[8,8],
                    1/covariance_matrix.iloc[9,9],
                    ]
myvar.P3_weights = myvar.P3_weights/Inv_variance

myvar.P3_return.append((myvar.P3_weights @ (1+(working_monthly_returns.iloc[-1])/100) - 1 ) * 100)

#4) the portfolio where the weight of each asset is inversely related to its volatility;
Inv_volatility =  sum(1/np.diag(np.sqrt(covariance_matrix)))
myvar.P4_weights = [
                    1/covariance_matrix.iloc[0,0], 1/covariance_matrix.iloc[1,1], 1/covariance_matrix.iloc[2,2],
                    1/covariance_matrix.iloc[3,3], 1/covariance_matrix.iloc[4,4], 1/covariance_matrix.iloc[5,5],
                    1/covariance_matrix.iloc[6,6], 1/covariance_matrix.iloc[7,7], 1/covariance_matrix.iloc[8,8],
                    1/covariance_matrix.iloc[9,9],
                    ]
myvar.P4_weights = np.sqrt(myvar.P4_weights)/Inv_volatility

myvar.P4_return.append((myvar.P4_weights @ (1+(working_monthly_returns.iloc[-1])/100) - 1 ) * 100)


#5) the portfolio where assets have the same weight;
myvar.P5_weights = np.full((1,n_industries),1/n_industries)

#6) the portfolio where the weight of each is linearly related to its market capitalization;
total_market_cap = avg_firm_size @ num_firms
myvar.P6_weights = (avg_firm_size * num_firms) / total_market_cap

#7) the portfolio with the minimum variance;
(tmp1, tmp2, tmp3) = myf.minvarpf(working_monthly_returns, [], rf, risk_free_allowed = True, tangency = False)
myvar.P7_weights = tmp3
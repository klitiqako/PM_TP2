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

# Empty vectors
P1_weights = []                         # 10x1
P2_weights = []                         # 10x1
P3_weights = []                         # 10x1
P4_weights = []                         # 10x1
P5_weights = []                         # 10x1
P6_weights = []                         # 10x1
P7_weights = []                         # 10x1

P1_return = []                          # 1 x n
P2_return = []                          # 1 x n 
P3_return = []                          # 1 x n 
P4_return = []                          # 1 x n 
P5_return = []                          # 1 x n 
P6_return = []                          # 1 x n 
P7_return = []                          # 1 x n 

portfolio_sharpe_ratio = []             # 7 x 1 


#1) the portfolio that maximizes the Sharpe ratio without short-sale constraints
(tmp1, tmp2, tmp3) = myf.minvarpf(working_monthly_returns, 5, rf[0], risk_free_allowed = False, tangency = True)
P1_weights = tmp3
P1_return.append(myf.prtf_return(P1_weights,all_monthly_returns.loc[date_vec[idx+60],:]))


#2) the portfolio that maximizes the Sharpe ratio with short-sale constraints;
(tmp1, tmp2, tmp3) = myf.minvarpf_noshortsale(working_monthly_returns, 5, rf[0], risk_free_allowed = False, tangency = True)
P2_weights = tmp3
P2_return.append(myf.prtf_return(P2_weights,all_monthly_returns.loc[date_vec[idx+60],:]))


#3) the portfolio where the weight of each asset is inversely related to its variance;
P3_weights = 1/np.diag(covariance_matrix)
Inv_variance =  sum(P3_weights)              
P3_weights = P3_weights/Inv_variance
P3_return.append(myf.prtf_return(P3_weights,all_monthly_returns.loc[date_vec[idx+60],:]))


#4) the portfolio where the weight of each asset is inversely related to its volatility;
P4_weights = 1/np.diag(np.sqrt(covariance_matrix))
Inv_volatility =  sum(P4_weights)
P4_weights = P4_weights/Inv_volatility
P4_return.append(myf.prtf_return(P4_weights,all_monthly_returns.loc[date_vec[idx+60],:]))


#5) the portfolio where assets have the same weight;
P5_weights = np.full([n_industries],1/n_industries)
P5_return.append(myf.prtf_return(P5_weights,all_monthly_returns.loc[date_vec[idx+60],:]))


#6) the portfolio where the weight of each is linearly related to its market capitalization;
total_market_cap = avg_firm_size @ num_firms
P6_weights = (avg_firm_size * num_firms) / total_market_cap
P6_weights=P6_weights.to_numpy()
P6_return.append(myf.prtf_return(P6_weights,all_monthly_returns.loc[date_vec[idx+60],:]))


#7) the portfolio with the minimum variance;
(tmp1, tmp2, tmp3) = myf.minvarpf(working_monthly_returns,[], rf[0], risk_free_allowed = False, tangency = False)
P7_weights = tmp3
P7_return.append(myf.prtf_return(P7_weights,all_monthly_returns.loc[date_vec[idx+60],:]))

Table_weights=[P1_weights, P2_weights, P3_weights, P4_weights, P5_weights, P6_weights, P7_weights]
Table_ret=[P1_return, P2_return, P3_return, P4_return, P5_return, P6_return, P7_return]

print(Table_weights)
print(Table_ret)

# Version KEANU
# Import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools as it
from scipy.optimize import minimize
import random

# Import function
from function_tp2  import myf
from function_tp2  import read_data
from variables import myvar



# Importing Data
all_monthly_returns = read_data.read_1_data("TP2_monthly_returns.csv") / 100
industries = all_monthly_returns.columns
date_vec = all_monthly_returns.index
date_list = date_vec.tolist()
all_risk_free_rates = read_data.read_1_data("TP2_monthly_risk_free.csv") / 100
all_avg_firm_size = read_data.read_1_data("TP2_avg_firm_size.csv")
all_num_firms = read_data.read_1_data("TP2_numb_of_firms.csv")
risk_free_rate = all_risk_free_rates.iloc[:,-1]

## ---------- Part A ----------------------------------------------------------



# Parameter of the back testing
start_date="1931-07-01"
end_date="2020-01-01"
rolling_window = 59 # Number of week: 5 years
idx_start = list(date_vec.strftime("%Y-%m-%d")).index(start_date)
idx_end = list(date_vec.strftime("%Y-%m-%d")).index(end_date)
date_vec_btst = date_vec[idx_start:idx_end].strftime("%Y-%m-%d")

# Initialization
P1_weights = []                         # 10x1
P2_weights = []                         # 10x1
P3_weights = []                         # 10x1
P4_weights = []                         # 10x1
P5_weights = []                         # 10x1
P6_weights = []                         # 10x1
P7_weights = []                         # 10x1
P8_weights = []                         # 10x1

P1_return = []                          # 1 x n
P2_return = []                          # 1 x n 
P3_return = []                          # 1 x n 
P4_return = []                          # 1 x n 
P5_return = []                          # 1 x n 
P6_return = []                          # 1 x n 
P7_return = []                          # 1 x n 

portfolio_sharpe_ratio = []             # 7 x 1 

for date in date_vec_btst:
    print(date)
    # Cleaning and Preping the Data
    idx = date_vec.get_loc(date)
    working_monthly_returns = all_monthly_returns.loc[date_vec[idx - rolling_window]:date_vec[idx],:]
    montly_returns_tplus1 = all_monthly_returns.loc[date_vec[idx + 1],:]
    mu = working_monthly_returns.mean()
    covariance_matrix = working_monthly_returns.cov()
    volatilities = np.diagonal(np.sqrt(covariance_matrix))
    n_industries = mu.size
    working_risk_free_rates = all_risk_free_rates.loc[date_vec[idx - rolling_window]:date_vec[idx],['RF']]
    rf = working_risk_free_rates.iloc[-1]
    working_avg_firm_size = all_avg_firm_size.loc[date_vec[idx - rolling_window]:date_vec[idx],:]
    avg_firm_size = working_avg_firm_size.iloc[-1]
    working_num_firms = all_num_firms.loc[date_vec[idx - rolling_window]:date_vec[idx],:]
    num_firms = working_num_firms.iloc[-1]
    
    #1) the portfolio that maximizes the Sharpe ratio without short-sale constraints
    (tmp1, tmp2, tmp3) = myf.minvarpf(x = working_monthly_returns, mu = None, risk_free_rate = rf, 
                                      risk_free_allowed = True, tangency = True, minvar = True)
    P1_weights.append(tmp3)
    P1_return.append(myf.prtf_return(P1_weights[-1], montly_returns_tplus1, rf))
    
    
    #2) the portfolio that maximizes the Sharpe ratio with short-sale constraints;
    (tmp1, tmp2, tmp3) = myf.minvarpf_noshortsale(working_monthly_returns, [], rf, risk_free_allowed = True, tangency = True)
    P2_weights = tmp3
    P2_return.append(myf.prtf_return(P2_weights, montly_returns_tplus1, rf))
   
    
    #3) the portfolio where the weight of each asset is inversely related to its variance;
    Inv_variance =  sum(1/np.diag(covariance_matrix))
    P3_weights = [ 0,
                        1/covariance_matrix.iloc[0,0], 1/covariance_matrix.iloc[1,1], 1/covariance_matrix.iloc[2,2],
                        1/covariance_matrix.iloc[3,3], 1/covariance_matrix.iloc[4,4], 1/covariance_matrix.iloc[5,5],
                        1/covariance_matrix.iloc[6,6], 1/covariance_matrix.iloc[7,7], 1/covariance_matrix.iloc[8,8],
                        1/covariance_matrix.iloc[9,9],
                        ]
    P3_weights = P3_weights/Inv_variance
    P3_return.append(myf.prtf_return(P3_weights, montly_returns_tplus1, rf))
    
    
    #4) the portfolio where the weight of each asset is inversely related to its volatility;
    Inv_volatility =  sum(1/np.diag(np.sqrt(covariance_matrix)))
    P4_weights = [ 0,
                        1/covariance_matrix.iloc[0,0], 1/covariance_matrix.iloc[1,1], 1/covariance_matrix.iloc[2,2],
                        1/covariance_matrix.iloc[3,3], 1/covariance_matrix.iloc[4,4], 1/covariance_matrix.iloc[5,5],
                        1/covariance_matrix.iloc[6,6], 1/covariance_matrix.iloc[7,7], 1/covariance_matrix.iloc[8,8],
                        1/covariance_matrix.iloc[9,9],
                        ]
    P4_weights = np.sqrt(P4_weights)/Inv_volatility
    P4_return.append(myf.prtf_return(P4_weights, montly_returns_tplus1, rf))
    
    
    #5) the portfolio where assets have the same weight;
    P5_weights = np.append(0, np.repeat(1/n_industries, n_industries))
    P5_return.append(myf.prtf_return(P5_weights, montly_returns_tplus1, rf))
    
    
    #6) the portfolio where the weight of each is linearly related to its market capitalization;
    total_market_cap = avg_firm_size @ num_firms
    P6_weights = np.append(0, np.array((avg_firm_size * num_firms) / total_market_cap))
    P6_return.append(myf.prtf_return(P6_weights, montly_returns_tplus1, rf))
    
    
    #7) the portfolio with the minimum variance;
    (tmp1, tmp2, tmp3) = myf.minvarpf(x = working_monthly_returns, risk_free_rate = rf, risk_free_allowed = False, tangency = False, minvar = True)
    P7_weights = tmp3
    P7_return.append(myf.prtf_return(P7_weights, montly_returns_tplus1, rf))
P8_return = risk_free_rate.iloc[idx_start:idx_end]


Returns = np.array([P1_return, P2_return, P3_return, P4_return, P5_return, P6_return, P7_return, P8_return])
NAV = np.cumprod(1 + np.transpose(Returns), 0)
strategies = ['Max Sharpe (long short)', 'Max Sharpe (Long only)', 'Inverse Var', 'Inverse S-d', 'Equally weighted', 'Mkt Cap weighted', 'Min Var', 'Risk_free']

for idx, strat in enumerate(strategies):
    plt.plot(NAV[:,idx], label = strat)
plt.yscale('log')
plt.legend()
plt.title('Value of 1 $ invested the '+ str(date_vec_btst[1]))
plt.show()

# Performance measure

n_year = (idx_end - idx_start) / 12

for idx, strat in enumerate(strategies):
    #annualize return (geometric mean)
    returns
    Sharpe_ratio.append(NAV[:,idx])

(idx_end - idx_start) / 12



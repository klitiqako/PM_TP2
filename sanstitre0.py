# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 10:16:46 2020

@author: Keanu
"""
import numpy as np
import matplotlib.pyplot as plt
from function_tp2 import myf
from function_tp2  import read_data

#Import data
all_monthly_returns = read_data.read_1_data("TP2_monthly_returns.csv") / 100
all_risk_free_rates = read_data.read_1_data("TP2_monthly_risk_free.csv") / 100


for idx in range(10):
    print(idx)
    rf = all_risk_free_rates.iloc[59+idx,-1]
    working_monthly_returns = all_monthly_returns.iloc[idx:idx+59,:]
    
    
    mu_scale = np.arange(start = 0  , stop = 0.015, step = 0.0005)
    var_tmp= []
    var_tmp2 = []
    for mu in  mu_scale:
        (tmp1, tmp2, tmp3) = myf.minvarpf_noshortsale(working_monthly_returns, mu, rf, risk_free_allowed = False, tangency = False)
        var_tmp.append(tmp2)
        
        (tmp1, tmp2, tmp3) = myf.minvarpf_noshortsale(working_monthly_returns, mu, rf, risk_free_allowed = True, tangency = False)
        var_tmp2.append(tmp2)
    plt.plot(var_tmp, mu_scale)  
    plt.plot(var_tmp2, mu_scale)  
    (tmp1, tmp2, tmp3) = myf.minvarpf_noshortsale(working_monthly_returns,mu, rf, risk_free_allowed = True, tangency = True)
    plt.scatter(tmp2, tmp1)
    
    
    
    plt.show()
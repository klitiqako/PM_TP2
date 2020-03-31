
# Import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools as it
from scipy.optimize import minimize
import random
import datetime
import dateutil.relativedelta
import statsmodels.api as sm
import time
from function import myf

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
input_file_path = './data/Daily_Returns.CSV'
df_Daily_Returns = pd.read_csv(input_file_path,
                   index_col = 0)
input_file_path = './data/F-F_Research_Data_Factors_daily.CSV'
df_Fama = pd.read_csv(input_file_path,
                   index_col = 0)
input_file_path = './data/FAMA_3_Factors.CSV'
df_Fama_3 = pd.read_csv(input_file_path,
                   index_col = 0)
input_file_path = './data/FAMA_5_Factors.CSV'
df_Fama_5 = pd.read_csv(input_file_path,
                   index_col = 0)

## ----------- Data cleansing -------------------------------------------------
date_vec = myf.eomonth(pd.to_datetime(df_Returns.index, format= '%Y%m'))
df_Returns.index = date_vec
df_Firm_Size.index = myf.eomonth(pd.to_datetime(df_Firm_Size.index, format= '%Y%m'))
df_NB_Firms.index = myf.eomonth(pd.to_datetime(df_NB_Firms.index, format= '%Y%m'))
df_MktBook.index = myf.midyearfama(pd.to_datetime(df_MktBook.index, format= '%Y'))
df_Daily_Returns.index = pd.to_datetime(df_Daily_Returns.index, format= '%Y%m%d')
df_Fama.index = pd.to_datetime(df_Fama.index, format= '%Y%m%d')
df_Fama_3.index = myf.eomonth(pd.to_datetime(df_Fama_3.index, format= '%Y%m'))
df_Fama_5.index = myf.eomonth(pd.to_datetime(df_Fama_5.index, format= '%Y%m'))


df_Returns.columns = df_Returns.columns.str.replace(' ', '')
df_Firm_Size.columns = df_Firm_Size.columns.str.replace(' ', '')
df_NB_Firms.columns = df_NB_Firms.columns.str.replace(' ', '')
df_MktBook.columns = df_MktBook.columns.str.replace(' ', '')
df_Daily_Returns.columns = df_Daily_Returns.columns.str.replace(' ', '')
df_Fama.columns = df_Fama.columns.str.replace(' ', '')
df_Fama_3.columns = df_Fama_3.columns.str.replace(' ', '')
df_Fama_5.columns = df_Fama_5.columns.str.replace(' ', '')

df_Returns[df_Returns == -99.99] = np.nan
df_NB_Firms[df_NB_Firms == 0] = np.nan

industries = df_Returns.columns


###############Q1:
Market_cap_C = df_Firm_Size*df_NB_Firms
Momentum_C = df_Returns.rolling(12).mean()

Book_to_Mkt_C = np.zeros((len(df_Returns),len(industries)))
Book_to_Mkt_C= pd.DataFrame(Book_to_Mkt_C)
Book_to_Mkt_C.index = df_Returns.index
Book_to_Mkt_C.columns = industries


for i in range(0,len(Book_to_Mkt_C)):
    
    if Book_to_Mkt_C.index[i].year == df_MktBook.index[0].year:
        B_M_1 = df_MktBook[df_MktBook.index.year == Book_to_Mkt_C.index[i].year]
        Book_to_Mkt_C.iloc[i,:] = np.array(B_M_1)
    elif Book_to_Mkt_C.index[i].month <= 6:
        B_M_2 = df_MktBook[df_MktBook.index.year == Book_to_Mkt_C.index[i].year-1]
        Book_to_Mkt_C.iloc[i,:] = np.array(B_M_2)
    else:
        B_M_3 = df_MktBook[df_MktBook.index.year == Book_to_Mkt_C.index[i].year]
        Book_to_Mkt_C.iloc[i,:] = np.array(B_M_3)  
Book_to_Mkt_C[Book_to_Mkt_C == -99.99] = np.nan       
###############Q2
lag = 12 * 3


Months = df_Returns.index
Months = Months.insert(len(Months) + 1, pd.to_datetime('20200229', format= '%Y%m%d'))
betas = np.zeros((len(df_Returns)-lag - 1,len(industries)))
betas = pd.DataFrame(betas)
betas.index = df_Returns.index[lag+1:]
betas.columns = industries
df_Daily_Returns[df_Daily_Returns == -99.99] = np.nan
df_Daily_Excess_Returns = df_Daily_Returns.subtract(df_Fama['RF'], 0)
time_start = time.clock()
for d, i in betas.iterrows():
    print(d)
    
    d_Last12 = d - dateutil.relativedelta.relativedelta(months=lag)
    X = df_Fama['Mkt-RF'][(df_Fama.index < d)&(df_Fama.index >= d_Last12)]
    X_np = np.array(X)
    X_np = sm.add_constant(X)
    for j in industries:
        Y = np.array(df_Daily_Excess_Returns.loc[:,j][(df_Daily_Excess_Returns.index < d)&(df_Daily_Excess_Returns.index >= d_Last12)])
        Y_np = np.array(Y)
        betas.loc[d,j] =  np.linalg.solve(np.dot(X_np.T, X_np), np.dot(X_np.T, Y_np))[1]
time_elapsed = (time.clock() - time_start)
##Checking computation time difference between package and analatycal way

beta_avg = betas.mean()



##Analytical takes half as long

###############Q3
Idio_vol = np.zeros((len(df_Returns),len(industries)))
Idio_vol = pd.DataFrame(Idio_vol)
Idio_vol.index = df_Returns.index
Idio_vol.columns = industries

time_start = time.clock()
for d, i in Idio_vol.iterrows(): 
    print(d)
    d_Last12 = d - dateutil.relativedelta.relativedelta(months=1)
    X = np.array(df_Fama.iloc[:,:3][(df_Fama.index < d)&(df_Fama.index >= d_Last12)])
    X = sm.add_constant(X)
    for j in industries:
        Y = df_Daily_Excess_Returns.loc[:,j][(df_Daily_Excess_Returns.index < d)&(df_Daily_Excess_Returns.index >= d_Last12)]
        Y = np.array(Y)
        beta_Fama = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
        resid = Y -  X @ beta_Fama.T 
        std_resid = np.std(resid)
        Idio_vol.loc[d,j] = std_resid
time_elapsed1 = (time.clock() - time_start)       
# Checking time
time_start = time.clock()
est = sm.OLS(Y,X).fit()
std_resid = np.std(est.resid)
time_elapsed = (time.clock() - time_start)

time_start = time.clock()
beta_Fama = np.linalg.solve(np.dot(np.array(X).T, np.array(X)), np.dot(np.array(X).T, np.array(Y)))
resid = Y -  X @ beta_Fama.T
std_resid = np.std(resid)
time_elapsed1 = (time.clock() - time_start)

print(time_elapsed - time_elapsed1)


###############Q4

# Parameters

start_date = pd.to_datetime(datetime.datetime(1963, 7, 31))
end_date = pd.to_datetime(datetime.datetime(2020, 1, 1))

date_vec = df_Returns.index[(df_Returns.index >= start_date)&(df_Returns.index <= end_date)]

num_long = 5
num_short = 5

# 1 : Long low characteristic and short high characteristic
# -1 : Long high characteristic and short low characteristic
long_short = pd.Series([1, 1, -1, 1, -1])
strat = ['BAB', 'Low Vol', 'HML', 'SMB', 'MOM']
characteristic_tmp = {'BAB': betas, 'Low Vol':Idio_vol, 'HML':Book_to_Mkt_C, 'SMB': Market_cap_C, 'MOM': Momentum_C}
long_short.index = strat

weight_type = 'Market_cap' # Equal, Market_cap or Rank

# Variable
df_Returns_tplus1 = df_Returns.shift(-1)
df_Returns_tplus1[df_Returns_tplus1 == -99.99] = np.nan
NAV = np.zeros((len(date_vec), len(strat)))
NAV = pd.DataFrame(NAV)
NAV.index = date_vec
NAV.columns = strat

weight = {}
for s in strat:
    weight[s] = pd.DataFrame(np.zeros((len(date_vec), len(industries))))
    weight[s].index = date_vec
    weight[s].columns = industries

# Initialization
returns = pd.DataFrame(np.zeros((len(date_vec), len(strat))))
returns.index = date_vec
returns.columns = strat

returns_tmoins1 = pd.Series(np.zeros(len(strat)))
returns_tmoins1.index = strat
Nav_tmoins1 = pd.Series(np.ones(len(strat)))
Nav_tmoins1.index = strat
i = 0
for date, Nav in NAV.iterrows():
    print(date, end_date)
    i = i + 1 
    NAV.loc[date,:] = np.array(Nav_tmoins1.T) * np.array(1 + returns_tmoins1 / 100)
    for characteristic_name in strat:       
        # investable universe
        is_excluded = pd.isna(df_Returns_tplus1.loc[date,:])
        is_eligible = np.logical_not(is_excluded)
        
        # Low characteristic
        idx_low = characteristic_tmp[characteristic_name].loc[date, is_eligible].nsmallest(num_long).index
        if weight_type == 'Equal':
            weight[characteristic_name].loc[date, idx_low] = 1 / num_long * long_short.loc[characteristic_name]
        elif weight_type == 'Market_cap':
            market_cap_long = characteristic_tmp['SMB'].loc[date, idx_low]
            weight[characteristic_name].loc[date, idx_low] = market_cap_long \
            / sum(market_cap_long) * long_short.loc[characteristic_name]
        elif weight_type == 'Rank':
            low_rank = characteristic_tmp[characteristic_name].loc[date, idx_low].rank(ascending = False)
            weight[characteristic_name].loc[date, low_rank.index] = low_rank \
            /sum(low_rank) * long_short.loc[characteristic_name]
            
        # High characteristic
        idx_high = characteristic_tmp[characteristic_name].loc[date, is_eligible].nlargest(num_short).index
        if weight_type == 'Equal':
            weight[characteristic_name].loc[date, idx_high] = - 1 / num_short * long_short.loc[characteristic_name]
        elif weight_type == 'Market_cap':
            market_cap_short = characteristic_tmp['SMB'].loc[date, idx_high]
            weight[characteristic_name].loc[date, idx_high] =  - market_cap_short \
            / sum(market_cap_short) * long_short.loc[characteristic_name]
        elif weight_type == 'Rank':
            high_rank = characteristic_tmp[characteristic_name].loc[date, idx_high].rank()
            weight[characteristic_name].loc[date, high_rank.index] = - high_rank \
            / sum(high_rank) * long_short.loc[characteristic_name]
            
        # Returns
        idx_weight = idx_low.append(idx_high)
        returns.loc[date, characteristic_name] = weight[characteristic_name].loc[date, idx_weight].dot(df_Returns_tplus1.loc[date, idx_weight])
    
    # Transfert for the next iteration
    returns_tmoins1 = returns.loc[date, :]
    Nav_tmoins1 = NAV.loc[date,:]



  
market = df_Fama_3.loc[date_vec,'Mkt-RF'] + df_Fama_3.loc[date_vec,'RF']
NAV_market = np.cumprod(1 + market[market.index > '1963-01-31'] / 100)    
for characteristic_name in strat:
    plt.plot(NAV.loc[:,characteristic_name], label = characteristic_name)
plt.plot(NAV_market, label = 'Market')
plt.legend()
plt.yscale('log')
plt.title('NAV with weigth: ' + weight_type)
plt.show()


## Performance calculation
# Input: DataFrame of returns according to each strategies


period1 = [pd.to_datetime(datetime.datetime(1963, 7, 31)) , pd.to_datetime(datetime.datetime(2019, 12, 31))]
period2 = [pd.to_datetime(datetime.datetime(1990, 1, 1)) , pd.to_datetime(datetime.datetime(2019, 12, 31))]
period3 = [pd.to_datetime(datetime.datetime(2000, 1, 1)) , pd.to_datetime(datetime.datetime(2019, 12, 31))]
period = [period1, period2, period3]


for subperiod in period:
    start_date = subperiod[0]
    end_date = subperiod[1]
    date_vec_tmp = df_Returns.index[(df_Returns.index >= start_date)&(df_Returns.index <= end_date)]
    returns_tmp = returns.loc[date_vec_tmp, :]
    strat = returns.columns
    performance_measure_type = ['Sharpe_Ratio', 'Fama_3', 'Fama_4', 'Fama_5']
    nb_factor = [1, 4, 5, 6]
    factor = ['alpha', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
    performance_measure = {}
    for s in strat:
        performance_measure[s] = pd.DataFrame(np.zeros((len(factor), len(performance_measure_type))))
        performance_measure[s].columns = performance_measure_type
        performance_measure[s].index = factor
    
    for s in strat:
        print(s, subperiod)
        for i, p in enumerate(performance_measure_type):
            factor_tmp = factor[:nb_factor[i]]
            if p == 'Sharpe_Ratio':
                # Sharpe Ratio: Excess Return / Standard deviation
                tmp = returns_tmp.loc[:,s].subtract(df_Fama_3.loc[date_vec_tmp, 'RF']).mean() / returns_tmp.loc[:,s].std()
                
            else: #Fama French
                #Excess Return
                Y = np.array(returns_tmp.loc[:,s].subtract(df_Fama_3.loc[date_vec_tmp, 'RF']))
                if p == 'Fama_3':
                    X = df_Fama_3.loc[date_vec_tmp, factor_tmp[1:]]
                else: # Fama_4 and Fama_5
                    X = df_Fama_5.loc[date_vec_tmp, factor_tmp[1:]]
               
                X = sm.add_constant(X)
                tmp = np.linalg.solve(np.dot(np.array(X).T, np.array(X)), np.dot(np.array(X).T, np.array(Y)))
    
            performance_measure[s].loc[factor_tmp,p] = tmp
    
        print(performance_measure[s])


# Import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools as it
from scipy.optimize import minimize
import random
import datetime
import statsmodels.api as sm

# Import function
from function import myf
from function import read_data
from variables import myvar

# Importing Data
all_monthly_returns = read_data.read_1_data("TP2_monthly_returns.csv")
industries          = all_monthly_returns.columns
date_vec            = all_monthly_returns.index
date_list           = date_vec.tolist()
all_risk_free_rates = read_data.read_1_data("TP2_monthly_risk_free.csv")
all_avg_firm_size   = read_data.read_1_data("TP2_avg_firm_size.csv")
all_num_firms       = read_data.read_1_data("TP2_numb_of_firms.csv")
sum_BE_div_sum_ME   = read_data.read_2_data("TP2_sum_BE_div_sum_ME.csv")

## ---------- Part A ----------------------------------------------------------

# Parameter of the back testing
TE_threshold        = 1
rolling_window      = 60  # Number of months in period used for estimation 5 years

start_date_1        = "1931-07-01"
start_date_2        = "1990-01-01"
start_date_3        = "2000-01-01"
end_date            = "2020-01-01"


idx_start_1         = list(date_vec.strftime("%Y-%m-%d")).index(start_date_1)
idx_start_2         = list(date_vec.strftime("%Y-%m-%d")).index(start_date_2)
idx_start_3         = list(date_vec.strftime("%Y-%m-%d")).index(start_date_3)
idx_end             = list(date_vec.strftime("%Y-%m-%d")).index(end_date) + 1
date_vec_btst       = date_vec[idx_start_1:idx_end].strftime("%Y-%m-%d")
date_vec_prd2       = date_vec[idx_start_2:idx_end].strftime("%Y-%m-%d")
date_vec_prd3       = date_vec[idx_start_3:idx_end].strftime("%Y-%m-%d")

# Constraint functions defined as dictionaries that don't change
constraint_weights          = {'type': 'eq', 'fun': myf.constraint_on_weights}
constraint_short_sell       = {'type': 'ineq', 'fun': myf.constraint_on_short_sell}
constraint_short_sell_lim   = {'type': 'ineq', 'fun': myf.constraint_on_short_sell_lim}

# Initialization Empty vectors
P1_weights                  = []               # 10 x n
P2_weights                  = []               # 10 x n
P3_weights                  = []               # 10 x n
P4_weights                  = []               # 10 x n
P5_weights                  = []               # 10 x n
P6_weights                  = []               # 10 x n
P7_weights                  = []               # 10 x n

PB_3_1_weights              = []        # Part B #3 bench 1
PB_3_2_weights              = []        # Part B #3 bench 2
PB_4_1_weights              = []        # Part B #4 bench 1
PB_4_2_weights              = []

P1_return                   = []                # 1 x n
P2_return                   = []                # 1 x n
P3_return                   = []                # 1 x n
P4_return                   = []                # 1 x n
P5_return                   = []                # 1 x n
P6_return                   = []                # 1 x n
P7_return                   = []                # 1 x n

PB_3_1_return               = []        # Part B #3 bench 1
PB_3_2_return               = []        # Part B #3 bench 2
PB_4_1_return               = []        # Part B #4 bench 1
PB_4_2_return               = []        # Part B #4 bench 2

P1_alpha                    = []                # 1 x n
P2_alpha                    = []                # 1 x n
P3_alpha                    = []                # 1 x n
P4_alpha                    = []                # 1 x n
P5_alpha                    = []                # 1 x n
P6_alpha                    = []                # 1 x n
P7_alpha                    = []                # 1 x n

PB_3_1_alpha                = []        # Part B #3 bench 1
PB_3_2_alpha                = []        # Part B #3 bench 2
PB_4_1_alpha                = []        # Part B #4 bench 1
PB_4_2_alpha                = []        # Part B #4 bench 2

Mrkt_Cap                    = []                # 10 x n

##--------------------- Building the different portfolios in 1 loop-----------

for date in date_vec_btst:
    print(date)

    # Cleaning and Preping the Data
    idx                     = date_vec.get_loc(date)

    working_monthly_returns = all_monthly_returns.loc[date_vec[idx-60]:date_vec[idx-1], :]
    montly_returns_tplus1   = all_monthly_returns.loc[date_vec[idx], :]
    mu                      = working_monthly_returns.mean()
    covariance_matrix       = working_monthly_returns.cov()
    volatilities            = np.diagonal(np.sqrt(covariance_matrix))
    n_industries            = mu.size

    working_risk_free_rates = all_risk_free_rates.loc[date_vec[idx-60]:date_vec[idx-1], ['RF']]
    rf                      = working_risk_free_rates.iloc[-1]
    rf_tplus1               = all_risk_free_rates.loc[date_vec[idx], ['RF']]

    working_avg_firm_size   = all_avg_firm_size.loc[date_vec[idx-60]:date_vec[idx-1], :]
    avg_firm_size           = working_avg_firm_size.iloc[-1]
    working_num_firms       = all_num_firms.loc[date_vec[idx-60]:date_vec[idx-1], :]
    num_firms               = working_num_firms.iloc[-1]


    #5) the portfolio where assets have the same weight;
    P5_weights = np.full([n_industries], 1 / n_industries)
    P5_return.append(myf.prtf_return(P5_weights, montly_returns_tplus1))
    P5_alpha.append(P5_return[-1]-rf_tplus1[0])

    # Using the loop to calculate the prtf of Part B #3) (without ss) and #4) (with ss)
    # which track the benchmark portfolios with a maximum tracking error of 1% monthly
    constraint_TE = {'type': 'ineq', 'fun': myf.constraint_on_TE, 'args': (P5_weights, covariance_matrix, TE_threshold)}
    TE_without_ss_constraints = [constraint_weights, constraint_TE, constraint_short_sell_lim]
    sol_3_2 = minimize(myf.tangency_objective, [0, 0, 0, 0, 0, 0, 0, 0, 0, 1], args=(rf[0], covariance_matrix, mu), method="SLSQP", constraints=TE_without_ss_constraints)
    PB_3_2_weights = sol_3_2.x
    PB_3_2_return.append(myf.prtf_return(PB_3_2_weights, montly_returns_tplus1))
    PB_3_2_alpha.append(PB_3_2_return[-1]-rf_tplus1[0])

    TE_with_ss_constraints = [constraint_weights, constraint_TE, constraint_short_sell]
    sol_4_2 = minimize(myf.tangency_objective, P5_weights, args=(rf[0], covariance_matrix, mu), method="SLSQP", constraints=TE_with_ss_constraints)
    PB_4_2_weights = sol_4_2.x
    PB_4_2_return.append(myf.prtf_return(PB_4_2_weights, montly_returns_tplus1))
    PB_4_2_alpha.append(PB_4_2_return[-1]-rf_tplus1[0])


    #7) the portfolio with the minimum variance;
    (sol_7_1, sol_7_2, sol_7_3) = myf.minvarpf(working_monthly_returns, [], rf[0], risk_free_allowed=False, tangency=False)
    P7_weights = sol_7_3
    P7_return.append(myf.prtf_return(P7_weights, montly_returns_tplus1))
    P7_alpha.append(P7_return[-1]-rf_tplus1[0])


    #1) the portfolio that maximizes the Sharpe ratio without short-sale constraints
    #if rf[0] < P7_return[-1]:
    #    (tmp1, tmp2, tmp3) = myf.minvarpf(working_monthly_returns, 5, rf[0], risk_free_allowed = False, tangency = True)
    #    P1_weights = tmp3
    #    P1_return.append(myf.prtf_return(P1_weights,montly_returns_tplus1))
    #else:
    tangency_constraints = [constraint_weights, constraint_short_sell_lim]
    sol_1 = minimize(myf.tangency_objective, P5_weights, args=(rf[0], covariance_matrix, mu), method="SLSQP", constraints=tangency_constraints)
    P1_weights = sol_1.x
    P1_return.append(myf.prtf_return(P1_weights, montly_returns_tplus1))
    P1_alpha.append(P1_return[-1]-rf_tplus1[0])


    #2) the portfolio that maximizes the Sharpe ratio with short-sale constraints;
    (sol_2_1, sol_2_2, sol_2_3) = myf.minvarpf_noshortsale(working_monthly_returns, 5, rf[0], risk_free_allowed=False, tangency=True)
    P2_weights = sol_2_3
    P2_return.append(myf.prtf_return(P2_weights, montly_returns_tplus1))
    P2_alpha.append(P2_return[-1]-rf_tplus1[0])


    #3) the portfolio where the weight of each asset is inversely related to its variance;
    P3_weights = 1/np.diag(covariance_matrix)
    Inv_variance = sum(P3_weights)
    P3_weights = P3_weights/Inv_variance
    P3_return.append(myf.prtf_return(P3_weights, montly_returns_tplus1))
    P3_alpha.append(P3_return[-1]-rf_tplus1[0])


    #4) the portfolio where the weight of each asset is inversely related to its volatility;
    P4_weights = 1/np.sqrt(np.diag(covariance_matrix))
    Inv_volatility = sum(P4_weights)
    P4_weights = P4_weights/Inv_volatility
    P4_return.append(myf.prtf_return(P4_weights, montly_returns_tplus1))
    P4_alpha.append(P4_return[-1]-rf_tplus1[0])


    #6) the portfolio where the weight of each is linearly related to its market capitalization;
    total_market_cap = avg_firm_size @ num_firms
    Mrkt_Cap.append(avg_firm_size * num_firms)
    P6_weights = Mrkt_Cap[-1] / total_market_cap
    P6_weights = P6_weights.to_numpy()
    P6_return.append(myf.prtf_return(P6_weights, montly_returns_tplus1))
    P6_alpha.append(P6_return[-1] - rf_tplus1[0])

    # Using the loop to calculate the prtf of Part B #3) (without ss) and #4) (with ss)
    # which track the benchmark portfolios with a maximum tracking error of 1% monthly
    constraint_TE = {'type': 'ineq', 'fun': myf.constraint_on_TE, 'args': (P6_weights, covariance_matrix, TE_threshold)}
    TE_without_ss_constraints = [constraint_weights, constraint_TE, constraint_short_sell_lim]
    sol_3_1 = minimize(myf.tangency_objective, P5_weights, args=(rf[0], covariance_matrix, mu), method="SLSQP", constraints=TE_without_ss_constraints)
    PB_3_1_weights = sol_3_1.x
    PB_3_1_return.append(myf.prtf_return(PB_3_1_weights, montly_returns_tplus1))
    PB_3_1_alpha.append(PB_3_1_return[-1] - rf_tplus1[0])

    TE_with_ss_constraints = [constraint_weights, constraint_TE, constraint_short_sell]
    sol_4_1 = minimize(myf.tangency_objective, P5_weights, args=(rf[0], covariance_matrix, mu), method="SLSQP", constraints=TE_with_ss_constraints)
    PB_4_1_weights = sol_4_1.x
    PB_4_1_return.append(myf.prtf_return(PB_4_1_weights, montly_returns_tplus1))
    PB_4_1_alpha.append(PB_4_1_return[-1] - rf_tplus1[0])


# Computing and comparing performance Sharpe Ratios accross periods
P1_SR_OS1               = np.asarray(P1_alpha).mean() / np.std(np.asarray(P1_return))
P2_SR_OS1               = np.asarray(P2_alpha).mean() / np.std(np.asarray(P2_return))
P3_SR_OS1               = np.asarray(P3_alpha).mean() / np.std(np.asarray(P3_return))
P4_SR_OS1               = np.asarray(P4_alpha).mean() / np.std(np.asarray(P4_return))
P5_SR_OS1               = np.asarray(P5_alpha).mean() / np.std(np.asarray(P5_return))
P6_SR_OS1               = np.asarray(P6_alpha).mean() / np.std(np.asarray(P6_return))
P7_SR_OS1               = np.asarray(P7_alpha).mean() / np.std(np.asarray(P7_return))
PB_3_1_SR_OS1           = np.asarray(PB_3_1_alpha).mean() / np.std(np.asarray(PB_3_1_return))
PB_3_2_SR_OS1           = np.asarray(PB_3_2_alpha).mean() / np.std(np.asarray(PB_3_2_return))
PB_4_1_SR_OS1           = np.asarray(PB_4_1_alpha).mean() / np.std(np.asarray(PB_4_1_return))
PB_4_2_SR_OS1           = np.asarray(PB_4_2_alpha).mean() / np.std(np.asarray(PB_4_2_return))


P1_SR_OS2               = np.asarray(P1_alpha[idx_start_2-idx_start_1:]).mean() / np.std(np.asarray(P1_return[idx_start_2-idx_start_1:]))
P2_SR_OS2               = np.asarray(P2_alpha[idx_start_2-idx_start_1:]).mean() / np.std(np.asarray(P2_return[idx_start_2-idx_start_1:]))
P3_SR_OS2               = np.asarray(P3_alpha[idx_start_2-idx_start_1:]).mean() / np.std(np.asarray(P3_return[idx_start_2-idx_start_1:]))
P4_SR_OS2               = np.asarray(P4_alpha[idx_start_2-idx_start_1:]).mean() / np.std(np.asarray(P4_return[idx_start_2-idx_start_1:]))
P5_SR_OS2               = np.asarray(P5_alpha[idx_start_2-idx_start_1:]).mean() / np.std(np.asarray(P5_return[idx_start_2-idx_start_1:]))
P6_SR_OS2               = np.asarray(P6_alpha[idx_start_2-idx_start_1:]).mean() / np.std(np.asarray(P6_return[idx_start_2-idx_start_1:]))
P7_SR_OS2               = np.asarray(P7_alpha[idx_start_2-idx_start_1:]).mean() / np.std(np.asarray(P7_return[idx_start_2-idx_start_1:]))
PB_3_1_SR_OS2           = np.asarray(PB_3_1_alpha[idx_start_2-idx_start_1:]).mean() / np.std(np.asarray(PB_3_1_return[idx_start_2-idx_start_1:]))
PB_3_2_SR_OS2           = np.asarray(PB_3_2_alpha[idx_start_2-idx_start_1:]).mean() / np.std(np.asarray(PB_3_2_return[idx_start_2-idx_start_1:]))
PB_4_1_SR_OS2           = np.asarray(PB_4_1_alpha[idx_start_2-idx_start_1:]).mean() / np.std(np.asarray(PB_4_1_return[idx_start_2-idx_start_1:]))
PB_4_2_SR_OS2           = np.asarray(PB_4_2_alpha[idx_start_2-idx_start_1:]).mean() / np.std(np.asarray(PB_4_2_return[idx_start_2-idx_start_1:]))


P1_SR_OS3               = np.asarray(P1_alpha[idx_start_3-idx_start_1:]).mean() / np.std(np.asarray(P1_return[idx_start_3-idx_start_1:]))
P2_SR_OS3               = np.asarray(P2_alpha[idx_start_3-idx_start_1:]).mean() / np.std(np.asarray(P2_return[idx_start_3-idx_start_1:]))
P3_SR_OS3               = np.asarray(P3_alpha[idx_start_3-idx_start_1:]).mean() / np.std(np.asarray(P3_return[idx_start_3-idx_start_1:]))
P4_SR_OS3               = np.asarray(P4_alpha[idx_start_3-idx_start_1:]).mean() / np.std(np.asarray(P4_return[idx_start_3-idx_start_1:]))
P5_SR_OS3               = np.asarray(P5_alpha[idx_start_3-idx_start_1:]).mean() / np.std(np.asarray(P5_return[idx_start_3-idx_start_1:]))
P6_SR_OS3               = np.asarray(P6_alpha[idx_start_3-idx_start_1:]).mean() / np.std(np.asarray(P6_return[idx_start_3-idx_start_1:]))
P7_SR_OS3               = np.asarray(P7_alpha[idx_start_3-idx_start_1:]).mean() / np.std(np.asarray(P7_return[idx_start_3-idx_start_1:]))
PB_3_1_SR_OS3           = np.asarray(PB_3_1_alpha[idx_start_3-idx_start_1:]).mean() / np.std(np.asarray(PB_3_1_return[idx_start_3-idx_start_1:]))
PB_3_2_SR_OS3           = np.asarray(PB_3_2_alpha[idx_start_3-idx_start_1:]).mean() / np.std(np.asarray(PB_3_2_return[idx_start_3-idx_start_1:]))
PB_4_1_SR_OS3           = np.asarray(PB_4_1_alpha[idx_start_3-idx_start_1:]).mean() / np.std(np.asarray(PB_4_1_return[idx_start_3-idx_start_1:]))
PB_4_2_SR_OS3           = np.asarray(PB_4_2_alpha[idx_start_3-idx_start_1:]).mean() / np.std(np.asarray(PB_4_2_return[idx_start_3-idx_start_1:]))

# Presenting SR in a table
test_array  = [[P1_SR_OS1, P1_SR_OS2, P1_SR_OS3], [P2_SR_OS1, P2_SR_OS2, P2_SR_OS3], [P3_SR_OS1, P3_SR_OS2, P3_SR_OS3], [P4_SR_OS1, P4_SR_OS2, P4_SR_OS3], [P5_SR_OS1, P5_SR_OS2, P5_SR_OS3], [P6_SR_OS1, P6_SR_OS2, P6_SR_OS3], [P7_SR_OS1, P7_SR_OS2, P7_SR_OS3],[PB_3_1_SR_OS1, PB_3_1_SR_OS2, PB_3_1_SR_OS3],[PB_3_2_SR_OS1, PB_3_2_SR_OS2, PB_3_2_SR_OS3],[PB_4_1_SR_OS1, PB_4_1_SR_OS2, PB_4_1_SR_OS3],[PB_4_2_SR_OS1, PB_4_2_SR_OS2, PB_4_2_SR_OS3]]
SR_table    = pd.DataFrame(np.asarray(test_array), columns=['SR for Jul 1931-Dec 2019', 'SR for Jan 1990-Dec 2019', 'SR for Jan 2000-Dec 2019'])
print(SR_table)


# Computing compounded return for the 1st period
NAV_P1_return           = np.cumprod(1 + np.array(P1_return) / 100)
NAV_P2_return           = np.cumprod(1 + np.array(P2_return) / 100)
NAV_P3_return           = np.cumprod(1 + np.array(P3_return) / 100)
NAV_P4_return           = np.cumprod(1 + np.array(P4_return) / 100)
NAV_P5_return           = np.cumprod(1 + np.array(P5_return) / 100)
NAV_P6_return           = np.cumprod(1 + np.array(P6_return) / 100)
NAV_P7_return           = np.cumprod(1 + np.array(P7_return) / 100)
NAV_PB_3_1_return       = np.cumprod(1 + np.array(PB_3_1_return) / 100)
NAV_PB_3_2_return       = np.cumprod(1 + np.array(PB_3_2_return) / 100)
NAV_PB_4_1_return       = np.cumprod(1 + np.array(PB_4_1_return) / 100)
NAV_PB_4_2_return       = np.cumprod(1 + np.array(PB_4_2_return) / 100)


# Computing compounded return for the 2nd period
NAV_P1_return_OS2       = np.cumprod(1 + np.array(P1_return[idx_start_2-idx_start_1:]) / 100)
NAV_P2_return_OS2       = np.cumprod(1 + np.array(P2_return[idx_start_2-idx_start_1:]) / 100)
NAV_P3_return_OS2       = np.cumprod(1 + np.array(P3_return[idx_start_2-idx_start_1:]) / 100)
NAV_P4_return_OS2       = np.cumprod(1 + np.array(P4_return[idx_start_2-idx_start_1:]) / 100)
NAV_P5_return_OS2       = np.cumprod(1 + np.array(P5_return[idx_start_2-idx_start_1:]) / 100)
NAV_P6_return_OS2       = np.cumprod(1 + np.array(P6_return[idx_start_2-idx_start_1:]) / 100)
NAV_P7_return_OS2       = np.cumprod(1 + np.array(P7_return[idx_start_2-idx_start_1:]) / 100)
NAV_PB_3_1_return_OS2   = np.cumprod(1 + np.array(PB_3_1_return[idx_start_2-idx_start_1:]) / 100)
NAV_PB_3_2_return_OS2   = np.cumprod(1 + np.array(PB_3_2_return[idx_start_2-idx_start_1:]) / 100)
NAV_PB_4_1_return_OS2   = np.cumprod(1 + np.array(PB_4_1_return[idx_start_2-idx_start_1:]) / 100)
NAV_PB_4_2_return_OS2   = np.cumprod(1 + np.array(PB_4_2_return[idx_start_2-idx_start_1:]) / 100)


# Computing compounded return for the 3rd period and Graph
NAV_P1_return_OS3       = np.cumprod(1 + np.array(P1_return[idx_start_3-idx_start_1:]) / 100)
NAV_P2_return_OS3       = np.cumprod(1 + np.array(P2_return[idx_start_3-idx_start_1:]) / 100)
NAV_P3_return_OS3       = np.cumprod(1 + np.array(P3_return[idx_start_3-idx_start_1:]) / 100)
NAV_P4_return_OS3       = np.cumprod(1 + np.array(P4_return[idx_start_3-idx_start_1:]) / 100)
NAV_P5_return_OS3       = np.cumprod(1 + np.array(P5_return[idx_start_3-idx_start_1:]) / 100)
NAV_P6_return_OS3       = np.cumprod(1 + np.array(P6_return[idx_start_3-idx_start_1:]) / 100)
NAV_P7_return_OS3       = np.cumprod(1 + np.array(P7_return[idx_start_3-idx_start_1:]) / 100)
NAV_PB_3_1_return_OS3   = np.cumprod(1 + np.array(PB_3_1_return[idx_start_3-idx_start_1:]) / 100)
NAV_PB_3_2_return_OS3   = np.cumprod(1 + np.array(PB_3_2_return[idx_start_3-idx_start_1:]) / 100)
NAV_PB_4_1_return_OS3   = np.cumprod(1 + np.array(PB_4_1_return[idx_start_3-idx_start_1:]) / 100)
NAV_PB_4_2_return_OS3   = np.cumprod(1 + np.array(PB_4_2_return[idx_start_3-idx_start_1:]) / 100)


################### Graphs of Accrued returns accross the different periods

date_vec_btst = pd.to_datetime(date_vec_btst)
date_vec_prd2 = pd.to_datetime(date_vec_prd2)
date_vec_prd3 = pd.to_datetime(date_vec_prd3)

#   NAV_P1_return,     Keeping this series in cmt as doesn't fit properly
NAV = np.transpose([NAV_P1_return, NAV_P2_return, NAV_P3_return, NAV_P4_return, NAV_P5_return, NAV_P6_return, NAV_P7_return])
fig1 = plt.figure()
axes1 = fig1.add_axes([0.01, 0.01, 0.9, 0.9])  # left, bottom, width, height (range 0 to 1)
x = date_vec_btst
y = NAV
axes1.plot(x, y)
axes1.legend(["max Sharpe without ssc", "max Sharpe with ssc", "1/var", "1/vol", "1/N", "MC weighted", "min var"])
axes1.set_xlabel('Dates')
axes1.set_ylabel('Prtf Value')
axes1.set_title('Portfolio values for the period July 1931 to December 2019')
fig1.show()
fig1.savefig('7_Portfolios_OS1.png')


#NAV_P1_return_OS2,
NAV_OS2 = np.transpose([NAV_P1_return_OS2, NAV_P2_return_OS2, NAV_P3_return_OS2, NAV_P4_return_OS2, NAV_P5_return_OS2, NAV_P6_return_OS2, NAV_P7_return_OS2])
fig2 = plt.figure()
axes2 = fig2.add_axes([0.1, 0.1, 0.8, 0.8])  # left, bottom, width, height (range 0 to 1)
x = date_vec_prd2
y = NAV_OS2
axes2.plot(x, y)
axes2.legend(["max Sharpe without ssc", "max Sharpe with ssc", "1/var", "1/vol", "1/N", "MC weighted", "min var"])
axes2.set_xlabel('Dates')
axes2.set_ylabel('Prtf Value')
axes2.set_title('Portfolio values for the period January 1990 to December 2019')
fig2.show()
fig2.savefig('7_Portfolios_OS2.png')

#NAV_P1_return_OS3,
NAV_OS3 = np.transpose([NAV_P1_return_OS3, NAV_P2_return_OS3, NAV_P3_return_OS3, NAV_P4_return_OS3, NAV_P5_return_OS3, NAV_P6_return_OS3, NAV_P7_return_OS3])
fig3 = plt.figure()
axes3 = fig3.add_axes([0.1, 0.1, 0.8, 0.8])  # left, bottom, width, height (range 0 to 1)
x = date_vec_prd3
y = NAV_OS3
axes3.plot(x, y)
axes3.legend(["max Sharpe without ssc", "max Sharpe with ssc", "1/var", "1/vol", "1/N", "MC weighted", "min var"])
axes3.set_xlabel('Dates')
axes3.set_ylabel('Prtf Value')
axes3.set_title('Portfolio values for the period January 2000 to December 2019')
fig3.show()
fig3.savefig('7_Portfolios_OS3.png')


# Graph of NAV of MC prtfs during enitre period
#NAV_PB_3_1_return,     in legend   "Prtf B3: TE<1% without short-sale const",
NAV_PartB_MC = np.transpose([NAV_P6_return, NAV_PB_3_1_return, NAV_PB_4_1_return])
figvw1 = plt.figure()
axesvw1 = figvw1.add_axes([0.1, 0.1, 0.8, 0.8])  # left, bottom, width, height (range 0 to 1)
x = date_vec_btst
y = NAV_PartB_MC
axesvw1.plot(x, y)
axesvw1.legend(["Bench: MC weighted", "Tracking without short-sale const", "Tracking with short-sale const"])
axesvw1.set_xlabel('Dates')
axesvw1.set_ylabel('Prtf Value')
axesvw1.set_title('Portfolio values for the period July 1931 to December 2019')
figvw1.show()
figvw1.savefig('VW_MV_OS1.png')

# Graph of NAV of MC prtfs during 2rd period
# NAV_PB_3_1_return_OS2, "Prtf B3: TE<1% without short-sale const",
NAV_PartB_MC_OS2 = np.transpose([NAV_P6_return_OS2, NAV_PB_3_1_return_OS2, NAV_PB_4_1_return_OS2])
figvw2 = plt.figure()
axesvw2 = figvw2.add_axes([0.1, 0.1, 0.8, 0.8])  # left, bottom, width, height (range 0 to 1)
x = date_vec_prd2
y = NAV_PartB_MC_OS2
axesvw2.plot(x, y)
axesvw2.legend(["Bench: MC weighted", "Tracking without short-sale const", "Tracking with short-sale const"])
axesvw2.set_xlabel('Dates')
axesvw2.set_ylabel('Prtf Value')
axesvw2.set_title('Portfolio values for the period January 1990 to December 2019')
figvw2.show()
figvw2.savefig('VW_MV_OS2.png')


# Graph of NAV of MC prtfs during 3rd period
# NAV_PB_3_1_return_OS3,   "Prtf B3: TE<1% without short-sale const",
NAV_PartB_MC_OS3 = np.transpose([NAV_P6_return_OS3, NAV_PB_3_1_return_OS3, NAV_PB_4_1_return_OS3])
figvw3 = plt.figure()
axesvw3 = figvw3.add_axes([0.1, 0.1, 0.8, 0.8])  # left, bottom, width, height (range 0 to 1)
x = date_vec_prd3
y = NAV_PartB_MC_OS3
axesvw3.plot(x, y)
axesvw3.legend(["Bench: MC weighted", "Tracking without short-sale const", "Tracking with short-sale const"])
axesvw3.set_xlabel('Dates')
axesvw3.set_ylabel('Prtf Value')
axesvw3.set_title('Portfolio values for the period January 2000 to December 2019')
figvw3.show()
figvw3.savefig('VW_MV_OS3.png')


# Graph of NAV of EQW prtfs during enitre period
#NAV_PB_3_2_return,     in legend   "Prtf B3: TE<1% without short-sale const",
NAV_PartB_EQW = np.transpose([NAV_P5_return, NAV_PB_3_2_return, NAV_PB_4_2_return])
figeqw1 = plt.figure()
axeseqw1 = figeqw1.add_axes([0.1, 0.1, 0.8, 0.8])  # left, bottom, width, height (range 0 to 1)
x = date_vec_btst
y = NAV_PartB_EQW
axeseqw1.plot(x, y)
axeseqw1.legend(["Bench: EQW weighted", "Tracking without short-sale const", "Tracking with short-sale const"])
axeseqw1.set_xlabel('Dates')
axeseqw1.set_ylabel('Prtf Value')
plt.yscale('log')
axeseqw1.set_title('Portfolio values for the period July 1931 to December 2019')
figeqw1.show()
figeqw1.savefig('EQW_OS1.png')


# Graph of NAV of EQW prtfs during 2rd period
# NAV_PB_3_2_return_OS2, "Prtf B3: TE<1% without short-sale const",
NAV_PartB_EQW_OS2 = np.transpose([NAV_P5_return_OS2, NAV_PB_3_2_return_OS2, NAV_PB_4_2_return_OS2])
figeqw2 = plt.figure()
axeseqw2 = figeqw2.add_axes([0.1, 0.1, 0.8, 0.8])  # left, bottom, width, height (range 0 to 1)
x = date_vec_prd2
y = NAV_PartB_EQW_OS2
axeseqw2.plot(x, y)
axeseqw2.legend(["Bench: EQW weighted", "Tracking without short-sale const", "Tracking with short-sale const"])
axeseqw2.set_xlabel('Dates')
axeseqw2.set_ylabel('Prtf Value')
axeseqw2.set_title('Portfolio values for the period January 1990 to December 2019')
figeqw2.show()
figeqw2.savefig('EQW_OS2.png')

# Graph of NAV of EQW prtfs during 3rd period
# NAV_PB_3_2_return_OS3,   "Prtf B3: TE<1% without short-sale const",
NAV_PartB_EQW_OS3 = np.transpose([NAV_P5_return_OS3, NAV_PB_3_2_return_OS3, NAV_PB_4_2_return_OS3])
figeqw3 = plt.figure()
axeseqw3 = figeqw3.add_axes([0.1, 0.1, 0.8, 0.8])  # left, bottom, width, height (range 0 to 1)
x = date_vec_prd3
y = NAV_PartB_EQW_OS3
axeseqw3.plot(x, y)
axeseqw3.legend(["Bench: EQW weighted", "Tracking without short-sale const", "Tracking with short-sale const"])
axeseqw3.set_xlabel('Dates')
axeseqw3.set_ylabel('Prtf Value')
axeseqw3.set_title('Portfolio values for the period January 2000 to December 2019')
figeqw3.show()
figeqw3.savefig('EQW_OS3.png')


## ---------- Part_B #5 #6 #7 #8 ----------------------------------------------------------

# Parameters for Part B
risk_aversion = 5


# Calculating the portfolio characteristics for the all period T and not just for the OS period
Size_MC         = all_avg_firm_size * all_num_firms

Value_BM        = sum_BE_div_sum_ME.loc[sum_BE_div_sum_ME.index.repeat(12)]
Value_BM        = Value_BM.iloc[:-5, :]
Value_BM.index  = Size_MC.index

Momentum        = all_monthly_returns.rolling(12, min_periods=1).mean()

#-------------------------------------------------------------------------------------------
# Recalculating the benchmark porfolios weights
Bench_EQW_weights = np.full((rolling_window,10),0.1)
Total_Mrkt_Caps     = []    # Total_Mrkt_Caps   T x 1
Bench_MC_weights    = []    # Total_Mrkt_Caps   T x 10

for dates in Size_MC.index:
    Total_Mrkt_Caps.append(all_avg_firm_size.loc[dates, :] @ all_num_firms.loc[dates, :])
    den = pd.DataFrame(Total_Mrkt_Caps).iloc[-1][0]
    Bench_MC_weights.append(Size_MC.loc[dates, :].div(den))
Bench_MC_weights = pd.DataFrame(Bench_MC_weights)

OS_period = Value_BM.index["1963-07-01" <= Value_BM.index]
Theta_mc = []
Theta_eqw = []
P8_MC_weights = []
P8_MC_return = []
P8_EQW_weights = []
P8_EQW_return = []

for dates in OS_period:
    print(dates)
    idx_date = Value_BM.index.get_loc(dates)
    Bench_MC_weights_IS = Bench_MC_weights.iloc[idx_date-rolling_window-1:idx_date-1]
    Size_MC_IS =  Size_MC.iloc[idx_date-rolling_window-1:idx_date-1]
    Value_BM_IS =  Value_BM.iloc[idx_date-rolling_window-1:idx_date-1]
    Momentum_IS =  Momentum.iloc[idx_date-rolling_window-1:idx_date-1]
    all_monthly_returns_IS =  all_monthly_returns.iloc[idx_date-rolling_window-1:idx_date]

    # Transforming data to numpy array as quicker to compute than dataframes
    Bench_MC_weights_IS = np.asarray(Bench_MC_weights_IS)
    Size_MC_IS = Size_MC_IS.to_numpy()
    Value_BM_IS = Value_BM_IS.to_numpy()
    Momentum_IS = Momentum_IS.to_numpy()
    all_monthly_returns_IS = all_monthly_returns_IS.to_numpy()

    all_monthly_returns_IS = all_monthly_returns_IS[1:,:]             # removing first month of returns (since we sum starting at t=1)
    #Bench_MC_weights = Bench_MC_weights[:-1,:]                       # removing last month of weights (since summed to T-1)
    #Size_MC = Size_MC[:-1, :]
    #Value_BM = Value_BM[:-1, :]
    #Momentum = Momentum[:-1, :]

    # Getting weights and returns of MC Portf
    fun = minimize(myf.objective_8, [0.4, 0.4, 0.2], args=(Bench_MC_weights_IS, Size_MC_IS, Value_BM_IS, Momentum_IS, all_monthly_returns_IS, risk_aversion),
                   method="SLSQP")

    tt_mc = fun.x
    Theta_mc.append(tt_mc)


    (tmp1, tmp2) = myf.prtf8(tt_mc, Bench_MC_weights.iloc[idx_date,:], Size_MC.iloc[idx_date,:], Value_BM.iloc[idx_date,:], Momentum.iloc[idx_date,:], all_monthly_returns.iloc[idx_date,:])
    P8_MC_weights.append(tmp1)
    P8_MC_return.append(tmp2)


    # Getting weights and returns of EQW Portf

    fun = minimize(myf.objective_8, [0.4, 0.4, 0.2], args=(Bench_EQW_weights, Size_MC_IS, Value_BM_IS, Momentum_IS, all_monthly_returns_IS, risk_aversion),
                   method="SLSQP")

    tt_eqw = fun.x
    Theta_eqw.append(tt_eqw)

    (tmp1, tmp2)= myf.prtf8(tt_eqw, Bench_EQW_weights[1,:], Size_MC.iloc[idx_date,:], Value_BM.iloc[idx_date,:], Momentum.iloc[idx_date,:], all_monthly_returns.iloc[idx_date,:])
    P8_EQW_weights.append(tmp1)
    P8_EQW_return.append(tmp2)

print(Theta_mc)
print(Theta_eqw)

#
## Aggregate Portfolio Returns in lists in order to evaluate Performance
#

# Market Cap Weighted portfolios

P8_MC_return    = pd.DataFrame(P8_MC_return, columns=['MC Tilt'], index=OS_period)
#8_MC_return    = P8_MC_return * 100
P8_MC_return = P8_MC_return.loc["1963-07-01":]
zippedList_mc = list(zip(P6_return, PB_3_1_return, PB_4_1_return))
MC_ports = pd.DataFrame(zippedList_mc, columns=['Mkt Cap', 'TE w/o ssc', 'TE w ssc'], index=date_vec_btst)
MC_ports = MC_ports.loc["1963-07-01":, :]
MC_ports = pd.concat([MC_ports, P8_MC_return], axis=1)

NAV_P8_1_return = np.cumprod(1 + np.array(MC_ports) / 100, axis=0)
fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])  # left, bottom, width, height (range 0 to 1)
x = OS_period
y = NAV_P8_1_return
axes.plot(x,y)
axes.legend(["Bench: MC weighted", "Tracking without short-sale const", "Tracking with short-sale const", "MC Tilted"])
axes.set_xlabel('Dates')
axes.set_ylabel('Prtf Value')
axes.set_title('Portfolio values for the period July 1963 to December 2019')
fig.show()
fig.savefig('B8_MC_1.png')

MC_ports_2 = MC_ports.loc["1990-01-01":, :]
OS_period_2 = Value_BM.index["1990-01-01" <= Value_BM.index]
NAV_P8_1_return_2 = np.cumprod(1 + np.array(MC_ports_2) / 100, axis=0)
fig2 = plt.figure()
axes = fig2.add_axes([0.1, 0.1, 0.8, 0.8])  # left, bottom, width, height (range 0 to 1)
x = OS_period_2
y = NAV_P8_1_return_2
axes.plot(x,y)
axes.legend(["Bench: MC weighted", "Tracking without short-sale const", "Tracking with short-sale const", "MC Tilted"])
axes.set_xlabel('Dates')
axes.set_ylabel('Prtf Value')
axes.set_title('Portfolio values for the period January 1990 to December 2019')
fig2.show()
fig2.savefig('B8_MC_2.png')

MC_ports_3 = MC_ports.loc["2000-01-01":, :]
OS_period_3 = Value_BM.index["2000-01-01" <= Value_BM.index]
NAV_P8_1_return_3 = np.cumprod(1 + np.array(MC_ports_3) / 100, axis=0)
fig3 = plt.figure()
axes = fig3.add_axes([0.1, 0.1, 0.8, 0.8])  # left, bottom, width, height (range 0 to 1)
x = OS_period_3
y = NAV_P8_1_return_3
axes.plot(x,y)
axes.legend(["Bench: MC weighted", "Tracking without short-sale const", "Tracking with short-sale const", "MC Tilted"])
axes.set_xlabel('Dates')
axes.set_ylabel('Prtf Value')
axes.set_title('Portfolio values for the period January 2000 to December 2019')
fig3.show()
fig3.savefig('B8_MC_3.png')

MC_ports.index = myf.eomonth(MC_ports.index)

## Performance

# Input: DataFrame of returns according to each strategies
input_file_path = './data/FAMA_3_Factors.CSV'
df_Fama_3 = pd.read_csv(input_file_path,
                   index_col = 0)
input_file_path = './data/FAMA_5_Factors.CSV'
df_Fama_5 = pd.read_csv(input_file_path,
                   index_col = 0)
df_Fama_3.index = myf.eomonth(pd.to_datetime(df_Fama_3.index, format= '%Y%m'))
df_Fama_5.index = myf.eomonth(pd.to_datetime(df_Fama_5.index, format= '%Y%m'))


period1 = [pd.to_datetime(datetime.datetime(1963, 7, 31)) , pd.to_datetime(datetime.datetime(2019, 12, 31))]
period2 = [pd.to_datetime(datetime.datetime(1990, 1, 1)) , pd.to_datetime(datetime.datetime(2019, 12, 31))]
period3 = [pd.to_datetime(datetime.datetime(2000, 1, 1)) , pd.to_datetime(datetime.datetime(2019, 12, 31))]
period = [period1, period2, period3] #, period2, period3]


for subperiod in period:
    start_date = subperiod[0]
    end_date = subperiod[1]
    date_vec_tmp = MC_ports.index[(MC_ports.index >= start_date)&(MC_ports.index <= end_date)]
    returns_tmp = MC_ports.loc[date_vec_tmp, :]
    strat = MC_ports.columns
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


# Equally weighted portfolios
P8_EQW_return    = pd.DataFrame(P8_EQW_return, columns=['EQW Tilt'], index=OS_period)
#P8_EQW_return    = P8_EQW_return * 100
P8_EQW_return = P8_EQW_return.loc["1963-07-01":]
zippedList_eqw = list(zip(P5_return, PB_3_2_return, PB_4_2_return))
EQW_ports = pd.DataFrame(zippedList_eqw, columns=['EQW', 'TE w/o ssc', 'TE w ssc'], index=date_vec_btst)
EQW_ports = EQW_ports.loc["1963-07-01":, :]
EQW_ports = pd.concat([EQW_ports, P8_EQW_return], axis=1)

NAV_P8_2_return = np.cumprod(1 + np.array(EQW_ports) / 100, axis=0)
fig4 = plt.figure()
axes = fig4.add_axes([0.1, 0.1, 0.8, 0.8])  # left, bottom, width, height (range 0 to 1)
x = OS_period
y = NAV_P8_2_return
axes.plot(x,y)
axes.legend(["Bench: EQW", "Tracking without short-sale const", "Tracking with short-sale const", "EQW Tilted"])
axes.set_xlabel('Dates')
axes.set_ylabel('Prtf Value')
axes.set_title('Portfolio values for the period July 1963 to December 2019')
fig4.show()
fig4.savefig('B8_EQW_1.png')

EQW_ports_2 = EQW_ports.loc["1990-01-01":, :]
EQW_period_2 = Value_BM.index["1990-01-01" <= Value_BM.index]
NAV_P8_2_return_2 = np.cumprod(1 + np.array(EQW_ports_2) / 100, axis=0)
fig5 = plt.figure()
axes = fig5.add_axes([0.1, 0.1, 0.8, 0.8])  # left, bottom, width, height (range 0 to 1)
x = EQW_period_2
y = NAV_P8_2_return_2
axes.plot(x,y)
axes.legend(["Bench: EQW weighted", "Tracking without short-sale const", "Tracking with short-sale const", "EQW Tilted"])
axes.set_xlabel('Dates')
axes.set_ylabel('Prtf Value')
axes.set_title('Portfolio values for the period January 1990 to December 2019')
fig5.show()
fig5.savefig('B8_EQW_2.png')

EQW_ports_3 = EQW_ports.loc["2000-01-01":, :]
EQW_period_3 = Value_BM.index["2000-01-01" <= Value_BM.index]
NAV_P8_2_return_3 = np.cumprod(1 + np.array(EQW_ports_3) / 100, axis=0)
fig6 = plt.figure()
axes = fig6.add_axes([0.1, 0.1, 0.8, 0.8])  # left, bottom, width, height (range 0 to 1)
x = EQW_period_3
y = NAV_P8_2_return_3
axes.plot(x,y)
axes.legend(["Bench: EQW weighted", "Tracking without short-sale const", "Tracking with short-sale const", "EQW Tilted"])
axes.set_xlabel('Dates')
axes.set_ylabel('Prtf Value')
axes.set_title('Portfolio values for the period January 2000 to December 2019')
fig6.show()
fig6.savefig('B8_EQW_3.png')


date_tmp = myf.eomonth(EQW_ports.index)
EQW_ports.index = date_tmp


## Performance calculation


period1 = [pd.to_datetime(datetime.datetime(1963, 7, 31)) , pd.to_datetime(datetime.datetime(2019, 12, 31))]
period2 = [pd.to_datetime(datetime.datetime(1990, 1, 1)) , pd.to_datetime(datetime.datetime(2019, 12, 31))]
period3 = [pd.to_datetime(datetime.datetime(2000, 1, 1)) , pd.to_datetime(datetime.datetime(2019, 12, 31))]
period = [period1, period2, period3]


for subperiod in period:
    start_date = subperiod[0]
    end_date = subperiod[1]
    date_vec_tmp = EQW_ports.index[(EQW_ports.index >= start_date)&(EQW_ports.index <= end_date)]
    returns_tmp = EQW_ports.loc[date_vec_tmp, :]
    strat = EQW_ports.columns
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

print(True)
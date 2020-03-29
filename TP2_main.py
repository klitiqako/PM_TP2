# Import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools as it
from scipy.optimize import minimize
import random

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
rolling_window      = 60  # Number of weeks in period used for estimation 5 years

start_date_1        = "1931-07-01"
start_date_2        = "1990-01-01"
start_date_3        = "2000-01-01"
end_date            = "2020-01-01"

#To test shorter period
start_date_2        = "1933-01-01"
start_date_3        = "1934-01-01"
end_date            = "1935-01-01"


idx_start_1         = list(date_vec.strftime("%Y-%m-%d")).index(start_date_1)
idx_start_2         = list(date_vec.strftime("%Y-%m-%d")).index(start_date_2)
idx_start_3         = list(date_vec.strftime("%Y-%m-%d")).index(start_date_3)
idx_end             = list(date_vec.strftime("%Y-%m-%d")).index(end_date)
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


#   NAV_P1_return,     Keeping this series in cmt as doesn't fit properly
NAV = np.transpose([NAV_P1_return, NAV_P2_return, NAV_P3_return, NAV_P4_return, NAV_P5_return, NAV_P6_return, NAV_P7_return])
fig1 = plt.figure()
axes1 = fig1.add_axes([0.1, 0.1, 2, 1])  # left, bottom, width, height (range 0 to 1)
x = date_vec_btst
y = NAV
axes1.plot(x, y)
axes1.legend(["Prtf 1", "Prtf 2", "Prtf 3", "Prtf 4", "Prtf 5", "Prtf 6", "Prtf 7"])
axes1.set_xlabel('Dates')
axes1.set_ylabel('Prtf Value')
axes1.set_title('Portfolio values for the period July 1931 to December 2019')
fig1.show()
fig1.savefig('7_Portfolios_OS1.png')


#NAV_P1_return_OS2,
NAV_OS2 = np.transpose([NAV_P1_return_OS2, NAV_P2_return_OS2, NAV_P3_return_OS2, NAV_P4_return_OS2, NAV_P5_return_OS2, NAV_P6_return_OS2, NAV_P7_return_OS2])
fig2 = plt.figure()
axes2 = fig2.add_axes([0.1, 0.1, 2, 1])  # left, bottom, width, height (range 0 to 1)
x = date_vec_prd2
y = NAV_OS2
axes2.plot(x, y)
axes2.legend(["Prtf 1", "Prtf 2", "Prtf 3", "Prtf 4", "Prtf 5", "Prtf 6", "Prtf 7"])
axes2.set_xlabel('Dates')
axes2.set_ylabel('Prtf Value')
axes2.set_title('Portfolio values for the period January 1990 to December 2019')
fig2.show()
fig2.savefig('7_Portfolios_OS2.png')

#NAV_P1_return_OS3,
NAV_OS3 = np.transpose([NAV_P1_return_OS3, NAV_P2_return_OS3, NAV_P3_return_OS3, NAV_P4_return_OS3, NAV_P5_return_OS3, NAV_P6_return_OS3, NAV_P7_return_OS3])
fig3 = plt.figure()
axes3 = fig3.add_axes([0.1, 0.1, 2, 1])  # left, bottom, width, height (range 0 to 1)
x = date_vec_prd3
y = NAV_OS3
axes3.plot(x, y)
axes3.legend(["Prtf 1", "Prtf 2", "Prtf 3","Prtf 4","Prtf 5","Prtf 6","Prtf 7"])
axes3.set_xlabel('Dates')
axes3.set_ylabel('Prtf Value')
axes3.set_title('Portfolio values for the period January 2000 to December 2019')
fig3.show()
fig3.savefig('7_Portfolios_OS3.png')


# Graph of NAV of MC prtfs during enitre period
#NAV_PB_3_1_return,     in legend   "Prtf B3: TE<1% without short-sale const",
NAV_PartB_MC = np.transpose([NAV_P6_return, NAV_PB_3_1_return, NAV_PB_4_1_return])
figvw1 = plt.figure()
axesvw1 = figvw1.add_axes([0.1, 0.1, 2, 1])  # left, bottom, width, height (range 0 to 1)
x = date_vec_btst
y = NAV_PartB_MC
axesvw1.plot(x, y)
axesvw1.legend(["Prtf 6: MC weighted", "Prtf B3: TE<1% without short-sale const", "Prtf B4: TE<1% with short-sale const"])
axesvw1.set_xlabel('Dates')
axesvw1.set_ylabel('Prtf Value')
axesvw1.set_title('Portfolio values for the period July 1931 to December 2019')
figvw1.show()
figvw1.savefig('VW_MV_OS1.png')

# Graph of NAV of MC prtfs during 2rd period
# NAV_PB_3_1_return_OS2, "Prtf B3: TE<1% without short-sale const",
NAV_PartB_MC_OS2 = np.transpose([NAV_P6_return_OS2, NAV_PB_3_1_return_OS2, NAV_PB_4_1_return_OS2])
figvw2 = plt.figure()
axesvw2 = figvw2.add_axes([0.1, 0.1, 2, 1])  # left, bottom, width, height (range 0 to 1)
x = date_vec_prd2
y = NAV_PartB_MC_OS2
axesvw2.plot(x, y)
axesvw2.legend(["Prtf 6: MC weighted", "Prtf B3: TE<1% without short-sale const", "Prtf B4: TE<1% with short-sale const"])
axesvw2.set_xlabel('Dates')
axesvw2.set_ylabel('Prtf Value')
axesvw2.set_title('Portfolio values for the period January 1990 to December 2019')
figvw2.show()
figvw2.savefig('VW_MV_OS2.png')


# Graph of NAV of MC prtfs during 3rd period
# NAV_PB_3_1_return_OS3,   "Prtf B3: TE<1% without short-sale const",
NAV_PartB_MC_OS3 = np.transpose([NAV_P6_return_OS3, NAV_PB_3_1_return_OS3, NAV_PB_4_1_return_OS3])
figvw3 = plt.figure()
axesvw3 = figvw3.add_axes([0.1, 0.1, 2, 1])  # left, bottom, width, height (range 0 to 1)
x = date_vec_prd3
y = NAV_PartB_MC_OS3
axesvw3.plot(x, y)
axesvw3.legend(["Prtf 6: MC weighted", "Prtf B3: TE<1% without short-sale const", "Prtf B4: TE<1% with short-sale const"])
axesvw3.set_xlabel('Dates')
axesvw3.set_ylabel('Prtf Value')
axesvw3.set_title('Portfolio values for the period January 2000 to December 2019')
figvw3.show()
figvw3.savefig('VW_MV_OS3.png')


# Graph of NAV of EQW prtfs during enitre period
#NAV_PB_3_2_return,     in legend   "Prtf B3: TE<1% without short-sale const",
NAV_PartB_EQW = np.transpose([NAV_P5_return, NAV_PB_3_2_return, NAV_PB_4_2_return])
figeqw1 = plt.figure()
axeseqw1 = figeqw1.add_axes([0.1, 0.1, 2, 1])  # left, bottom, width, height (range 0 to 1)
x = date_vec_btst
y = NAV_PartB_EQW
axeseqw1.plot(x, y)
axeseqw1.legend(["Prtf 5: EQW weighted", "Prtf B3: TE<1% without short-sale const","Prtf B4: TE<1% with short-sale const"])
axeseqw1.set_xlabel('Dates')
axeseqw1.set_ylabel('Prtf Value')
axeseqw1.set_title('Portfolio values for the period July 1931 to December 2019')
figeqw1.show()
figeqw1.savefig('EQW_OS1.png')


# Graph of NAV of EQW prtfs during 2rd period
# NAV_PB_3_2_return_OS2, "Prtf B3: TE<1% without short-sale const",
NAV_PartB_EQW_OS2 = np.transpose([NAV_P5_return_OS2, NAV_PB_3_2_return_OS2, NAV_PB_4_2_return_OS2])
figeqw2 = plt.figure()
axeseqw2 = figeqw2.add_axes([0.1, 0.1, 2, 1])  # left, bottom, width, height (range 0 to 1)
x = date_vec_prd2
y = NAV_PartB_EQW_OS2
axeseqw2.plot(x, y)
axeseqw2.legend(["Prtf 5: EQW weighted", "Prtf B3: TE<1% without short-sale const", "Prtf B4: TE<1% with short-sale const"])
axeseqw2.set_xlabel('Dates')
axeseqw2.set_ylabel('Prtf Value')
axeseqw2.set_title('Portfolio values for the period January 1990 to December 2019')
figeqw2.show()
figeqw2.savefig('EQW_OS2.png')

# Graph of NAV of EQW prtfs during 3rd period
# NAV_PB_3_2_return_OS3,   "Prtf B3: TE<1% without short-sale const",
NAV_PartB_EQW_OS3 = np.transpose([NAV_P5_return_OS3, NAV_PB_3_2_return_OS3, NAV_PB_4_2_return_OS3])
figeqw3 = plt.figure()
axeseqw3 = figeqw3.add_axes([0.1, 0.1, 2, 1])  # left, bottom, width, height (range 0 to 1)
x = date_vec_prd3
y = NAV_PartB_EQW_OS3
axeseqw3.plot(x, y)
axeseqw3.legend(["Prtf 5: EQW weighted", "Prtf B3: TE<1% without short-sale const", "Prtf B4: TE<1% with short-sale const"])
axeseqw3.set_xlabel('Dates')
axeseqw3.set_ylabel('Prtf Value')
axeseqw3.set_title('Portfolio values for the period January 2000 to December 2019')
figeqw3.show()
figeqw3.savefig('EQW_OS3.png')


## ---------- Part_B #5 #6 #7 #8 ----------------------------------------------------------

# Calculating the portfolio characteristics for the all period T and not just for the OS period

Size_MC         = all_avg_firm_size * all_num_firms

Value_BM        = sum_BE_div_sum_ME.loc[sum_BE_div_sum_ME.index.repeat(12)]
Value_BM        = Value_BM.iloc[:-5, :]
Value_BM.index  = Size_MC.index


Momentum        = all_monthly_returns.rolling(12, min_periods=1).mean()

#-------------------------------------------------------------------------------------------
# Recalculating the benchmark porfolio weights by Market Cap
Total_Mrkt_Caps     = []    # Total_Mrkt_Caps   T x 1
Bench_MC_weights    = []    # Total_Mrkt_Caps   T x 10

for dates in Size_MC.index:
    Total_Mrkt_Caps.append(all_avg_firm_size.loc[dates, :] @ all_num_firms.loc[dates, :])
    den = pd.DataFrame(Total_Mrkt_Caps).iloc[-1][0]
    Bench_MC_weights.append(Size_MC.loc[dates, :].div(den))

# Transforming data to numpy array as quicker to compute than dataframes
# Bench_MC_weights = np.asarray(Bench_MC_weights)
# Size_MC = Size_MC.to_numpy()
# Value_BM = Value_BM.to_numpy()
# Momentum = Momentum.to_numpy()
# all_monthly_returns = all_monthly_returns.to_numpy()
#
# all_monthly_returns = all_monthly_returns[1:,:]            # removing first month of returns (since we sum starting at t=1)
# Bench_MC_weights = Bench_MC_weights[:-1,:]                 # removing last month of weights (since summed to T-1)
# Size_MC = Size_MC[:-1, :]
# Value_BM = Value_BM[:-1, :]
# Momentum = Momentum[:-1, :]

#fun = myf.objective_8([0.4, 0.4, 0.2], Bench_MC_weights, Size_MC, Value_BM, Momentum, all_monthly_returns)
fun = minimize(myf.objective_8, [0.4, 0.4, 0.2], args=(Bench_MC_weights, Size_MC, Value_BM, Momentum, all_monthly_returns),
               method="SLSQP")

Theta = fun.x
print(Theta)

P8_weights = myf.prtf8(Theta, Bench_MC_weights, Size_MC, Value_BM, Momentum, all_monthly_returns)

print(True)
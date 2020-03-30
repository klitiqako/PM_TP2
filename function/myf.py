# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 22:52:32 2020
@author: Keanu Vivish
"""
import numpy as np
import random
import pandas as pd
from scipy.optimize import minimize
from scipy import stats

## ----- Min variance portfolio by mimimizing the variance ----- ##
def objective(weights_vector, covariance_matrix):
    weights = np.asarray(weights_vector)
    weights_transposed = weights.reshape(-1, 1)
    return ((weights.dot(covariance_matrix)).dot(weights_transposed))


## ----- Tangency portfolio by maximizing sharpe ratio ----- ##
def tangency_objective(weights_vector, risk_free_rate, covariance_matrix, mu):
    weights = np.asarray(weights_vector)
    weights_transposed = weights.reshape(-1, 1)
    ## HINT: There is a problem with solving this ....
    return (risk_free_rate - (weights.dot(mu)) / (np.sqrt((weights.dot(covariance_matrix)).dot(weights_transposed))))


## ----- Defining constraints ----- ##
def constraint_on_weights(weights):
    return sum(weights) - 1


def constraint_on_expected_return(weights, mu, expected_return):
    return mu.dot(weights) - expected_return


def constraint_on_short_sell(weights):
    return weights

def constraint_on_short_sell_lim(weights):
    return weights + 1


def constraint_on_TE(weights, bench_weights, covariance_matrix, TE_threshold):
    return -((weights - bench_weights).dot(covariance_matrix)).dot(weights - bench_weights) + (TE_threshold ** 2)


def minvarpf(x, mu, risk_free_rate=0., risk_free_allowed=False, tangency=False):
    # Description: Function that compute the min var portfolio for a given E[R]

    # Input:
    ## x : Matrix [T x N] Matrix of returns of N assets for time T
    ## risk_free_rate: Scalar Riskfree rate
    ## tangency: Boolean Compute the tangency portfolio if true
    ## short: Boolean True if short is allowed

    # Analytical solution!
    covariance_matrix = np.cov(x, rowvar=False)
    E_ret = np.mean(x, 0)
    nb_ind = np.size(covariance_matrix, 1)
    covariance_matrixinv = np.linalg.inv(covariance_matrix)
    one_vec = np.ones(nb_ind)
    A = np.dot(np.dot(one_vec, covariance_matrixinv), np.ones(nb_ind))
    C = np.dot(np.dot(E_ret, covariance_matrixinv), E_ret)
    B = np.dot(np.dot(one_vec, covariance_matrixinv), E_ret)
    Delta = A * C - B ** 2

    if mu == []:
        mu = B / A
        min_var = (A * mu ** 2 - 2 * B * mu + C) / Delta
        lmbda = (C - mu * B) / Delta
        gma = (mu * A - B) / Delta
        weight = lmbda * np.dot(covariance_matrixinv, one_vec) + gma * np.dot(covariance_matrixinv, E_ret)

        return (mu, np.sqrt(min_var), weight)
    else:

        if tangency == False:
            if risk_free_allowed == False:
                # Analitical solution
                Delta = A * C - B ** 2
                lmbda = (C - mu * B) / Delta
                gma = (mu * A - B) / Delta
                min_var = (A * (mu ** 2) - 2 * B * mu + C) / Delta
                weight = lmbda * np.dot(covariance_matrixinv, one_vec) + gma * np.dot(covariance_matrixinv, E_ret)
                wrf = np.array([0.])
                weight = np.concatenate((wrf, weight))
            else:
                gma = (mu - risk_free_rate) / (C - 2 * risk_free_rate * B + risk_free_rate ** 2 * A)
                min_var = (mu - risk_free_rate) ** 2 / (C - 2 * risk_free_rate * B + risk_free_rate ** 2 * A)
                weight = gma * np.dot(covariance_matrixinv, E_ret - risk_free_rate * one_vec)
                wrf = np.array([1 - np.dot(one_vec, weight)])
                weight = np.concatenate((wrf, weight))

            return (mu, np.sqrt(min_var), weight)

        else:
            # Mat2=E_ret - risk_free_rate * one_vec
            # Mat3= (B - risk_free_rate * A)
            weight = np.matmul(covariance_matrixinv, E_ret - risk_free_rate * np.transpose(one_vec)) / (
                        B - risk_free_rate * A)
            mu = (C - B * risk_free_rate) / (B - A * risk_free_rate)
            min_var = (C - 2 * risk_free_rate * B + risk_free_rate ** 2 * A) / (B - A * risk_free_rate) ** 2
            return (mu, np.sqrt(min_var), weight)


def minvarpf_noshortsale(x, mu, risk_free_rate=0., risk_free_allowed=False, tangency=False):
    if tangency == False:
        if risk_free_allowed == False:
            covariance_matrix = np.cov(x, rowvar=False)
            expected_return = np.mean(x, 0)
            nb_ind = np.size(covariance_matrix, 1)
            initial_weights = np.repeat((1 / nb_ind), nb_ind)

            ## ----- Because of difference between descrete and continous we need to find min(Var) ----- ##
            ## ----- Reframing the constraints for optimization ----- ##
            constraint_weights = {'type': 'eq', 'fun': constraint_on_weights}
            constraint_expected_return = {'type': 'eq',
                                          'fun': constraint_on_expected_return,
                                          'args': (expected_return, mu,)}
            constraint_short_sell = {'type': 'ineq', 'fun': constraint_on_short_sell}

            constraints = [constraint_weights, constraint_expected_return, constraint_short_sell]
            tmp = minimize(objective, initial_weights, args=(covariance_matrix),
                           method="SLSQP", constraints=constraints)
            min_var = tmp.fun
            weight = tmp.x
            wrf = np.array([0.])
            weight = np.concatenate((wrf, weight))

            return (mu, np.sqrt(min_var), weight)

        else:
            y = x.copy()
            y.insert(0, "Risk free", risk_free_rate)
            covariance_matrix = y.cov()
            expected_return = np.mean(y, 0)
            nb_ind = np.size(covariance_matrix, 1)
            initial_weights = np.repeat((1 / nb_ind), nb_ind)

            ## ----- Because of difference between descrete and continous we need to find min(Var) ----- ##
            ## ----- Reframing the constraints for optimization ----- ##
            constraint_weights = {'type': 'eq', 'fun': constraint_on_weights}
            constraint_expected_return = {'type': 'eq',
                                          'fun': constraint_on_expected_return,
                                          'args': (expected_return, mu,)}
            constraint_short_sell = {'type': 'ineq', 'fun': constraint_on_short_sell}

            constraints = [constraint_weights, constraint_expected_return,
                           constraint_short_sell]
            tmp = minimize(objective, initial_weights, args=(covariance_matrix),
                           method="SLSQP", constraints=constraints)
            min_var = tmp.fun
            weight = tmp.x

        return (mu, np.sqrt(min_var), weight)
    else:

        covariance_matrix = np.cov(x, rowvar=False)
        expected_return = np.mean(x, 0)
        nb_ind = np.size(covariance_matrix, 1)
        initial_weights = np.repeat((1 / nb_ind), nb_ind)
        ## ----- Reframing the constraints for optimization ----- ##
        constraint_weights = {'type': 'eq', 'fun': constraint_on_weights}
        constraint_short_sell = {'type': 'ineq', 'fun': constraint_on_short_sell}

        constraints = [constraint_weights, constraint_short_sell]

        tmp = minimize(tangency_objective, initial_weights,
                       args=(risk_free_rate, covariance_matrix, expected_return),
                       method="SLSQP", constraints=constraints)
        weight = tmp.x
        min_var = np.sqrt((weight.dot(covariance_matrix)).dot(weight))
        mu = weight.dot(expected_return)
        return (mu, min_var, weight)


def f_neighbour(x, n_industries, n_sup):
    xn = np.copy(x)
    ix = random.sample(range(n_industries), 1)
    xn[ix] = np.invert(xn[ix])
    out = np.copy(xn)
    if sum(xn) > n_sup:
        out = np.copy(x)
    return out


def f_SR(x, MyInput):
    mu = MyInput.mu
    covariance_matrix = np.array(MyInput.covariance_matrix)
    rf = MyInput.rf

    covariance_matrix = covariance_matrix[x, :]
    covariance_matrix = covariance_matrix[:, x]
    mu = np.array(mu[x])

    constraint_weights = {'type': 'eq', 'fun': constraint_on_weights}
    constraint_short_sell = {'type': 'ineq', 'fun': constraint_on_short_sell}

    constraints = [constraint_weights, constraint_short_sell]

    initial_weights = np.repeat((1 / sum(x)), sum(x))
    solution = minimize(tangency_objective, initial_weights, args=(rf, covariance_matrix, mu), method="SLSQP",
                        constraints=constraints)
    y = solution.fun
    return (y)


def f_SA(SR_N, SR_O, d, i):
    T = d / np.log(i + 2)
    P = min(1, np.exp(-(SR_N - SR_O) / T))
    rnd = np.random.uniform()
    if rnd < P:
        out = True
    else:
        out = False
    return (out)


def f_MAXSR(MyInput, n_sim, d):
    x0 = np.zeros(MyInput.n_industries, dtype=bool)
    x0 = x0.astype(dtype=bool)
    ix = random.sample(range(MyInput.n_industries), MyInput.n_sup)
    x0[ix] = np.invert(x0[ix])
    x = np.zeros((MyInput.n_industries, n_sim), dtype=bool)
    x[:, 0] = x0
    SR = np.zeros(n_sim)
    SR_new = np.zeros(n_sim)
    for i in range(n_sim - 1):
        x_new = f_neighbour(x[:, i], MyInput.n_industries, MyInput.n_sup)
        SR_new[i] = f_SR(x_new, MyInput)
        if f_SA(SR_new[i], SR[i], d, i) == True:
            # print('Accepted', SR_new)
            x[:, i + 1] = x_new
            SR[i + 1] = SR_new[i]
        else:
            # print('Rejected', SR_new)
            SR[i + 1] = SR[i]
            x[:, i + 1] = x[:, i]

    return ([x[:, i + 1], -SR[i + 1]])


def prtf_return(weights, industry_returns):
    gross_ret = 1 + (industry_returns) / 100
    ret = weights @ (gross_ret - 1) * 100
    return (ret)



##----- Part B, Q 8  Replicating Parametric Paper

# TO DO fix r t+1 -- done
# standardize charact -- done
# add constraint on weights -- done

def objective_8(theta, w_bar, x1, x2, x3, monthly_ret, gamma):
    # # Transforming data to numpy array as quicker to compute than dataframes
    # w_bar = np.asarray(w_bar)
    # x1 = x1.to_numpy()
    # x2 = x2.to_numpy()
    # x3 = x3.to_numpy()
    # monthly_ret = monthly_ret.to_numpy()
    #
    # monthly_ret = monthly_ret[1:,:]            # removing first month of returns (since we sum starting at t=1)
    # w_bar = w_bar[:-1,:]                       # removing last month of weights (since summed to T-1)
    # x1 = x1[:-1, :]
    # x2 = x2[:-1, :]
    # x3 = x3[:-1, :]

    # Standardizing characteristics crosssectionally (accross columns)
    x1 = stats.zscore(x1, axis=1)
    x2 = stats.zscore(x2, axis=1)
    x3 = stats.zscore(x3, axis=1)

    theta_x1 = x1 * theta[0]
    theta_x2 = x2 * theta[1]
    theta_x3 = x3 * theta[2]

    theta_x = theta_x1 + theta_x2 + theta_x3
    theta_x = theta_x / 10

    opt_weight = w_bar + theta_x

    opt_weight[opt_weight < 0] = 0
    sum_weight = np.sum(opt_weight, axis=1)
    sum_weight = sum_weight[np.newaxis]
    opt_weight = opt_weight / sum_weight.T

    ret_i_t = opt_weight * ( monthly_ret / 100)

    ret_t = np.sum(ret_i_t, axis=1)         # summing over the col

    utility = ((1 + ret_t) ** (1 - gamma)) / (1 - gamma)

    f = np.mean(utility)

    return(-f)

def prtf8(theta, w_bar, x1, x2, x3, monthly_ret):
    # Transforming data to numpy array as quicker to compute than dataframes
    # w_bar = np.asarray(w_bar)
    # x1 = x1.to_numpy()
    # x2 = x2.to_numpy()
    # x3 = x3.to_numpy()
    # monthly_ret = monthly_ret.to_numpy()
    #
    # monthly_ret = monthly_ret[1:, :]  # removing first month of returns (since we sum starting at t=1)
    # w_bar = w_bar[:-1, :]  # removing last month of weights (since summed to T-1)
    # x1 = x1[:-1, :]
    # x2 = x2[:-1, :]
    # x3 = x3[:-1, :]

    # Standardizing characteristics crosssectionally (accross columns)
    x1 = stats.zscore(x1, axis=1)
    x2 = stats.zscore(x2, axis=1)
    x3 = stats.zscore(x3, axis=1)

    theta_x1 = x1 * theta[0]
    theta_x2 = x2 * theta[1]
    theta_x3 = x3 * theta[2]

    theta_x = theta_x1 + theta_x2 + theta_x3
    theta_x = theta_x / 10

    opt_weight = w_bar + theta_x

    ret_i_t = opt_weight * (monthly_ret / 100)

    ret_t = np.sum(ret_i_t, axis=1)  # summing over the col

    return opt_weight, ret_t


import datetime

def last_day_of_month(any_day):
    next_month = any_day.replace(day=28) + datetime.timedelta(days=4)  # this will never fail
    out = next_month - datetime.timedelta(days=next_month.day)
    return out

def eomonth(date_vec):
    tmp = date_vec.date
    out = []
    for i in  range(len(date_vec)):
        out.append(last_day_of_month(tmp[i]))
    return pd.to_datetime(out)

def midyearfama(date_vec):
    tmp = date_vec
    out = []
    for i in  range(len(date_vec)):
        out.append(datetime.datetime(tmp.year[i], 7, 31))
    return pd.to_datetime(out)
# -*- coding: utf-8 -*-
"""
Spyder Editor
Created on Fri Jan 17 22:52:32 2020
Created by : Keanu Vivish
TP 1 of Portfolio Management: Portfolio Allocation à la Markowitz
"""
# Import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools as it
from scipy.optimize import minimize
import random

# Import function
#from function import myf

#####################################################################################################################
## ----------- Import data -------------------------------------------------

input_file_path = './data/data.CSV'
df = pd.read_csv(input_file_path,
                   index_col = 0)

## ----------- Data cleansing -------------------------------------------------
df.index = pd.to_datetime(df.index, format= '%Y%m')
df.columns = df.columns.str.replace(' ', '')
industries = df.columns
date_vec = df.index
# Parameters

desired_industries = ['Food','Smoke','Toys','Books','Steel']
desired_industries_rf = desired_industries[:]
desired_industries_rf.insert(0, 'Risk Free Asset')
lastyears = '5Y'


df = df.last(lastyears)
df = pd.DataFrame.sort_index(df, ascending = False)
working_data = df.loc[:,desired_industries]
## ---------- Part A ----------------------------------------------------------
## Parameters and data storage

#Creation of a class to store everything
class MyInput:
    mu = working_data.mean()
    covariance_matrix = working_data.cov()
    volatilities = np.diagonal(np.sqrt(covariance_matrix))
    n_industries = mu.size
    rf = 0.001

risk_free_rate = 0.001
mu = working_data.mean()
covariance_matrix = working_data.cov()
volatilities = np.diagonal(np.sqrt(covariance_matrix))
n_industries = mu.size
#Plot and optimization parameters
min_expected_return = -50
max_expected_return = 300

# =============================================================================
# # Question 1: 
# Graph the "mean-variance locus" (without the risk-free asset) of these 5 
# industry portfolios. Specify each industry portfolio in the chart.
# =============================================================================
x_volatilityQ1 = []
y_returnQ1 = []
weightsQ1 = []
for expected_return in range(min_expected_return, max_expected_return):
    expected_return = expected_return/100
    (tmp1, tmp2, tmp3) = myf.minvarpf(working_data, expected_return, risk_free_rate)
    ## ----- Appending results for plotting ----- ##
    y_returnQ1.append(tmp1)
    x_volatilityQ1.append(tmp2)
    weightsQ1.append(tmp3)

(tmp1, tmp2, tmp3) = myf.minvarpf(working_data, [], risk_free_rate)
min_var_returnQ1 = tmp1
min_var_volatilityQ1 = tmp2 

## ----- Saving plots for using in Q3 ----- ##
plot_answer1 = plt.plot(x_volatilityQ1, y_returnQ1, label = 'Frontier')
industries_color = []    
for x_point, y_point, ind in zip(volatilities, mu, desired_industries):
    colortmp = plt.scatter(x_point, y_point, marker= "*",
                label = ind, s=30)
    ## Saving the colors for each industries
    tmp = colortmp.get_facecolors()
    tmp_tuple = tuple(tmp[0])
    industries_color.append(tmp_tuple)

plt.scatter(min_var_volatilityQ1, min_var_returnQ1, marker = '^',
            label = 'Minimum variance portfolio', s=60)
## ----- Naming axis ----- ##
plt.xlabel('Volatility')
plt.ylabel('Return') 
plt.xlim([0,10]) 
 
## ----- Title of the graph ----- ##
plt.title('Mean-Variance Locus Without Rf') 

## ----- Adding legend to the graph ----- ##
plt.legend(loc='upper left')
 
## ----- Showing plot ----- ##
if __name__ == "__main__":
    plt.grid()
    plt.show() 
    
    
# Area plot of weight in mean variance to volatility
    
x_volatilityQ1 = []
y_returnQ1 = []
weightsQ1 = []
for expected_return in range(int(round(min_var_returnQ1*100)), max_expected_return):
    expected_return = expected_return/100
    (tmp1, tmp2, tmp3) = myf.minvarpf(working_data, expected_return, risk_free_rate)
    ## ----- Appending results for plotting ----- ##
    y_returnQ1.append(tmp1)
    x_volatilityQ1.append(tmp2)
    weightsQ1.append(tmp3)

tmp_weight = np.asarray(weightsQ1)
tmp_weight = tmp_weight[:,1:]
tmp_weight_positive = np.copy(tmp_weight)
tmp_weight_positive[tmp_weight_positive < 0] = 0
tmp_weight_negative = np.copy(tmp_weight)
tmp_weight_negative[tmp_weight_negative > 0] = 0

tmp_volatility = np.asarray(x_volatilityQ1)

stackplotQ1 = plt.stackplot(tmp_volatility, tmp_weight_positive.T, 
                            colors = industries_color, labels = desired_industries)
stackplotQ1 = plt.stackplot(tmp_volatility, tmp_weight_negative.T,
                            colors = industries_color)


plt.xlabel('Volatility')
plt.ylabel('Weights') 

  
## ----- Title of the graph ----- ##
plt.title('Weight of each assets on the Mean-Variance Frontier Without Rf') 

## ----- Adding legend to the graph ----- ##
plt.legend(loc='upper left')
 
## ----- Showing plot ----- ##
if __name__ == "__main__":
    plt.grid()
    plt.show() 
    

# =============================================================================
# # Question 2: 
# Graph the "mean-variance locus" (with the risk-free asset) of these 5 industry 
# portfolios.
# =============================================================================

x_volatilityQ2 = []
y_returnQ2 = []
weightsQ2 = []
for expected_return in range(min_expected_return, max_expected_return):
    expected_return = expected_return/100
    (tmp1, tmp2, tmp3) = myf.minvarpf(working_data, expected_return, risk_free_rate, 
    risk_free_allowed = True)
    ## ----- Appending results for plotting ----- ##
    y_returnQ2.append(tmp1)
    x_volatilityQ2.append(tmp2)
    weightsQ2.append(tmp3)
    
## ----- Saving plots for using in Q3 ----- ##
plot_answer2 = plt.plot(x_volatilityQ2, y_returnQ2, label = 'Frontier')

for x_point, y_point, ind, col in zip(volatilities, mu, desired_industries, industries_color):
    plt.scatter(x_point, y_point, marker= "*",
                label = ind, color = col , s=30) 
plt.scatter(0, risk_free_rate, marker= "<",
                label = 'Risk free rate', s=80, color = 'b') 
## ----- Naming axis ----- ##
plt.xlabel('Volatility')
plt.ylabel('Return') 
plt.xlim([0,10])
  
## ----- Title of the graph ----- ##
plt.title('Mean-Variance Locus With rf') 

## ----- Adding legend to the graph ----- ##
plt.legend(loc='upper left')
 
## ----- Showing plot ----- ##
if __name__ == "__main__":
    plt.grid()
    plt.show() 
    

x_volatilityQ2 = []
y_returnQ2 = []
weightsQ2 = []
for expected_return in range(int(round(risk_free_rate)), max_expected_return):
    expected_return = expected_return/100
    (tmp1, tmp2, tmp3) = myf.minvarpf(working_data, expected_return, risk_free_rate, 
    risk_free_allowed = True)
    ## ----- Appending results for plotting ----- ##
    y_returnQ2.append(tmp1)
    x_volatilityQ2.append(tmp2)
    weightsQ2.append(tmp3)

#Add risk free asset in industries color
industries_color_rf = industries_color[:]
industries_color_rf.insert(0,('b'))


tmp_weight = np.asarray(weightsQ2)
tmp_weight_positive = np.copy(tmp_weight)
tmp_weight_positive[tmp_weight_positive < 0] = 0
tmp_weight_negative = np.copy(tmp_weight)
tmp_weight_negative[tmp_weight_negative > 0] = 0

tmp_volatility = np.asarray(x_volatilityQ2)

stackplotQ2 = plt.stackplot(tmp_volatility, tmp_weight_positive.T, 
                            colors = industries_color_rf, labels = desired_industries_rf)
stackplotQ2 = plt.stackplot(tmp_volatility, tmp_weight_negative.T,
                            colors = industries_color_rf)


plt.xlabel('Volatility')
plt.ylabel('Weights') 

  
## ----- Title of the graph ----- ##
plt.title('Weight of each assets on the Mean-Variance Frontier With Rf') 

## ----- Adding legend to the graph ----- ##
plt.legend(loc='upper left')
 
## ----- Showing plot ----- ##
if __name__ == "__main__":
    plt.grid()
    plt.show() 



# =============================================================================
# # Question 3
# Describe the tangent portfolio and its characteristics such as its mean
# and variance and the weights of each asset. 
# How can you check the tangent portfolio is the portfolio that maximizes
# the Sharpe ratio?
# =============================================================================

(tmp1, tmp2, tmp3) = myf.minvarpf(working_data, expected_return, risk_free_rate,
risk_free_allowed = True, tangency = True)

expected_return_tangancy1 = tmp1
volatility_tangency1 = tmp2
weights_tangency1 = tmp3
    
    
plt.plot(x_volatilityQ1, y_returnQ1, label = 'Frontier without rf')

plt.plot(x_volatilityQ2, y_returnQ2, label = 'Frontier with rf')

plt.scatter(volatility_tangency1, expected_return_tangancy1, 
            label ='Tangency Portfolio', c = 'k')

for x_point, y_point, ind in zip(volatilities, mu, desired_industries):
    plt.scatter(x_point, y_point, marker= "*",
                label = ind, s=30) 
plt.scatter(0, risk_free_rate, marker= "<",
                label = 'Risk free rate', s=80, color = 'b')   
## ----- Naming axis ----- ##
plt.xlabel('Volatility')
plt.ylabel('Return') 
plt.xlim(0)
  
## ----- Title of the graph ----- ##
plt.title('Mean-Variance frontier') 

## ----- Adding legend to the graph ----- ##
plt.legend(loc='best')
 
## ----- Showing plot ----- ##
if __name__ == "__main__":
    plt.grid()
    plt.show() 







# Check the Sharpe Ratio for each unit of risk on the portfolio
SR_Q3 = (np.asarray(y_returnQ1) - risk_free_rate) / np.asarray(x_volatilityQ1)
max_SR_Q3 = max(SR_Q3)
argmax_SR_Q3 = np.argmax(SR_Q3)
y_return_max_SR_Q3 = y_returnQ1[argmax_SR_Q3]

plot_answer3_2 = plt.plot(y_returnQ1, SR_Q3, label = 'Frontier')
plt.scatter(y_return_max_SR_Q3, max_SR_Q3, marker= "o",
                label = 'Tangency Pf', c = 'k', s=80)  
  
## ----- Naming axis ----- ##
plt.xlabel('Expected Returns')
plt.ylabel('Sharpe Ratio') 
plt.xlim(0)
  
## ----- Title of the graph ----- ##
plt.title('Sharpe Ratio') 

## ----- Adding legend to the graph ----- ##
plt.legend(loc='upper left')
 
## ----- Showing plot ----- ##
if __name__ == "__main__":
    plt.grid()
    plt.show() 
 
 
# =============================================================================
# Question 4
# Graph the "mean-variance locus" (without the risk-free asset) with 
# the short-sale constraints on each industry portfolio. 
# Specify each industry portfolio in the chart.
# =============================================================================
    
    
x_volatilityQ4 = []
y_returnQ4 = []
weightsQ4 = []

(tmp1, tmp2, tmp3) = myf.minvarpf_noshortsale(working_data, [], risk_free_rate)
var_min = tmp2
tmp3 = tmp3[1:]
min_var_returnQ4 = tmp3.dot(working_data.mean())

for expected_return in np.arange(min_var_returnQ4,max(mu), 0.01):
    expected_return = expected_return
    (tmp1, tmp2, tmp3) = myf.minvarpf_noshortsale(working_data, expected_return, risk_free_rate)
    ## ----- Appending results for plotting ----- ##
    y_returnQ4.append(tmp1)
    x_volatilityQ4.append(tmp2)
    weightsQ4.append(tmp3)
tmp_weight = np.asarray(weightsQ4)
tmp_volatility = np.asarray(x_volatilityQ4)



plot_answer4 = plt.plot(x_volatilityQ4, y_returnQ4, label = 'Frontier')

for x_point, y_point, ind in zip(volatilities, mu, desired_industries):
    plt.scatter(x_point, y_point, marker= "*",
                label = ind, s=30) 
  
## ----- Naming axis ----- ##
plt.xlabel('Volatility')
plt.ylabel('Return') 
plt.xlim(0)
  
## ----- Title of the graph ----- ##
plt.title('Mean-Variance Frontier Without Rf') 

## ----- Adding legend to the graph ----- ##
plt.legend(loc='upper left')
 
## ----- Showing plot ----- ##
if __name__ == "__main__":
    plt.grid()
    plt.show() 
 
    
    
    
    
 # Area plot of weight in mean variance to volatility
    
x_volatilityQ4 = []
y_returnQ4 = []
weightsQ4 = []
for expected_return in np.arange(min_var_returnQ4,max(mu), 0.01):
    expected_return = expected_return
    (tmp1, tmp2, tmp3) =  myf.minvarpf_noshortsale(working_data, expected_return, risk_free_rate)
    ## ----- Appending results for plotting ----- ##
    y_returnQ4.append(tmp1)
    x_volatilityQ4.append(tmp2)
    weightsQ4.append(tmp3)

tmp_weight = np.asarray(weightsQ4)
tmp_weight = tmp_weight[:,1:]
tmp_weight_positive = np.copy(tmp_weight)
tmp_weight_positive[tmp_weight_positive < 0] = 0
tmp_weight_negative = np.copy(tmp_weight)
tmp_weight_negative[tmp_weight_negative > 0] = 0

tmp_volatility = np.asarray(x_volatilityQ4)

stackplotQ4 = plt.stackplot(tmp_volatility, tmp_weight_positive.T, 
                            colors = industries_color, labels = desired_industries)
stackplotQ4 = plt.stackplot(tmp_volatility, tmp_weight_negative.T,
                            colors = industries_color)


plt.xlabel('Volatility')
plt.ylabel('Weights') 

  
## ----- Title of the graph ----- ##
plt.title('Weight of each assets on the Mean-Variance Frontier Without Rf') 

## ----- Adding legend to the graph ----- ##
plt.legend(loc='upper right')
 
## ----- Showing plot ----- ##
if __name__ == "__main__":
    plt.grid()
    plt.show() 
    
   
    
    
# =============================================================================
# Question 5
# Graph the "mean-variance locus" (with the risk-free asset) with the short-sale 
# constraints oneach industry portfolio. 
# Specify each industry portfolio in the chart. 
# Explain how the meanvariance locus has changed with the risk-free asset.
# Question 6
# Describe the tangent portfolio and its characteristics such as its mean
# and variance and the weights of each asset. 
# How can you check the tangent portfolio is the portfolio that maximizes
# the Sharpe ratio?
# =============================================================================

 
x_volatilityQ5 = []
y_returnQ5 = []
weightsQ5 = []
for expected_return in range(round(risk_free_rate*100), round(max(mu)*100)):
    expected_return = expected_return/100
    (tmp1, tmp2, tmp3) = myf.minvarpf_noshortsale(working_data, expected_return, risk_free_rate,
    risk_free_allowed = True)
    ## ----- Appending results for plotting ----- ##
    y_returnQ5.append(tmp1)
    x_volatilityQ5.append(tmp2)
    weightsQ5.append(tmp3)

## ----- Tangency Portfolio ----- ##

(tmp1, tmp2, tmp3) = myf.minvarpf_noshortsale(working_data, expected_return, risk_free_rate,
    risk_free_allowed = True, tangency= True)

y_tangency_expected_return = tmp1
x_tangency_volatility = tmp2

   

plot_answer5 = plt.plot(x_volatilityQ5, y_returnQ5, label = 'Frontier')

for x_point, y_point, ind in zip(volatilities, mu, desired_industries):
    plt.scatter(x_point, y_point, marker= "*",
                label = ind, s=30)
plt.scatter(0, risk_free_rate, marker= "<",
                label = 'Risk free rate', s=80, color = 'b') 
plt.scatter(tmp2, tmp1, marker= "o",
            label = 'Tangency Pf', c = 'k', s=80) 
  
## ----- Naming axis ----- ##
plt.xlabel('Volatility')
plt.ylabel('Return') 
plt.xlim(0)
  
## ----- Title of the graph ----- ##
plt.title('Mean-Variance Frontier With Rf') 

## ----- Adding legend to the graph ----- ##
plt.legend(loc='upper left')
 
## ----- Showing plot ----- ##
if __name__ == "__main__":
    plt.grid()
    plt.show()

#

x_volatilityQ5 = []
y_returnQ5 = []
weightsQ5 = []
for expected_return in range(int(round(risk_free_rate)), round(max(mu)*100)):
    expected_return = expected_return/100
    (tmp1, tmp2, tmp3) = myf.minvarpf_noshortsale(working_data, expected_return, risk_free_rate,
    risk_free_allowed = True)
    ## ----- Appending results for plotting ----- ##
    y_returnQ5.append(tmp1)
    x_volatilityQ5.append(tmp2)
    weightsQ5.append(tmp3)

#Add risk free asset in industries color
industries_color_rf = industries_color[:]
industries_color_rf.insert(0,('b'))


tmp_weight = np.asarray(weightsQ5)
tmp_weight_positive = np.copy(tmp_weight)
tmp_weight_positive[tmp_weight_positive < 0] = 0
tmp_weight_negative = np.copy(tmp_weight)
tmp_weight_negative[tmp_weight_negative > 0] = 0

tmp_volatility = np.asarray(x_volatilityQ5)

stackplotQ6 = plt.stackplot(tmp_volatility, tmp_weight_positive.T, 
                            colors = industries_color_rf, labels = desired_industries_rf)
stackplotQ6 = plt.stackplot(tmp_volatility, tmp_weight_negative.T,
                            colors = industries_color_rf)


plt.xlabel('Volatility')
plt.ylabel('Weights') 

  
## ----- Title of the graph ----- ##
plt.title('Weight of each assets on the Mean-Variance Frontier With Rf') 

## ----- Adding legend to the graph ----- ##
plt.legend(loc='upper left')
 
## ----- Showing plot ----- ##
if __name__ == "__main__":
    plt.grid()
    plt.show() 


# Check the Sharpe Ratio for each unit of risk on the portfolio
SR_Q6 = (np.asarray(y_returnQ4) - risk_free_rate) / np.asarray(x_volatilityQ4)
max_SR_Q6 = max(SR_Q6)
argmax_SR_Q6 = np.argmax(SR_Q6)
y_return_max_SR_Q6 = y_returnQ4[argmax_SR_Q6]

plot_answer3_2 = plt.plot(y_returnQ4, SR_Q6, label = 'Frontier')
plt.scatter(y_return_max_SR_Q6, max_SR_Q6, marker= "o",
                label = 'Tangency Pf', c = 'k', s=40)  
  
## ----- Naming axis ----- ##
plt.xlabel('Expected Returns')
plt.ylabel('Sharpe Ratio') 
plt.xlim(0)
  
## ----- Title of the graph ----- ##
plt.title('Sharpe Ratio') 

## ----- Adding legend to the graph ----- ##
plt.legend(loc='upper left')
 
## ----- Showing plot ----- ##
if __name__ == "__main__":
    plt.grid()
    plt.show() 
 







## ----- Consolidation of plot with and without riskfree ----- ##
plot_answer5 = plt.plot(x_volatilityQ4, y_returnQ4, label = 'Frontier without rf')
plot_answer6 = plt.plot(x_volatilityQ5, y_returnQ5, label = 'Frontier with rf')

for x_point, y_point, ind in zip(volatilities, mu, desired_industries):
    plt.scatter(x_point, y_point, marker= "*",
                label = ind, s=30)
plt.scatter(0, risk_free_rate, marker= "<",
                label = 'Risk free rate', s=80, color = 'b') 
plt.scatter(x_tangency_volatility, y_tangency_expected_return, marker= "o",
            label = 'Tangency Pf', s=40, c = 'k') 
  
## ----- Naming axis ----- ##
plt.xlabel('Volatility')
plt.ylabel('Return') 
plt.xlim(0)
  
## ----- Title of the graph ----- ##
plt.title('Mean-Variance Frontier With Rf') 

## ----- Adding legend to the graph ----- ##
plt.legend(loc='upper left')
 
## ----- Showing plot ----- ##
if __name__ == "__main__":
    plt.grid()
    plt.show()


# =============================================================================
# Question 7
# Repeat the same calculations in 1-6 adding 5 other industry portfolios to 
# the original list of 5 industry portfolios you chose at the start. 
# Compare the results and discuss the advantages and disadvantages of 
# using 10 portfolios instead of 5.
# =============================================================================

desired_industries_Q7 = ['Food','Smoke','Toys','Books','Steel', 'Agric', 'Fin', 
                      'Hlth', 'Drugs', 'Banks']
desired_industries_rf_Q7 = desired_industries_Q7[:]
desired_industries_rf_Q7.insert(0, 'Risk Free Asset')
lastyears = '5Y'


df = df.last(lastyears)
df = pd.DataFrame.sort_index(df, ascending = False)
working_data = df.loc[:,desired_industries_Q7]



#Creation of a class to store everything
class MyInput:
    mu = working_data.mean()
    covariance_matrix = working_data.cov()
    volatilities = np.diagonal(np.sqrt(covariance_matrix))
    n_industries = mu.size
    rf = 0.001

risk_free_rate = 0.001
mu = working_data.mean()
covariance_matrix = working_data.cov()
volatilities = np.diagonal(np.sqrt(covariance_matrix))
n_industries = mu.size
#Plot and optimization parameters
min_expected_return = -50
max_expected_return = 300

initial_weights = np.repeat((1 / n_industries), n_industries)


# =============================================================================
# # Question 1: 
# Graph the "mean-variance locus" (without the risk-free asset) of these 5 
# industry portfolios. Specify each industry portfolio in the chart.
# =============================================================================
# =============================================================================
# # Question 1: 
# Graph the "mean-variance locus" (without the risk-free asset) of these 5 
# industry portfolios. Specify each industry portfolio in the chart.
# =============================================================================
x_volatilityQ7_1 = []
y_returnQ7_1 = []
weightsQ7_1 = []
for expected_return in range(min_expected_return, max_expected_return):
    expected_return = expected_return/100
    (tmp1, tmp2, tmp3) = myf.minvarpf(working_data, expected_return, risk_free_rate)
    ## ----- Appending results for plotting ----- ##
    y_returnQ7_1.append(tmp1)
    x_volatilityQ7_1.append(tmp2)
    weightsQ7_1.append(tmp3)

(tmp1, tmp2, tmp3) = myf.minvarpf(working_data, [], risk_free_rate)
min_var_returnQ7_1 = tmp1
min_var_volatilityQ7_1 = tmp2 

## ----- Saving plots for using in Q3 ----- ##
plot_answer1 = plt.plot(x_volatilityQ7_1, y_returnQ7_1, label = 'Frontier')
industries_color = []    
for x_point, y_point, ind in zip(volatilities, mu, desired_industries_Q7):
    colortmp = plt.scatter(x_point, y_point, marker= "*",
                label = ind, s=30)
    ## Saving the colors for each industries
    tmp = colortmp.get_facecolors()
    tmp_tuple = tuple(tmp[0])
    industries_color.append(tmp_tuple)

plt.scatter(min_var_volatilityQ7_1, min_var_returnQ7_1, marker = '^',
            label = 'Minimum variance portfolio', s=60)
## ----- Naming axis ----- ##
plt.xlabel('Volatility')
plt.ylabel('Return') 
plt.xlim([0,10]) 
 
## ----- Title of the graph ----- ##
plt.title('Mean-Variance Locus Without Rf') 

## ----- Adding legend to the graph ----- ##
plt.legend(loc='upper left')
 
## ----- Showing plot ----- ##
if __name__ == "__main__":
    plt.grid()
    plt.show() 
    
    
# Area plot of weight in mean variance to volatility
    
x_volatilityQ7_1 = []
y_returnQ7_1 = []
weightsQ7_1 = []
for expected_return in range(int(round(min_var_returnQ7_1*100)), max_expected_return):
    expected_return = expected_return/100
    (tmp1, tmp2, tmp3) = myf.minvarpf(working_data, expected_return, risk_free_rate)
    ## ----- Appending results for plotting ----- ##
    y_returnQ7_1.append(tmp1)
    x_volatilityQ7_1.append(tmp2)
    weightsQ7_1.append(tmp3)

tmp_weight = np.asarray(weightsQ7_1)
tmp_weight = tmp_weight[:,1:]
tmp_weight_positive = np.copy(tmp_weight)
tmp_weight_positive[tmp_weight_positive < 0] = 0
tmp_weight_negative = np.copy(tmp_weight)
tmp_weight_negative[tmp_weight_negative > 0] = 0

tmp_volatility = np.asarray(x_volatilityQ7_1)

stackplotQ7_1 = plt.stackplot(tmp_volatility, tmp_weight_positive.T, 
                            colors = industries_color, labels = desired_industries_Q7)
stackplotQ7_1 = plt.stackplot(tmp_volatility, tmp_weight_negative.T,
                            colors = industries_color)


plt.xlabel('Volatility')
plt.ylabel('Weights') 

  
## ----- Title of the graph ----- ##
plt.title('Weight of each assets on the Mean-Variance Frontier Without Rf') 

## ----- Adding legend to the graph ----- ##
plt.legend(loc='upper left')
 
## ----- Showing plot ----- ##
if __name__ == "__main__":
    plt.grid()
    plt.show() 
    

# =============================================================================
# # Question 2: 
# Graph the "mean-variance locus" (with the risk-free asset) of these 5 industry 
# portfolios.
# =============================================================================

x_volatilityQ7_2 = []
y_returnQ7_2 = []
weightsQ7_2 = []
for expected_return in range(min_expected_return, max_expected_return):
    expected_return = expected_return/100
    (tmp1, tmp2, tmp3) = myf.minvarpf(working_data, expected_return, risk_free_rate, 
    risk_free_allowed = True)
    ## ----- Appending results for plotting ----- ##
    y_returnQ7_2.append(tmp1)
    x_volatilityQ7_2.append(tmp2)
    weightsQ7_2.append(tmp3)
    
## ----- Saving plots for using in Q3 ----- ##
plot_answer2 = plt.plot(x_volatilityQ7_2, y_returnQ7_2, label = 'Frontier')

for x_point, y_point, ind, col in zip(volatilities, mu, desired_industries_Q7, industries_color):
    plt.scatter(x_point, y_point, marker= "*",
                label = ind, color = col , s=30) 
plt.scatter(0, risk_free_rate, marker= "<",
                label = 'Risk free rate', s=80, color = 'b') 
## ----- Naming axis ----- ##
plt.xlabel('Volatility')
plt.ylabel('Return') 
plt.xlim([0,10])
  
## ----- Title of the graph ----- ##
plt.title('Mean-Variance Locus With rf') 

## ----- Adding legend to the graph ----- ##
plt.legend(loc='upper left')
 
## ----- Showing plot ----- ##
if __name__ == "__main__":
    plt.grid()
    plt.show() 
    

x_volatilityQ7_2 = []
y_returnQ7_2 = []
weightsQ7_2 = []
for expected_return in range(int(round(risk_free_rate)), max_expected_return):
    expected_return = expected_return/100
    (tmp1, tmp2, tmp3) = myf.minvarpf(working_data, expected_return, risk_free_rate, 
    risk_free_allowed = True)
    ## ----- Appending results for plotting ----- ##
    y_returnQ7_2.append(tmp1)
    x_volatilityQ7_2.append(tmp2)
    weightsQ7_2.append(tmp3)

#Add risk free asset in industries color
industries_color_rf = industries_color[:]
industries_color_rf.insert(0,('b'))


tmp_weight = np.asarray(weightsQ7_2)
tmp_weight_positive = np.copy(tmp_weight)
tmp_weight_positive[tmp_weight_positive < 0] = 0
tmp_weight_negative = np.copy(tmp_weight)
tmp_weight_negative[tmp_weight_negative > 0] = 0

tmp_volatility = np.asarray(x_volatilityQ7_2)

stackplotQ7_2 = plt.stackplot(tmp_volatility, tmp_weight_positive.T, 
                            colors = industries_color_rf, labels = desired_industries_rf_Q7)
stackplotQ7_2 = plt.stackplot(tmp_volatility, tmp_weight_negative.T,
                            colors = industries_color_rf)


plt.xlabel('Volatility')
plt.ylabel('Weights') 

  
## ----- Title of the graph ----- ##
plt.title('Weight of each assets on the Mean-Variance Frontier With Rf') 

## ----- Adding legend to the graph ----- ##
plt.legend(loc='upper left')
 
## ----- Showing plot ----- ##
if __name__ == "__main__":
    plt.grid()
    plt.show() 



# =============================================================================
# # Question 3
# Describe the tangent portfolio and its characteristics such as its mean
# and variance and the weights of each asset. 
# How can you check the tangent portfolio is the portfolio that maximizes
# the Sharpe ratio?
# =============================================================================

(tmp1, tmp2, tmp3) = myf.minvarpf(working_data, expected_return, risk_free_rate,
risk_free_allowed = True, tangency = True)

expected_return_tangancyQ7_1 = tmp1
volatility_tangencyQ7_1 = tmp2
weights_tangencyQ7_1 = tmp3
    
    
plt.plot(x_volatilityQ7_1, y_returnQ7_1, label = 'Frontier without rf')

plt.plot(x_volatilityQ7_2, y_returnQ7_2, label = 'Frontier with rf')

plt.scatter(volatility_tangencyQ7_1, expected_return_tangancyQ7_1, 
            label ='Tangency Portfolio', c = 'k')

for x_point, y_point, ind in zip(volatilities, mu, desired_industries_Q7):
    plt.scatter(x_point, y_point, marker= "*",
                label = ind, s=30) 
plt.scatter(0, risk_free_rate, marker= "<",
                label = 'Risk free rate', s=80, color = 'b')   
## ----- Naming axis ----- ##
plt.xlabel('Volatility')
plt.ylabel('Return') 
plt.xlim(0)
  
## ----- Title of the graph ----- ##
plt.title('Mean-Variance frontier') 

## ----- Adding legend to the graph ----- ##
plt.legend(loc='best')
 
## ----- Showing plot ----- ##
if __name__ == "__main__":
    plt.grid()
    plt.show() 







# Check the Sharpe Ratio for each unit of risk on the portfolio
SR_Q7_3 = (np.asarray(y_returnQ7_1) - risk_free_rate) / np.asarray(x_volatilityQ7_1)
max_SR_Q7_3 = max(SR_Q7_3)
argmax_SR_Q7_3 = np.argmax(SR_Q7_3)
y_return_max_SR_Q7_3 = y_returnQ7_1[argmax_SR_Q7_3]

plot_answer3_2 = plt.plot(y_returnQ7_1, SR_Q7_3, label = 'Frontier')
plt.scatter(y_return_max_SR_Q7_3, max_SR_Q7_3, marker= "o",
                label = 'Tangency Pf', c = 'k', s=80)  
  
## ----- Naming axis ----- ##
plt.xlabel('Expected Returns')
plt.ylabel('Sharpe Ratio') 
plt.xlim(0)
  
## ----- Title of the graph ----- ##
plt.title('Sharpe Ratio') 

## ----- Adding legend to the graph ----- ##
plt.legend(loc='upper left')
 
## ----- Showing plot ----- ##
if __name__ == "__main__":
    plt.grid()
    plt.show() 
 
 
# =============================================================================
# Question 4
# Graph the "mean-variance locus" (without the risk-free asset) with 
# the short-sale constraints on each industry portfolio. 
# Specify each industry portfolio in the chart.
# =============================================================================
    
    
x_volatilityQ7_4 = []
y_returnQ7_4 = []
weightsQ7_4 = []

(tmp1, tmp2, tmp3) = myf.minvarpf_noshortsale(working_data, [], risk_free_rate)
var_min = tmp2
tmp3 = tmp3[1:]
min_var_returnQ7_4 = tmp3.dot(working_data.mean())

for expected_return in np.arange(min_var_returnQ7_4,max(mu), 0.01):
    expected_return = expected_return
    (tmp1, tmp2, tmp3) = myf.minvarpf_noshortsale(working_data, expected_return, risk_free_rate)
    ## ----- Appending results for plotting ----- ##
    y_returnQ7_4.append(tmp1)
    x_volatilityQ7_4.append(tmp2)
    weightsQ7_4.append(tmp3)
tmp_weight = np.asarray(weightsQ7_4)
tmp_volatility = np.asarray(x_volatilityQ7_4)



plot_answer4 = plt.plot(x_volatilityQ7_4, y_returnQ7_4, label = 'Frontier')

for x_point, y_point, ind in zip(volatilities, mu, desired_industries_Q7):
    plt.scatter(x_point, y_point, marker= "*",
                label = ind, s=30) 
  
## ----- Naming axis ----- ##
plt.xlabel('Volatility')
plt.ylabel('Return') 
plt.xlim(0)
  
## ----- Title of the graph ----- ##
plt.title('Mean-Variance Frontier Without Rf') 

## ----- Adding legend to the graph ----- ##
plt.legend(loc='upper left')
 
## ----- Showing plot ----- ##
if __name__ == "__main__":
    plt.grid()
    plt.show() 
 
    
    
    
    
 # Area plot of weight in mean variance to volatility
    
x_volatilityQ7_4 = []
y_returnQ7_4 = []
weightsQ7_4 = []
for expected_return in np.arange(min_var_returnQ7_4,max(mu), 0.01):
    expected_return = expected_return
    (tmp1, tmp2, tmp3) =  myf.minvarpf_noshortsale(working_data, expected_return, risk_free_rate)
    ## ----- Appending results for plotting ----- ##
    y_returnQ7_4.append(tmp1)
    x_volatilityQ7_4.append(tmp2)
    weightsQ7_4.append(tmp3)

tmp_weight = np.asarray(weightsQ7_4)
tmp_weight = tmp_weight[:,1:]
tmp_weight_positive = np.copy(tmp_weight)
tmp_weight_positive[tmp_weight_positive < 0] = 0
tmp_weight_negative = np.copy(tmp_weight)
tmp_weight_negative[tmp_weight_negative > 0] = 0

tmp_volatility = np.asarray(x_volatilityQ7_4)

stackplotQ7_4 = plt.stackplot(tmp_volatility, tmp_weight_positive.T, 
                            colors = industries_color, labels = desired_industries_Q7)
stackplotQ7_4 = plt.stackplot(tmp_volatility, tmp_weight_negative.T,
                            colors = industries_color)


plt.xlabel('Volatility')
plt.ylabel('Weights') 

  
## ----- Title of the graph ----- ##
plt.title('Weight of each assets on the Mean-Variance Frontier Without Rf') 

## ----- Adding legend to the graph ----- ##
plt.legend(loc='upper right')
 
## ----- Showing plot ----- ##
if __name__ == "__main__":
    plt.grid()
    plt.show() 
    
   
    
    
# =============================================================================
# Question 5
# Graph the "mean-variance locus" (with the risk-free asset) with the short-sale 
# constraints oneach industry portfolio. 
# Specify each industry portfolio in the chart. 
# Explain how the meanvariance locus has changed with the risk-free asset.
# Question 6
# Describe the tangent portfolio and its characteristics such as its mean
# and variance and the weights of each asset. 
# How can you check the tangent portfolio is the portfolio that maximizes
# the Sharpe ratio?
# =============================================================================

 
x_volatilityQ7_5 = []
y_returnQ7_5 = []
weightsQ7_5 = []
for expected_return in range(round(risk_free_rate*100), round(max(mu)*100)):
    expected_return = expected_return/100
    (tmp1, tmp2, tmp3) = myf.minvarpf_noshortsale(working_data, expected_return, risk_free_rate,
    risk_free_allowed = True)
    ## ----- Appending results for plotting ----- ##
    y_returnQ7_5.append(tmp1)
    x_volatilityQ7_5.append(tmp2)
    weightsQ7_5.append(tmp3)

## ----- Tangency Portfolio ----- ##

(tmp1, tmp2, tmp3) = myf.minvarpf_noshortsale(working_data, expected_return, risk_free_rate,
    risk_free_allowed = True, tangency= True)

y_tangency_expected_returnQ7 = tmp1
x_tangency_volatilityQ7 = tmp2

   

plot_answer5 = plt.plot(x_volatilityQ7_5, y_returnQ7_5, label = 'Frontier')

for x_point, y_point, ind in zip(volatilities, mu, desired_industries_Q7):
    plt.scatter(x_point, y_point, marker= "*",
                label = ind, s=30)
plt.scatter(0, risk_free_rate, marker= "<",
                label = 'Risk free rate', s=80, color = 'b') 
plt.scatter(tmp2, tmp1, marker= "o",
            label = 'Tangency Pf', c = 'k', s=80) 
  
## ----- Naming axis ----- ##
plt.xlabel('Volatility')
plt.ylabel('Return') 
plt.xlim(0)
  
## ----- Title of the graph ----- ##
plt.title('Mean-Variance Frontier With Rf') 

## ----- Adding legend to the graph ----- ##
plt.legend(loc='upper left')
 
## ----- Showing plot ----- ##
if __name__ == "__main__":
    plt.grid()
    plt.show()

#

x_volatilityQ7_5 = []
y_returnQ7_5 = []
weightsQ7_5 = []
for expected_return in range(int(round(risk_free_rate)), round(max(mu)*100)):
    expected_return = expected_return/100
    (tmp1, tmp2, tmp3) = myf.minvarpf_noshortsale(working_data, expected_return, risk_free_rate,
    risk_free_allowed = True)
    ## ----- Appending results for plotting ----- ##
    y_returnQ7_5.append(tmp1)
    x_volatilityQ7_5.append(tmp2)
    weightsQ7_5.append(tmp3)

#Add risk free asset in industries color
industries_color_rf = industries_color[:]
industries_color_rf.insert(0,('b'))


tmp_weight = np.asarray(weightsQ7_5)
tmp_weight_positive = np.copy(tmp_weight)
tmp_weight_positive[tmp_weight_positive < 0] = 0
tmp_weight_negative = np.copy(tmp_weight)
tmp_weight_negative[tmp_weight_negative > 0] = 0

tmp_volatility = np.asarray(x_volatilityQ7_5)

stackplotQ7_6 = plt.stackplot(tmp_volatility, tmp_weight_positive.T, 
                            colors = industries_color_rf, labels = desired_industries_rf_Q7)
stackplotQ7_6 = plt.stackplot(tmp_volatility, tmp_weight_negative.T,
                            colors = industries_color_rf)


plt.xlabel('Volatility')
plt.ylabel('Weights') 

  
## ----- Title of the graph ----- ##
plt.title('Weight of each assets on the Mean-Variance Frontier With Rf') 

## ----- Adding legend to the graph ----- ##
plt.legend(loc='upper left')
 
## ----- Showing plot ----- ##
if __name__ == "__main__":
    plt.grid()
    plt.show() 


# Check the Sharpe Ratio for each unit of risk on the portfolio
SR_Q7_6 = (np.asarray(y_returnQ7_4) - risk_free_rate) / np.asarray(x_volatilityQ7_4)
max_SR_Q7_6 = max(SR_Q7_6)
argmax_SR_Q7_6 = np.argmax(SR_Q7_6)
y_return_max_SR_Q7_6 = y_returnQ7_4[argmax_SR_Q7_6]

plot_answer3_2 = plt.plot(y_returnQ7_4, SR_Q7_6, label = 'Frontier')
plt.scatter(y_return_max_SR_Q7_6, max_SR_Q7_6, marker= "o",
                label = 'Tangency Pf', c = 'k', s=40)  
  
## ----- Naming axis ----- ##
plt.xlabel('Expected Returns')
plt.ylabel('Sharpe Ratio') 
plt.xlim(0)
  
## ----- Title of the graph ----- ##
plt.title('Sharpe Ratio') 

## ----- Adding legend to the graph ----- ##
plt.legend(loc='upper left')
 
## ----- Showing plot ----- ##
if __name__ == "__main__":
    plt.grid()
    plt.show() 
 







## ----- Consolidation of plot with and without riskfree ----- ##
plot_answer5 = plt.plot(x_volatilityQ7_4, y_returnQ7_4, label = 'Frontier without rf')
plot_answer6 = plt.plot(x_volatilityQ7_5, y_returnQ7_5, label = 'Frontier with rf')

for x_point, y_point, ind in zip(volatilities, mu, desired_industries_Q7):
    plt.scatter(x_point, y_point, marker= "*",
                label = ind, s=30)
plt.scatter(0, risk_free_rate, marker= "<",
                label = 'Risk free rate', s=80, color = 'b') 
plt.scatter(x_tangency_volatilityQ7, y_tangency_expected_returnQ7, marker= "o",
            label = 'Tangency Pf', s=40, c = 'k') 
  
## ----- Naming axis ----- ##
plt.xlabel('Volatility')
plt.ylabel('Return') 
plt.xlim(0)
  
## ----- Title of the graph ----- ##
plt.title('Mean-Variance Frontier With Rf') 

## ----- Adding legend to the graph ----- ##
plt.legend(loc='upper left')
 
## ----- Showing plot ----- ##
if __name__ == "__main__":
    plt.grid()
    plt.show()


## -------- COMPARASON ------- ##

# Question 1-3
    
plt.plot(x_volatilityQ1, y_returnQ1, label = 'Frontier without rf (5)', c='b', linestyle = '--')
plt.plot(x_volatilityQ2, y_returnQ2, label = 'Frontier with rf (5)')
plt.scatter(volatility_tangency1, expected_return_tangancy1, 
            label ='Tangency Portfolio', c = 'b')

plt.plot(x_volatilityQ7_1, y_returnQ7_1, label = 'Frontier without rf (10)', c = 'r', linestyle = '--')
plt.plot(x_volatilityQ7_2, y_returnQ7_2, label = 'Frontier with rf (10)', c = 'r')
plt.scatter(volatility_tangencyQ7_1, expected_return_tangancyQ7_1, 
            label ='Tangency Portfolio', c = 'r')
for x_point, y_point, ind in zip(volatilities, mu, desired_industries_Q7):
    plt.scatter(x_point, y_point, marker= "*", s=30)
plt.scatter(0, risk_free_rate, marker= "<",
                label = 'Risk free rate', s=80, color = 'b') 
  
## ----- Naming axis ----- ##
plt.xlabel('Volatility')
plt.ylabel('Return') 
plt.xlim(0)
  
## ----- Title of the graph ----- ##
plt.title('Mean-Variance Frontier Without Short Sale Constraint') 

## ----- Adding legend to the graph ----- ##
plt.legend(loc='best')
 
## ----- Showing plot ----- ##
if __name__ == "__main__":
    plt.grid()
    plt.show()

# Question 4-6
    
plt.plot(x_volatilityQ4, y_returnQ4, label = 'Frontier without rf (5)', c='b', linestyle = '--')
plt.plot(x_volatilityQ5, y_returnQ5, label = 'Frontier with rf (5)')
plt.scatter(x_tangency_volatility, y_tangency_expected_return, 
            label ='Tangency Portfolio', c = 'b')

plt.plot(x_volatilityQ7_4, y_returnQ7_4, label = 'Frontier without rf (10)', c = 'r', linestyle = '--')
plt.plot(x_volatilityQ7_5, y_returnQ7_5, label = 'Frontier with rf (10)', c = 'r')
plt.scatter(x_tangency_volatilityQ7, y_tangency_expected_returnQ7, 
            label ='Tangency Portfolio', c = 'r')
for x_point, y_point, ind in zip(volatilities, mu, desired_industries_Q7):
    plt.scatter(x_point, y_point, marker= "*", s=30)
plt.scatter(0, risk_free_rate, marker= "<",
                label = 'Risk free rate', s=80, color = 'b') 
  
## ----- Naming axis ----- ##
plt.xlabel('Volatility')
plt.ylabel('Return') 
plt.xlim(0)
  
## ----- Title of the graph ----- ##
plt.title('Mean-Variance Frontier With Short Sale Constraint') 

## ----- Adding legend to the graph ----- ##
plt.legend(loc='best')
 
## ----- Showing plot ----- ##
if __name__ == "__main__":
    plt.grid()
    plt.show()




##
##
## ---------- Part B ----------------------------------------------------------
##
##    

#Partie B: BOOTSTRAPPING


#Partie B: Question 1 (1-6)


## ----- Parameters ----- ##
desired_industries = ['Food','Smoke','Toys','Books','Steel']
desired_industries_rf_Q7 = desired_industries[:]
desired_industries_rf.insert(0, 'Risk Free Asset')
lastyears = '5Y'


df = df.last(lastyears)
df = pd.DataFrame.sort_index(df, ascending = False)
working_data = df.loc[:,desired_industries]



#########Q1
#Fonction de bootstrap et while loop pour générer diff mu
def bootstrapQ1(working_data, simulation = 100):
    X_plotQ1 = np.zeros((max_expected_return-min_expected_return,simulation-1))
    Y_plotQ1 = np.zeros((max_expected_return-min_expected_return,simulation-1))
    i = 1
    while i<simulation:
            X = working_data.sample(len(working_data),replace=True)
            x_volatilityQ1 = []
            y_returnQ1 = []
            weightsQ1 = []
            for expected_return in range(min_expected_return, max_expected_return):
                expected_return = expected_return/100
                (tmp1, tmp2, tmp3) = myf.minvarpf(X, expected_return, MyInput.rf)
                ## ----- Appending results for plotting ----- ##
                y_returnQ1.append(tmp1)
                x_volatilityQ1.append(tmp2)
                weightsQ1.append(tmp3)
            #
            X_plotQ1[:,i-1] = x_volatilityQ1
            Y_plotQ1[:,i-1] = y_returnQ1       
            i += 1
    return X_plotQ1,Y_plotQ1

Fct = bootstrapQ1(working_data,30)
X_plotQ1_ = Fct[0]
Y_plotQ1_ = Fct[1]

Shape = X_plotQ1_.shape

i = 0

plt.figure()
while i<Shape[1]:
    plt.plot(X_plotQ1_[:,i],Y_plotQ1_[:,i],c='#C0C0C0')
    i += 1

plt.plot(x_volatilityQ1, y_returnQ1, label = 'Original efficient frontier')
plt.ylabel("Expected Return")
plt.xlabel("Volatility")
plt.title("Bootstrap Efficient Locus (without the risk-free asset)")
plt.ylim(-0.5,2.5)
plt.xlim(0,20)

## ----- Title of the graph ----- ##
plt.title('Mean-Variance Frontier Without Short Sale Constraint') 

## ----- Adding legend to the graph ----- ##
plt.legend(loc='upper right')
 
## ----- Showing plot ----- ##
if __name__ == "__main__":
    plt.grid()
    plt.show()


#########Q2
#Fonction de bootstrap et while loop pour générer diff mu
def bootstrapQ2(working_data, simulation = 100):
    X_plotQ2 = np.zeros((max_expected_return-min_expected_return,simulation-1))
    Y_plotQ2 = np.zeros((max_expected_return-min_expected_return,simulation-1))
    i = 1
    while i<simulation:
            X = working_data.sample(len(working_data),replace=True)
            x_volatilityQ2 = []
            y_returnQ2 = []
            weightsQ2 = []
            for expected_return in range(min_expected_return, max_expected_return):
                expected_return = expected_return/100
                (tmp1, tmp2, tmp3) = myf.minvarpf(X, expected_return, MyInput.rf, 
                risk_free_allowed = True)
                ## ----- Appending results for plotting ----- ##
                y_returnQ2.append(tmp1)
                x_volatilityQ2.append(tmp2)
                weightsQ2.append(tmp3)
            #
            X_plotQ2[:,i-1] = x_volatilityQ2
            Y_plotQ2[:,i-1] = y_returnQ2       
            i += 1
    return X_plotQ2,Y_plotQ2

Fct = bootstrapQ2(working_data,30)
X_plotQ2_ = Fct[0]
Y_plotQ2_ = Fct[1]

Shape = X_plotQ2_.shape

i = 0

plt.figure()
while i<Shape[1]:
    plt.plot(X_plotQ2_[:,i],Y_plotQ2_[:,i],c='#C0C0C0')
    i += 1

plt.plot(x_volatilityQ2, y_returnQ2, label = 'Original efficient frontier')
plt.ylabel("Expected Return")
plt.xlabel("Volatility")
plt.title("Bootstrap Efficient Locus (with the risk-free asset)")
plt.ylim(-0.5,2)
plt.xlim(0,15)

## ----- Adding legend to the graph ----- ##
plt.legend(loc='upper right')
 
## ----- Showing plot ----- ##
if __name__ == "__main__":
    plt.grid()
    plt.show()

#########Q4

x_volatilityQ4 = []
y_returnQ4 = []
weightsQ4 = []

(tmp1, tmp2, tmp3) = myf.minvarpf_noshortsale(working_data, [], risk_free_rate)
var_min = tmp2
tmp3 = tmp3[1:]

for expected_return in np.arange(min(mu),max(mu), 0.01):
    expected_return = expected_return
    (tmp1, tmp2, tmp3) = myf.minvarpf_noshortsale(working_data, expected_return, risk_free_rate)
    ## ----- Appending results for plotting ----- ##
    y_returnQ4.append(tmp1)
    x_volatilityQ4.append(tmp2)
    weightsQ4.append(tmp3)
    
#Fonction de bootstrap et while loop pour générer diff mu
def bootstrapQ5(working_data, simulation = 100):
    mu = working_data.mean()
    X_plotQ5 = np.zeros((1000,simulation-1))
    X_plotQ5.fill(np.nan)
    Y_plotQ5 = np.zeros((1000,simulation-1))
    Y_plotQ5.fill(np.nan)
    i = 1
    while i<simulation:
            X = working_data.sample(len(working_data),replace=True)
            mu = X.mean()
            x_volatilityQ5 = []
            y_returnQ5 = []
            weightsQ5 = []
            for expected_return in np.arange(min(mu), max(mu), 0.05):
                expected_return = expected_return
                (tmp1, tmp2, tmp3) = myf.minvarpf_noshortsale(X, expected_return, MyInput.rf)
                ## ----- Appending results for plotting ----- ##
                y_returnQ5.append(tmp1)
                x_volatilityQ5.append(tmp2)
                weightsQ5.append(tmp3)
            #
            X_plotQ5[:len(x_volatilityQ5),i-1] = x_volatilityQ5
            Y_plotQ5[:len(x_volatilityQ5),i-1] = y_returnQ5       
            i += 1
    return X_plotQ5,Y_plotQ5

Fct = bootstrapQ5(working_data,30)
X_plotQ5_ = Fct[0]
Y_plotQ5_ = Fct[1]

Shape = X_plotQ5_.shape

i = 0

plt.figure()
while i<Shape[1]:
    plt.plot(X_plotQ5_[:,i],Y_plotQ5_[:,i],c='#C0C0C0')
    i += 1

plt.plot(x_volatilityQ4, y_returnQ4, label = 'Original efficient frontier')
plt.ylabel("Expected Return")
plt.xlabel("Volatility")
plt.title("Bootstrap Efficient Locus (without the rf asset & with short-sale constraint)")
plt.ylim(-.5,2)
plt.xlim(0,6)

## ----- Adding legend to the graph ----- ##
plt.legend(loc='upper right')
 
## ----- Showing plot ----- ##
if __name__ == "__main__":
    plt.grid()
    plt.show()

#########Q5
#Fonction de bootstrap et while loop pour générer diff mu
def bootstrapQ6(working_data, simulation = 100):
    X_plotQ6 = np.zeros((1000,simulation-1))
    X_plotQ6.fill(np.nan)
    Y_plotQ6 = np.zeros((1000,simulation-1))
    Y_plotQ6.fill(np.nan)
    i = 1
    while i<simulation:
            X = working_data.sample(len(working_data),replace=True)
            mu = X.mean()
            x_volatilityQ6 = []
            y_returnQ6 = []
            weightsQ6 = []
            for expected_return in range(round(MyInput.rf*100), round(max(mu)*100)):
                expected_return = expected_return/100
                (tmp1, tmp2, tmp3) = myf.minvarpf_noshortsale(X, expected_return, MyInput.rf,
                risk_free_allowed = True)
                ## ----- Appending results for plotting ----- ##
                y_returnQ6.append(tmp1)
                x_volatilityQ6.append(tmp2)
                weightsQ6.append(tmp3)
            #
            X_plotQ6[:len(x_volatilityQ6),i-1] = x_volatilityQ6
            Y_plotQ6[:len(x_volatilityQ6),i-1] = y_returnQ6    
            i += 1
    return X_plotQ6,Y_plotQ6

Fct = bootstrapQ6(working_data,30)
X_plotQ6_ = Fct[0]
Y_plotQ6_ = Fct[1]

Shape = X_plotQ6_.shape

i = 0

plt.figure()
while i<Shape[1]:
    plt.plot(X_plotQ6_[:,i],Y_plotQ6_[:,i],c='#C0C0C0')
    i += 1

plt.plot(x_volatilityQ5, y_returnQ5, label = 'Original efficient frontier')
plt.ylabel("Expected Return")
plt.xlabel("Volatility")
plt.title("Bootstrap Efficient Locus (with the rf asset & short-sale constraint)")
plt.ylim(0,2.5)
plt.xlim(0)

## ----- Adding legend to the graph ----- ##
plt.legend(loc='upper right')
 
## ----- Showing plot ----- ##
if __name__ == "__main__":
    plt.grid()
    plt.show()

#########Q7
desired_industries_Q7 = ['Food','Smoke','Toys','Books','Steel', 'Agric', 'Fin', 
                      'Hlth', 'Drugs', 'Banks']
desired_industries_rf_Q7 = desired_industries_Q7[:]
desired_industries_rf_Q7.insert(0, 'Risk Free Asset')
lastyears = '5Y'


df = df.last(lastyears)
df = pd.DataFrame.sort_index(df, ascending = False)
working_data = df.loc[:,desired_industries_Q7]
    
    
#########Q7_1
#Fonction de bootstrap et while loop pour générer diff mu
def bootstrapQ1(working_data, simulation = 100):
    X_plotQ1 = np.zeros((max_expected_return-min_expected_return,simulation-1))
    Y_plotQ1 = np.zeros((max_expected_return-min_expected_return,simulation-1))
    i = 1
    while i<simulation:
            X = working_data.sample(len(working_data),replace=True)
            x_volatilityQ1 = []
            y_returnQ1 = []
            weightsQ1 = []
            for expected_return in range(min_expected_return, max_expected_return):
                expected_return = expected_return/100
                (tmp1, tmp2, tmp3) = myf.minvarpf(X, expected_return, MyInput.rf)
                ## ----- Appending results for plotting ----- ##
                y_returnQ1.append(tmp1)
                x_volatilityQ1.append(tmp2)
                weightsQ1.append(tmp3)
            #
            X_plotQ1[:,i-1] = x_volatilityQ1
            Y_plotQ1[:,i-1] = y_returnQ1       
            i += 1
    return X_plotQ1,Y_plotQ1

Fct = bootstrapQ1(working_data,30)
X_plotQ1_ = Fct[0]
Y_plotQ1_ = Fct[1]

Shape = X_plotQ1_.shape

i = 0

plt.figure()
while i<Shape[1]:
    plt.plot(X_plotQ1_[:,i],Y_plotQ1_[:,i],c='#C0C0C0')
    i += 1

plt.plot(x_volatilityQ7_1, y_returnQ7_1, label = 'Original efficient frontier (10 industries)')
plt.ylabel("Expected Return")
plt.xlabel("Volatility")
plt.title("Bootstrap Efficient Locus (without the risk-free asset)")
plt.ylim(-0.5,2.5)
plt.xlim(0,20)

## ----- Title of the graph ----- ##
plt.title('Mean-Variance Frontier Without Short Sale Constraint') 

## ----- Adding legend to the graph ----- ##
plt.legend(loc='upper right')
 
## ----- Showing plot ----- ##
if __name__ == "__main__":
    plt.grid()
    plt.show()


#########Q7_2
#Fonction de bootstrap et while loop pour générer diff mu
def bootstrapQ2(working_data, simulation = 100):
    X_plotQ2 = np.zeros((max_expected_return-min_expected_return,simulation-1))
    Y_plotQ2 = np.zeros((max_expected_return-min_expected_return,simulation-1))
    i = 1
    while i<simulation:
            X = working_data.sample(len(working_data),replace=True)
            x_volatilityQ2 = []
            y_returnQ2 = []
            weightsQ2 = []
            for expected_return in range(min_expected_return, max_expected_return):
                expected_return = expected_return/100
                (tmp1, tmp2, tmp3) = myf.minvarpf(X, expected_return, MyInput.rf, 
                risk_free_allowed = True)
                ## ----- Appending results for plotting ----- ##
                y_returnQ2.append(tmp1)
                x_volatilityQ2.append(tmp2)
                weightsQ2.append(tmp3)
            #
            X_plotQ2[:,i-1] = x_volatilityQ2
            Y_plotQ2[:,i-1] = y_returnQ2       
            i += 1
    return X_plotQ2,Y_plotQ2

Fct = bootstrapQ2(working_data,30)
X_plotQ2_ = Fct[0]
Y_plotQ2_ = Fct[1]

Shape = X_plotQ2_.shape

i = 0

plt.figure()
while i<Shape[1]:
    plt.plot(X_plotQ2_[:,i],Y_plotQ2_[:,i],c='#C0C0C0')
    i += 1

plt.plot(x_volatilityQ7_2, y_returnQ7_2, label = 'Original efficient frontier (10 industries)')
plt.ylabel("Expected Return")
plt.xlabel("Volatility")
plt.title("Bootstrap Efficient Locus (with the risk-free asset)")
plt.ylim(-0.5,2)
plt.xlim(0,15)

## ----- Adding legend to the graph ----- ##
plt.legend(loc='upper right')
 
## ----- Showing plot ----- ##
if __name__ == "__main__":
    plt.grid()
    plt.show()

#########Q7_4
#Fonction de bootstrap et while loop pour générer diff mu
def bootstrapQ5(working_data, simulation = 100):
    mu = working_data.mean()
    X_plotQ5 = np.zeros((1000,simulation-1))
    X_plotQ5.fill(np.nan)
    Y_plotQ5 = np.zeros((1000,simulation-1))
    Y_plotQ5.fill(np.nan)
    i = 1
    while i<simulation:
            X = working_data.sample(len(working_data),replace=True)
            mu = X.mean()
            x_volatilityQ5 = []
            y_returnQ5 = []
            weightsQ5 = []
            for expected_return in np.arange(min(mu), max(mu), 0.05):
                expected_return = expected_return
                (tmp1, tmp2, tmp3) = myf.minvarpf_noshortsale(X, expected_return, MyInput.rf)
                ## ----- Appending results for plotting ----- ##
                y_returnQ5.append(tmp1)
                x_volatilityQ5.append(tmp2)
                weightsQ5.append(tmp3)
            #
            X_plotQ5[:len(x_volatilityQ5),i-1] = x_volatilityQ5
            Y_plotQ5[:len(x_volatilityQ5),i-1] = y_returnQ5       
            i += 1
    return X_plotQ5,Y_plotQ5

Fct = bootstrapQ5(working_data,30)
X_plotQ5_ = Fct[0]
Y_plotQ5_ = Fct[1]

Shape = X_plotQ5_.shape

i = 0

plt.figure()
while i<Shape[1]:
    plt.plot(X_plotQ5_[:,i],Y_plotQ5_[:,i],c='#C0C0C0')
    i += 1

plt.plot(x_volatilityQ7_4, y_returnQ7_4, label = 'Original efficient frontier (10 industries)')
plt.ylabel("Expected Return")
plt.xlabel("Volatility")
plt.title("Bootstrap Efficient Locus (without the rf asset & with short-sale constraint)")
plt.ylim(-.5,2)
plt.xlim(0,7)

## ----- Adding legend to the graph ----- ##
plt.legend(loc='upper right')
 
## ----- Showing plot ----- ##
if __name__ == "__main__":
    plt.grid()
    plt.show()

#########Q7_5
#Fonction de bootstrap et while loop pour générer diff mu
def bootstrapQ6(working_data, simulation = 100):
    X_plotQ6 = np.zeros((1000,simulation-1))
    X_plotQ6.fill(np.nan)
    Y_plotQ6 = np.zeros((1000,simulation-1))
    Y_plotQ6.fill(np.nan)
    i = 1
    while i<simulation:
            X = working_data.sample(len(working_data),replace=True)
            mu = X.mean()
            x_volatilityQ6 = []
            y_returnQ6 = []
            weightsQ6 = []
            for expected_return in range(round(MyInput.rf*100), round(max(mu)*100)):
                expected_return = expected_return/100
                (tmp1, tmp2, tmp3) = myf.minvarpf_noshortsale(X, expected_return, MyInput.rf,
                risk_free_allowed = True)
                ## ----- Appending results for plotting ----- ##
                y_returnQ6.append(tmp1)
                x_volatilityQ6.append(tmp2)
                weightsQ6.append(tmp3)
            #
            X_plotQ6[:len(x_volatilityQ6),i-1] = x_volatilityQ6
            Y_plotQ6[:len(x_volatilityQ6),i-1] = y_returnQ6    
            i += 1
    return X_plotQ6,Y_plotQ6

Fct = bootstrapQ6(working_data,30)
X_plotQ6_ = Fct[0]
Y_plotQ6_ = Fct[1]

Shape = X_plotQ6_.shape

i = 0

plt.figure()
while i<Shape[1]:
    plt.plot(X_plotQ6_[:,i],Y_plotQ6_[:,i],c='#C0C0C0')
    i += 1

plt.plot(x_volatilityQ7_5, y_returnQ7_5, label = 'Original efficient frontier (10 industries)')
plt.ylabel("Expected Return")
plt.xlabel("Volatility")
plt.title("Bootstrap Efficient Locus (with the rf asset & short-sale constraint)")
plt.ylim(0,2.5)
plt.xlim(0)

## ----- Adding legend to the graph ----- ##
plt.legend(loc='upper right')
 
## ----- Showing plot ----- ##
if __name__ == "__main__":
    plt.grid()
    plt.show()
    
# =============================================================================
# Question B2
# You want to minimize transaction costs. You want to invest in up to 
# 3 portfolios from the original list of 5 industry portfolios. Repeat the 
# same calculations in 1-6 with this constraint on the maximum number of assets.
# =============================================================================

# Finding the filtered industries of the original 5 that max the Sharpe ratio
# for questions 1-3 (without short sale constraints)

# Parameters
# Original list of industries
desired_industries = ['Food','Smoke','Toys','Books','Steel']
#desired_industries = ['Food','Smoke','Agric','Soda','Beer']
lastyears = '5Y'

df = df.last(lastyears)
df = pd.DataFrame.sort_index(df, ascending = False)

## ---- Finding Best 3 Industries----------------------------------------------


positions = list(range(0,len(working_data.columns)))

max_SR_weights_b2 = []
max_SR_b2 = []
subsets = []

# For 3 industries
working_data = df.loc[:,desired_industries]

short_list=["","",""]
for subset in it.combinations(positions,3):
    subsets.append(subset)
    short_list[0] = desired_industries[subset[0]]
    short_list[1] = desired_industries[subset[1]]
    short_list[2] = desired_industries[subset[2]]
      
    working_data_short = working_data.loc[:,short_list] 
    
    mu = working_data_short.mean()
    covariance_matrix = np.array(working_data_short.cov())
    volatilities = np.diagonal(np.sqrt(covariance_matrix))
    n_industries = mu.size
    
    temp_sr =  myf.minvarpf(working_data_short, [], risk_free_rate, risk_free_allowed = False , tangency = True)
    max_SR_weights_b2.append(temp_sr[2])
    max_SR_b2.append((temp_sr[0] - risk_free_rate)/temp_sr[1])

# For 2 industries
working_data = df.loc[:,desired_industries]

short_list=["",""]
for subset in it.combinations(positions,2):
    subsets.append(subset)
    short_list[0] = desired_industries[subset[0]]
    short_list[1] = desired_industries[subset[1]]
      
    working_data_short = working_data.loc[:,short_list]
    
    mu = working_data_short.mean()
    covariance_matrix = np.array(working_data_short.cov())
    volatilities = np.diagonal(np.sqrt(covariance_matrix))
    n_industries = mu.size
    
    temp_sr =  myf.minvarpf(working_data_short, [], risk_free_rate, risk_free_allowed = False , tangency = True)
    max_SR_weights_b2.append(temp_sr[2])
    max_SR_b2.append((temp_sr[0] - risk_free_rate)/temp_sr[1])    

# =============================================================================
#
# # For 1 industries    
# working_data = df.loc[:,desired_industries]
# positions = list(range(0,len(working_data.columns)))
# short_list=[""]
# for subset in positions:
# #for subset in range(0,4):
#     subsets.append(subset)
#     short_list[0] = desired_industries[subset]
# 
#     working_data_short = working_data.loc[:,short_list]
#     
#     mu = working_data_short.mean()
#     covariance_matrix = np.array(working_data_short.cov())
#     volatilities = np.diagonal(np.sqrt(covariance_matrix))
#     n_industries = mu.size
#     
#     temp_sr =  myf.minvarpf(working_data_short, [], risk_free_rate, risk_free_allowed = False , tangency = True)
#     max_SR_weights_b2.append(temp_sr[2])
#     max_SR_b2.append((temp_sr[0] - risk_free_rate)/temp_sr[1])        
# 
# =============================================================================


# Index of Max Sharpe
max_index = max_SR_b2.index(max(max_SR_b2))

if max_index <= len(list(it.combinations(positions,3))):
    max_desired_industries = ["","",""]    
    max_desired_industries[0] = desired_industries[subsets[max_index][0]]
    max_desired_industries[1] = desired_industries[subsets[max_index][1]]
    max_desired_industries[2] = desired_industries[subsets[max_index][2]]
    
else:
    max_desired_industries = ["",""]    
    max_desired_industries[0] = desired_industries[subsets[max_index][0]]
    max_desired_industries[1] = desired_industries[subsets[max_index][1]]

print(max_desired_industries)


#------------------------------------------------------------------------------
working_data = df.loc[:,max_desired_industries]
desired_industries_rf = max_desired_industries[:]
desired_industries_rf.insert(0, 'Risk Free Asset')

## Parameters and data storage

risk_free_rate = 0.001
mu = working_data.mean()
covariance_matrix = working_data.cov()
volatilities = np.diagonal(np.sqrt(covariance_matrix))
n_industries = mu.size
#Plot and optimization parameters
min_expected_return = -50
max_expected_return = 300


# =============================================================================
# # Question B2_1: 
# Graph the "mean-variance locus" (without the risk-free asset) of these 3 
# industry portfolios. Specify each industry portfolio in the chart.
# =============================================================================
x_volatilityQ1 = []
y_returnQ1 = []
weightsQ1 = []
for expected_return in range(min_expected_return, max_expected_return):
    expected_return = expected_return/100
    (tmp1, tmp2, tmp3) = myf.minvarpf(working_data, expected_return, risk_free_rate)
    ## ----- Appending results for plotting ----- ##
    y_returnQ1.append(tmp1)
    x_volatilityQ1.append(tmp2)
    weightsQ1.append(tmp3)

(tmp1, tmp2, tmp3) = myf.minvarpf(working_data, [], risk_free_rate)
min_var_returnQ1 = tmp1
min_var_volatilityQ1 = tmp2 

## ----- Saving plots for using in Q3 ----- ##
plot_answer1 = plt.plot(x_volatilityQ1, y_returnQ1, label = 'Frontier')
industries_color = []    
for x_point, y_point, ind in zip(volatilities, mu, max_desired_industries):
    colortmp = plt.scatter(x_point, y_point, marker= "*",
                label = ind, s=30)
    ## Saving the colors for each industries
    tmp = colortmp.get_facecolors()
    tmp_tuple = tuple(tmp[0])
    industries_color.append(tmp_tuple)

plt.scatter(min_var_volatilityQ1, min_var_returnQ1, marker = '^',
            label = 'Minimum variance portfolio', s=60)
## ----- Naming axis ----- ##
plt.xlabel('Volatility')
plt.ylabel('Return') 
plt.xlim([0,10]) 
 
## ----- Title of the graph ----- ##
plt.title('Mean-Variance Locus Without Rf') 

## ----- Adding legend to the graph ----- ##
plt.legend(loc='upper left')
 
## ----- Showing plot ----- ##
if __name__ == "__main__":
    plt.grid()
    plt.show() 
    
    
# Area plot of weight in mean variance to volatility
    
x_volatilityQ1 = []
y_returnQ1 = []
weightsQ1 = []
for expected_return in range(int(round(min_var_returnQ1*100)), max_expected_return):
    expected_return = expected_return/100
    (tmp1, tmp2, tmp3) = myf.minvarpf(working_data, expected_return, risk_free_rate)
    ## ----- Appending results for plotting ----- ##
    y_returnQ1.append(tmp1)
    x_volatilityQ1.append(tmp2)
    weightsQ1.append(tmp3)

tmp_weight = np.asarray(weightsQ1)
tmp_weight = tmp_weight[:,1:]
tmp_weight_positive = np.copy(tmp_weight)
tmp_weight_positive[tmp_weight_positive < 0] = 0
tmp_weight_negative = np.copy(tmp_weight)
tmp_weight_negative[tmp_weight_negative > 0] = 0

tmp_volatility = np.asarray(x_volatilityQ1)

stackplotQ1 = plt.stackplot(tmp_volatility, tmp_weight_positive.T, 
                            colors = industries_color, labels = max_desired_industries)
stackplotQ1 = plt.stackplot(tmp_volatility, tmp_weight_negative.T,
                            colors = industries_color)


plt.xlabel('Volatility')
plt.ylabel('Weights') 

  
## ----- Title of the graph ----- ##
plt.title('Weight of each assets on the Mean-Variance Frontier Without Rf') 

## ----- Adding legend to the graph ----- ##
plt.legend(loc='upper left')
 
## ----- Showing plot ----- ##
if __name__ == "__main__":
    plt.grid()
    plt.show()

# =============================================================================
# # Question B2_2: 
# Graph the "mean-variance locus" (with the risk-free asset) of these 5 industry 
# portfolios.
# =============================================================================

x_volatilityQ2 = []
y_returnQ2 = []
weightsQ2 = []
for expected_return in range(min_expected_return, max_expected_return):
    expected_return = expected_return/100
    (tmp1, tmp2, tmp3) = myf.minvarpf(working_data, expected_return, risk_free_rate, 
    risk_free_allowed = True)
    ## ----- Appending results for plotting ----- ##
    y_returnQ2.append(tmp1)
    x_volatilityQ2.append(tmp2)
    weightsQ2.append(tmp3)
    
## ----- Saving plots for using in Q3 ----- ##
plot_answer2 = plt.plot(x_volatilityQ2, y_returnQ2, label = 'Frontier')

for x_point, y_point, ind, col in zip(volatilities, mu, max_desired_industries, industries_color):
    plt.scatter(x_point, y_point, marker= "*",
                label = ind, color = col , s=30) 
plt.scatter(0, risk_free_rate, marker= "<",
                label = 'Risk free rate', s=80, color = 'b') 
## ----- Naming axis ----- ##
plt.xlabel('Volatility')
plt.ylabel('Return') 
plt.xlim([0,10])
  
## ----- Title of the graph ----- ##
plt.title('Mean-Variance Locus With rf') 

## ----- Adding legend to the graph ----- ##
plt.legend(loc='upper left')
 
## ----- Showing plot ----- ##
if __name__ == "__main__":
    plt.grid()
    plt.show() 
    

x_volatilityQ2 = []
y_returnQ2 = []
weightsQ2 = []
for expected_return in range(int(round(risk_free_rate)), max_expected_return):
    expected_return = expected_return/100
    (tmp1, tmp2, tmp3) = myf.minvarpf(working_data, expected_return, risk_free_rate, 
    risk_free_allowed = True)
    ## ----- Appending results for plotting ----- ##
    y_returnQ2.append(tmp1)
    x_volatilityQ2.append(tmp2)
    weightsQ2.append(tmp3)

#Add risk free asset in industries color
industries_color_rf = industries_color[:]
industries_color_rf.insert(0,('b'))


tmp_weight = np.asarray(weightsQ2)
tmp_weight_positive = np.copy(tmp_weight)
tmp_weight_positive[tmp_weight_positive < 0] = 0
tmp_weight_negative = np.copy(tmp_weight)
tmp_weight_negative[tmp_weight_negative > 0] = 0

tmp_volatility = np.asarray(x_volatilityQ2)

stackplotQ2 = plt.stackplot(tmp_volatility, tmp_weight_positive.T, 
                            colors = industries_color_rf, labels = desired_industries_rf)
stackplotQ2 = plt.stackplot(tmp_volatility, tmp_weight_negative.T,
                            colors = industries_color_rf)


plt.xlabel('Volatility')
plt.ylabel('Weights') 

  
## ----- Title of the graph ----- ##
plt.title('Weight of each assets on the Mean-Variance Frontier With Rf') 

## ----- Adding legend to the graph ----- ##
plt.legend(loc='upper left')
 
## ----- Showing plot ----- ##
if __name__ == "__main__":
    plt.grid()
    plt.show()

# =============================================================================
# # Question B2_3
# Describe the tangent portfolio and its characteristics such as its mean
# and variance and the weights of each asset. 
# How can you check the tangent portfolio is the portfolio that maximizes
# the Sharpe ratio?
# =============================================================================

(tmp1, tmp2, tmp3) = myf.minvarpf(working_data, expected_return, risk_free_rate,
risk_free_allowed = True, tangency = True)

expected_return_tangancy1 = tmp1
volatility_tangency1 = tmp2
weights_tangency1 = tmp3
    
    
plt.plot(x_volatilityQ1, y_returnQ1, label = 'Frontier without rf')

plt.plot(x_volatilityQ2, y_returnQ2, label = 'Frontier with rf')

plt.scatter(volatility_tangency1, expected_return_tangancy1, 
            label ='Tangency Portfolio', c = 'k')

for x_point, y_point, ind in zip(volatilities, mu, max_desired_industries):
    plt.scatter(x_point, y_point, marker= "*",
                label = ind, s=30) 
plt.scatter(0, risk_free_rate, marker= "<",
                label = 'Risk free rate', s=80, color = 'b')   
## ----- Naming axis ----- ##
plt.xlabel('Volatility')
plt.ylabel('Return') 
plt.xlim(0)
  
## ----- Title of the graph ----- ##
plt.title('Mean-Variance frontier') 

## ----- Adding legend to the graph ----- ##
plt.legend(loc='best')
 
## ----- Showing plot ----- ##
if __name__ == "__main__":
    plt.grid()
    plt.show()    

# Check the Sharpe Ratio for each unit of risk on the portfolio
SR_Q3 = (np.asarray(y_returnQ1) - risk_free_rate) / np.asarray(x_volatilityQ1)
max_SR_Q3 = max(SR_Q3)
argmax_SR_Q3 = np.argmax(SR_Q3)
y_return_max_SR_Q3 = y_returnQ1[argmax_SR_Q3]

plot_answer3_2 = plt.plot(y_returnQ1, SR_Q3, label = 'Frontier')
plt.scatter(y_return_max_SR_Q3, max_SR_Q3, marker= "o",
                label = 'Tangency Pf', c = 'k', s=80)  
  
## ----- Naming axis ----- ##
plt.xlabel('Expected Returns')
plt.ylabel('Sharpe Ratio') 
plt.xlim(0)
  
## ----- Title of the graph ----- ##
plt.title('Sharpe Ratio') 

## ----- Adding legend to the graph ----- ##
plt.legend(loc='upper left')
 
## ----- Showing plot ----- ##
if __name__ == "__main__":
    plt.grid()
    plt.show() 

#-----------------------------------------------------------------------------
# Finding the filtered industries of the original 5 that max the Sharpe ratio
# for questions 4-6 (with short sale constraints)

# Parameters
# Original list of industries

working_data = df.loc[:,desired_industries]

## ---- Finding Best 3 Industries----------------------------------------------

max_SR_weights_b2 = []
max_SR_b2 = []
subsets = []

# For 3 industry
short_list=["","",""]
for subset in it.combinations(positions,3):
    subsets.append(subset)
    short_list[0] = desired_industries[subset[0]]
    short_list[1] = desired_industries[subset[1]]
    short_list[2] = desired_industries[subset[2]]
    
    working_data_short = working_data.loc[:,short_list]
    
    mu = working_data_short.mean()
    covariance_matrix = np.array(working_data_short.cov())
    volatilities = np.diagonal(np.sqrt(covariance_matrix))
    n_industries = mu.size
    
    temp_sr =  myf.minvarpf_noshortsale(working_data_short, [], risk_free_rate, risk_free_allowed = False , tangency = True)
    max_SR_weights_b2.append(temp_sr[2])
    max_SR_b2.append((temp_sr[0] - risk_free_rate)/temp_sr[1])
    
# For 2 industries
working_data = df.loc[:,desired_industries]
short_list=["",""]
for subset in it.combinations(positions,2):
    subsets.append(subset)
    short_list[0] = desired_industries[subset[0]]
    short_list[1] = desired_industries[subset[1]]
    
    working_data_short = working_data.loc[:,short_list]
    
    mu = working_data_short.mean()
    covariance_matrix = np.array(working_data_short.cov())
    volatilities = np.diagonal(np.sqrt(covariance_matrix))
    n_industries = mu.size
    
    temp_sr =  myf.minvarpf_noshortsale(working_data_short, [], risk_free_rate, risk_free_allowed = False , tangency = True)
    max_SR_weights_b2.append(temp_sr[2])
    max_SR_b2.append((temp_sr[0] - risk_free_rate)/temp_sr[1])
    
# Index of Max Sharpe    
max_index = max_SR_b2.index(max(max_SR_b2))

if max_index <= len(list(it.combinations(positions,3))):
    max_desired_industries = ["","",""]    
    max_desired_industries[0] = desired_industries[subsets[max_index][0]]
    max_desired_industries[1] = desired_industries[subsets[max_index][1]]
    max_desired_industries[2] = desired_industries[subsets[max_index][2]]
    
else:
    max_desired_industries = ["",""]    
    max_desired_industries[0] = desired_industries[subsets[max_index][0]]
    max_desired_industries[1] = desired_industries[subsets[max_index][1]]

print(max_desired_industries)
#------------------------------------------------------------------------------

working_data = df.loc[:,max_desired_industries]
desired_industries_rf = max_desired_industries[:]
desired_industries_rf.insert(0, 'Risk Free Asset')
mu = working_data.mean()
covariance_matrix = working_data.cov()
volatilities = np.diagonal(np.sqrt(covariance_matrix))
n_industries = mu.size



# =============================================================================
# Question B2_4
# Graph the "mean-variance locus" (without the risk-free asset) with 
# the short-sale constraints on each industry portfolio. 
# Specify each industry portfolio in the chart.
# =============================================================================
    
    
x_volatilityQ4 = []
y_returnQ4 = []
weightsQ4 = []

(tmp1, tmp2, tmp3) = myf.minvarpf_noshortsale(working_data, [], risk_free_rate)
var_min = tmp2
tmp3 = tmp3[1:]
min_var_returnQ4 = tmp3.dot(working_data.mean())

for expected_return in np.arange(min_var_returnQ4,max(mu), 0.01):
    expected_return = expected_return
    (tmp1, tmp2, tmp3) = myf.minvarpf_noshortsale(working_data, expected_return, risk_free_rate)
    ## ----- Appending results for plotting ----- ##
    y_returnQ4.append(tmp1)
    x_volatilityQ4.append(tmp2)
    weightsQ4.append(tmp3)
tmp_weight = np.asarray(weightsQ4)
tmp_volatility = np.asarray(x_volatilityQ4)



plot_answer4 = plt.plot(x_volatilityQ4, y_returnQ4, label = 'Frontier')

for x_point, y_point, ind in zip(volatilities, mu, max_desired_industries):
    plt.scatter(x_point, y_point, marker= "*",
                label = ind, s=30) 
  
## ----- Naming axis ----- ##
plt.xlabel('Volatility')
plt.ylabel('Return') 
plt.xlim(0)
  
## ----- Title of the graph ----- ##
plt.title('Mean-Variance Frontier Without Rf') 

## ----- Adding legend to the graph ----- ##
plt.legend(loc='upper left')
 
## ----- Showing plot ----- ##
if __name__ == "__main__":
    plt.grid()
    plt.show() 
 
# Area plot of weight in mean variance to volatility
    
x_volatilityQ4 = []
y_returnQ4 = []
weightsQ4 = []
for expected_return in np.arange(min_var_returnQ4,max(mu), 0.01):
    expected_return = expected_return
    (tmp1, tmp2, tmp3) =  myf.minvarpf_noshortsale(working_data, expected_return, risk_free_rate)
    ## ----- Appending results for plotting ----- ##
    y_returnQ4.append(tmp1)
    x_volatilityQ4.append(tmp2)
    weightsQ4.append(tmp3)

tmp_weight = np.asarray(weightsQ4)
tmp_weight = tmp_weight[:,1:]
tmp_weight_positive = np.copy(tmp_weight)
tmp_weight_positive[tmp_weight_positive < 0] = 0
tmp_weight_negative = np.copy(tmp_weight)
tmp_weight_negative[tmp_weight_negative > 0] = 0

tmp_volatility = np.asarray(x_volatilityQ4)

stackplotQ4 = plt.stackplot(tmp_volatility, tmp_weight_positive.T, 
                            colors = industries_color, labels = max_desired_industries)
stackplotQ4 = plt.stackplot(tmp_volatility, tmp_weight_negative.T,
                            colors = industries_color)


plt.xlabel('Volatility')
plt.ylabel('Weights') 

  
## ----- Title of the graph ----- ##
plt.title('Weight of each assets on the Mean-Variance Frontier Without Rf') 

## ----- Adding legend to the graph ----- ##
plt.legend(loc='upper right')
 
## ----- Showing plot ----- ##
if __name__ == "__main__":
    plt.grid()
    plt.show() 
    
 
# =============================================================================
# Question B2_5
# Graph the "mean-variance locus" (with the risk-free asset) with the short-sale 
# constraints oneach industry portfolio. 
# Specify each industry portfolio in the chart. 
# Explain how the meanvariance locus has changed with the risk-free asset.
# Question 6
# Describe the tangent portfolio and its characteristics such as its mean
# and variance and the weights of each asset. 
# How can you check the tangent portfolio is the portfolio that maximizes
# the Sharpe ratio?
# =============================================================================

 
x_volatilityQ5 = []
y_returnQ5 = []
weightsQ5 = []
for expected_return in range(round(risk_free_rate*100), round(max(mu)*100)):
    expected_return = expected_return/100
    (tmp1, tmp2, tmp3) = myf.minvarpf_noshortsale(working_data, expected_return, risk_free_rate,
    risk_free_allowed = True)
    ## ----- Appending results for plotting ----- ##
    y_returnQ5.append(tmp1)
    x_volatilityQ5.append(tmp2)
    weightsQ5.append(tmp3)

## ----- Tangency Portfolio ----- ##

(tmp1, tmp2, tmp3) = myf.minvarpf_noshortsale(working_data, expected_return, risk_free_rate,
    risk_free_allowed = True, tangency= True)

y_tangency_expected_return = tmp1
x_tangency_volatility = tmp2

   

plot_answer5 = plt.plot(x_volatilityQ5, y_returnQ5, label = 'Frontier')

for x_point, y_point, ind in zip(volatilities, mu, max_desired_industries):
    plt.scatter(x_point, y_point, marker= "*",
                label = ind, s=30)
plt.scatter(0, risk_free_rate, marker= "<",
                label = 'Risk free rate', s=80, color = 'b') 
plt.scatter(tmp2, tmp1, marker= "o",
            label = 'Tangency Pf', c = 'k', s=80) 
  
## ----- Naming axis ----- ##
plt.xlabel('Volatility')
plt.ylabel('Return') 
plt.xlim(0)
  
## ----- Title of the graph ----- ##
plt.title('Mean-Variance Frontier With Rf') 

## ----- Adding legend to the graph ----- ##
plt.legend(loc='upper left')
 
## ----- Showing plot ----- ##
if __name__ == "__main__":
    plt.grid()
    plt.show()

#

x_volatilityQ5 = []
y_returnQ5 = []
weightsQ5 = []
for expected_return in range(int(round(risk_free_rate)), round(max(mu)*100)):
    expected_return = expected_return/100
    (tmp1, tmp2, tmp3) = myf.minvarpf_noshortsale(working_data, expected_return, risk_free_rate,
    risk_free_allowed = True)
    ## ----- Appending results for plotting ----- ##
    y_returnQ5.append(tmp1)
    x_volatilityQ5.append(tmp2)
    weightsQ5.append(tmp3)

#Add risk free asset in industries color
industries_color_rf = industries_color[:]
industries_color_rf.insert(0,('b'))


tmp_weight = np.asarray(weightsQ5)
tmp_weight_positive = np.copy(tmp_weight)
tmp_weight_positive[tmp_weight_positive < 0] = 0
tmp_weight_negative = np.copy(tmp_weight)
tmp_weight_negative[tmp_weight_negative > 0] = 0

tmp_volatility = np.asarray(x_volatilityQ5)

stackplotQ6 = plt.stackplot(tmp_volatility, tmp_weight_positive.T, 
                            colors = industries_color_rf, labels = desired_industries_rf)
stackplotQ6 = plt.stackplot(tmp_volatility, tmp_weight_negative.T,
                            colors = industries_color_rf)


plt.xlabel('Volatility')
plt.ylabel('Weights') 

  
## ----- Title of the graph ----- ##
plt.title('Weight of each assets on the Mean-Variance Frontier With Rf') 

## ----- Adding legend to the graph ----- ##
plt.legend(loc='upper left')
 
## ----- Showing plot ----- ##
if __name__ == "__main__":
    plt.grid()
    plt.show() 

## ----- Consolidation of plot with and without riskfree ----- ##
plot_answer5 = plt.plot(x_volatilityQ4, y_returnQ4, label = 'Frontier without rf')
plot_answer6 = plt.plot(x_volatilityQ5, y_returnQ5, label = 'Frontier with rf')

for x_point, y_point, ind in zip(volatilities, mu, max_desired_industries):
    plt.scatter(x_point, y_point, marker= "*",
                label = ind, s=30)
plt.scatter(0, risk_free_rate, marker= "<",
                label = 'Risk free rate', s=80, color = 'b') 
plt.scatter(x_tangency_volatility, y_tangency_expected_return, marker= "o",
            label = 'Tangency Pf', s=40, c = 'k') 
  
## ----- Naming axis ----- ##
plt.xlabel('Volatility')
plt.ylabel('Return') 
plt.xlim(0)
  
## ----- Title of the graph ----- ##
plt.title('Mean-Variance Frontier With Rf') 

## ----- Adding legend to the graph ----- ##
plt.legend(loc='upper left')
 
## ----- Showing plot ----- ##
if __name__ == "__main__":
    plt.grid()
    plt.show()

# Check the Sharpe Ratio for each unit of risk on the portfolio
SR_Q6 = (np.asarray(y_returnQ4) - risk_free_rate) / np.asarray(x_volatilityQ4)
max_SR_Q6 = max(SR_Q6)
argmax_SR_Q6 = np.argmax(SR_Q6)
y_return_max_SR_Q6 = y_returnQ4[argmax_SR_Q6]

plot_answer3_2 = plt.plot(y_returnQ4, SR_Q6, label = 'Frontier')
plt.scatter(y_return_max_SR_Q6, max_SR_Q6, marker= "o",
                label = 'Tangency Pf', c = 'k', s=40)  
  
## ----- Naming axis ----- ##
plt.xlabel('Expected Returns')
plt.ylabel('Sharpe Ratio') 
plt.xlim(0)
  
## ----- Title of the graph ----- ##
plt.title('Sharpe Ratio') 

## ----- Adding legend to the graph ----- ##
plt.legend(loc='upper left')
 
## ----- Showing plot ----- ##
if __name__ == "__main__":
    plt.grid()
    plt.show()
    
# =============================================================================
# Question 3
# Instead of choosing 5 industries randomly as you did in part A, you now 
# want to find the industries among the 48 industries that maximize the 
# Sharpe ratio with and without short selling constraints. The investment 
# policy requires a maximum number of 5 assets. Propose and implement methods 
# to identify industries and their weight.
# =============================================================================


#Input of the function

working_data = df
class MyInput:
    mu = working_data.mean()
    covariance_matrix = np.array(working_data.cov())
    volatilities = np.diagonal(np.sqrt(covariance_matrix))
    n_industries = mu.size
    rf = 0.001
    n_sim = 1000
    n_sup = 5


def f_neighbour(x, n_industries, n_sup):
   xn = np.copy(x)
   if sum(xn) == n_sup:
       tmp = np.array(range(n_industries))
       tmp = tmp[xn]
       ix_out = np.random.choice(tmp)
       xn[ix_out] = False
       out = xn
   
   else:
       ix = random.choice(range(n_industries))
       xn[ix] = np.invert(xn[ix])
       out = np.copy(xn)
   
   if sum(xn) > n_sup :
       out = np.copy(x)
   return out

def f_SR(x, MyInput, short_constraint):
    mu = MyInput.mu
    covariance_matrix = np.array(MyInput.covariance_matrix)
    rf = MyInput.rf
    
    covariance_matrix = covariance_matrix[x, :]
    covariance_matrix = covariance_matrix[:, x]
    mu = np.array(mu[x])
    if short_constraint == True:
        constraint_weights = {'type': 'eq', 'fun': myf.constraint_on_weights}
        constraint_short_sell = {'type': 'ineq', 'fun': myf.constraint_on_short_sell}
    
        constraints = [constraint_weights, constraint_short_sell]
        
        initial_weights = np.repeat((1 / sum(x)), sum(x))
        solution = minimize(myf.tangency_objective, initial_weights, 
                            args=(rf, covariance_matrix, mu), method="SLSQP", 
                            constraints = constraints)
        SR = solution.fun
        return(SR) 
    else:
        
        [tmp1, tmp2, tmp3] = myf.minvarpf(working_data.loc[:,x], 1 , rf, True, True)
        SR = (rf - tmp1) / tmp2 
        return(SR)
        
def f_TresholdAcceptance(SR_N, SR_O, d, i):
    T = d / np.log(i + 2)
    P = min(1, np.exp(-(SR_N - SR_O) / T))
    rnd = np.random.uniform()
    if rnd < P:
        out = True
    else:
        out = False
    return(out)
    

def f_MAXSR(MyInput, n_sim, short_constraint, cooler):
    x0 = np.zeros(MyInput.n_industries, dtype = bool)
    x0 = x0.astype(dtype = bool)
    ix = random.sample(range(MyInput.n_industries), MyInput.n_sup)
    x0[ix] = np.invert(x0[ix])
    x = np.zeros((MyInput.n_industries, n_sim), dtype = bool)
    x[:,0] = x0
    SR = np.zeros(n_sim)
    SR_new = np.zeros(n_sim)
    for i in range(n_sim - 1):
        x_new = f_neighbour(x[:, i], MyInput.n_industries, MyInput.n_sup)
        SR_new[i] = f_SR(x_new, MyInput, short_constraint)
        if f_TresholdAcceptance(SR_new[i], SR[i], cooler, i) == True:
            #print('Accepted', SR_new)
            x[: ,i + 1] = x_new
            SR[i + 1] = SR_new[i]
        else:
            #print('Rejected', SR_new)
            SR[i + 1] = SR[i]
            x[: , i + 1] = x[: , i]

    idx = np.argmin(SR)
    return([x[: , idx], -SR[idx]])

n_optimization = 10

selected_industries_short = np.zeros((n_optimization, MyInput.n_industries))
sharpe_ratio_short = np.zeros(n_optimization)

for i in range(n_optimization):

    [selected_industries_short[i, :] , sharpe_ratio_short[i]] = f_MAXSR(MyInput,
    5000, False, 0.1)
    #print("Iteration ", i)
    #print("Sharpe Ratio" , sharpe_ratio_short[i])
idx = sharpe_ratio_short.argmax()   
    
final_sharpe_short = sharpe_ratio_short.max()
final_industries_short = selected_industries_short[idx]
final_industries_short = np.bool8(final_industries_short)
print('Our best industries are :', industries[final_industries_short])
print('The Sharpe Ratio is :', final_sharpe_short)

n_optimization = 10


selected_industries_noshort = np.zeros((n_optimization, MyInput.n_industries))
sharpe_ratio_noshort = np.zeros(n_optimization)

for i in range(n_optimization):

    [selected_industries_noshort[i, :] , sharpe_ratio_noshort[i]] = f_MAXSR(MyInput,
    5000, True, 0.1)
    #print("Iteration ", i)
    #print("Sharpe Ratio" , sharpe_ratio_noshort[i])
idx = sharpe_ratio_noshort.argmax()   
    
final_sharpe_noshort = sharpe_ratio_noshort.max()
final_industries_noshort = selected_industries_noshort[idx]
final_industries_noshort = np.bool8(final_industries_noshort)
print('Our best industries are :',industries[final_industries_noshort])
print('The Sharpe Ratio is:', final_sharpe_noshort)


###############3 Brut Force #################


SR_bf = []

combin = list(it.combinations(industries, 5))

i = 0
for idx in combin:
    tmp = df.loc[:,idx]
    i = i + 1
    print(i)
    [tmp1, tmp2, tmp3] = myf.minvarpf(tmp, 1 , risk_free_rate, True, True)
    SR = (risk_free_rate - tmp1) / tmp2 
    SR_bf.append(SR)
 
SR_BF_NSC = min(SR_bf)
result_BF_NSC = combin[np.argmin(SR_bf)]







SR_bf_NS = []

combin = list(it.combinations(industries, 5))


for idx in combin:
    tmp = df.loc[:,idx]
    [tmp1, tmp2, tmp3] = myf.minvarpf_noshortsale(tmp, 1 , risk_free_rate, True, True)
    SR = (risk_free_rate - tmp1) / tmp2 
    SR_bf_NS.append(SR)
 
SR_BF_NS = min(SR_bf_NS)
result_BF_NS = combin[np.argmin(SR_bf_NS)]
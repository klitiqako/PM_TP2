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
df_Returns.index = pd.to_datetime(df_Returns.index, format= '%Y%m')
df_Firm_Size.index = pd.to_datetime(df_Firm_Size.index, format= '%Y%m')
df_NB_Firms.index = pd.to_datetime(df_NB_Firms.index, format= '%Y%m')
df_MktBook.index = pd.to_datetime(df_MktBook.index, format= '%Y')
df_Daily_Returns.index = pd.to_datetime(df_Daily_Returns.index, format= '%Y%m%d')
df_Fama.index = pd.to_datetime(df_Fama.index, format= '%Y%m%d')
df_Fama_3.index = pd.to_datetime(df_Fama_3.index, format= '%Y%m')
df_Fama_5.index = pd.to_datetime(df_Fama_5.index, format= '%Y%m')


df_Returns.columns = df_Returns.columns.str.replace(' ', '')
df_Firm_Size.columns = df_Firm_Size.columns.str.replace(' ', '')
df_NB_Firms.columns = df_NB_Firms.columns.str.replace(' ', '')
df_MktBook.columns = df_MktBook.columns.str.replace(' ', '')
df_Daily_Returns.columns = df_Daily_Returns.columns.str.replace(' ', '')
df_Fama.columns = df_Fama.columns.str.replace(' ', '')
df_Fama_3.columns = df_Fama_3.columns.str.replace(' ', '')
df_Fama_5.columns = df_Fama_5.columns.str.replace(' ', '')



industries = df_Returns.columns


###############Q1:
Market_cap_C = df_Firm_Size*df_NB_Firms
Momentum_C = df_Returns.rolling(12).mean()
Momentum_C = pd.DataFrame.dropna(Momentum_C)
Book_to_Mkt_C = np.zeros((len(df_Returns),len(industries)))
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
            
###############Q2
Months = df_Returns.index
Months = Months.insert(1123,pd.to_datetime('20200201', format= '%Y%m%d'))
betas = np.zeros((len(df_Returns)-11,len(industries)))
betas= pd.DataFrame(betas)
betas.index = df_Returns.index[11:]
betas.columns = industries
df_Daily_Returns[df_Daily_Returns == -99.99] = np.nan


for i in range(0,len(df_Returns)-11): 
    for j in range(0,len(industries)):
        d = Months[12+i]
        d_Last12 = d - dateutil.relativedelta.relativedelta(months=11)
        X = df_Fama['Mkt-RF'][(df_Fama.index < d)&(df_Fama.index >= d_Last12)]
        X = sm.add_constant(X)
        Y = df_Daily_Returns.iloc[:,j][(df_Fama.index < d)&(df_Fama.index >= d_Last12)]
        beta = np.linalg.solve(np.dot(np.array(X).T, np.array(X)), np.dot(np.array(X).T, np.array(Y)))[1]
        betas.iloc[i,j] = beta

##Checking computation time difference between package and analatycal way
time_start = time.clock()
a = np.linalg.solve(np.dot(np.array(X).T, np.array(X)), np.dot(np.array(X).T, np.array(Y)))
time_elapsed1 = (time.clock() - time_start)

time_start = time.clock()
est = sm.OLS(Y,X).fit()
beta = est.params[1]
time_elapsed2 = (time.clock() - time_start)
##Analytical takes half as long

###############Q3
Idio_vol = np.zeros((len(df_Returns)-11,len(industries)))
Idio_vol= pd.DataFrame(Idio_vol)
Idio_vol.index = df_Returns.index[11:]
Idio_vol.columns = industries


for i in range(0,len(Idio_vol)): 
    for j in range(0,len(industries)):
        d = Months[i+1]
        d_Last12 = d - dateutil.relativedelta.relativedelta(months=1)
        X = df_Fama.iloc[:,:3][(df_Fama.index < d)&(df_Fama.index >= d_Last12)]
        X = sm.add_constant(X)
        Y = df_Daily_Returns.iloc[:,j][(df_Fama.index < d)&(df_Fama.index >= d_Last12)]
        est = sm.OLS(Y,X).fit()
        std_resid = np.std(est.resid)
        Idio_vol.iloc[i,j] = std_resid

###############Q4
Volatility_Q4 = Idio_vol
Book_to_Mkt_Q4 = Book_to_Mkt_C[11:]
Market_cap_Q4 = Market_cap_C[11:]

Book_to_Mkt_Q4[Book_to_Mkt_Q4 == -99.99] = np.nan
Market_cap_Q4[Market_cap_Q4 == -0] = np.nan
Momentum_C[Momentum_C == -99.99] = np.nan



Rank_Vol = np.zeros((len(betas),len(industries)))
Rank_Book_Mkt = np.zeros((len(betas),len(industries)))
Rank_Mkt_cap = np.zeros((len(betas),len(industries)))
Rank_Momemtum = np.zeros((len(betas),len(industries)))
Rank_Betas = np.zeros((len(betas),len(industries)))

Rank_Vol= pd.DataFrame(Rank_Vol)
Rank_Book_Mkt= pd.DataFrame(Rank_Book_Mkt)
Rank_Mkt_cap= pd.DataFrame(Rank_Mkt_cap)
Rank_Momemtum= pd.DataFrame(Rank_Momemtum)
Rank_Betas= pd.DataFrame(Rank_Betas)

Rank_Vol.columns = industries
Rank_Book_Mkt.columns = industries
Rank_Mkt_cap.columns = industries
Rank_Momemtum.columns = industries
Rank_Betas.columns = industries

Rank_Vol.index = betas.index
Rank_Book_Mkt.index = betas.index
Rank_Mkt_cap.index = betas.index
Rank_Momemtum.index = betas.index
Rank_Betas.index = betas.index


for i in range(0,len(betas)):
    rank1 = Volatility_Q4.iloc[i,:].rank()
    Rank_Vol.iloc[i,:] = rank1
    rank2 = Book_to_Mkt_Q4.iloc[i,:].rank()
    Rank_Book_Mkt.iloc[i,:] = rank2
    rank3 = Market_cap_Q4.iloc[i,:].rank()
    Rank_Mkt_cap.iloc[i,:] = rank3
    rank4 = Momentum_C.iloc[i,:].rank()
    Rank_Momemtum.iloc[i,:] = rank4
    rank5 = betas.iloc[i,:].rank()
    Rank_Betas.iloc[i,:] = rank5


#a = np.where(Rank_Betas>30,0.2,(np.where(Rank_Betas<=5,-0.2,0)))


#Equal-weighted
Weight_Vol = np.zeros((len(betas),len(industries)))
Weight_Book_Mkt = np.zeros((len(betas),len(industries)))
Weight_Mkt_cap = np.zeros((len(betas),len(industries)))
Weight_Momemtum = np.zeros((len(betas),len(industries)))
Weight_Betas = np.zeros((len(betas),len(industries)))

Weight_Vol= pd.DataFrame(Weight_Vol)
Weight_Book_Mkt= pd.DataFrame(Weight_Book_Mkt)
Weight_Mkt_cap= pd.DataFrame(Weight_Mkt_cap)
Weight_Momemtum= pd.DataFrame(Weight_Momemtum)
Weight_Betas= pd.DataFrame(Weight_Betas)

Weight_Vol.columns = industries
Weight_Book_Mkt.columns = industries
Weight_Mkt_cap.columns = industries
Weight_Momemtum.columns = industries
Weight_Betas.columns = industries

Weight_Vol.index = betas.index
Weight_Book_Mkt.index = betas.index
Weight_Mkt_cap.index = betas.index
Weight_Momemtum.index = betas.index
Weight_Betas.index = betas.index

for i in range(0,len(betas)):
    max1 = min(Rank_Vol.iloc[i,:].nlargest(5))
    min1 = max(Rank_Vol.iloc[i,:].nsmallest(5))
    Weight_Vol.iloc[i,:] = np.where(Rank_Vol.iloc[i,:]>=max1,0.2,(np.where(Rank_Vol.iloc[i,:]<=min1,-0.2,0)))
    max2 = min(Rank_Book_Mkt.iloc[i,:].nlargest(5))
    min2 = max(Rank_Book_Mkt.iloc[i,:].nsmallest(5))
    Weight_Book_Mkt.iloc[i,:] = np.where(Rank_Book_Mkt.iloc[i,:]>=max2,0.2,(np.where(Rank_Book_Mkt.iloc[i,:]<=min2,-0.2,0)))
    max3 = min(Rank_Mkt_cap.iloc[i,:].nlargest(5))
    min3 = max(Rank_Mkt_cap.iloc[i,:].nsmallest(5))
    Weight_Mkt_cap.iloc[i,:] = np.where(Rank_Mkt_cap.iloc[i,:]>=max3,0.2,(np.where(Rank_Mkt_cap.iloc[i,:]<=min3,-0.2,0)))
    max4 = min(Rank_Momemtum.iloc[i,:].nlargest(5))
    min4 = max(Rank_Momemtum.iloc[i,:].nsmallest(5))
    Weight_Momemtum.iloc[i,:] = np.where(Rank_Momemtum.iloc[i,:]>=max4,0.2,(np.where(Rank_Momemtum.iloc[i,:]<=min4,-0.2,0)))
    max5 = min(Rank_Betas.iloc[i,:].nlargest(5))
    min5 = max(Rank_Betas.iloc[i,:].nsmallest(5))
    Weight_Betas.iloc[i,:] = np.where(Rank_Betas.iloc[i,:]>=max5,0.2,(np.where(Rank_Betas.iloc[i,:]<=min5,-0.2,0)))
    
#Market_Cap-weighted
Weight_VW_Vol = np.zeros((len(betas),len(industries)))
Weight_VW_Book_Mkt = np.zeros((len(betas),len(industries)))
Weight_VW_Mkt_cap = np.zeros((len(betas),len(industries)))
Weight_VW_Momemtum = np.zeros((len(betas),len(industries)))
Weight_VW_Betas = np.zeros((len(betas),len(industries)))

Weight_VW_Vol= pd.DataFrame(Weight_VW_Vol)
Weight_VW_Book_Mkt= pd.DataFrame(Weight_VW_Book_Mkt)
Weight_VW_Mkt_cap= pd.DataFrame(Weight_VW_Mkt_cap)
Weight_VW_Momemtum= pd.DataFrame(Weight_VW_Momemtum)
Weight_VW_Betas= pd.DataFrame(Weight_VW_Betas)

Weight_VW_Vol.columns = industries
Weight_VW_Book_Mkt.columns = industries
Weight_VW_Mkt_cap.columns = industries
Weight_VW_Momemtum.columns = industries
Weight_VW_Betas.columns = industries

Weight_VW_Vol.index = betas.index
Weight_VW_Book_Mkt.index = betas.index
Weight_VW_Mkt_cap.index = betas.index
Weight_VW_Momemtum.index = betas.index
Weight_VW_Betas.index = betas.index

for i in range(0,len(betas)):
    large1 = Rank_Vol.iloc[i,:].nlargest(5)
    Market_Cap_L1 = sum(Market_cap_Q4.loc[betas.index[i],(large1.index)])
    small1 = Rank_Vol.iloc[i,:].nsmallest(5)
    Market_Cap_S1 = sum(Market_cap_Q4.loc[betas.index[i],(small1.index)])
    max1 = min(large1)
    min1 = max(small1)
    
    large2 = Rank_Book_Mkt.iloc[i,:].nlargest(5)
    Market_Cap_L2 = sum(Market_cap_Q4.loc[betas.index[i],(large2.index)])
    small2 = Rank_Book_Mkt.iloc[i,:].nsmallest(5)
    Market_Cap_S2 = sum(Market_cap_Q4.loc[betas.index[i],(small2.index)])
    max2 = min(large2)
    min2 = max(small2)
    
    large3 = Rank_Mkt_cap.iloc[i,:].nlargest(5)
    Market_Cap_L3 = sum(Market_cap_Q4.loc[betas.index[i],(large3.index)])
    small3 = Rank_Mkt_cap.iloc[i,:].nsmallest(5)
    Market_Cap_S3 = sum(Market_cap_Q4.loc[betas.index[i],(small3.index)])
    max3 = min(large3)
    min3 = max(small3)
    
    large4 = Rank_Momemtum.iloc[i,:].nlargest(5)
    Market_Cap_L4 = sum(Market_cap_Q4.loc[betas.index[i],(large4.index)])
    small4 = Rank_Momemtum.iloc[i,:].nsmallest(5)
    Market_Cap_S4 = sum(Market_cap_Q4.loc[betas.index[i],(small4.index)])
    max4 = min(large4)
    min4 = max(small4)
    
    large5 = Rank_Betas.iloc[i,:].nlargest(5)
    Market_Cap_L5 = sum(Market_cap_Q4.loc[betas.index[i],(large5.index)])
    small5 = Rank_Betas.iloc[i,:].nsmallest(5)
    Market_Cap_S5 = sum(Market_cap_Q4.loc[betas.index[i],(small5.index)])
    max5 = min(large5)
    min5 = max(small5)
    for j in range(0,5):
        Weight_VW_Vol.loc[betas.index[i],large1.index[j]] = Market_cap_Q4.loc[betas.index[i],(large1.index[j])]/Market_Cap_L1
        Weight_VW_Vol.loc[betas.index[i],small1.index[j]] = -Market_cap_Q4.loc[betas.index[i],(small1.index[j])]/Market_Cap_S1
        
        Weight_VW_Book_Mkt.loc[betas.index[i],large2.index[j]] = Market_cap_Q4.loc[betas.index[i],(large2.index[j])]/Market_Cap_L2
        Weight_VW_Book_Mkt.loc[betas.index[i],small2.index[j]] = -Market_cap_Q4.loc[betas.index[i],(small2.index[j])]/Market_Cap_S2
        
        Weight_VW_Mkt_cap.loc[betas.index[i],large3.index[j]] = Market_cap_Q4.loc[betas.index[i],(large3.index[j])]/Market_Cap_L3
        Weight_VW_Mkt_cap.loc[betas.index[i],small3.index[j]] = -Market_cap_Q4.loc[betas.index[i],(small3.index[j])]/Market_Cap_S3
        
        Weight_VW_Momemtum.loc[betas.index[i],large4.index[j]] = Market_cap_Q4.loc[betas.index[i],(large4.index[j])]/Market_Cap_L4
        Weight_VW_Momemtum.loc[betas.index[i],small4.index[j]] = -Market_cap_Q4.loc[betas.index[i],(small4.index[j])]/Market_Cap_S4
        
        Weight_VW_Betas.loc[betas.index[i],large5.index[j]] = Market_cap_Q4.loc[betas.index[i],(large5.index[j])]/Market_Cap_L5
        Weight_VW_Betas.loc[betas.index[i],small5.index[j]] = -Market_cap_Q4.loc[betas.index[i],(small5.index[j])]/Market_Cap_S5
   
    Weight_VW_Vol.iloc[i,:][Weight_VW_Vol.iloc[i,:]==1] = 0
    Weight_VW_Book_Mkt.iloc[i,:][Weight_VW_Book_Mkt.iloc[i,:]==1] = 0
    Weight_VW_Mkt_cap.iloc[i,:][Weight_VW_Mkt_cap.iloc[i,:]==1] = 0
    Weight_VW_Momemtum.iloc[i,:][Weight_VW_Momemtum.iloc[i,:]==1] = 0
    Weight_VW_Betas.iloc[i,:][Weight_VW_Betas.iloc[i,:]==1] = 0


#Return calculations
#Equal_Weighted
Weights_EW = [Weight_Betas,Weight_Book_Mkt,Weight_Mkt_cap,Weight_Momemtum,Weight_Vol]
Return_EW = np.zeros((len(Weight_Betas),5))
Return_EW= pd.DataFrame(Return_EW)
Return_EW.columns = ['Weight_Betas','Weight_Book_Mkt','Weight_Mkt_cap','Weight_Momentum','Weight_Vol']
Return_EW.index = betas.index

for k in range(0,5):
    for i in range(0,len(Weight_Betas)):
        Return_EW.iloc[i,k] = np.nansum(Weights_EW[k].iloc[i,:]*df_Returns.iloc[i,:]/100)

EW_Strategy_Returns = np.cumprod(1 + Return_EW / 100).iloc[-1,:]



#Value_Weighted
Weights_VW = [Weight_VW_Betas,Weight_VW_Book_Mkt,Weight_VW_Mkt_cap,Weight_VW_Momemtum,Weight_VW_Vol]
Return_VW = np.zeros((len(Weight_Betas),5))
Return_VW= pd.DataFrame(Return_VW)
Return_VW.columns = ['Weight_VW_Betas','Weight_VW_Book_Mkt','Weight_VW_Mkt_cap','Weight_VW_Momentum','Weight_VW_Vol']
Return_VW.index = betas.index


for k in range(0,5):
    for i in range(0,len(Weight_Betas)):
        Return_VW.iloc[i,k] = np.nansum(Weights_VW[k].iloc[i,:]*df_Returns.iloc[i,:]/100)

VW_Strategy_Returns = np.cumprod(1 + Return_VW / 100).iloc[-1,:]
        
Market_Return = np.cumprod(1+(df_Fama['Mkt-RF'][df_Fama.index >= '1950-06-01'] + df_Fama['RF'][df_Fama.index >= '1950-06-01'])/100)[-1]

#We can see that the Market_Return is much higher than our current strategies let's try to fix that

#Equal-weighted
Weight_Vol_LowStrat = np.zeros((len(betas),len(industries)))
Weight_Book_Mkt_LowStrat = np.zeros((len(betas),len(industries)))
Weight_Mkt_cap_LowStrat = np.zeros((len(betas),len(industries)))
Weight_Momemtum_LowStrat = np.zeros((len(betas),len(industries)))
Weight_Betas_LowStrat = np.zeros((len(betas),len(industries)))

Weight_Vol_LowStrat= pd.DataFrame(Weight_Vol_LowStrat)
Weight_Book_Mkt_LowStrat= pd.DataFrame(Weight_Book_Mkt_LowStrat)
Weight_Mkt_cap_LowStrat= pd.DataFrame(Weight_Mkt_cap_LowStrat)
Weight_Momemtum_LowStrat= pd.DataFrame(Weight_Momemtum_LowStrat)
Weight_Betas_LowStrat= pd.DataFrame(Weight_Betas_LowStrat)

Weight_Vol_LowStrat.columns = industries
Weight_Book_Mkt_LowStrat.columns = industries
Weight_Mkt_cap_LowStrat.columns = industries
Weight_Momemtum_LowStrat.columns = industries
Weight_Betas_LowStrat.columns = industries

Weight_Vol_LowStrat.index = betas.index
Weight_Book_Mkt_LowStrat.index = betas.index
Weight_Mkt_cap_LowStrat.index = betas.index
Weight_Momemtum_LowStrat.index = betas.index
Weight_Betas_LowStrat.index = betas.index

for i in range(0,len(betas)):
    max1 = min(Rank_Vol.iloc[i,:].nlargest(5))
    min1 = max(Rank_Vol.iloc[i,:].nsmallest(5))
    Weight_Vol_LowStrat.iloc[i,:] = np.where(Rank_Vol.iloc[i,:]>=max1,-0.2,(np.where(Rank_Vol.iloc[i,:]<=min1,0.2,0)))
    max2 = min(Rank_Book_Mkt.iloc[i,:].nlargest(5))
    min2 = max(Rank_Book_Mkt.iloc[i,:].nsmallest(5))
    Weight_Book_Mkt_LowStrat.iloc[i,:] = np.where(Rank_Book_Mkt.iloc[i,:]>=max2,-0.2,(np.where(Rank_Book_Mkt.iloc[i,:]<=min2,0.2,0)))
    max3 = min(Rank_Mkt_cap.iloc[i,:].nlargest(5))
    min3 = max(Rank_Mkt_cap.iloc[i,:].nsmallest(5))
    Weight_Mkt_cap_LowStrat.iloc[i,:] = np.where(Rank_Mkt_cap.iloc[i,:]>=max3,-0.2,(np.where(Rank_Mkt_cap.iloc[i,:]<=min3,0.2,0)))
    max4 = min(Rank_Momemtum.iloc[i,:].nlargest(5))
    min4 = max(Rank_Momemtum.iloc[i,:].nsmallest(5))
    Weight_Momemtum_LowStrat.iloc[i,:] = np.where(Rank_Momemtum.iloc[i,:]>=max4,-0.2,(np.where(Rank_Momemtum.iloc[i,:]<=min4,0.2,0)))
    max5 = min(Rank_Betas.iloc[i,:].nlargest(5))
    min5 = max(Rank_Betas.iloc[i,:].nsmallest(5))
    Weight_Betas_LowStrat.iloc[i,:] = np.where(Rank_Betas.iloc[i,:]>=max5,-0.2,(np.where(Rank_Betas.iloc[i,:]<=min5,0.2,0)))

Weights_EW_LowStrat = [Weight_Betas_LowStrat,Weight_Book_Mkt_LowStrat,Weight_Mkt_cap_LowStrat,Weight_Momemtum_LowStrat,Weight_Vol_LowStrat]
Return_EW_LowStrat = np.zeros((len(Weight_Betas),5))
Return_EW_LowStrat= pd.DataFrame(Return_EW_LowStrat)
Return_EW_LowStrat.columns = ['Weight_Betas','Weight_Book_Mkt','Weight_Mkt_cap','Weight_Momentum','Weight_Vol']
Return_EW_LowStrat.index = betas.index
  
for k in range(0,5):
    for i in range(0,len(Weight_Betas)):
        Return_EW_LowStrat.iloc[i,k] = np.nansum(Weights_EW_LowStrat[k].iloc[i,:]*df_Returns.iloc[i,:]/100)

EW_Low_Strategy_Returns = np.cumprod(1 + Return_EW_LowStrat / 100).iloc[-1,:]
plt.plot(np.cumprod(1 + Return_EW_LowStrat / 100))

#Market_Cap-weighted
Weight_VW_Vol_LowStrat = np.zeros((len(betas),len(industries)))
Weight_VW_Book_Mkt_LowStrat = np.zeros((len(betas),len(industries)))
Weight_VW_Mkt_cap_LowStrat = np.zeros((len(betas),len(industries)))
Weight_VW_Momemtum_LowStrat = np.zeros((len(betas),len(industries)))
Weight_VW_Betas_LowStrat = np.zeros((len(betas),len(industries)))

Weight_VW_Vol_LowStrat= pd.DataFrame(Weight_VW_Vol_LowStrat)
Weight_VW_Book_Mkt_LowStrat= pd.DataFrame(Weight_VW_Book_Mkt_LowStrat)
Weight_VW_Mkt_cap_LowStrat= pd.DataFrame(Weight_VW_Mkt_cap_LowStrat)
Weight_VW_Momemtum_LowStrat= pd.DataFrame(Weight_VW_Momemtum_LowStrat)
Weight_VW_Betas_LowStrat= pd.DataFrame(Weight_VW_Betas_LowStrat)

Weight_VW_Vol_LowStrat.columns = industries
Weight_VW_Book_Mkt_LowStrat.columns = industries
Weight_VW_Mkt_cap_LowStrat.columns = industries
Weight_VW_Momemtum_LowStrat.columns = industries
Weight_VW_Betas_LowStrat.columns = industries

Weight_VW_Vol_LowStrat.index = betas.index
Weight_VW_Book_Mkt_LowStrat.index = betas.index
Weight_VW_Mkt_cap_LowStrat.index = betas.index
Weight_VW_Momemtum_LowStrat.index = betas.index
Weight_VW_Betas_LowStrat.index = betas.index

for i in range(0,len(betas)):
    large1 = Rank_Vol.iloc[i,:].nlargest(5)
    Market_Cap_L1 = sum(Market_cap_Q4.loc[betas.index[i],(large1.index)])
    small1 = Rank_Vol.iloc[i,:].nsmallest(5)
    Market_Cap_S1 = sum(Market_cap_Q4.loc[betas.index[i],(small1.index)])
    max1 = min(large1)
    min1 = max(small1)
    
    large2 = Rank_Book_Mkt.iloc[i,:].nlargest(5)
    Market_Cap_L2 = sum(Market_cap_Q4.loc[betas.index[i],(large2.index)])
    small2 = Rank_Book_Mkt.iloc[i,:].nsmallest(5)
    Market_Cap_S2 = sum(Market_cap_Q4.loc[betas.index[i],(small2.index)])
    max2 = min(large2)
    min2 = max(small2)
    
    large3 = Rank_Mkt_cap.iloc[i,:].nlargest(5)
    Market_Cap_L3 = sum(Market_cap_Q4.loc[betas.index[i],(large3.index)])
    small3 = Rank_Mkt_cap.iloc[i,:].nsmallest(5)
    Market_Cap_S3 = sum(Market_cap_Q4.loc[betas.index[i],(small3.index)])
    max3 = min(large3)
    min3 = max(small3)
    
    large4 = Rank_Momemtum.iloc[i,:].nlargest(5)
    Market_Cap_L4 = sum(Market_cap_Q4.loc[betas.index[i],(large4.index)])
    small4 = Rank_Momemtum.iloc[i,:].nsmallest(5)
    Market_Cap_S4 = sum(Market_cap_Q4.loc[betas.index[i],(small4.index)])
    max4 = min(large4)
    min4 = max(small4)
    
    large5 = Rank_Betas.iloc[i,:].nlargest(5)
    Market_Cap_L5 = sum(Market_cap_Q4.loc[betas.index[i],(large5.index)])
    small5 = Rank_Betas.iloc[i,:].nsmallest(5)
    Market_Cap_S5 = sum(Market_cap_Q4.loc[betas.index[i],(small5.index)])
    max5 = min(large5)
    min5 = max(small5)
    for j in range(0,5):
        Weight_VW_Vol_LowStrat.loc[betas.index[i],large1.index[j]] = -(Market_cap_Q4.loc[betas.index[i],(large1.index[j])]/Market_Cap_L1)
        Weight_VW_Vol_LowStrat.loc[betas.index[i],small1.index[j]] = Market_cap_Q4.loc[betas.index[i],(small1.index[j])]/Market_Cap_S1
        
        Weight_VW_Book_Mkt_LowStrat.loc[betas.index[i],large2.index[j]] = -(Market_cap_Q4.loc[betas.index[i],(large2.index[j])]/Market_Cap_L2)
        Weight_VW_Book_Mkt_LowStrat.loc[betas.index[i],small2.index[j]] = Market_cap_Q4.loc[betas.index[i],(small2.index[j])]/Market_Cap_S2
        
        Weight_VW_Mkt_cap_LowStrat.loc[betas.index[i],large3.index[j]] = -(Market_cap_Q4.loc[betas.index[i],(large3.index[j])]/Market_Cap_L3)
        Weight_VW_Mkt_cap_LowStrat.loc[betas.index[i],small3.index[j]] = Market_cap_Q4.loc[betas.index[i],(small3.index[j])]/Market_Cap_S3
        
        Weight_VW_Momemtum_LowStrat.loc[betas.index[i],large4.index[j]] = -(Market_cap_Q4.loc[betas.index[i],(large4.index[j])]/Market_Cap_L4)
        Weight_VW_Momemtum_LowStrat.loc[betas.index[i],small4.index[j]] = Market_cap_Q4.loc[betas.index[i],(small4.index[j])]/Market_Cap_S4
        
        Weight_VW_Betas_LowStrat.loc[betas.index[i],large5.index[j]] = -(Market_cap_Q4.loc[betas.index[i],(large5.index[j])]/Market_Cap_L5)
        Weight_VW_Betas_LowStrat.loc[betas.index[i],small5.index[j]] = Market_cap_Q4.loc[betas.index[i],(small5.index[j])]/Market_Cap_S5
   
    Weight_VW_Vol_LowStrat.iloc[i,:][Weight_VW_Vol_LowStrat.iloc[i,:]==1] = 0
    Weight_VW_Book_Mkt_LowStrat.iloc[i,:][Weight_VW_Book_Mkt_LowStrat.iloc[i,:]==1] = 0
    Weight_VW_Mkt_cap_LowStrat.iloc[i,:][Weight_VW_Mkt_cap_LowStrat.iloc[i,:]==1] = 0
    Weight_VW_Momemtum_LowStrat.iloc[i,:][Weight_VW_Momemtum_LowStrat.iloc[i,:]==1] = 0
    Weight_VW_Betas_LowStrat.iloc[i,:][Weight_VW_Betas_LowStrat.iloc[i,:]==1] = 0
    
#Value_Weighted
Weights_VW_LowStrat = [Weight_VW_Betas_LowStrat,Weight_VW_Book_Mkt_LowStrat,Weight_VW_Mkt_cap_LowStrat,Weight_VW_Momemtum_LowStrat,Weight_VW_Vol_LowStrat]
Return_VW_LowStrat = np.zeros((len(Weight_Betas),5))
Return_VW_LowStrat= pd.DataFrame(Return_VW_LowStrat)
Return_VW_LowStrat.columns = ['Weight_VW_Betas','Weight_VW_Book_Mkt','Weight_VW_Mkt_cap','Weight_VW_Momentum','Weight_VW_Vol']
Return_VW_LowStrat.index = betas.index


for k in range(0,5):
    for i in range(0,len(Weight_Betas)):
        Return_VW_LowStrat.iloc[i,k] = np.nansum(Weights_VW_LowStrat[k].iloc[i,:]*df_Returns.iloc[i,:]/100)

VW_Low_Strategy_Returns = np.cumprod(1 + Return_VW_LowStrat / 100).iloc[-1,:]

#Equal_weighted in each portfolio test
Return_Test = pd.DataFrame(np.zeros((len(Weight_Betas),1)))
for i in range(0,len(Weight_Betas)):
    Return_Test.iloc[i,0] = np.nansum(1/48*df_Returns.iloc[i,:]/100)
    
np.cumprod(1 + Return_Test / 100).iloc[-1,:]
###

#Equal-weighted
Weight_Vol_Strat2 = pd.DataFrame(np.zeros((len(betas),len(industries))))
Weight_Book_Mkt_Strat2 = pd.DataFrame(np.zeros((len(betas),len(industries))))
Weight_Mkt_cap_Strat2 = pd.DataFrame(np.zeros((len(betas),len(industries))))
Weight_Momemtum_Strat2 = pd.DataFrame(np.zeros((len(betas),len(industries))))
Weight_Betas_Strat2 = pd.DataFrame(np.zeros((len(betas),len(industries))))

Weight_Vol_Strat2.columns = industries
Weight_Book_Mkt_Strat2.columns = industries
Weight_Mkt_cap_Strat2.columns = industries
Weight_Momemtum_Strat2.columns = industries
Weight_Betas_Strat2.columns = industries

Weight_Vol_Strat2.index = betas.index
Weight_Book_Mkt_Strat2.index = betas.index
Weight_Mkt_cap_Strat2.index = betas.index
Weight_Momemtum_Strat2.index = betas.index
Weight_Betas_Strat2.index = betas.index

for i in range(0,len(betas)):
    max1 = min(Rank_Vol.iloc[i,:].nlargest(1))
    min1 = max(Rank_Vol.iloc[i,:].nsmallest(1))
    Weight_Vol_Strat2.iloc[i,:] = np.where(Rank_Vol.iloc[i,:]>=max1,-1,(np.where(Rank_Vol.iloc[i,:]<=min1,1,0)))
    max2 = min(Rank_Book_Mkt.iloc[i,:].nlargest(1))
    min2 = max(Rank_Book_Mkt.iloc[i,:].nsmallest(1))
    Weight_Book_Mkt_Strat2.iloc[i,:] = np.where(Rank_Book_Mkt.iloc[i,:]>=max2,1,(np.where(Rank_Book_Mkt.iloc[i,:]<=min2,-1,0)))
    max3 = min(Rank_Mkt_cap.iloc[i,:].nlargest(1))
    min3 = max(Rank_Mkt_cap.iloc[i,:].nsmallest(1))
    Weight_Mkt_cap_Strat2.iloc[i,:] = np.where(Rank_Mkt_cap.iloc[i,:]>=max3,-1,(np.where(Rank_Mkt_cap.iloc[i,:]<=min3,1,0)))
    max4 = min(Rank_Momemtum.iloc[i,:].nlargest(1))
    min4 = max(Rank_Momemtum.iloc[i,:].nsmallest(1))
    Weight_Momemtum_Strat2.iloc[i,:] = np.where(Rank_Momemtum.iloc[i,:]>=max4,1,(np.where(Rank_Momemtum.iloc[i,:]<=min4,-1,0)))
    max5 = min(Rank_Betas.iloc[i,:].nlargest(1))
    min5 = max(Rank_Betas.iloc[i,:].nsmallest(1))
    Weight_Betas_Strat2.iloc[i,:] = np.where(Rank_Betas.iloc[i,:]>=max5,-1,(np.where(Rank_Betas.iloc[i,:]<=min5,1,0)))
    
Weights_EW_Strat2 = [Weight_Betas_Strat2,Weight_Book_Mkt_Strat2,Weight_Mkt_cap_Strat2,Weight_Momemtum_Strat2,Weight_Vol_Strat2]
Return_EW_Strat2 = pd.DataFrame(np.zeros((len(Weight_Betas),5)))
Return_EW_Strat2.columns = ['Weight_Betas','Weight_Book_Mkt','Weight_Mkt_cap','Weight_Momentum','Weight_Vol']
Return_EW_Strat2.index = betas.index
  
for k in range(0,5):
    for i in range(0,len(Weight_Betas)):
        Return_EW_Strat2.iloc[i,k] = np.nansum(Weights_EW_Strat2[k].iloc[i,:]*df_Returns.iloc[i,:]/100)

EW_Strat2_Returns = np.cumprod(1 + Return_EW_Strat2/ 100).iloc[-1,:]
plt.plot(np.cumprod(1 + Return_EW_Strat2 / 100))

#Regression on FAMA factors
#3-FACTORS

Periods = ['1963-06-01','1989-12-01','1999-12-01']
alpha_3_EW = pd.DataFrame(np.zeros((3,5)))
alpha_3_EW.columns = EW_Strat2_Returns.index
alpha_3_VW = pd.DataFrame(np.zeros((3,5)))
alpha_3_VW.columns = EW_Strat2_Returns.index
alpha_3_EW_LowStrat = pd.DataFrame(np.zeros((3,5)))
alpha_3_EW_LowStrat.columns = EW_Strat2_Returns.index
alpha_3_EW_Strat2 = pd.DataFrame(np.zeros((3,5)))
alpha_3_EW_Strat2.columns = EW_Strat2_Returns.index
alpha_3_VW_LowStrat = pd.DataFrame(np.zeros((3,5)))
alpha_3_VW_LowStrat.columns = EW_Strat2_Returns.index

Strategies_Returns = [Return_EW, Return_VW, Return_EW_LowStrat, Return_EW_Strat2, Return_VW_LowStrat]
alphas_3_Factors = [alpha_3_EW,alpha_3_VW,alpha_3_EW_LowStrat,alpha_3_EW_Strat2,alpha_3_VW_LowStrat]

for S in range(0, len(alphas_3_Factors)):
    for k in range(0,5):
        for i in range(0,len(Periods)):
            X = df_Fama_3.iloc[:,:3][(df_Fama_3.index < '2020-01-01')&(df_Fama_3.index > Periods[i])]/100
            X = sm.add_constant(X)
            Y = Strategies_Returns[S].iloc[:,k][(Strategies_Returns[S].index < '2020-01-01')&(Strategies_Returns[S].index > Periods[i])] - df_Fama_3.iloc[:,3][(df_Fama_3.index < '2020-01-01')&(df_Fama_3.index > Periods[i])]/100
            alpha = np.linalg.solve(np.dot(np.array(X).T, np.array(X)), np.dot(np.array(X).T, np.array(Y)))[0]
            alphas_3_Factors[S].iloc[i,k] = alpha

#4 & 5 Factors
        
Periods = ['1963-06-01','1989-12-01','1999-12-01']
alpha_4_EW = pd.DataFrame(np.zeros((3,5)))
alpha_4_EW.columns = EW_Strat2_Returns.index
alpha_4_VW = pd.DataFrame(np.zeros((3,5)))
alpha_4_VW.columns = EW_Strat2_Returns.index
alpha_4_EW_LowStrat = pd.DataFrame(np.zeros((3,5)))
alpha_4_EW_LowStrat.columns = EW_Strat2_Returns.index
alpha_4_EW_Strat2 = pd.DataFrame(np.zeros((3,5)))
alpha_4_EW_Strat2.columns = EW_Strat2_Returns.index
alpha_4_VW_LowStrat = pd.DataFrame(np.zeros((3,5)))
alpha_4_VW_LowStrat.columns = EW_Strat2_Returns.index

alphas_4_Factors = [alpha_4_EW,alpha_4_VW,alpha_4_EW_LowStrat,alpha_4_EW_Strat2,alpha_4_VW_LowStrat]

for S in range(0,len(alphas_4_Factors)):
    for k in range(0,5):
        for i in range(0,len(Periods)):
            X = df_Fama_5.iloc[:,:4][(df_Fama_5.index < '2020-01-01')&(df_Fama_5.index > Periods[i])]/100
            X = sm.add_constant(X)
            Y = Strategies_Returns[S].iloc[:,k][(Strategies_Returns[S].index < '2020-01-01')&(Strategies_Returns[S].index > Periods[i])] - df_Fama_5.iloc[:,4][(df_Fama_5.index < '2020-01-01')&(df_Fama_5.index > Periods[i])]/100
            alpha = np.linalg.solve(np.dot(np.array(X).T, np.array(X)), np.dot(np.array(X).T, np.array(Y)))[0]
            alphas_4_Factors[S].iloc[i,k] = alpha
        
alpha_5_EW = pd.DataFrame(np.zeros((3,5)))
alpha_5_EW.columns = EW_Strat2_Returns.index
alpha_5_VW = pd.DataFrame(np.zeros((3,5)))
alpha_5_VW.columns = EW_Strat2_Returns.index
alpha_5_EW_LowStrat = pd.DataFrame(np.zeros((3,5)))
alpha_5_EW_LowStrat.columns = EW_Strat2_Returns.index
alpha_5_EW_Strat2 = pd.DataFrame(np.zeros((3,5)))
alpha_5_EW_Strat2.columns = EW_Strat2_Returns.index
alpha_5_VW_LowStrat = pd.DataFrame(np.zeros((3,5)))
alpha_5_VW_LowStrat.columns = EW_Strat2_Returns.index

alphas_5_Factors = [alpha_5_EW,alpha_5_VW,alpha_5_EW_LowStrat,alpha_5_EW_Strat2,alpha_5_VW_LowStrat]

for S in range(0,len(alphas_5_Factors)):
    for k in range(0,5):
        for i in range(0,len(Periods)):
            X = df_Fama_5.iloc[:,:5][(df_Fama_5.index < '2020-01-01')&(df_Fama_5.index > Periods[i])]/100
            X = sm.add_constant(X)
            Y = Strategies_Returns[S].iloc[:,k][(Strategies_Returns[S].index < '2020-01-01')&(Strategies_Returns[S].index > Periods[i])] - df_Fama_5.iloc[:,4][(df_Fama_5.index < '2020-01-01')&(df_Fama_5.index > Periods[i])]/100
            alpha = np.linalg.solve(np.dot(np.array(X).T, np.array(X)), np.dot(np.array(X).T, np.array(Y)))[0]
            alphas_5_Factors[S].iloc[i,k] = alpha
        
#Sharpe

Periods = ['1963-06-01','1989-12-01','1999-12-01']
Sharpe_EW = pd.DataFrame(np.zeros((3,5)))
Sharpe_EW.columns = EW_Strat2_Returns.index
Sharpe_VW = pd.DataFrame(np.zeros((3,5)))
Sharpe_VW.columns = EW_Strat2_Returns.index
Sharpe_EW_LowStrat = pd.DataFrame(np.zeros((3,5)))
Sharpe_EW_LowStrat.columns = EW_Strat2_Returns.index
Sharpe_EW_Strat2 = pd.DataFrame(np.zeros((3,5)))
Sharpe_EW_Strat2.columns = EW_Strat2_Returns.index
Sharpe_VW_LowStrat = pd.DataFrame(np.zeros((3,5)))
Sharpe_VW_LowStrat.columns = EW_Strat2_Returns.index

Sharpe = [Sharpe_EW, Sharpe_VW, Sharpe_EW_LowStrat, Sharpe_EW_Strat2, Sharpe_VW_LowStrat]

for S in range(0, len(Strategies_Returns)):
    for k in range(0,5):
        for i in range(0,len(Periods)):
            Sharpe_i = (np.mean(Strategies_Returns[S].iloc[:,k][(Strategies_Returns[S].index < '2020-01-01')&(Strategies_Returns[S].index > Periods[i])] - df_Fama_3.iloc[:,3][(df_Fama_3.index < '2020-01-01')&(df_Fama_3.index > Periods[i])]/100))/np.std(Strategies_Returns[S].iloc[:,k][(Strategies_Returns[S].index < '2020-01-01')&(Strategies_Returns[S].index > Periods[i])])
            Sharpe[S].iloc[i,k] = Sharpe_i
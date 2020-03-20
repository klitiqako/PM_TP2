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

## ----------- Data cleansing -------------------------------------------------
df_Returns.index = pd.to_datetime(df_Returns.index, format= '%Y%m')
df_Firm_Size.index = pd.to_datetime(df_Firm_Size.index, format= '%Y%m')
df_NB_Firms.index = pd.to_datetime(df_NB_Firms.index, format= '%Y%m')
df_MktBook.index = pd.to_datetime(df_MktBook.index, format= '%Y')
df_Daily_Returns.index = pd.to_datetime(df_Daily_Returns.index, format= '%Y%m%d')
df_Fama.index = pd.to_datetime(df_Fama.index, format= '%Y%m%d')


df_Returns.columns = df_Returns.columns.str.replace(' ', '')
df_Firm_Size.columns = df_Firm_Size.columns.str.replace(' ', '')
df_NB_Firms.columns = df_NB_Firms.columns.str.replace(' ', '')
df_MktBook.columns = df_MktBook.columns.str.replace(' ', '')
df_Daily_Returns.columns = df_Daily_Returns.columns.str.replace(' ', '')
df_Fama.columns = df_Fama.columns.str.replace(' ', '')


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
Idio_vol = np.zeros((len(df_Returns),len(industries)))
Idio_vol= pd.DataFrame(Idio_vol)
Idio_vol.index = df_Returns.index
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
Volatility_Q4 = Idio_vol[11:]
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
        Weight_VW_Vol.loc[betas.index[i],large1.index[j]] = round(Market_cap_Q4.loc[betas.index[i],(large1.index[j])]/Market_Cap_L1,3)
        Weight_VW_Vol.loc[betas.index[i],small1.index[j]] = -round(Market_cap_Q4.loc[betas.index[i],(small1.index[j])]/Market_Cap_S1,3)
        
        Weight_VW_Book_Mkt.loc[betas.index[i],large2.index[j]] = round(Market_cap_Q4.loc[betas.index[i],(large2.index[j])]/Market_Cap_L2,3)
        Weight_VW_Book_Mkt.loc[betas.index[i],small2.index[j]] = -round(Market_cap_Q4.loc[betas.index[i],(small2.index[j])]/Market_Cap_S2,3)
        
        Weight_VW_Mkt_cap.loc[betas.index[i],large3.index[j]] = round(Market_cap_Q4.loc[betas.index[i],(large3.index[j])]/Market_Cap_L3,3)
        Weight_VW_Mkt_cap.loc[betas.index[i],small3.index[j]] = -round(Market_cap_Q4.loc[betas.index[i],(small3.index[j])]/Market_Cap_S3,3)
        
        Weight_VW_Momemtum.loc[betas.index[i],large4.index[j]] = round(Market_cap_Q4.loc[betas.index[i],(large4.index[j])]/Market_Cap_L4,3)
        Weight_VW_Momemtum.loc[betas.index[i],small4.index[j]] = -round(Market_cap_Q4.loc[betas.index[i],(small4.index[j])]/Market_Cap_S4,3)
        
        Weight_VW_Betas.loc[betas.index[i],large5.index[j]] = round(Market_cap_Q4.loc[betas.index[i],(large5.index[j])]/Market_Cap_L5,3)
        Weight_VW_Betas.loc[betas.index[i],small5.index[j]] = -round(Market_cap_Q4.loc[betas.index[i],(small5.index[j])]/Market_Cap_S5,3)
   
    Weight_VW_Vol.iloc[i,:][Weight_VW_Vol.iloc[i,:]==1] = 0
    Weight_VW_Book_Mkt.iloc[i,:][Weight_VW_Book_Mkt.iloc[i,:]==1] = 0
    Weight_VW_Mkt_cap.iloc[i,:][Weight_VW_Mkt_cap.iloc[i,:]==1] = 0
    Weight_VW_Momemtum.iloc[i,:][Weight_VW_Momemtum.iloc[i,:]==1] = 0
    Weight_VW_Betas.iloc[i,:][Weight_VW_Betas.iloc[i,:]==1] = 0

#Return calculations
np.nansum(Weight_Betas.iloc[0,:]*df_Daily_Returns.iloc[0,:]/100)+1
#Equal_Weighted
Weights_EW = [Weight_Betas,Weight_Book_Mkt,Weight_Mkt_cap,Weight_Momemtum,Weight_Vol]
Return_EW = np.zeros((len(Weight_Betas),5))
Return_EW= pd.DataFrame(Return_EW)
Return_EW.columns = ['Weight_Betas','Weight_Book_Mkt','Weight_Mkt_cap','Weight_Momentum','Weight_Vol']
Return_EW.index = betas.index

Return_Month = np.zeros((31,1))

for k in range(0,5):
    for i in range(0,len(Weight_Betas)):
        d = Weights_EW[k].index[i]+dateutil.relativedelta.relativedelta(months=1)
        d_Last12 = d - dateutil.relativedelta.relativedelta(months=1)
        X = df_Daily_Returns[(df_Daily_Returns.index < d)&(df_Daily_Returns.index >= d_Last12)]
        for j in range(0,len(X)):
            Return_Month[j] = np.nansum(Weights_EW[k].iloc[i,:]*X.iloc[j,:]/100)
        Return_EW.iloc[i,k] = sum(Return_Month)

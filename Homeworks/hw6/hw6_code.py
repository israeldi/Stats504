#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 00:40:04 2019

@author: israeldiego
"""
# import matplotlib
# matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# from scipy.special import gammaln
# from scipy.stats import kendalltau

data_dir = "./data/"
# The date to process
dt = "2012-04-01"

# Create a set of graphs in one pdf file.
# pdf = PdfPages("internet_%s.pdf" % dt)

# Load the traffic statistics (one record per minute
# within a day).
df = pd.read_csv("%straffic_stats_%s.csv" % (data_dir, dt))

# Rename the columns
cname = {"Traffic": "Total traffic", "UDP": "UDP", "TCP": "TCP",
         "Sources": "Unique sources"}

# Perform log transformation
df.iloc[:,2:6] = np.log(df.iloc[:,2:6])


def fracDiff_FFD(series, d, threshold= 1e-5):
    '''
    Constant width window (new solution)
    Note 1: thres determines the cut-off weight for the window
    Note 2: d can be any positive fractional, not necessarily bounded [0,1].
    '''
    #1) Compute weights for the longest series
    weights = getWeights_FFD(d, len(series), threshold)
    width = len(weights) - 1
    
    #2) Apply weights to values
    df={}
    
    # for each series to be differenced, apply weights to appropriate prices 
    # and save
    for name in series.columns:
        curr_series = series[[name]].fillna(method='ffill').dropna()
        # df_temp = pd.DataFrame(columns=[name])
        df_temp = pd.DataFrame()
        
        # loop through all values that fall into range to be fractionally 
        # differenced
        for iloc1 in range(width, curr_series.shape[0]):
            
            # set values for first and last time-series point to be used in 
            # current pass of fractional differences
            loc0 = curr_series.index[iloc1 - width]
            loc1= curr_series.index[iloc1]
            
            # make sure current value is valid
            if not np.isfinite(curr_series.loc[loc1,name]):
                continue # exclude NAs
            
            # dot product of weights with values from first and last indices
            frac_val = np.dot(weights.T, curr_series.loc[loc0:loc1])[0,0]
            # data = pd.DataFrame([frac_val], columns=[name], index=[loc1])
            data = pd.DataFrame([frac_val], index=[loc1])
            df_temp = df_temp.append(data)
        
        df[name]=df_temp.copy(deep=True)
    df=pd.concat(df,axis=1)
    df.columns = series.columns
    return df

def getWeights_FFD(d, length, threshold= 1e-5):
    '''
    Computes the weights for our fractionally differenced features up to a given
    threshold requirement for fixed-window fractional differencing.
    Args:
        d: A float representing the differencing factor
        length: AAn int length of series to differenced
        theshold: A float representing the minimum theshold to include weights
    Returns: 
        A numpy array containing the weights to be applied to our time series
    '''
    
    # set first weight to be a 1 aand k to be 1
    w, k = [1.], 1
    w_curr = 1

    # while we still have more weights to process, do the following
    while(k < length):
        w_curr = -w[-1] / k*(d-k+1)

        # if the current weight is below threshold, exit
        if (abs(w_curr) <= threshold):
            break
        w.append(w_curr)
        k += 1
    # Make sure to convert it to a numpy array and reshape from a single row to 
    # a single column so they can be applied to time-series values easier
    w = np.array(w[::-1]).reshape(-1,1)
    return w

def plotMinFFD(series, threshold = 0.01):
    from statsmodels.tsa.stattools import adfuller
    # path,instName='./','ES1_Index_Method12'
    out = pd.DataFrame(columns=['adfStat','pVal','lags','nObs','95% conf','corr'])
    df0 = series
    
    for d in np.linspace(0, 1, 11):
        df1 = np.log(df0) # downcast to daily obs
        df2 = fracDiff_FFD(df1, d, threshold)
        data1 = df1.iloc[:,0].loc[df2.index]
        data2 = df2.iloc[:,0]
        
        corr = np.corrcoef(data1, data2)[0,1]
        df2 = adfuller(data2, maxlag=1,regression='c', autolag=None)
        out.loc[d] = list(df2[:4])+[df2[4]['5%']]+[corr] # with critical value
    
    # out.to_csv(path+instName+'_testMinFFD.csv')
    out[['adfStat','corr']].plot(secondary_y='adfStat')
    plt.axhline(out['95% conf'].mean(),linewidth=1,color='r',linestyle='dotted')
    plt.xlabel("Order of difference: d")
    # plt.ylabel("ADF-stat")
    plt.title("Plot of ADF-stat and Correlation for %s" % (series.columns[0]))
    # ax = plt.gca() 
    # ax.yaxis.tick_right("ADF Stat")
    # plt.savefig(path+instName+'_testMinFFD.png')
    return out

'''
def plotWeights(dRange,nPlots,size):
    w = pd.DataFrame()
    for d in np.linspace(dRange[0],dRange[1],nPlots):
        w_ = getWeights_FFD(d, size=size)
        w_ = pd.DataFrame(w_,index=range(w_.shape[0])[::-1],columns=[d])
        w = w.join(w_,how='outer')
    ax = w.plot()
    ax.legend(loc='upper left');
    plt.show()
    return
'''

# Perform fractional differencing 
name = "UDP"
thresh = 0.001

out = plotMinFFD(pd.DataFrame(df[name]), thresh)

# Get optimal order of differentiation
crit_val = out['95% conf'].mean()
d_star = out[out['adfStat'] < crit_val].index[0]
corr_star = out.loc[d_star, "corr"]

# Plot optimal difference vs first difference
frac_df1 = fracDiff_FFD(pd.DataFrame(df[['UDP', 'TCP']]), d_star, thresh)
frac_df2 = fracDiff_FFD(pd.DataFrame(df[['UDP', 'TCP']]), 1, thresh)

# get starting index since we removed first few observations to perform
# fractional differencing
ind_start = max(frac_df1.index[0], frac_df2.index[0])

# plot no differencing
plt.figure(0)
plt.plot(df.loc[ind_start:, name] - df.loc[ind_start:, name].mean())

# plot optimal difference
plt.plot(frac_df1.loc[ind_start:, name] - frac_df1.loc[ind_start:, name].mean())

# plot first-difference
plt.plot(frac_df2.loc[ind_start:, name])
plt.legend(["d=0","d*=%.2f" % (d_star), "d=1"])
plt.title("Fractional Differencing on %s time-series" % (name))
plt.show()


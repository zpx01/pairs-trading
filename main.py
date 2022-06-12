import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import streamlit as st
import statsmodels
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
from pandas_datareader import data as pdr
import yfinance as yf
import datetime
yf.pdr_override()


st.title("Pairs Trading Statistical Arbitrage")
st.write(" ")
st.write("**In this app, you can select a pair of stocks to conduct pairs trading on within a certain time period. The overall trading strategy is detailed below.**")
start = datetime.datetime(2015, 1, 1)
end = datetime.datetime(2022, 1, 1)
tickers = ['AAPL', 'GOOG', 'MSFT', "FB", "CRM", "TWTR", "MCD", "KO", "HD", "TGT", "SBUX", "NKE", "TSLA", "LOW"]
df = pd.read_csv('data.csv')
st.write(df.head())

pairs = [(i, j) for idx, i in enumerate(tickers) for j in tickers[idx + 1:]]

coint_check = []
for pair in pairs:
  s1, s2 = df[pair[0]], df[pair[1]]
  score, pval, _ = coint(s1, s2)
  coint_check.append((pair[0], pair[1], pval))

coint_check.sort(key=lambda x: x[2])
st.write("Top 4 cointegrated pairs: ", coint_check[0:4])

def zscore(data):
  return (data - data.mean()) / np.std(data)

def coint_pairs(data):
    dim = data.shape[1]
    scores, pvalues = np.zeros((dim, dim)), np.ones((dim, dim))
    keys = data.keys()
    pairs = []
    for i in range(dim):
        for j in range(i+1, dim):
            stock_1 = data[keys[i]]
            stock_2 = data[keys[j]]
            test = coint(stock_1, stock_2)
            score, pvalue = test[0], test[1]
            scores[i, j] = score
            pvalues[i, j] = pvalue
            if pvalue < 0.05:
                pairs.append((keys[i], keys[j]))
    return scores, pvalues, pairs

# Trade using a simple strategy
def trade(stock1, stock2, window1, window2):
    if (window1 == 0) or (window2 == 0):
        return 0
    ratios = stock1/stock2
    maWindow1 = ratios.rolling(window=window1, center=False).mean()
    maWindow2 = ratios.rolling(window=window2, center=False).mean()                           
    std = ratios.rolling(window=window2, center=False).std()
    zscore = (maWindow1 - maWindow2)/std
    money = 0
    countStock1 = 0
    countStock2 = 0
    for i in range(len(ratios)):
        if zscore[i] < -1:
            money += stock1[i] - stock2[i] * ratios[i]
            countStock1 -= 1
            countStock2 += ratios[i]
        elif zscore[i] > 1:
            money -= stock1[i] - stock2[i] * ratios[i]
            countStock1 += 1
            countStock2 -= ratios[i]
        elif abs(zscore[i]) < 0.75:
            money += stock1[i] * countStock1 + stock2[i] * countStock2
            countStock1 = 0
            countStock2 = 0            
            
    return money

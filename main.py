import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import math
import streamlit as st
import statsmodels
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
from pandas_datareader import data as pdr
import yfinance as yf
import datetime
from trader import Trader
yf.pdr_override()


st.title("Pairs Trading Statistical Arbitrage")
st.write(" ")
st.write("This website provides a framework for implementing a simple pairs trading strategy for statistical arbitrage between the consumer discretionary and technology sectors. We chose these two sectors as they both had a high correlation with the S&P 500. To form pairs of stocks to trade upon, we chose several stocks of major companies within both sectors and performed cointegration tests to find pairs with high cointegration between their time series.")
start = datetime.datetime(2015, 1, 1)
end = datetime.datetime(2022, 1, 1)
tickers = ['AAPL', 'GOOG', 'MSFT', "FB", "CRM", "TWTR", "MCD", "KO", "HD", "TGT", "SBUX", "NKE", "TSLA", "LOW"]
df = pd.read_csv('data.csv')
st.write(" ")
st.markdown("### Stock Price Dataset (Pulled from Yahoo Finance API):")
st.write(df)

pairs = [(i, j) for idx, i in enumerate(tickers) for j in tickers[idx + 1:]]
coint_check = []
for pair in pairs:
  s1, s2 = df[pair[0]], df[pair[1]]
  score, pval, _ = coint(s1, s2)
  coint_check.append((pair[0], pair[1], pval))

st.markdown("### Correlation vs. Cointegration")
st.write(" ")
st.write("""In a pairs trading strategy, we want to see if there is some sort of relationship between the two stocks were are analyzing. One way of doing this is using the correlation between the two time series of the stocks. However, correlation is highly unstable as in the short-term, it may be possible that correlation is high, yet the two series diverge in the long run.\n

A perhaps more robust measure of how closely related two stocks are is cointegration, which checks for the existence of a long-run relationship of two time series. For this reason, we chose to use cointegration over correlation to measure the relationship between the two time series.\n

Our cointegration test provides us with a p-value which helps us determine how cointegrated two stocks are. We have kept our significance level α = 0.05. In our tests, a lower p-values indicates higher cointegration between two stocks.""")

coint_check.sort(key=lambda x: x[2])
st.markdown("### Top 4 cointegrated pairs: ")
for i in range(4):
    st.code(f"Pair {i+1}: {coint_check[i][0]} & {coint_check[i][1]}, p-value={round(coint_check[i][2], 4)}")

trader = Trader()
pairs = [("HD", "MSFT"), ("AAPL", "LOW"), ("FB", "HD"), ("MSFT", "NKE")]
ratios = []
for pair in pairs:
  ratios.append(trader.zscore(df[pair[0]] / df[pair[1]]))

days = mdates.drange(start,end, datetime.timedelta(days=10))

st.markdown("### Time Series Analysis")
st.write(" ")
st.write("We will use the price ratio between the two pairs of stocks as our measure of spread. After we normalize the price ratios, we can then try to find trends in the z-scores of the price ratios over time.\nAdditionally, we would like to find features that are important to the movement of the price ratio (we only care whether the ratio goes up or down). We will use the 5 and 60 day moving averages of the price ratios as our features.")

for i in range(len(ratios)):
  st.markdown("### Pair {}: {} & {}".format(i+1, pairs[i][0], pairs[i][1]))
  fig, ax = plt.subplots(figsize=(12,7))
  fig.gca().xaxis.set_major_formatter(mdates.DateFormatter(r'%Y'))
  fig.gca().xaxis.set_major_locator(mdates.DayLocator(interval=365))  
  ax.plot(days, ratios[i][0:1536:6])
  ax.axhline(-1, color="green")
  ax.axhline(1, color="black")
  ax.axhline(ratios[i].mean(), color="red")
  ax.legend(["Price Ratio"])
  st.pyplot(fig)

# Train Test Split
for i in range(len(pairs)):
  pair_s1 = df[pairs[i][0]]
  pair_s2 = df[pairs[i][1]]
  ratios = pair_s1 / pair_s2
  x = math.floor(len(ratios) * 0.8)
  train = ratios[:x]
  test = ratios[x:]

  # Using 5 and 60 day moving averages as features
  mavg5 = train.rolling(window=5, center=False).mean()
  mavg60 = train.rolling(window=60, center=False).mean()
  std60 = train.rolling(window=60, center=False).std()
  zscore_60_5 = (mavg5 - mavg60)/std60
  fig, ax = plt.subplots(figsize=(12,6))
  fig.gca().xaxis.set_major_formatter(mdates.DateFormatter(r'%Y'))
  fig.gca().xaxis.set_major_locator(mdates.DayLocator(interval=365))
  ax.plot(days, train.values[0:1280:5])
  ax.plot(days, mavg5.values[0:1280:5])
  ax.plot(days, mavg60.values[0:1280:5])
  st.markdown("### Pair {}: {} & {}".format(i+1, pairs[i][0], pairs[i][1]))
  ax.legend(['Ratio', '5d Ratio MA', '60d Ratio MA'])
  st.pyplot(fig)
  fig, ax = plt.subplots(figsize=(12,6))
  fig.gca().xaxis.set_major_formatter(mdates.DateFormatter(r'%Y'))
  fig.gca().xaxis.set_major_locator(mdates.DayLocator(interval=365))
  ax.plot(days, zscore_60_5.values[0:1280:5])
  ax.axhline(0, color='black')
  ax.axhline(1.0, color='red', linestyle='--')
  ax.axhline(-1.0, color='green', linestyle='--')
  ax.legend(['Rolling Ratio z-Score', 'Mean', '+1', '-1'])
  st.pyplot(fig)


st.write(" ")
st.markdown("### Trading Strategy")
st.write(" ")
st.write("We can notice in the Rolling Ratio Z-Score graphs that the ratio for all 4 four pairs tends to return back to the mean as it crosses +1 or -1 SD. We can then create a strategy in which we buy when the ratio crosses -1 SD and we sell when we cross +1 SD. This simple strategy should be able to provide us with some returns in theory as long as our past assumptions are correct on average.")

for i in range(len(pairs)):
  fig, ax = plt.subplots(figsize=(12,6))
  fig.gca().xaxis.set_major_formatter(mdates.DateFormatter(r'%Y'))
  fig.gca().xaxis.set_major_locator(mdates.DayLocator(interval=365))
  data = df[pairs[i][0]] / df[pairs[i][1]]
  train = data[:math.floor(len(data)*0.8)]
  buy = train.copy()
  sell = train.copy()
  mavg5 = train.rolling(window=5, center=False).mean()
  mavg60 = train.rolling(window=60, center=False).mean()
  std60 = train.rolling(window=60, center=False).std()
  zscore_60_5 = (mavg5 - mavg60)/std60
  buy[zscore_60_5>-1] = 0
  sell[zscore_60_5<1] = 0
  stock1 = df[pairs[i][0]].iloc[:math.floor(len(data)*0.8)]
  stock2 = df[pairs[i][1]].iloc[:math.floor(len(data)*0.8)]

  ax.plot(days, stock1.values[0:1280:5], color='blue')
  ax.plot(days, stock2.values[0:1280:5], color='cyan')
  buyRatio = 0*stock1.copy()
  sellRatio = 0*stock2.copy()

  # Buy the ratio -> Buy S1, Sell S2
  buyRatio[buy!=0] = stock1[buy!=0]
  sellRatio[buy!=0] = stock2[buy!=0]

  # Sell the ratio -> Sell S1, Buy S2
  buyRatio[sell!=0] = stock2[sell!=0]
  sellRatio[sell!=0] = stock1[sell!=0]
  ax.plot(days, buyRatio.values[0:1280:5], color='g', linestyle='None', marker='.')
  ax.plot(days, sellRatio.values[0:1280:5], color='r', linestyle='None', marker='.')
  x1, x2, y1, y2 = plt.axis()
  st.markdown("### Pair {}: {} & {}".format(i+1, pairs[i][0], pairs[i][1]))
  ax.set(xlim=(x1, x2), ylim=(min(stock1.min(), stock2.min()), max(stock1.max(), stock2.max())))
  ax.legend([pairs[i][0], pairs[i][1], 'Buy Signal', 'Sell Signal'])
  st.pyplot(fig)

st.write(" ")
st.write("### Trading Results")
st.write("Using our trading strategy, we were able to generate the following returns:")
for i in range(len(pairs)):
  returns = trader.trade(df[pairs[i][0]].iloc[math.floor(len(df[pairs[i][0]]) * 0.8):], df[pairs[i][1]].iloc[math.floor(len(df[pairs[i][0]]) * 0.8):], 60, 5)
  st.code("Pair {} ({} & {}) Returns: {}".format(i+1, pairs[i][0], pairs[i][1], round(returns, 3)))

st.write(" ")
st.markdown("### Conclusion")
st.write(" ")
st.write("Our trading strategy gave us a decent amount of profit for each of the four pairs, with the most profit coming from the MSFT & NKE pair and the least coming from AAPL & LOW. We can develop a more comprehensive strategy by analyzing different features of the time series data and using those to influence our trading decisions. Having more features can help us make better trading decisions which can help further optimize the returns we generate with this strategy. Additionally, we can quantify the risk of our investments through the sharpe ratio. We can also utilize data from more years to backtest our strategy further.")
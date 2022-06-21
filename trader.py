import numpy as np
import pandas as pd
class Trader:
    def __init__(self):
        pass

    def zscore(self, data):
        return (data - data.mean()) / np.std(data)

    def trade(self, stock1, stock2, window1, window2):
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
            if zscore.values[i] < -1:
                money += stock1.values[i] - stock2.values[i] * ratios.values[i]
                countStock1 -= 1
                countStock2 += ratios.values[i]
            elif zscore.values[i] > 1:
                money -= stock1.values[i] - stock2.values[i] * ratios.values[i]
                countStock1 += 1
                countStock2 -= ratios.values[i]
            elif abs(zscore.values[i]) < 0.75:
                money += stock1.values[i] * countStock1 + stock2.values[i] * countStock2
                countStock1 = 0
                countStock2 = 0            
        return money

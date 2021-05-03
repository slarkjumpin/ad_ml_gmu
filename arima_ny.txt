import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
from pandas import read_csv
from statsmodels.tsa.stattools import adfuller
import math
from sklearn.metrics import mean_squared_error

# import csv data
ca_series = read_csv('ny_ili.csv', engine='python')
df = ca_series

# stationary test
test_result=adfuller(df['ili_rate'])

# first order difference
df['First Difference'] = df['ili_rate'] - df['ili_rate'].shift(1)

# plot pacf 
fig = plt.figure(figsize=(12,8))
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df['First Difference'].dropna(),lags=40,ax=ax2)

# build model
model=ARIMA(df['ili_rate'],order=(1,1,1))
model_fit=model.fit()
model_fit.summary()

# make prediction
model=sm.tsa.statespace.SARIMAX(df['ili_rate'],order=(1, 1, 1),seasonal_order=(1,1,1,52))
results=model.fit()
df['forecast']=results.predict(start=327,end=482,dynamic=True)

# plot the actual data and prediction data
plt.figure(figsize=(12,4))
plt.title("ARIMA Prediction")
plt.ylabel("ILI Rate")
plt.grid(True)
plt.plot(df['ili_rate'])
# plt.plot(trainPredictPlot)
plt.plot(df['forecast'])
plt.show()

# claculate the rmse
true = df['ili_rate'][327:482]
predict = df['forecast'][327:482]
math.sqrt(mean_squared_error(true, predict))


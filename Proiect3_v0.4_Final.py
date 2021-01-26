import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from fbprophet import Prophet
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA

test = pd.read_csv('EURRON.csv')

print('*****TEST*****')
print(test.head())
print('*****Cativa parametrii ai seriei*****')
print(test.describe())

X1 = test["Value"].values
X2 = test["Date"].values

#Augumented Dickey-Fuller Test
result = adfuller(X1)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

def adf_result(res):
    if res[0] < res[4]["5%"]:
        print ("Respingem ipoteza de nul - Seria este stationara")
    else:
        print ("Nu s-a putut respinge ipoteza de nul - Seria nu este stationara")

adf_result(result)
window=100  
xs=pd.Series(X1,X2)
date=pd.date_range(X2[0],X2[len(X2)-1])
dfr = pd.DataFrame({'date' : date,
                   'value' : X1})
x = np.arange(date.size)
fit = np.polyfit(x, dfr['value'], 1)
fit_fn = np.poly1d(fit)
sma = xs.rolling(window).mean()
mae = mean_absolute_error(xs[window:], sma[window:])
deviation = np.std(xs[window:] - sma[window:])
lower_bound = sma - (mae + 1.5 * deviation)
upper_bound = sma + (mae + 1.5 * deviation)
plt.figure()
plt.title("Evolutia cursului pe toata perioada")
plt.plot(xs,label='Evolutie curs')
plt.plot(sma,'g',label='SMA 100' )
plt.plot(upper_bound, 'r--', label='Limita superioara/inferioara anomalie')
plt.plot(lower_bound, 'r--')
plt.legend(loc='best')

plt.figure()
plt.title("Evolutie curs si regresie liniara")
plt.plot(dfr['date'], fit_fn(x), 'k-')
plt.plot(dfr['date'], dfr['value'], 'go', ms=2)

test_stat=xs.diff().dropna()
result2=adfuller(test_stat)
print('ADF Statistic: %f' % result2[0])
print('p-value: %f' % result2[1])
print('Critical Values:')
for key, value in result2[4].items():
    print('\t%s: %.3f' % (key, value))

adf_result(result2)

plt.figure()
plt.title("Seria dorita stationara")
plt.plot(test_stat)

# data = X1
# model = ARIMA(data, order=(2, 0, 1))
# model_fit = model.fit()
# yhat = model_fit.predict(len(data)-window,len(data))
# plt.figure()
# plt.plot(yhat)

df=test
#Setul de antrenat
df.columns = ['y', 'ds']
prediction_size = window
train_df = df[:-prediction_size]


# Antrenarea unui model
m = Prophet()
m.fit(train_df)

# Prezicere
future = m.make_future_dataframe(periods=prediction_size)
forecast = m.predict(future)
forecast.head()

m.plot(forecast)
plt.title('Predictie')


# Eroare predictie
def make_comparison_dataframe(historical, forecast):
    return forecast.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper']].join(historical.set_index('ds'))

cmp_df = make_comparison_dataframe(df, forecast)
cmp_df.head()

def calculate_forecast_errors(df, prediction_size):
    
    df = df.copy()
    
    df['e'] = df['y'] - df['yhat']
    df['s'] = (df['y'] - df['yhat'])**2
    df['p'] = 100 * df['e'] / df['y']
    
    predicted_part = df[-prediction_size:]
    
    error_mean = lambda error_name: np.mean(np.abs(predicted_part[error_name]))
    
    return {'Eroare medie absoluta (MAE)': error_mean('e'),'Eroare patratica medie (MSE)': error_mean('s')}

for err_name, err_value in calculate_forecast_errors(cmp_df, prediction_size).items():
    print(err_name, err_value)
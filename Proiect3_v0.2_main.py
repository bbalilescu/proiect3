import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

test = pd.read_csv('EURRON.csv')

print('*****TEST*****')
print(test.head())
print('*****Cativa parametrii ai seriei*****')
print(test.describe())
#print('****')
#a=test.mean()
#print(a)
#print(type(a))
# plt.plot(test["Date"],test["Value"])
# plt.title("Evolutia cursului pe toata perioada")
# plt.show()

X1 = test["Value"].values
X2 = test["Date"].values
# print(X2)
# X=X1[0:1500]
# X22=X2[0:1500]
# plt.figure()
# plt.plot(X22,X)
# print(X)

#Augumented Dickey-Fuller Test
result = adfuller(X1)
# print(adfuller(X1[0:200])[4]["5%"])
# print(result)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

if result[0] < result[4]["5%"]:
    print ("Respingem ipoteza de nul - Seria este stationara")
else:
    print ("Nu s-a putut respinge ipoteza de nul - Seria nu este stationara")

window=300    
xs=pd.Series(X1,X2)
sma = xs.rolling(300).mean()
mae = mean_absolute_error(xs[window:], sma[window:])
deviation = np.std(xs[window:] - sma[window:])
lower_bound = sma - (mae + 1.5 * deviation)
upper_bound = sma + (mae + 1.5 * deviation)
plt.figure()
plt.title("Evolutia cursului pe toata perioada")
plt.plot(xs,label='Evolutie curs')
plt.plot(sma,'g',label='SMA 300' )
plt.plot(upper_bound, 'r--', label='Limita superioara/inferioara anomalie')
plt.plot(lower_bound, 'r--')
plt.legend(loc='best')

test_stat=xs.diff().dropna()
result2=adfuller(test_stat)
print('ADF Statistic: %f' % result2[0])
print('p-value: %f' % result2[1])
print('Critical Values:')
for key, value in result2[4].items():
    print('\t%s: %.3f' % (key, value))

if result2[0] < result2[4]["5%"]:
    print ("Respingem ipoteza de nul - Seria noua este stationara")
else:
    print ("Nu s-a putut respinge ipoteza de nul - Seria noua nu este stationara")

plt.figure()
plt.title("Seria dorita stationara")
plt.plot(test_stat)

#Dimensiune fereastra maxima stationaritate - nu a reusit

#dmax=0
# while(True):
#     if(adfuller(X1[i:j])[0]<adfuller(X1[i:j])[4]["5%"]):
#         j+=1
#     else:
#         if(dmax<j-i):
#             dmax=j-i
#             imax=i
#             jmax=j
        
#     if(j>=len(X1) and i>=len(X1)-1):
#         break
# for i in range(len(X1)-14):
#     for j in range(i+14,len(X1)):
#         if(adfuller(X1[i:j])[0]<adfuller(X1[i:j])[4]["5%"] and dmax<j-i):
#             dmax=j-i
#             imax=i
#             jmax=j
        
# print("Dimensiune maxima fereastra stationaritate: ")
# print(dmax)
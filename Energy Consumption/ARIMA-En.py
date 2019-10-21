#Import LIbraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import rcParams
from statsmodels.tsa.arima_model import ARIMA
get_ipython().magic(u'matplotlib inline')

import timeit
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error

#Import Data
df = pd.read_csv("~/Documents/Codes/Data/AEP_hourly.csv") 

#Remove NA records
df_idx = df.dropna()
df_idx.head(5)

#Choose & Plot Wighted Prices
dt = df_idx[['AEP_MW']]
dt.plot(y='AEP_MW')

#Select new data range
print len(dt)
#dt_test = dt.iloc[5543:]
dt_test = dt

#Define the Border Record between train and test sets
border = int (len(dt_test)*0.75)
print border

#Define Train and  Test sets
train = dt_test.iloc[:border]
test = dt_test.iloc[border:]

#Plot the Sets
ax = train.plot(figsize=(20,10))
test.plot(ax=ax)
plt.legend(['train', 'test'])
plt.show()

#Convert to NumpyArrays
np_train = np.array(train)
np_test = np.array(test)

#ARIMA model - Creation, Fitting, Predictions
start = timeit.default_timer()

history = [x for x in np_train]
predictions = list()
for i in range(len(np_test)):
    model = ARIMA(history, order=(2,1,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    pred = output[0]
    predictions.append(pred)
    exp = np_test[i]
    history.append(exp)
    print('predicted=%f, expected=%f' % (pred, exp))
    
stop = timeit.default_timer()

print('Time: ', stop - start) 

#Plot Results
rcParams['figure.figsize'] = 40, 20
plt.legend(['Actual Prices', 'Predicted Prices'])
plt.plot(np_test)
plt.plot(predictions)

#Print Metrics
print('R-Squared: %f'%(r2_score(np_test, predictions)))
print ('RMSE: %f'%(sqrt(mean_squared_error(np_test, predictions))))
print('MAE: %f'%(mean_absolute_error(np_test, predictions)))
print('medAE: %f'%(median_absolute_error(np_test, predictions)))
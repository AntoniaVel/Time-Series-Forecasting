#Import LIbraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import rcParams
from statsmodels.tsa.arima_model import ARIMA
get_ipython().magic(u'matplotlib inline')

from sklearn.preprocessing import MinMaxScaler


#Sliding Window
def sliding_window(array, interval):
    x = []
    y = []
    
    for i in range(len(array)-interval):
        temp = array[i:i+interval]
        t = np.reshape(temp, (interval/10,10))
        x.append(t)
        y.append(array[i+interval])
    
    return x, y

#Import Data (First 600000 records eliminated)
df = pd.read_csv("~/Documents/Codes/Data/bitstampUSD_1-min_data_2012-01-01_to_2019-03-13.csv") 
df_idx = df.iloc[600000:, :]

#Remove NA records
df_idx = df_idx.dropna()
df_idx.head(5)

#Choose & Plot Wighted Prices
dt = df_idx[['Weighted_Price']]
dt.plot(y='Weighted_Price')

#Define the Border Record between train and test sets
border = int (len(dt)*0.75)
print border

#Define Train and  Test sets
train = dt.iloc[:border]
test = dt.iloc[border:]

#Plot the Sets
ax = train.plot(figsize=(20,10))
test.plot(ax=ax)
plt.legend(['train', 'test'])
plt.show()

#Convert to NumpyArrays
np_train = np.array(train)
np_test = np.array(test)

#ARIMA model - Creation, Fitting, Predictions
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
    if i < 5:
        print('predicted=%f, expected=%f' % (pred, exp))


#Plot Results
rcParams['figure.figsize'] = 100, 72
plt.legend(['Actual Prices', 'Predicted Prices'])
plt.plot(test)
plt.plot(predictions)

#Print Metrics
print('R-Squared: %f'%(r2_score(test, predictions)))
print ('RMSE: %f'%(sqrt(mean_squared_error(test, predictions))))
print('MAE: %f'%(mean_absolute_error(test, predictions)))
print('medAE: %f'%(median_absolute_error(test, predictions)))
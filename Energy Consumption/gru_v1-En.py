#Import LIbraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import rcParams
get_ipython().magic(u'matplotlib inline')

from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import GRU  
from keras.layers import Dropout
from keras.optimizers import Adam

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

#Import Data
df = pd.read_csv("~/Documents/Codes/Data/AEP_hourly.csv") 

#Remove NA records
df_idx = df.dropna()
df_idx.head(5)

#Choose & Plot Wighted Prices
dt = df_idx[['AEP_MW']]
dt.plot(y='AEP_MW')

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

#Data Normalization
sc = MinMaxScaler(feature_range = (-1, 1))
train_sc = sc.fit_transform(train)
test_sc = sc.transform(test)

#Apply Sliding Window Function for Train Data
x, y = sliding_window(train_sc, 100)

#Shuffle the Matrices
X, Y = shuffle(x, y, random_state=0)

#Convert to NumpyArrays
features = np.array(X)
labels = np.array(Y)

print features.shape[0], features.shape[1], features.shape[2]

#Create the Model
model = Sequential()

model.add(GRU(units=50, input_shape=(features.shape[1], features.shape[2]), activation='tanh', return_sequences=False))
model.add(Dropout(0.25))

model.add(Dense(1))

#Compiling
model.compile(optimizer = 'adam', loss = 'mean_squared_error')  
model.fit(features, labels, epochs = 30, batch_size = 50)  

#Save Model as JSON
model_json = model.to_json()
with open("gru1_en.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("gru1_en.h5")
print("Saved model to disk")

#Load Model
from keras.models import model_from_json
json_file = open('gru1_en.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("gru1_en.h5")
print("Loaded model from disk")

#Apply Sliding Window Function for Test Data
tx, ty = sliding_window(train_sc, 100)

#Convert to NumpyArrays
t_features = np.array(tx)
t_labels = np.array(ty)

#Prediction
pred = loaded_model.predict(t_features, verbose = 0)

#Used for Metrics' Calculations
ac = []
a_pred = []

for i in range(len(pred)-1):
        if (pred[i] > pred[i+1] and t_labels[i] > t_labels[i+1]) or (pred[i] < pred[i+1] and t_labels[i] < t_labels[i+1])  :
            #rise/fall predicted correctly
            a_pred.append(1)
        else:
            #rise/fall not predicted correctly
            a_pred.append(0)
        
for i in range(len(a_pred)):
        ac.append(1)

#Plot Results
rcParams['figure.figsize'] = 100, 72
plt.legend(['Actual Prices', 'Predicted Prices'])
plt.plot(t_labels)
plt.plot(pred)

#Save as png
plt.savefig('gru1_gr_en.png')

#Print Metrics
print('R-Squared: %f'%(r2_score(t_labels, pred)))
print ('RMSE: %f'%(sqrt(mean_squared_error(t_labels, pred))))
print('MAE: %f'%(mean_absolute_error(t_labels, pred)))
print('medAE: %f'%(median_absolute_error(t_labels, pred)))
print('Accuracy: %.2f'%(accuracy_score(ac, a_pred, normalize = True)))
print('Precision: %f'%(precision_score(ac, a_pred, average = 'macro')))
print('Recall: %f'%(recall_score(ac, a_pred, average = 'macro')))
print('F1: %f'%(f1_score(ac, a_pred, average = 'macro')))
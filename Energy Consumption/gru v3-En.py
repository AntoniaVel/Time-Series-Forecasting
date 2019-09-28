#Import LIbraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import rcParams
get_ipython().magic(u'matplotlib inline')

from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from sklearn.metrics import r2_score

from keras.models import Sequential#Import LIbraries
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

from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import GRU  
from keras.layers import Dropout
from keras.optimizers import Adam
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
model.add(Dense(1))

model.add(GRU(units=50, input_shape=(features.shape[1], features.shape[2]), activation='tanh', return_sequences=False))
model.add(Dropout(0.25))

model.add(Dense(1))

#Compiling
model.compile(optimizer = 'adam', loss = 'mean_squared_error')  
model.fit(features, labels, epochs = 30, batch_size = 50)  

#Save Model as JSON
model_json = model.to_json()
with open("gru3_en.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("gru3_en.h5")
print("Saved model to disk")

#Load Model
from keras.models import model_from_json
json_file = open('gru3_en.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("gru3_en.h5")
print("Loaded model from disk")

#Apply Sliding Window Function for Test Data
tx, ty = sliding_window(train_sc, 100)

#Convert to NumpyArrays
t_features = np.array(tx)
t_labels = np.array(ty)

#Prediction
pred = loaded_model.predict(t_features, verbose = 0)

#Print Results
rcParams['figure.figsize'] = 20, 12
plt.plot(t_labels)#Plot Results
rcParams['figure.figsize'] = 100, 72
plt.legend(['Actual Prices', 'Predicted Prices'])
plt.plot(t_labels)
plt.plot(pred)

#Save as png
plt.savefig('gru1_gr_en.png')
plt.plot(pred)
print('R-Squared: %f'%(r2_score(t_labels, pred)))

#Print Metrics
print('R-Squared: %f'%(r2_score(t_labels, pred)))
print ('RMSE: %f'%(sqrt(mean_squared_error(t_labels, pred))))
print('MAE: %f'%(mean_absolute_error(t_labels, pred)))
print('medAE: %f'%(median_absolute_error(t_labels, pred)))

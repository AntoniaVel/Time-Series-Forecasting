#Import LIbraries
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

# evaluate an ARIMA model for a given order (p,d,q)
def evaluate_arima_model(train, test, arima_order):
    history = [x for x in train]
    # make predictions
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit(disp=0)
        out = model_fit.forecast()[0]
        predictions.append(out)
        history.append(test[t])
    # calculate out of sample error
    error = mean_squared_error(test, predictions)
    return error

# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(train, test, p_values, d_values, q_values):
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    mse = evaluate_arima_model(train, test, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    print('ARIMA%s MSE=%.3f' % (order,mse))
                except:
                    continue
    print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))

#Import Data (First 600000 records eliminated)
df = pd.read_csv("~/Documents/Codes/Data/bitstampUSD_1-min_data_2012-01-01_to_2019-03-13.csv") 
df_idx = df.iloc[600000:, :]

#Remove NA records
df_idx = df_idx.dropna()

#Choose Wighted Prices
dt = df_idx[['Weighted_Price']]
dt_test = dt.iloc[2515730-200:]

#Define the Border Record between train and test sets
border = int (len(dt_test)*0.75)

#Define Train and  Test sets
train = dt_test.iloc[:border]
test = dt_test.iloc[border:]

#Convert to NumpyArrays
np_train = np.array(train)
np_test = np.array(test)

#ARIMA tets parameters
p_values = [0, 1, 2, 4, 6, 8, 10]
d_values = range(0, 3)
q_values = range(0, 3)
warnings.filterwarnings("ignore")
evaluate_models(np_train, np_test, p_values, d_values, q_values)
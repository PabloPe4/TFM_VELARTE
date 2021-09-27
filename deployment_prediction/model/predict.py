import pandas as pd
import numpy as np
import datetime
import catboost as cb
import pickle

from sklearn.model_selection import  train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from catboost import CatBoostRegressor
from catboost import Pool
from catboost import CatBoost


import warnings
warnings.filterwarnings("ignore")

def predict():
    '''if option=='transform':
        path= 'DF_CLEAN_VELARTE.csv'
        delimiter = ','
        df = pd.read_csv(path, header='infer', delimiter=delimiter)

    elif option=='predict':
        path='articulos_a_predecir.csv'
        delimiter = ','
        df = pd.read_csv(path, header='infer', delimiter=delimiter)

    else:
        path = 'DF_CLEAN_VELARTE.csv'
        delimiter = ','
        df = pd.read_csv(path, header='infer', delimiter=delimiter)
        '''
    path = 'DF_CLEAN_VELARTE.csv'
    delimiter = ','
    df = pd.read_csv(path, header='infer', delimiter=delimiter)
    # creating independent variables as X and target/dependent variable as y
    X = df.loc[:, df.columns != 'Cantidad Merma']
    y = df['Cantidad Merma']

    # Let’s split X and y using Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
    # get shape of train and test data

    #my_model = cb.CatBoostRegressor(loss_function='RMSE')

    tuned_model = pickle.load(open('catboost.pickle', 'rb'))


    y_pred = tuned_model.predict(X_test)
    rmse = (np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = r2_score(y_test, y_pred)
    print('Model performance')
    print('RMSE: {:.2f}'.format(rmse))
    print('R2: {:.2f}'.format(r2))

    print("Estas son las correspondientes mermas esperadas:", y_pred)

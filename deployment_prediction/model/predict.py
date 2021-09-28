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

def predict(option):
    if option=='1':
        path= 'DF_CLEAN_VELARTE.csv'
        delimiter = ','
        df = pd.read_csv(path, header='infer', delimiter=delimiter)

        X = df.loc[:, df.columns != 'Cantidad Merma']
        y = df['Cantidad Merma']

        # Let’s split X and y using Train test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

        tuned_model = pickle.load(open('catboost.pickle', 'rb'))

        y_pred = tuned_model.predict(X_test)
        rmse = (np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = r2_score(y_test, y_pred)
        print('Model performance')
        print('RMSE: {:.2f}'.format(rmse))
        print('R2: {:.2f}'.format(r2))

        X_test['pred'] = y_pred
        X_test.to_csv('resultados_predicciones_produccion.csv', index=False)
        print('¡Csv con predicciones generado con éxito!')

    elif option=='2':
        path='articulos_a_predecir.csv'
        delimiter = ','
        df = pd.read_csv(path, header='infer', delimiter=delimiter)

        X_test = df.loc[:, df.columns != 'Cantidad Merma']
        y_test = df['Cantidad Merma']

        tuned_model = pickle.load(open('catboost.pickle', 'rb'))
        y_pred = tuned_model.predict(X_test)

        df['pred'] = y_pred
        df.to_csv('resultados_articulos_a_predecir.csv', index=False)
        print('¡Csv con predicciones generado con éxito!')

    else:
        path= 'DF_CLEAN_VELARTE.csv'
        delimiter = ','
        df = pd.read_csv(path, header='infer', delimiter=delimiter)

        X = df.loc[:, df.columns != 'Cantidad Merma']
        y = df['Cantidad Merma']

        # Let’s split X and y using Train test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

        tuned_model = pickle.load(open('catboost.pickle', 'rb'))
        y_pred = tuned_model.predict(X_test)
        rmse = (np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = r2_score(y_test, y_pred)
        print('Model performance')
        print('RMSE: {:.2f}'.format(rmse))
        print('R2: {:.2f}'.format(r2))

        X_test['pred'] = y_pred
        X_test.to_csv('resultados_predicciones_produccion.csv', index=False)
        print('¡Csv con predicciones generado con éxito!')



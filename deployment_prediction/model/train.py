import pandas as pd
import numpy as np
import datetime
import catboost as cb
import pickle

from sklearn import preprocessing
from numpy import asarray

from sklearn.model_selection import  train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance
import math

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

#creating independent variables as X and target/dependent variable as y
X= df.loc[:, df.columns!= 'Cantidad Merma']
y= df['Cantidad Merma']


#Let’s split X and y using Train test split
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.8,random_state= 42)
#get shape of train and test data
print("train data size:",X_train.shape)
print("test data size:",X_test.shape)


train_dataset = cb.Pool(X_train, y_train, cat_features = ['Articulo_Orden','Cod_ Turno Trabajo', 'Location Code', 'Codigo_rechazo',	'Causa', 'Weekday', 'Day', 'Month','Year'])
test_dataset = cb.Pool(X_test, y_test, cat_features = ['Articulo_Orden','Cod_ Turno Trabajo', 'Location Code', 'Codigo_rechazo',	'Causa', 'Weekday', 'Day', 'Month','Year'])

my_model = cb.CatBoostRegressor(loss_function='RMSE')

grid = {'iterations': [300, 400],
        'learning_rate': [0.04,0.05],
        'depth': [6, 8],
        'l2_leaf_reg': [4, 5],
        'loss_function':['RMSE']}

grid_search = GridSearchCV(estimator=my_model, param_grid = grid, cv = 2, n_jobs=-1)
best_model = grid_search.fit(X_train, y_train)

print(" Results from Grid Search " )
print("\n The best estimator across ALL searched params:\n", best_model.best_estimator_)
print("\n The best score across ALL searched params:\n", best_model.best_score_)
print("\n The best parameters across ALL searched params:\n", best_model.best_params_)


tuned_model = CatBoost(params=best_model.best_params_)
tuned_model = tuned_model.fit(X_train, y_train)

# It is important to use binary access
with open('catboost.pickle', 'wb') as f:
    pickle.dump(tuned_model, f)
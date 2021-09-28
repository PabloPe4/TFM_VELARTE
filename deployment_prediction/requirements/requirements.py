import pandas as pd
import numpy as np
import datetime
import catboost as cb
import pickle

from sklearn import preprocessing
from numpy import asarray
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import  train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from catboost import CatBoostRegressor
from catboost import Pool
from catboost import CatBoost
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance
import math

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore") 
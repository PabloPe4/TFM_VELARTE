#TRANSFORMACIONES DESDE LA TABLA DE PRODUCCIÓN CREADA POR ÁLVARO
import pandas as pd
import numpy as np
import datetime
import catboost as cb
import pickle

from sklearn import preprocessing
import math

import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

def transformations(option):

    path= 'VELARTE_PROD_MERMA.csv'
    delimiter = ','
    df = pd.read_csv(path, header='infer', delimiter=delimiter)



    del df['Orden']

    df['Codigo_rechazo']=100

    #Reemplazamos según los insights
    df['Codigo_rechazo'] = np.where((df['Articulo_Orden']==9050542), 35, df['Codigo_rechazo'])
    df['Codigo_rechazo'] = np.where((df['Articulo_Orden']==9050541), 31, df['Codigo_rechazo'])
    df['Codigo_rechazo'] = np.where((df['Articulo_Orden']==9050272), 32, df['Codigo_rechazo'])

    df['Codigo_rechazo'] = np.where((df['Articulo_Orden']==9010470), 71, df['Codigo_rechazo'])
    df['Codigo_rechazo'] = np.where((df['Articulo_Orden']==9010527), 71, df['Codigo_rechazo'])
    df['Codigo_rechazo'] = np.where((df['Articulo_Orden']==9010610), 71, df['Codigo_rechazo'])
    df['Codigo_rechazo'] = np.where((df['Articulo_Orden']==9010414), 71, df['Codigo_rechazo'])

    df['Codigo_rechazo'] = np.where((df['Articulo_Orden']==9020101), 81, df['Codigo_rechazo'])
    df['Codigo_rechazo'] = np.where((df['Cantidad Merma']<8), 372, df['Codigo_rechazo'])

    # Creamos la columna causa
    df['Causa'] = 100
    # Ahora rellenamos la columna causa en funcion del codigo rechazo
    df['Causa'] = np.where((df['Codigo_rechazo'] == 35), 2, df['Causa'])  # Envasado
    df['Causa'] = np.where((df['Codigo_rechazo'] == 31), 2, df['Causa'])
    df['Causa'] = np.where((df['Codigo_rechazo'] == 32), 2, df['Causa'])
    df['Causa'] = np.where((df['Codigo_rechazo'] == 71), 1, df['Causa'])  # Film
    df['Causa'] = np.where((df['Codigo_rechazo'] == 81), 4, df['Causa'])  # Horno
    df['Causa'] = np.where((df['Codigo_rechazo'] == 372), 3, df['Causa'])  # Elaboración

    df['Tipo']=100

    #Reemplazamos las que teníamos por insights y existen en el excel
    df['Tipo'] = np.where((df['Articulo_Orden']==9050542), 2, df['Tipo'])
    df['Tipo'] = np.where((df['Articulo_Orden']==9050541), 2, df['Tipo'])
    df['Tipo'] = np.where((df['Articulo_Orden']==9050272), 2, df['Tipo'])

    df['Tipo'] = np.where((df['Articulo_Orden']==9010470), 2, df['Tipo'])
    df['Tipo'] = np.where((df['Articulo_Orden']==9010527), 1, df['Tipo'])
    df['Tipo'] = np.where((df['Articulo_Orden']==9010610), 1, df['Tipo'])

    #Las top 20 que tienen más merma
    df['Tipo'] = np.where((df['Articulo_Orden']==9010610), 1, df['Tipo'])
    df['Tipo'] = np.where((df['Articulo_Orden']==9010496), 2, df['Tipo'])
    df['Tipo'] = np.where((df['Articulo_Orden']==9020110), 1, df['Tipo'])
    df['Tipo'] = np.where((df['Articulo_Orden']==9050270), 2, df['Tipo'])
    df['Tipo'] = np.where((df['Articulo_Orden']==9510325), 2, df['Tipo'])
    df['Tipo'] = np.where((df['Articulo_Orden']==9010323), 1, df['Tipo'])
    df['Tipo'] = np.where((df['Articulo_Orden']==9010467), 1, df['Tipo'])
    df['Tipo'] = np.where((df['Articulo_Orden']==9510415), 2, df['Tipo'])
    df['Tipo'] = np.where((df['Articulo_Orden']==9010466), 1, df['Tipo'])
    df['Tipo'] = np.where((df['Articulo_Orden']==9050574), 2, df['Tipo'])
    df['Tipo'] = np.where((df['Articulo_Orden']==9010494), 2, df['Tipo'])
    df['Tipo'] = np.where((df['Articulo_Orden']==9010611), 1, df['Tipo'])
    df['Tipo'] = np.where((df['Articulo_Orden']==9020111), 1, df['Tipo'])
    df['Tipo'] = np.where((df['Articulo_Orden']==9010475), 2, df['Tipo'])
    df['Tipo'] = np.where((df['Articulo_Orden']==9010321), 1, df['Tipo'])
    df['Tipo'] = np.where((df['Articulo_Orden']==9010105), 1, df['Tipo'])
    df['Tipo'] = np.where((df['Articulo_Orden']==9010486), 2, df['Tipo'])
    df['Tipo'] = np.where((df['Articulo_Orden']==9010499), 1, df['Tipo'])
    df['Tipo'] = np.where((df['Articulo_Orden']==9050547), 2, df['Tipo'])
    df['Tipo'] = np.where((df['Articulo_Orden']==9010321), 1, df['Tipo'])


    df['Fecha'] = pd.to_datetime(df['Fecha'])
    df['Weekday'] = df['Fecha'].dt.dayofweek
    df['Day'] = pd.DatetimeIndex(df['Fecha']).day
    df['Month'] = pd.DatetimeIndex(df['Fecha']).month
    df['Year'] = pd.DatetimeIndex(df['Fecha']).year
    del df['Fecha']


    index_na = df[df['Cod_ Turno Trabajo'].isna()].index
    # drop these row indexes
    # from dataFrame
    df.drop(index_na, inplace = True)

    index_na2 = df[df['Location Code'].isna()].index
    # drop these row indexes
    # from dataFrame
    df.drop(index_na2, inplace = True)

    index_LR = df[df['Location Code']=='LR'].index
    # drop these row indexes
    # from dataFrame
    df.drop(index_LR, inplace = True)

    df['Location Code']= df['Location Code'].replace(['L1'], value='1')
    df['Location Code']= df['Location Code'].replace(['L2'], value='2')
    df['Location Code']= df['Location Code'].replace(['L3'], value='3')
    df['Cod_ Turno Trabajo']= df['Cod_ Turno Trabajo'].replace(['L1-1'], value='1')
    df['Cod_ Turno Trabajo']= df['Cod_ Turno Trabajo'].replace(['L1-2'], value='2')
    df['Cod_ Turno Trabajo']= df['Cod_ Turno Trabajo'].replace(['L1-3'], value='3')
    df['Cod_ Turno Trabajo']= df['Cod_ Turno Trabajo'].replace(['L2-1'], value='1')
    df['Cod_ Turno Trabajo']= df['Cod_ Turno Trabajo'].replace(['L2-2'], value='2')
    df['Cod_ Turno Trabajo']= df['Cod_ Turno Trabajo'].replace(['L2-3'], value='3')
    df['Cod_ Turno Trabajo']= df['Cod_ Turno Trabajo'].replace(['L3-1'], value='1')
    df['Cod_ Turno Trabajo']= df['Cod_ Turno Trabajo'].replace(['L3-2'], value='2')
    df['Cod_ Turno Trabajo']= df['Cod_ Turno Trabajo'].replace(['L3-3'], value='3')
    df['Cod_ Turno Trabajo']= df['Cod_ Turno Trabajo'].replace(['AJUSTE  L1', 'AJUSTE  L2', 'MASA  L2', 'M', 'L2-FD-3',
           'L2-FD-2', 'L2-FD-1', 'L1-FD-1', 'L1-TE-1', 'L1-FD-3', 'L1-TE-3',
           'L1-FD-2', 'L2-TE-1', 'L2-TE-3', 'T', 'L1-TE-2'], value='OTRO')

    index_m = df[df['Cod_ Turno Trabajo']=='OTRO'].index
    # drop these row indexes
    # from dataFrame
    df.drop(index_m, inplace = True)

    index_names100 = df[df['Codigo_rechazo']==100].index
    # drop these row indexes
    # from dataFrame
    df.drop(index_names100, inplace = True)

    df['Articulo_Orden'] = df['Articulo_Orden'].astype(int)
    df['Cod_ Turno Trabajo'] = df['Cod_ Turno Trabajo'].astype(int)
    df['Location Code'] = df['Location Code'].astype(int)
    df['Day'] = df['Day'].astype(int)
    df['Month'] = df['Month'].astype(int)
    df['Year'] = df['Year'].astype(int)

    df = df[df['Year']!=2020]
    df = df[(df['Year']!=2021) & (df['Month']>=1) & (df['Month']<7)]

    df.head()

    df.to_csv('DF_CLEAN_VELARTE.csv', index=False)

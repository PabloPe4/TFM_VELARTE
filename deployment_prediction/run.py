from requirements.requirements import *
from ETL.transformations import *
from model.predict import *


import warnings
warnings.filterwarnings("ignore")
# RUN

print('')
print('Escriba una de las siguientes opciones:')
print('transform --> para ver la predicción sobre la tabla de producción')
print('predict --> para ver la predicción sobre las filas que usted haya creado en un csv')
print('')
dispatcher = {
    'transform': transformations, 'predict': predict
}

action = input('Escriba su opción: ')
dispatcher[action]()

transformations()

predict()

if __name__ == '__run__':
    run()
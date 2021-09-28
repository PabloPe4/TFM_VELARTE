from requirements.requirements import *
from ETL.transformations import *
from model.predict import *
import warnings
warnings.filterwarnings("ignore")

print('')
print('Escriba una de las siguientes opciones:')
print('1: para ver la predicción sobre la tabla de producción')
print('2: para ver la predicción sobre las filas que usted haya creado en un csv')
print('')

dispatcher = {'1': transformations, '2': predict}
action = input('Escriba su opción: ')
option = action
dispatcher[action](option)

transformations(option)
predict(option)

if __name__ == '__run__':
    run()
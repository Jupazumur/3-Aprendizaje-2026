import utileria as ut
import bosque_aleatorio as ba
import numpy as np
from collections import Counter
import os
import random

# Descarga y descomprime los datos

url = "https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip"
archivo = "datos/cancer.zip"
archivo_datos = "datos/wdbc.data"
atributos = ['ID', 'Diagnosis'] + [f'feature_{i}' for i in range(1, 31)]

# Descarga datos
if not os.path.exists("datos"):
    os.makedirs("datos")
if not os.path.exists(archivo):
    ut.descarga_datos(url, archivo)
    ut.descomprime_zip(archivo)

#Extrae datos y convierte a numericos
datos = ut.lee_csv(
    archivo_datos,
    atributos=atributos,
    separador=","
)
for d in datos:
    d['Diagnosis'] = 1 if d['Diagnosis'] == 'M' else 0
    for i in range(1, 31):
        d[f'feature_{i}'] = float(d[f'feature_{i}'])
    del(d['ID'])

# Selecciona los artributos
target = 'Diagnosis'
atributos = [target] + [f'feature_{i}' for i in range(1, 31)]

# Selecciona un conjunto de entrenamiento y de validación
random.seed(42)
random.shuffle(datos)
N = int(0.8*len(datos))
datos_entrenamiento = datos[:N]
datos_validacion = datos[N:]

# Para bosque
clases = Counter(d[target] for d in datos_entrenamiento)
clase_default = clases.most_common(1)[0][0]
#M = 10
vars_optimas = int(np.sqrt(len(atributos)-1))

### Para diferentes profundidades ###
errores_d = []
for profundidad in [1, 3, 5, 10, 15, 20, 30]:
    bosque = ba.entrenar_bosque(
        datos_entrenamiento, 
        target,
        clase_default,
        max_profundidad=profundidad,
        acc_nodo=1.0,
        min_ejemplos=0,
        M=10,
        variables_seleccionadas=vars_optimas
    )

    predicciones_in = ba.predecir_bosque(bosque, datos_entrenamiento)
    aciertos_in = sum(1 for p, d in zip(predicciones_in, datos_entrenamiento) if p == d[target])
    e_in = 1.0 - (aciertos_in / len(datos_entrenamiento))

    predicciones_out = ba.predecir_bosque(bosque, datos_validacion)
    aciertos_out = sum(1 for p, d in zip(predicciones_out, datos_validacion) if p == d[target])
    e_out = 1.0 - (aciertos_out / len(datos_validacion))

    errores_d.append( (profundidad, e_in, e_out) )
    
# Erorres
print("Errores a diferentes profundidades\n")
print('d'.center(10) + 'E_in'.center(15) + 'E_out'.center(15))
print('-' * 40)
for profundidad, error_entrenamiento, error_validacion in errores_d:
    print(
        f'{profundidad}'.center(10) 
        + f'{error_entrenamiento:.2f}'.center(15) 
        + f'{error_validacion:.2f}'.center(15)
    )
print('-' * 40 + '\n')

### Para diferentes cantidades de arboles (M) ###
errores_M = []
for M in [1, 5, 10, 20, 30, 40, 50]:
    bosque = ba.entrenar_bosque(
        datos_entrenamiento, 
        target,
        clase_default,
        max_profundidad=None,
        acc_nodo=1.0,
        min_ejemplos=0,
        M=M,
        variables_seleccionadas=vars_optimas
    )

    predicciones_in = ba.predecir_bosque(bosque, datos_entrenamiento)
    aciertos_in = sum(1 for p, d in zip(predicciones_in, datos_entrenamiento) if p == d[target])
    e_in = 1.0 - (aciertos_in / len(datos_entrenamiento))

    predicciones_out = ba.predecir_bosque(bosque, datos_validacion)
    aciertos_out = sum(1 for p, d in zip(predicciones_out, datos_validacion) if p == d[target])
    e_out = 1.0 - (aciertos_out / len(datos_validacion))

    errores_M.append( (M, e_in, e_out) )

## Errores
print("Errores a diferentes cantidades de arboles (M)\n")
print('M'.center(10) + 'E_in'.center(15) + 'E_out'.center(15))
print('-' * 40)
for M, error_entrenamiento, error_validacion in errores_M:
    print(
        f'{M}'.center(10) 
        + f'{error_entrenamiento:.2f}'.center(15) 
        + f'{error_validacion:.2f}'.center(15)
    )
print('-' * 40 + '\n')

### Para diferentes cantidades de variables seleccionadas ###
print("Errores a diferentes cantidades de variables seleccionadas\n")
errores_var = []
for vs in [2, 4, 8, 12, 16, 20, 24]:
    bosque = ba.entrenar_bosque(
        datos_entrenamiento, 
        target,
        clase_default,
        max_profundidad=None,
        acc_nodo=1.0,
        min_ejemplos=0,
        M=10,
        variables_seleccionadas=vs
    )

    predicciones_in = ba.predecir_bosque(bosque, datos_entrenamiento)
    aciertos_in = sum(1 for p, d in zip(predicciones_in, datos_entrenamiento) if p == d[target])
    e_in = 1.0 - (aciertos_in / len(datos_entrenamiento))

    predicciones_out = ba.predecir_bosque(bosque, datos_validacion)
    aciertos_out = sum(1 for p, d in zip(predicciones_out, datos_validacion) if p == d[target])
    e_out = 1.0 - (aciertos_out / len(datos_validacion))

    errores_var.append( (vs, e_in, e_out) )

# Errores
print('vars'.center(10) + 'E_in'.center(15) + 'E_out'.center(15))
print('-' * 40)
for vs, error_entrenamiento, error_validacion in errores_var:
    print(
        f'{vs}'.center(10) 
        + f'{error_entrenamiento:.2f}'.center(15) 
        + f'{error_validacion:.2f}'.center(15)
    )
print('-' * 40 + '\n')

# Entrena con la mejor profundidad
#arbol = an.entrena_arbol(datos, target, atributos, max_profundidad=3)
#error = an.evalua_arbol(arbol, datos_entrenamiento, target)
#print(f'Error del modelo seleccionado entrenado con TODOS los datos: {error:.2f}')
#an.imprime_arbol(arbol)
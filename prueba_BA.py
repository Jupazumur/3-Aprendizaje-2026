import utileria as ut
import bosque_aleatorio as ba
from numpy import sqrt
from collections import Counter
import os
from random import seed, shuffle
from time import perf_counter

def main():

    # Timer
    t_comienzo = perf_counter()

    # Descarga y descomprime los datos
    url = "https://archive.ics.uci.edu/static/public/697/predict+students+dropout+and+academic+success.zip"
    archivo = "datos/dropout.zip"
    archivo_datos = "datos/data.csv"
    atributos = [f'feature_{i}' for i in range(1, 37)] + ['Target']

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
        separador=";"
    )

    for d in datos:
        for i in range(1, 37):
            d[f'feature_{i}'] = float(d[f'feature_{i}'])

    # Selecciona los artributos
    target = 'Target'
    atributos = [target] + [f'feature_{i}' for i in range(1, 37)]

    # Selecciona un conjunto de entrenamiento y de validación
    seed(42)
    shuffle(datos)

    # Truncamiento para pruebas más rápidas
    datos = datos[:100]

    # Split de datos de entrenamiento y validación
    N = int(0.8*len(datos))
    datos_entrenamiento = datos[:N]
    datos_validacion = datos[N:]

    # Para bosque
    clases = Counter(d[target] for d in datos_entrenamiento)
    clase_default = clases.most_common(1)[0][0]
    vars_optimas = int(sqrt(len(atributos)-1))

    errores_d = [test_profundidad(datos_entrenamiento, datos_validacion, target, clase_default, profundidad, vars_optimas)
                 for profundidad in [1, 3, 5, 10, 15, 20, 30]]
    
    errores_M = [test_M(datos_entrenamiento, datos_validacion, target, clase_default, num_arboles, vars_optimas)
                 for num_arboles in [1, 5, 10, 20, 30, 40, 50]]
    
    errores_var = [test_vs(datos_entrenamiento, datos_validacion, target, clase_default, num_vars)
                 for num_vars in [2, 4, 8, 12, 16, 20, 24]]

    print("Errores para diferentes profundidades\n")
    imprimir_error(errores_d, 'd')

    print("Errores para diferentes cantidades de arboles (M)\n")
    imprimir_error(errores_M, 'M')

    print("Errores para diferentes cantidades de variables seleccionadas\n")
    imprimir_error(errores_var, 'vars')

    t_final = perf_counter()

    print(f"Tiempo ejecución: {(t_final - t_comienzo):.4f} seconds")

def imprimir_error(errores, aspecto):
    print(aspecto.center(10) + 'E_in'.center(15) + 'E_out'.center(15))
    print('-' * 40)

    for medicion, error_entrenamiento, error_validacion in errores:
        print(
            f'{medicion}'.center(10) 
            + f'{error_entrenamiento:.2f}'.center(15) 
            + f'{error_validacion:.2f}'.center(15)
        )
    print('-' * 40 + '\n')

def test_profundidad(datos_e, datos_v, target, clase_default, max_profundidad, variables_seleccionadas):
    bosque = ba.entrenar_bosque(
        datos_e, 
        target,
        clase_default,
        max_profundidad=max_profundidad,
        acc_nodo=1.0,
        min_ejemplos=0,
        M=10,
        variables_seleccionadas=variables_seleccionadas
    )

    predicciones_in = ba.predecir_bosque(bosque, datos_e)
    aciertos_in = sum(1 for p, d in zip(predicciones_in, datos_e) if p == d[target])
    e_in = 1.0 - (aciertos_in / len(datos_e))

    predicciones_out = ba.predecir_bosque(bosque, datos_v)
    aciertos_out = sum(1 for p, d in zip(predicciones_out, datos_v) if p == d[target])
    e_out = 1.0 - (aciertos_out / len(datos_v))

    return (max_profundidad, e_in, e_out)

def test_M(datos_e, datos_v, target, clase_default, M, variables_seleccionadas):
    bosque = ba.entrenar_bosque(
        datos_e, 
        target,
        clase_default,
        max_profundidad=None,
        acc_nodo=1.0,
        min_ejemplos=0,
        M=M,
        variables_seleccionadas=variables_seleccionadas
    )

    predicciones_in = ba.predecir_bosque(bosque, datos_e)
    aciertos_in = sum(1 for p, d in zip(predicciones_in, datos_e) if p == d[target])
    e_in = 1.0 - (aciertos_in / len(datos_e))

    predicciones_out = ba.predecir_bosque(bosque, datos_v)
    aciertos_out = sum(1 for p, d in zip(predicciones_out, datos_v) if p == d[target])
    e_out = 1.0 - (aciertos_out / len(datos_v))

    return (M, e_in, e_out)

def test_vs(datos_e, datos_v, target, clase_default, vs):
    bosque = ba.entrenar_bosque(
        datos_e, 
        target,
        clase_default,
        max_profundidad=None,
        acc_nodo=1.0,
        min_ejemplos=0,
        M=10,
        variables_seleccionadas=vs
    )

    predicciones_in = ba.predecir_bosque(bosque, datos_e)
    aciertos_in = sum(1 for p, d in zip(predicciones_in, datos_e) if p == d[target])
    e_in = 1.0 - (aciertos_in / len(datos_e))

    predicciones_out = ba.predecir_bosque(bosque, datos_v)
    aciertos_out = sum(1 for p, d in zip(predicciones_out, datos_v) if p == d[target])
    e_out = 1.0 - (aciertos_out / len(datos_v))

    return (vs, e_in, e_out)

if __name__ == '__main__':
    main()
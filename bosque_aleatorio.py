import arboles_numericos as an
import random
from collections import Counter

def entrenar_bosque(datos, target, clase_default, max_profundidad, acc_nodo, min_ejemplos, M, variables_seleccionadas):
    """
    Entrena un bosque de M árboles, cada árbol entrenado con
    un subconjunto de datos
    """
    #bosque = []
    muestras = _bootstrap(datos, M)

    #for muestra in muestras:
    #    arbol = an.entrena_arbol(
    #        muestra,
    #        target,
    #        clase_default,
    #        max_profundidad=max_profundidad,
    #        acc_nodo=acc_nodo,
    #        min_ejemplos=min_ejemplos,
    #        variables_seleccionadas=variables_seleccionadas,
    #    )
    #    bosque.append(arbol)

    bosque = [an.entrena_arbol(
                muestra,
                target,
                clase_default,
                max_profundidad=max_profundidad,
                acc_nodo=acc_nodo,
                min_ejemplos=min_ejemplos,
                variables_seleccionadas=variables_seleccionadas)
            
            for muestra in muestras]

    return bosque

def _bootstrap(datos, M):
    """
    Realiza el proceso de Bootstrapping en conjunto de datos
    """
    #muestras = []
    k=len(datos)

    #for _ in range(M):
    #    muestras.append(random.choices(datos, k=k))

    #return muestras
    return [random.choices(datos, k=k) for _ in range(M)]

def predecir_bosque(bosque, datos):
    """
    Reúne las predicciones de un bosque
    """
    #predicciones = []

    #for instancia in datos:
        #votos = []

        #for arbol in bosque:
        #    votos.append(arbol.predice(instancia))

        #votos = [arbol.predice(instancia) for arbol in bosque]
        
        #ganador = max(votos, key=votos.count)
        #predicciones.append(ganador)

    #return predicciones

    return [_obtener_ganador(instancia, bosque) for instancia in datos]

def _obtener_ganador(instancia, bosque):
    """
    Obtiene la clase mayoritaria del bosque para una instancia
    """
    votos = [arbol.predice(instancia) for arbol in bosque]
    return Counter(votos).most_common(1)[0][0]
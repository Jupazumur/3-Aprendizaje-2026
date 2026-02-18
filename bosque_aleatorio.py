import arboles_numericos as an
import random

def entrenar_bosque(datos, target, clase_default, max_profundidad, acc_nodo, min_ejemplos, M, variables_seleccionadas):
    bosque = []
    muestras = __entrenar_bosque_helper(datos, M)

    for muestra in muestras:
        arbol = an.entrena_arbol(
            muestra,
            target,
            clase_default,
            max_profundidad=max_profundidad,
            acc_nodo=acc_nodo,
            min_ejemplos=min_ejemplos,
            variables_seleccionadas=variables_seleccionadas,
        )
        bosque.append(arbol)

    return bosque

def __entrenar_bosque_helper(datos, M):
    """
    Bootstrapping
    """
    muestras = []

    for _ in range(M):
        muestras.append(random.choices(datos, k=len(datos)))

    return muestras

def predecir_bosque(bosque, datos):
    predicciones = []

    for instancia in datos:
        votos = []

        for arbol in bosque:
            votos.append(arbol.predice(instancia))
        
        ganador = max(votos, key=votos.count)
        predicciones.append(ganador)

    return predicciones
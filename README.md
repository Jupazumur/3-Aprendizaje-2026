# Miniproyecto 3: Aprendizaje Automático con árboles de desición

En este proyecto vamos a repasar un poco las ideas básicas de aprendizaje supervisado, 
tratando de ejemplificar los métodos utilizando árboles de desición y bosques aleatorios.

Para este proyecto se tiene como base documental:

1. Un modulo `utileria.py`, el cual sirve para definir algunas funciones necesarias,
   (como las que se usar para cargar datos) pero que no son propias de los modelos de 
   aprendizaje.
   
2. Árboles cualitativos, los cuales se encuentran definidos en el módulo 
   `arboles_cualitativos.py` y que luego se utilizan en `prueba_arbol_Q.py`. 
   Para eso, vamos a buscar algunas bases de datos muy sencillas que permitan ilustrar 
   como funciona el algoritmo de aprendizaje, así como su uso en predicción. 
   Estos conjuntos de datos se obtuvieron del venerable pero todavía interesante 
   [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/datasets/), 
   donde se encuentran una serie de bases de datos abiertas para diferentes tareas de 
   aprendizaje. Este repositorio es un lugar que siempre debe uno conocer en el área de 
   Aprendizaje Automático.
   
3. Árboles numéricos, los cuales se encuentran en `arboles_numericos.py` y se puede ver como 
   utilizarlo en `prueba_arbol_N.py`, tambien con un conjunto de datos del mismo 
   [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/datasets/).

Para esta tarea, tienes como tarea el desarrollar y probar un algoritmo de 
*bosques aleatorios* basado en los algortimos ya presentados,
para esto hay que hacer lo siguiente:

1. En el archivo `arboles_numericos.py` se encuentra la difuiente función:
   
   ```python
    def entrena_arbol(datos, target, clase_default, 
                      max_profundidad=None, acc_nodo=1.0, min_ejemplos=0,
                      variables_seleccionadas=None):

   ```

   En el cual existe el parámetro `variables_seleccionadas` el cual no se usa (por default 
   usa el valor `None`). Modifica esta función (no tienes que modificar en otras 
   subfunciones o clases) de manera que si el valor de este parámetro es un número
   entero, entonces se seleccione solo ese número de atributos, seleccionados aleatoriamente,
   para seleccionar el par `(atributo, valor)` en ese nodo (solo en ese nodo). Recuerda que 
   a los hijos se les envía todos los atributos en forma recursiva.

2. En el archivo `bosque_aleatorio.py` desarrolla una función para entrenar busques 
   aleatorios, basándose en el uso de la función `entrena_arbol` del modulo 
   `arboles_numericos.py`. El algoritmo deberá:

   1. Separar los datos en subconjuntos con selección aleatoria con repetición 
      (para `M` subconjuntos).
   2. Por cada subconjunto, entrenar un árbol con un número limitado de variables en cada 
      nodo. Un bosque se puede representar como una lista de nodos raíz (árboles).
   3. Una función para poder hacer predicciones a partir del bosque (lista de nodos raíz).

    Puedes usar varias funciones, que separen un problema grande en varios más pequeños y tratables.

3. En el archivo `prueba_BA.py` usa un conjunto de datos **diferente** a los que ya 
   usamos para árboles cualitativos y numérico del 
   [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/datasets/). 
   Muestra que pasa conforme se aumenta el número de árboles en tu bosque, la máxima
   profundidad de los árboles, o la cantidad de variables a considerar en cada nodo.

Este proyecto solo se puede evaluar a partir de `prueba_BA.py`, por lo que es necesario desarrollar los 3 incisos y se evaluará en relación a la funcionalidad del algoritmo, y del estudio realizado en `prueba_BA.py`.

## Conclusión
A partir de los archivos de texto resultados. El número que le sigue a sus títulos es el número de datos usados en las pruebas.

- Al aumentar la profundidad se puede observar como se reduce rapidamente el $E_\textrm{in}$ , pero se estanca o empeora el $E_\textrm{out}$ . Esto es porque se memorizan los datos pero no generalizan mejor.

- Lo que pasa al aumentar la cantidad de árboles es simplemente que se llega a un punto de rendimientos decrecientes (el poder computacional utilizado para $M$ s mayores ya no mejora el error).

- Cuando se aumentan las variables seleccionadas el $E_\textrm{in}$ se mantiene muy bajo para todas las cantidades de datos. Para el $E_\textrm{out}$ , fluctúa mucho en cantidades menores de datos pero se mantiene estable en cantidades mayores (800-1000).


```sh
➜  ~ ssh -p 2522 root@localhost

Welcome to Alpine!

sd-GCDA:~# 
```

- [Computación de Alto Rendimiento (HPC)](#computación-de-alto-rendimiento-hpc)
  - [Práctica](#práctica)
    - [Objetivos específicos de la Práctica](#objetivos-específicos-de-la-práctica)
  - [Descripción del código secuencial: Tormenta de partículas](#descripción-del-código-secuencial-tormenta-de-partículas)
  - [Ejercicios](#ejercicios)
    - [Descripción de las carpetas](#descripción-de-las-carpetas)
  - [Ejercicio 1: Objetivo paralelización utilizando mpi4py (MPI)](#ejercicio-1-objetivo-paralelización-utilizando-mpi4py-mpi)
  - [Ejercicio 2: Objetivo paralelización utilizando Numba(OpenMP)](#ejercicio-2-objetivo-paralelización-utilizando-numbaopenmp)
  - [Ejercicio 3: Multiplicación de Matrices con MPI (mpi4py)](#ejercicio-3-multiplicación-de-matrices-con-mpi-mpi4py)
- [Bibliografía](#bibliografía)


## Computación de Alto Rendimiento (HPC)

La programación paralela emerge como un recurso invaluable en este contexto, capaz de reducir drásticamente el tiempo necesario para llevar a cabo estas tareas, a veces de horas o incluso días a un período acotado. La Computación de Alto Rendimiento (HPC) se erige como la solución que hace posible esta aceleración, brindando tiempos de cálculo excepcionalmente rápidos y escalabilidad en función de las necesidades particulares.

Este enfoque es esencial cuando se abordan problemas de gran envergadura y complejidad, ya que divide las cargas de trabajo extensas en tareas computacionales individuales que se procesan de manera concurrente. Además de la computación en paralelo, existen otros enfoques en HPC, como la computación distribuida, la computación por agrupamiento, y más recientemente, la infraestructura en la nube, que permite ampliar la capacidad local con recursos disponibles en la nube. En conjunto, estas tecnologías permiten un procesamiento eficiente y rápido en el ámbito de la computación de alto rendimiento. Los superordenadores y clústeres de HPC son ejemplos de plataformas que pueden ejecutar aplicaciones de alto rendimiento y realizar cálculos intensivos en paralelo, ya sea para simulaciones científicas, modelado de sistemas complejos, análisis de grandes conjuntos de datos, o incluso para tareas de inteligencia artificial y Deep Learning.

### Práctica 

En esta práctica está enfocada al problema de las tormentas de partículas. Las tormentas de partículas con alta energía son eventos en los que se liberan grandes cantidades de partículas cargadas, como protones y electrones, provenientes del Sol o de otras fuentes astronómicas (Figura 1). Estas partículas viajan a altas velocidades y pueden causar efectos significativos tanto en el espacio como en la Tierra, afectando satélites, tecnología espaciales y redes de comunicación, y aumentando también el riesgo de enfermedades por radiación en los astronautas.

Para mitigar los efectos adversos de estos eventos, se utilizan satélites de observación, modelos de clima espacial, y sistemas de alerta que permiten a las agencias espaciales y operadores de satélites tomar medidas preventivas. Una de las herramientas clave en este proceso es la **simulación computacional de estos eventos, la cual se realiza mediante aplicaciones especializadas** que simulan el comportamiento de las partículas cargadas. Estas aplicaciones permiten predecir la trayectoria, velocidad, y posibles impactos de las partículas, lo que es fundamental para desarrollar estrategias de protección efectivas y evitar daños en los satélites y otros sistemas críticos.

Sin embargo, la simulación de millones de partículas es una tarea computacionalmente costosa, que puede requerir un tiempo considerable si no se optimiza adecuadamente. El objetivo de esta práctica es abordar este desafío utilizando técnicas de paralelización en entornos de programación de alto rendimiento.

![](img/01.png)
Fig. 1 Efecto de las partículas sobre las superficies.

#### Objetivos específicos de la Práctica

En esta práctica, el objetivo general es paralelizar la aplicación de simulación de tormentas de partículas utilizando dos librerías: **[MPI](https://www.mpi-forum.org/)** (Message Passing Interface)** y **[OpenMP](https://www.open-mpi.org/)** (Open Multi-Processing). La paralelización es esencial para reducir los tiempos de simulación y permitir que los sistemas puedan manejar grandes volúmenes de datos de manera eficiente.

El primer objetivo específico es analizar la simulación que originalmente es secuencial, y posteriormente como segundo objetivo específico, implementaran MPI y OpenMP para mejorar su rendimiento, comparando los tiempos de ejecución y observando cómo la paralelización afecta la eficiencia del programa. A través de esta práctica, se busca que comprendan las ventajas de la computación paralela y adquieran habilidades prácticas en la implementación de estas técnicas en problemas reales de simulación.

En la actualidad MPI [5], acrónimo de Message Passing Interface, es el protocolo más utilizado en el mundo HPC, en especial para el cálculo de cómputo científico ya que tiene tres grandes metas: portabilidad, escalabilidad y alto rendimiento.

Esta práctica se enfoca en el ámbito de la paralelización, por un lado mediante MPI, ya que es un estándar que define la sintaxis y semántica de las funciones contenidas en una librería de paso de mensajes diseñada para ser usada en programas de múltiples procesadores que permite la comunicación de información en un ambiente con memoria distribuida con la ventaja de que, al establecer un estándar para el paso de mensajes tenemos portabilidad (ya que ha sido implementado para casi toda arquitectura de memoria distribuida) y rapidez (cada implementación de la librería ha sido optimizada para el hardware en la cual se ejecuta).

Por otro lado, además de MPI, se utilizará OpenMP [6] una API para programación paralela de memoria compartida. OpenMP facilita la creación de programas paralelos mediante la adición de directivas al código fuente que especifican cómo las tareas se deben dividir entre múltiples hilos, permitiendo aprovechar al máximo los recursos disponibles en sistemas con arquitecturas multicore. OpenMP destaca por su facilidad de uso y flexibilidad, ya que permite un desarrollo más rápido y eficiente de aplicaciones paralelas en comparación con otros modelos de programación paralela.

Python [3] se ha convertido en un lenguaje de programación dominante para áreas emergentes como Machine Learning (ML), Deep Learning (DL) y Data Science (DS). La librería [mpi4py](https://mpi4py.readthedocs.io/en/stable/mpi4py.html) [4] es una librería de comunicación basada en Python que da soporte y proporciona una interfaz a MPI, lo que permite a los desarrolladores de aplicaciones utilizar elementos de procesamiento paralelo. Mpi4py proporciona una solución para que muchas de las aplicaciones, por ejemplo, de Inteligencia Artificial (AI), sean capaces de escalar en múltiples nodos, resolviendo el problema de los recursos limitados cuando se simula en un entorno de memoria compartida (shared-memory).

De manera similar, Numba [1,2] es una librería de Python que permite la compilación just-in-time (JIT) para acelerar las funciones numéricas, utilizando OpenMP para paralelización de tareas en entornos de memoria compartida. Al proporcionar decoradores que facilitan la compilación de código Python a código máquina optimizado, Numba permite aprovechar las capacidades de paralelización de OpenMP, logrando un rendimiento cercano al del código escrito en C o Fortran. Esto es útil en aplicaciones que requieren cálculos intensivos y pueden beneficiarse de la ejecución en múltiples núcleos de procesadores.

### Descripción del código secuencial: Tormenta de partículas

Para esta práctica, se os proporciona un código secuencial que simula los efectos del bombardeo de partículas con alta energía en una superficie expuesta, utilizaremos una iniciativa nacida en el workshop EduPar de la conferencia IPDPS [8] y el workshop EduHPC de la conferencia SuperComputing [9]. Se trata de un conjunto de problemas que pueden ser utilizados en prácticas de asignaturas de HPC o Sistemas Distribuidos y que se han incluido dentro de un proyecto más ambicioso de la NSF/IEEE-TCPP para el diseño curricular y la elaboración de contenidos de asignaturas en el ámbito [10].

El programa proporcionado está simplificado, considerando solo una sección transversal de la superficie. La sección se representa mediante un número discreto de puntos de control distribuidos equitativamente en la capa más externa del material. Se utiliza un array para almacenar la cantidad de energía acumulada en cada punto. El programa calcula la energía acumulada en cada punto después del impacto de varias oleadas de partículas de alta energía (ver Figura 2). El programa calcula e informa, para cada oleada, el punto con la mayor energía acumulada, el cual presenta el mayor riesgo de ser dañado.

![](/img/02.png)
Figura 2. Un conjunto de partículas de alta energía se aproxima a la superficie objetivo. La partícula impactante transfiere energía al punto de impacto y a su entorno.

El código proporcionado incluye, además de la simulación, la lectura de la entrada y control de errores, definición de estructuras, creación de estructuras dinámicas, etc. Este código no es necesario modificarlo, por lo tanto, deberemos tener en cuenta cuando trabajemos en las diferentes paralelizaciones trabajar solo sobre la sección del código marcada por los comentarios y solo sobre el fichero `StudentTemplate.py`:

```py
# “START HERE: DO NOT CHANGE THE CODE ABOVE THIS POINT” 

# our implementation

# “STOP HERE: DO NOT CHANGE THE CODE BELOW THIS POINT”.
```

```sh
alpine:~# tree studentCode/

studentCode/
├── MPI
│   ├── EnergyStormMPIMain.py
│   ├── StudentTemplate.py
│   ├── __pycache__
│   └── test_files
├── OMP
├── SerialEnergyStorm.py
└── test_files
```

`EnergyStormMPIMain.py` :

```py
from mpi4py import MPI
import sys
import numpy as np
from math import sqrt
import time
import StudentTemplate
"""
/*
 * Simplified simulation of high-energy particle storms
 *
 * Parallel computing (Degree in Computer Engineering)
 * 
 *
 * Version: 2.0
 *
 * Code prepared to be used with the Tablon on-line judge.
 * The current Parallel Computing course includes contests using:
 * OpenMP, MPI, and CUDA.
 *
 * (c) 2018 Arturo Gonzalez-Escribano, Eduardo Rodriguez-Gutiez
 * Grupo Trasgo, Universidad de Valladolid (Spain)
 *
 * This work is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License.
 * https://creativecommons.org/licenses/by-sa/4.0/
" */
"""
THRESHOLD = 0.001

def cp_wtime():
    """ Function to get wall time """
    return time.time()

# Structure used to store data for one storm of particles 
class Storm:
    def __init__(self, size, posval):
        self.size = size #Number of particles
        self.posval = np.array(posval, dtype=int) #Positions and values


# Function: Read data of particle storms from a file
def read_storm_file(filename):
    posval = []
    try:
        with open(filename, 'r') as file:
            size = int(file.readline().strip())
            for line in file:
                parts = list(map(int, line.strip().split()))
                posval.extend(parts)
    except Exception as e:
        print(f"Error opening or reading storm file: {filename}, {str(e)}")
        MPI.COMM_WORLD.Abort(1)
    return Storm(size, posval)


# ANCILLARY FUNCTIONS: These are not called from the code section which is measured, leave untouched 
# DEBUG function: Prints the layer status 
def debug_print(layer_size, layer, positions, maximum, num_storms):
    """ Only print for array size up to 35 (change it for bigger sizes if needed) """
    if layer_size <= 35:
        # Traverse layer 
        for k in range(layer_size):
            #Print the energy value of the current cell 
            print(f"{layer[k]:10.4f} |", end="")

            # Compute the number of characters. 
            # This number is normalized, the maximum level is depicted with 60 characters 
            ticks = int(60 * layer[k] / np.max(maximum))

            # Print all characters except the last one 
            print("o" * (ticks - 1), end="")

            # If the cell is a local maximum print a special trailing character
            if k > 0 and k < layer_size - 1 and layer[k] > layer[k - 1] and layer[k] > layer[k + 1]:
                print("x", end="")
            else:
                print("o", end="")

            # If the cell is the maximum of any storm, print the storm mark
            for i in range(num_storms):
                if positions[i] == k:
                    print(f" M{i}", end="")

            # Line feed
            print()

# MAIN PROGRAM

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # 1.1. Read arguments
    if len(sys.argv) < 4:
        if rank == 0:
            print("Usage:", sys.argv[0], "<size> <seed> <storm_1_file> [ <storm_i_file> ] ...")
        sys.exit()

    # 1.2. Read storms information
    layer_size = int(sys.argv[1])
    seed = int(sys.argv[2])
    storm_files = sys.argv[3:]
    num_storms = len(storm_files)
    storms = [read_storm_file(f) for f in storm_files]

    np.random.seed(seed)
    random_factor = np.random.uniform(1000, 10000000) 
    THRESHOLD = 0.01 * random_factor  # Adjust THRESHOLD


    # 1.3. Intialize maximum levels to zero
    # 1.4. Allocate memory for the layer and initialize to zero
    layer = np.zeros(layer_size, dtype=float)
    layer_copy = np.zeros_like(layer)
    maximum = np.zeros(num_storms, dtype=float)
    positions = np.zeros(num_storms, dtype=int)

    comm.Barrier()  # Synchronize before timing

    # 2. Begin time measurement
    start_time = cp_wtime()

    maximum, positions, lastLayer= StudentTemplate.EnergyCore(layer_size, layer, layer_copy, num_storms,storms, maximum, positions,rank, size, comm, THRESHOLD)

    # 4. End time measurement
    end_time = cp_wtime()
    total_time = end_time - start_time

    energySumP = float(np.sum(lastLayer))
    totalEnergySum = comm.reduce(energySumP, op=MPI.SUM, root=0)

    # 5. Results output
    if rank == 0:
        # 5.1. Total computation time 
        print("\nTime:", total_time)
        # 5.2. Print the maximum levels
        print("Result:", end="")
        for i in range(num_storms):
            print(f" {positions[i]} {maximum[i]:.2f}", end="")
        print()


        # Optionally include debug print
        # debug_print(layer_size, layer, positions, maximum, num_storms)

if __name__ == "__main__":
    main()
```

En el codigo anterior se ejecuta `StudentTemplate`

`StudentTemplate.py`:

```py
from mpi4py import MPI
import sys
import numpy as np
from math import sqrt
import time
import math 



# START: Do NOT optimize/parallelize the code of the program above this point 


# THIS FUNCTION CAN BE MODIFIED 
# Function to update a single position of the layer 
def update(layer, layer_size, k, pos, energy, THRESHOLD):
    distance = float(abs(pos - k))
    distance += 1.0
    attenuation = math.sqrt(float(distance))
    energy_k = float(energy) / float(layer_size) / attenuation

    
    energy_k = round(energy_k, 6)

    if energy_k >= THRESHOLD / layer_size or energy_k <= -THRESHOLD / layer_size:
        layer[k] += energy_k

def EnergyCore(layer_size, layer, layer_copy, num_storms,storms, maximum, positions,rank, size, comm, THRESHOLD):
	

	# 3. Storms simulation
	
	for i in range(num_storms):
	    # 3.1. Add impacts energies to layer cells 
	    # For each particle 
	    for j in range(storms[i].size):
	        # Get impact energy (expressed in thousandths)
	        energy = round(float(storms[i].posval[2 * j + 1]) * 1000.0, 6)
	        # Get impact position
	        position = int(storms[i].posval[2 * j])
	        # For each cell in the layer 
	        for k in range(layer_size):
	            # Update the energy value for the cell
	            update(layer, layer_size, k, position, energy, THRESHOLD)

	    # 3.2. Energy relaxation between storms 
	    # 3.2.1. Copy values to the ancillary array
	    layer_copy[:] = layer
	    # 3.2.2. Update layer using the ancillary values.
	    #         Skip updating the first and last positions 
	    layer[1:-1] = (layer_copy[:-2] + layer_copy[1:-1] + layer_copy[2:]) / 3

	    # 3.3. Locate the maximum value in the layer, and its position
	    for k in range(1, layer_size - 1):
	        # Check it only if it is a local maximum
	        if layer[k] > layer[k-1] and layer[k] > layer[k+1] and layer[k] > maximum[i]:
	            maximum[i] = layer[k]
	            positions[i] = k

    # END: Do NOT optimize/parallelize the code below this point
	lastLayer= layer
	return maximum, positions, lastLayer
```

**Argumentos del programa**
El programa necesitará los siguientes argumentos:
1. **Tamaño** `10000` :Eltamañorepresentaelnúmerodepuntosdecontrol. 
2. **Semilla** `687`: Representa el umbral (THRESHOLD), determina si una cantidad de energía (calculada y almacenada) es suficientemente significativa para ser añadida.
1. **Lista de los archivos de las oleadas** `test_files/test_07*` : Es una lista con los nombres de los archivos que contienen la información de cada oleada. Se puede utilizar el mismo archivo varias veces.

```sh
alpine:~/studentCode# python3.9 SerialEnergyStorm.py 10000 687 test_files/test_07*

Total Energy Summary: 32240606224.67188

Time: 474.2194194793701
Result: 9280 885923.65 3686 1748313.47 9254 2579631.72 9203 3426454.41
```

El objetivo principal de la práctica consiste en utilizar los diferentes lenguajes de programación paralela para optimizar el código proporcionado y obtener una reducción del tiempo de ejecución.

**Formato de los archivos de partículas**

Los archivos de partículas contienen la información referente a cada oleada. La primera línea del archivo nos da el número total de partículas de esa oleada. Cada línea contiene la información de una partícula. Las líneas están formadas por dos números: el primero representa el índice del punto de impacto, y el segundo el valor de la energía de la partícula en miles de Joules. La información de cada partícula se lee del archivo y se almacena en un array. 

```sh
alpine:~/studentCode/MPI# head test_files/test_07_a1M_p5k_w1

# Formato del fichero de partículas. 
# En este caso, la ola está formada por 5000 partículas diferentes.
==> test_files/test_07_a1M_p5k_w1 <==
5000
396955 980791
930446 774721
617519 895205
971266 570794
168774 731630
651290 774832
988433 536625
428693 732645
884807 719825
```

**Descripción funcional**
La descripción funcional del programa se puede dividir en tres grandes bloques:
1. **Lectura**: En este bloque, el programa lee la información del archivo de partículas y la guarda en un array de partículas.
2. **Realización de los cálculos**: En este bloque es donde el programa realiza todos los cálculos necesarios. Podemos dividir esta parte en tres fases:
   1. **Fase de bombardeo**: Al impactar una partícula, esta transmite su energía al punto de control y a sus vecinos. Esta energía se va acumulando en cada punto. Existe un factor de atenuación que depende de la distancia al cuadrado. Se ha definido un umbral por debajo del cual se considera que no hay transferencia de energía.
   2. **Fase de relajación**:Enestafase,lasuperficiereaccionadistribuyendo ligeramente su carga. La energía de cada punto se actualiza a la media de los valores de tres puntos: el anterior, el mismo punto y el posterior. El resultado obtenido representará la energía acumulada en el punto de control.
   3. **Búsqueda de máximos** : En este bloque, el programa calcula el valor máximo de energía y su posición para cada oleada.
3. **Fase Final**: En la última fase se mostrarán los resultados obtenidos en pantalla, 

```sh
# Figura 4. La salida consta de dos líneas. 
Time: 474.2194194793701
Result: 9280 885923.65 3686 1748313.47 9254 2579631.72 9203 3426454.41
```

**Modo de depuración.**

El modo de depuración permite observar de cerca el comportamiento del programa y detectar posibles errores o ineficiencias en el código. Si el tamaño del array no es mayor a 35 celdas, se presenta una representación gráfica de los resultados, como se discutió anteriormente. Esta salida se puede comparar antes y después de las modificaciones del programa. Se recomienda ejecutar el programa varias veces con los mismos parámetros para intentar detectar cambios aleatorios en la salida originados por condiciones de carrera.

La salida gráfica es especialmente útil para comparar el estado del programa antes y después de realizar modificaciones en el código. De este modo, es posible asegurarse de que los cambios introducidos no afecten negativamente el comportamiento esperado del programa. Además, ejecutar el programa varias veces con los mismos parámetros es altamente recomendable para detectar cambios aleatorios en los resultados, que podrían ser indicativos de errores en la gestión de la memoria.

El modo de depuración no solo ayuda a identificar problemas, sino que también es una herramienta valiosa para optimizar el código, asegurando que las soluciones implementadas son robustas y se comportan de manera consistente bajo diferentes condiciones de ejecución.

![](/img/04.png)
Figura 5. Ejemplo de salida del programa en modo de depuración para un array de 30 posiciones y tres oleadas de partículas.

En la Figura 5 se muestra la salida del programa para un array de 30 puntos de control, después de tres oleadas con varias partículas aleatorias en cada una. Para cada punto de control, la salida muestra su valor de energía final y una cadena de caracteres que representa gráficamente el valor como una barra. La longitud de la barra está normalizada con la energía máxima después de la última oleada. Si el último carácter de la barra es una 'x', indica que este punto fue identificado como un máximo local después de la última oleada. Un carácter 'M' seguido de un número al final de la barra indica que el valor en el punto fue identificado como la energía más alta después de la oleada indicada por el número asociado. Esta representación gráfica se muestra opcionalmente cuando el programa se compila en modo de depuración. A continuación, se muestra el tiempo de ejecución del cálculo principal, excluyendo los tiempos de inicio, lectura de datos de archivos y escritura de la salida. La última línea muestra la lista de posiciones y valores de energía máximos después de cada oleada.

### Ejercicios

Esta práctica consta de dos ejercicios enfocados en la paralelización y optimización de un código secuencial que simula el bombardeo de partículas de alta energía en una superficie. El objetivo principal es que los estudiantes adquieran habilidades en la aplicación de modelos de programación paralela, tanto en entornos de memoria distribuida como compartida.

En el primer ejercicio, los estudiantes deberán utilizar MPI (Message Passing Interface) para dividir la carga de trabajo entre múltiples procesos, optimizando el rendimiento en sistemas con arquitecturas de memoria distribuida.

En el segundo ejercicio, los estudiantes utilizarán OpenMP mediante la librería Numba para paralelizar el código. A través de la compilación just-in-time (JIT) proporcionada por Numba, los estudiantes aprenderán a explotar el paralelismo en sistemas multicore, acelerando los cálculos.

#### Descripción de las carpetas

La estructura de carpetas se organiza de manera que el código y los archivos relacionados estén separados según su propósito. Dentro de la carpeta principal studentCode, se encuentran las subcarpetas MPI y OMP, que contienen el código de las plantillas que hacen uso de MPI y OpenMP. La carpeta test_files, está destinada a almacenar archivos de prueba necesarios para ejecutar y validar los programas. Fuera de estas subcarpetas, se encuentra el archivo “SerialEnergyStorm.py”, que contiene la versión secuencial del código antes de ser paralelizado.


```sh
alpine:~/studentCode# tree -P '*.py'
.
├── MPI
│   ├── EnergyStormMPIMain.py
│   ├── StudentTemplate.py
│   ├── __pycache__
│   └── test_files
├── OMP
│   ├── EnergyStormOMPMain.py
│   ├── StudentTemplate.py
│   └── test_files
├── SerialEnergyStorm.py
└── test_files
```

En ambos ejercicios solo se podrá modificar el fichero “StudentTemplate.py”. Para ambos ejercicios es necesario que la función **`EnergyCore()` devuelva los siguientes arreglos**:

```py
    # END: Do NOT optimize/parallelize the code below this point
	lastLayer= layer
	return maximum, positions, lastLayer
```

* `Maximum`: arreglo con los máximos valores de cada tormenta 
* `Positions`: arreglo con la posición de punto de control
* `lastLayer`: arreglo con los datos de energía de la última tormenta.

En la primera se muestra el tiempo de ejecución del bloque 2, sin incluir la lectura de datos ni la escritura en pantalla. En la segunda línea encontramos la posición con la máxima energía y el valor de esta para cada oleada de partículas.


### Ejercicio 1: Objetivo paralelización utilizando mpi4py (MPI)

En este ejercicio, se deberá paralelizar el código secuencial que simula el bombardeo de partículas de energía utilizando MPI (Message Passing Interface). El objetivo es dividir la carga de trabajo entre múltiples procesos, donde cada proceso será responsable de calcular la energía acumulada en una sección de la superficie. Este enfoque permitirá que el programa escale eficientemente en un entorno de memoria distribuida, optimizando el tiempo de ejecución para grandes volúmenes de datos.

**Descripción de la carpeta MPI**

La carpeta MPI dentro del directorio studentCode contiene los archivos de código relacionados con la paralelización utilizando MPI (Message Passing Interface). Se pueden observar dos archivos principales:


```sh
alpine:~/studentCode# tree -P '*.py'
.
├── MPI
│   ├── EnergyStormMPIMain.py
│   ├── StudentTemplate.py
│   ├── __pycache__
│   └── test_files
├── OMP
│   ├── EnergyStormOMPMain.py
│   ├── StudentTemplate.py
│   └── test_files
├── SerialEnergyStorm.py
└── test_files
```

[EnergyStormMPIMain.py](/studentCode/MPI/EnergyStormMPIMain.py): Este fichero contiene la función main() para cargar los datos e imprimir los resultados.

[StudentTemplate.py](/studentCode/MPI/StudentTemplate.py): Este fichero es la plantilla que se os proporciona, con una estructura de código inicial para comenzar a implementar la paralelización utilizando MPI. Este es el fichero sobre el que deberán trabajar para modificar y completar la paralelización.

**Cómo ejecutar el programa:**

```sh
alpine:~/studentCode/MPI# mpirun -np 4 python3.9 -m mpi4py EnergyStormMPIMain.py 10000 687 test_files/test_07*

Time: 659.5694713592529
Result: 9280 885923.65 3686 1748313.47 9254 2579631.72 9203 3426454.41
```

Como se puede ver en la imagen anterior, aunque hemos ejecutado el código con los mismos parámetros que en la versión secuencial, el tiempo de ejecución utilizando 4 procesadores es prácticamente igual al de la versión secuencial. Esto se debe a que el código proporcionado en el archivo "StudentTemplate.py" aún no ha sido paralelizado.

Dentro del fichero “StudentTemplate.py” se encuentran dos funciones `update()` y `EnergyCore()`, y dentro de estas, se encuentran las 3 etapas del programa. Paraleliza para generar la mejor distribución del trabajo entre los procesos de la aplicación.

**Resultados esperados después de la paralelización:**

```sh
alpine:~/studentCode/MPI# mpirun -np 4 python3.9 -m mpi4py EnergyStormMPIMain.py 10000 687 test_files/test_07*

Time: 44.2194194793701
Result: 9280 885923.65 3686 1748313.47 9254 2579631.72 9203 3426454.41
```

Como podemos observar, el tiempo de ejecución ha disminuido como resultado de la distribución del trabajo sobre los 4 procesadores manteniendo el mismo resultado que la versión serie.

**`PROPUESTA Y RESULTADOS`**

```py
def EnergyCore(layer_size, layer, layer_copy, num_storms, storms, maximum, positions, rank, size, comm, THRESHOLD):
    # Calculate chunk size for each process
    chunk_size = layer_size // size
    start = rank * chunk_size
    end = start + chunk_size if rank != size - 1 else layer_size  # Last process takes the remainder

    # Allocate local buffers for each process
    local_layer = np.zeros(chunk_size, dtype=float)
    local_layer_copy = np.zeros_like(local_layer)

    # 3. Storms simulation
    for i in range(num_storms):
        # 3.1. Add impacts energies to local cells (parallelized)
        for j in range(storms[i].size):
            energy = round(float(storms[i].posval[2 * j + 1]) * 1000.0, 6)
            position = int(storms[i].posval[2 * j])

            # Update only if the particle impacts this process's range
            if start <= position < end:
                local_position = position - start
                for k in range(chunk_size):
                    update(local_layer, layer_size, k, local_position + start, energy, THRESHOLD)

        # Synchronize all processes (global layer)
        comm.Allgather(local_layer, layer)

        # 3.2. Relax energy between storms
        local_layer_copy[:] = local_layer

        # Send/receive boundary data with neighbors
        if rank != 0:  # Send left boundary to the previous rank
            left_boundary_send = np.array([local_layer[0]], dtype=float)
            left_boundary_recv = np.zeros(1, dtype=float)
            comm.Sendrecv(sendbuf=left_boundary_send, dest=rank - 1,
                          recvbuf=left_boundary_recv, source=rank - 1)
            local_layer_copy[0] = left_boundary_recv[0]

        if rank != size - 1:  # Send right boundary to the next rank
            right_boundary_send = np.array([local_layer[-1]], dtype=float)
            right_boundary_recv = np.zeros(1, dtype=float)
            comm.Sendrecv(sendbuf=right_boundary_send, dest=rank + 1,
                          recvbuf=right_boundary_recv, source=rank + 1)
            local_layer_copy[-1] = right_boundary_recv[0]

        # Perform relaxation on local chunk
        for k in range(1, chunk_size - 1):
            local_layer[k] = (local_layer_copy[k - 1] + local_layer_copy[k] + local_layer_copy[k + 1]) / 3

        # 3.3. Locate the maximum value in the local layer
        local_max = -float('inf')
        local_pos = -1
        for k in range(chunk_size):
            if local_layer[k] > local_max:
                local_max = local_layer[k]
                local_pos = k + start

        # Reduce maximum values and positions to the root process
        global_max = comm.reduce(local_max, op=MPI.MAX, root=0)
        global_pos = comm.reduce(local_pos, op=MPI.MAX, root=0)

        if rank == 0:
            maximum[i] = global_max
            positions[i] = global_pos

    # Collect the final layer
    comm.Gather(local_layer, layer, root=0)

    # END: Do NOT optimize/parallelize the code below this point
    lastLayer = layer
    return maximum, positions, lastLayer

```

```sh
alpine:~/studentCode/MPI# mpirun -np 4 python3.9 -m mpi4py EnergyStormMPIMain.py 10000 687 test_files/test_07*

Time: 0.446976900100708
Result: 9999 119739.60 7501 176541.07 7501 291585.86 7501 379570.39
```




### Ejercicio 2: Objetivo paralelización utilizando Numba(OpenMP)

En este ejercicio, deberán paralelizar el mismo código secuencial, pero esta vez utilizando OpenMP a través de la librería Numba en Python. OpenMP permitirá explotar el paralelismo en entornos de memoria compartida, dividiendo la carga de trabajo entre múltiples hilos dentro de un mismo proceso. Este enfoque permitirá optimizar el rendimiento del código en sistemas multicore, proporcionando una solución eficiente y rápida para la simulación del bombardeo de partículas de alta energía.

**Descripción de la carpeta OMP**

La carpeta OMP (OpenMP) dentro del directorio studentCode contiene los archivos de código relacionados con la paralelización utilizando OpenMP. Se pueden observar también dos archivos principales:

* “EnergyStormOMPMain.py”: Este fichero contiene la función main() para cargar los datos e imprimir los resultados.

* “StudentTemplate.py”: Este fichero es la plantilla que se os proporciona, con una estructura de código inicial para comenzar a implementar la paralelización utilizando Numba. Este es el fichero sobre el que deberán trabajar para modificar y completar la paralelización.

```sh
alpine:~/studentCode# tree -P '*.py'
.
├── MPI
│   ├── EnergyStormMPIMain.py
│   ├── StudentTemplate.py
│   ├── __pycache__
│   └── test_files
├── OMP
│   ├── EnergyStormOMPMain.py
│   ├── StudentTemplate.py
│   └── test_files
├── SerialEnergyStorm.py
└── test_files
```

Cómo ejecutar el programa:

```sh
# NUMBA_NUM_THREADS=2 python3.8 EnergyStormOMPMain.py 10000 687 test_files/test_07* 
Number of threads configured: 2
Time: 116.18192148208618
Result: 9280 885923.65 3686 1748313.47 9254 2579631.72 9203 3426454.41

# NUMBA_NUM_THREADS=4 python3.8 EnergyStormOMPMain.py 10000 687 test_files/test_07* 
Number of threads configured: 4
Time: 125.17492604255676
Result: 9280 885923.65 3686 1748313.47 9254 2579631.72 9203 3426454.41
```

Aunque hemos ejecutado el código con los mismos parámetros que en la versión secuencial, el tiempo de ejecución utilizando 2 o 4 procesadores es prácticamente similar al de la versión serie. Esto se debe, de nuevo, a que el código proporcionado en el archivo "StudentTemplate.py" aún no ha sido paralelizado.

Resultados esperados después de la paralelización:

```sh
# NUMBA_NUM_THREADS=4 python3.8 EnergyStormOMPMain.py 10000 687 test_files/test_07* 
Number of threads configured: 4
Time: 0.9875888824462891
Result: 9280 885923.65 3686 1748313.47 9254 2579631.72 9203 3426454.41

# NUMBA_NUM_THREADS=2 python3.8 EnergyStormOMPMain.py 10000 687 test_files/test_07* 
Number of threads configured: 2
Time: 1.1121788024902344
Result: 9280 885923.65 3686 1748313.47 9254 2579631.72 9203 3426454.41
```

Como se puede observar, el tiempo de ejecución ha disminuido como resultado de la distribución del trabajo sobre los 4 o incluso 2 procesadores, manteniendo el mismo resultado que la versión serie.

**Arquitecturas:**

Podéis descargar del campus virtual 3 imágenes de máquinas virtuales con todas las herramientas necesarias instaladas para el desarrollo de la práctica:
1. Alpineparavirtualbox:paraprocesadoresx86: 
   1. Gestor de máquinas virtuales: https://www.virtualbox.org
1. *AlpineparaVmWare:procesadoresARM(AppleSilicon):
   1. Gestor de máquinas virtuales: https://www.vmware.com
1. AlpineparaUTM:procesadoresARM(AppleSilicon):
   1. Gestor de máquinas virtuales: https://mac.getutm.app

> Nota: Para la arquitectura ARM, por temas de compatibilidad es necesario utilizar la versión Python3.8 y en x86 Python3.9. Respecto a Alpine para VmWare, su uso es opcional, ya que puede requerir una licencia de pago. Sin embargo, si disponéis de licencia, podéis optar por utilizarlo.


### Ejercicio 3: Multiplicación de Matrices con MPI (mpi4py)

La paralelización de la multiplicación de matrices es una operación clave en muchas áreas de la ciencia y la ingeniería. Es un componente en aplicaciones que abarcan desde la simulación numérica hasta el aprendizaje automático y su optimización a través de la paralelización es importante para mejorar el rendimiento las aplicaciones computacionales intensivas. En otras palabras, esta operación subyace en una amplia gama de problemas matemáticos y científicos y su eficiencia puede variar significativamente dependiendo de la estructura y las dimensiones de las matrices involucradas.

En el contexto de los ejercicios anteriores sobre la paralelización de una tormenta de partículas mediante MPI y OpenMP, este ejercicio de multiplicación de matrices añade un enfoque complementario y práctico para seguir profundizando en las técnicas de paralelización. La mulriplicación de matrices implica operaciones repetitivas y dependientes de los datos, lo que la convierte en un reto para practicar la división del trabajo y el uso de la comunicación en MPI.

Objetivo del ejercicio

Implica:
* Aprender a distribuir el trabajo de la multiplicación de matrices que pueden ser ejecutadas de manera paralela por diferentes procesadores.
* Adquirir experiencia práctica en la programación paralela utilizando MPI para comunicarse entre procesos y compartir datos.
* Evaluar y comprender el rendimiento de la multiplicación de matrices en un entorno paralelo con respecto a una implementación secuencial, lo que permitirá comprender las ventajas de la paralelización en operaciones matriciales y su eficacia en función del tamaño de las matrices y la cantidad de recursos computacionales disponibles.

**Tarea**

Multiplicación de matrices en formato paralelo utilizando el paradigma Máster- Worker. Las matrices están definidas como matrices cuadradas del orden NxN. Con el fin de realizar una comparación en cuanto a tiempos de cálculo del producto, se desarrolló este ejercicio también de manera secuencial.

**Documentación para la realización del ejercicio:**
Para resolver el ejercicio, deberéis descargar los ficheros que se encuentran en el Aula de Laboratorio:
* `createMMMatrix.py`: Construye las matrices.
* `mm_serie.py`: Ejercicio de multiplicación de matrices en formato secuencial.
* `mm_parallel_main.py`: Es el fichero que contiene la función main() del programa.
* `mm_parallel_Plantilla.py`: Sobre este fichero se desarrollará el código en formato paralelo para realizar la multiplicación de matrices mediante MPI.


## Bibliografía

[1] NUMBA, https://numba.pydata.org  
[2] A ~5 minute guide to Numba. URL: https://numba.readthedocs.io/en/stable/user/5minguide.html  
[3] Python 3: https://www.python.org/doc/  
[4] Mpi4py: https://mpi4py.readthedocs.io/en/stable/  
[5] Standard MPI 3: https://www.mpi-forum.org/docs/mpi-3.0/mpi30-report.pdf    
[6] OpenMPI Open Source HPC: https://www.open-mpi.org    
[7] Virtualbox: http://virtualbox.org  
[8] https://www.ipdps.org/ipdps2022/  
[9] https://sc21.supercomputing.org/  
[10] https://tcpp.cs.gsu.edu/curriculum/?q=peachy  
import time
import numpy as np
import sys
from math import sqrt
import math
from decimal import Decimal, getcontext

# Función para obtener el tiempo (timestap)
def cp_wtime():
    return time.time()


# Estructura para almacenar datos de una tormenta de partículas
class Storm:
    def __init__(self, size, posval):
        self.size = size
        self.posval = posval

# Función para actualizar una única posición de la capa
def update(layer, layer_size, k, pos, energy, THRESHOLD):
    distance = float(abs(pos - k))
    distance += 1.0
    attenuation = sqrt(distance)
    energy_k = float(energy) / float(layer_size) / attenuation

    energy_k = round(energy_k, 6)
    
    if energy_k >= THRESHOLD / layer_size or energy_k <= -THRESHOLD / layer_size:
        layer[k] += energy_k


# Función para leer datos de tormentas de partículas desde un archivo
def read_storm_file(fname):
    with open(fname, 'r') as f:
        size = int(f.readline().strip())
        posval = []
        for line in f:
            parts = list(map(int, line.strip().split()))
            posval.extend(parts)
    return Storm(size, posval)


# Función de depuración para imprimir el estado de la capa
def debug_print(layer_size, layer, positions, maximum, num_storms):
    if layer_size <= 35:
        for k in range(layer_size):
            print(f"{layer[k]:10.4f} |", end="")
            ticks = int(60 * layer[k] / maximum[num_storms-1])
            for i in range(ticks - 1):
                print("o", end="")
            if k > 0 and k < layer_size - 1 and layer[k] > layer[k - 1] and layer[k] > layer[k + 1]:
                print("x", end="")
            else:
                print("o", end="")
            for i in range(num_storms):
                if positions[i] == k:
                    print(f" M{i}", end="")
            print()


def main(argv):

    # 1.1. Read arguments
    if len(sys.argv) < 4:
    
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
    
    maximum = np.zeros(num_storms)
    positions = np.zeros(num_storms, dtype=int)
    storms_size = np.array([storm.size for storm in storms])
    storms_posval = np.array([storm.posval for storm in storms]) 
    lastLayer = np.array([])
    # Medición del tiempo: inicio
    ttotal = cp_wtime()

    # Simulación de tormentas
    layer = np.zeros(layer_size, dtype=float)
    layer_copy = np.zeros_like(layer)

    for i in range(num_storms):
        for j in range(storms_size[i]):
            energy = round(float(storms[i].posval[2 * j + 1]) * 1000.0, 6)
            position = int(storms[i].posval[2 * j])

            for k in range(layer_size):
                pass
                update(layer, layer_size, k, position, energy, THRESHOLD)

        layer_copy[:] = layer[:]
        layer[1:-1] = (layer_copy[:-2] + layer_copy[1:-1] + layer_copy[2:]) / 3


        for k in range(1, layer_size - 1):
            if layer[k] > layer[k - 1] and layer[k] > layer[k + 1]:
                if layer[k] > maximum[i]:
                    maximum[i] = layer[k]
                    positions[i] = k
        
    # Medición del tiempo: finalización
    ttotal = cp_wtime() - ttotal

    lastLayer = layer
    totalEnergySum = np.sum(lastLayer)
    # Resultados
    print("Total Energy Summary:", totalEnergySum)
    print("\nTime:", ttotal)
    print("Result:", end="")
    for i in range(num_storms):
        print(f" {positions[i]} {maximum[i]:.2f}", end="")
    print()


if __name__ == "__main__":
    main(sys.argv)
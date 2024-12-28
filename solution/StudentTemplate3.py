from mpi4py import MPI  # Comunicación paralela usando MPI
import sys  # Manejo de argumentos y sistema
import numpy as np  # Procesamiento eficiente de arreglos numéricos
from math import sqrt  # Cálculos matemáticos básicos
import time  # Medir tiempos de ejecución
import math  # Operaciones matemáticas adicionales

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


def EnergyCore(layer_size, layer, layer_copy, num_storms, storms, maximum, positions, rank, size, comm, THRESHOLD):
    # Dividir el trabajo entre los procesos
    chunk_size = layer_size // size
    start = rank * chunk_size
    end = start + chunk_size if rank != size - 1 else layer_size
    local_chunk_size = end - start

    # Buffers locales para cada proceso
    local_layer = np.zeros(local_chunk_size, dtype=float)
    local_layer_copy = np.zeros_like(local_layer)

    # Simulación de tormentas
    for i in range(num_storms):
        # Procesar las partículas de la tormenta
        for j in range(storms[i].size):
            energy = round(float(storms[i].posval[2 * j + 1]) * 1000.0, 6)
            position = int(storms[i].posval[2 * j])

            # Verificar si la posición de impacto está en el rango local
            if start <= position < end:
                local_position = position - start
                for k in range(local_chunk_size):
                    update(local_layer, layer_size, k, local_position + start, energy, THRESHOLD)

        # Comunicación entre procesos vecinos para actualizar los límites
        if rank != 0:
            # Enviar el límite izquierdo al proceso anterior
            left_boundary_send = np.array([local_layer[0]], dtype=float)
            left_boundary_recv = np.zeros(1, dtype=float)
            comm.Sendrecv(sendbuf=left_boundary_send, dest=rank - 1,
                          recvbuf=left_boundary_recv, source=rank - 1)
            local_layer_copy[0] = left_boundary_recv[0]

        if rank != size - 1:
            # Enviar el límite derecho al siguiente proceso
            right_boundary_send = np.array([local_layer[-1]], dtype=float)
            right_boundary_recv = np.zeros(1, dtype=float)
            comm.Sendrecv(sendbuf=right_boundary_send, dest=rank + 1,
                          recvbuf=right_boundary_recv, source=rank + 1)
            local_layer_copy[-1] = right_boundary_recv[0]

        # Relajación dentro del rango local
        for k in range(1, local_chunk_size - 1):
            local_layer[k] = (local_layer_copy[k - 1] + local_layer_copy[k] + local_layer_copy[k + 1]) / 3

        # Encontrar el máximo local
        local_max = -float('inf')
        local_pos = -1
        for k in range(local_chunk_size):
            if local_layer[k] > local_max:
                local_max = local_layer[k]
                local_pos = k + start

        # Reducir los máximos a nivel global
        global_max = comm.reduce(local_max, op=MPI.MAX, root=0)
        global_pos = comm.reduce(local_pos, op=MPI.MAX, root=0)

        if rank == 0:
            maximum[i] = global_max
            positions[i] = global_pos

    # Reunir la capa final en el proceso raíz
    comm.Gather(local_layer, layer[start:end], root=0)

    # Retornar resultados finales
    lastLayer = layer
    return maximum, positions, lastLayer


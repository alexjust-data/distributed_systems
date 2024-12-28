from mpi4py import MPI
import numpy as np
import sys

np.set_printoptions(linewidth=np.inf)
np.set_printoptions(threshold=sys.maxsize)

# Función del proceso máster (rank 0)
def mm_rank_0(A, B, N, rank, size, comm):
    # Dividir las filas de A entre los procesos
    rows_per_process = N // size
    remaining_rows = N % size

    # Buffer para almacenar la matriz resultado
    mat_result = np.zeros((N, N), dtype=float)

    # Distribuir las filas de A a cada proceso
    current_row = 0
    for worker_rank in range(1, size):
        rows_to_send = rows_per_process + (1 if worker_rank < remaining_rows else 0)
        start_row = current_row
        end_row = start_row + rows_to_send

        # Enviar las filas de A al proceso trabajador
        comm.Send(A[start_row:end_row, :], dest=worker_rank, tag=0)

        # Enviar la matriz B al proceso trabajador
        comm.Send(B, dest=worker_rank, tag=1)

        current_row = end_row

    # Calcular las filas asignadas al proceso máster
    master_rows = rows_per_process + (1 if rank < remaining_rows else 0)
    mat_result[:master_rows, :] = np.dot(A[:master_rows, :], B)

    # Recibir los resultados de los procesos trabajadores
    current_row = master_rows
    for worker_rank in range(1, size):
        rows_to_receive = rows_per_process + (1 if worker_rank < remaining_rows else 0)
        recv_buffer = np.zeros((rows_to_receive, N), dtype=float)  # Buffer para recibir
        comm.Recv(recv_buffer, source=worker_rank, tag=2)
        mat_result[current_row:current_row + rows_to_receive, :] = recv_buffer
        current_row += rows_to_receive

    return mat_result


# Función de los procesos trabajadores (rank > 0)
def mm_rank_other(N, rank, size, comm):
    # Recibir las filas de A asignadas al proceso
    rows_to_receive = (N // size) + (1 if rank < (N % size) else 0)
    local_A = np.zeros((rows_to_receive, N), dtype=float)
    comm.Recv(local_A, source=0, tag=0)

    # Recibir la matriz B completa
    B = np.zeros((N, N), dtype=float)
    comm.Recv(B, source=0, tag=1)

    # Calcular el producto local
    local_result = np.dot(local_A, B)

    # Enviar el resultado parcial de vuelta al máster
    comm.Send(local_result, dest=0, tag=2)


from mpi4py import MPI
import numpy as np
import time
import sys
import os
from mm_parallel_functions import mm_rank_0, mm_rank_other

np.set_printoptions(linewidth=np.inf)
np.set_printoptions(threshold=sys.maxsize)

def readMatrix(Nmax):

    A = np.loadtxt('A.txt', usecols=range(Nmax))

    B = np.loadtxt('B.txt', usecols=range(Nmax))

    return A, B

'''
Function used to compute the matrix mult using a master-worker paradigm model.
'''
def MMmulti(size, rank, MAX_N,comm):
    tic = time.perf_counter()
    
    if rank == 0: # Master Process
        A = np.zeros((N,N))
        B = np.zeros((N,N))
        A,B = readMatrix(MAX_N)

        mat_result = mm_rank_0(A, B, MAX_N, rank, size,comm)

        result=np.sum(mat_result)
        toc = time.perf_counter()
        print(f"Running Time = {toc - tic:0.4f} seconds")
        print("Final result: ")
        print(result)
    else:
        mm_rank_other(MAX_N, rank, size,comm) # worker Processes
    


'''
Main Function which call the MMmulti function to compute the matrix Multiplication in parallel way.
'''
if __name__ == '__main__':
    N=0
    if len(sys.argv) == 2:
        N = int(sys.argv[1])
    else:
        print("ERROR: Invalid number of parameters. Usage: mm_parallel.py size")
        exit()

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    MMmulti(size, rank, N,comm)
    


import numpy as np
import math
import sys
import random
import os

def initMatrix(MAX_N):
    # Matriz A
    A = np.zeros((MAX_N,MAX_N))
    # Matriz B
    B = np.zeros((MAX_N,MAX_N))

    for i in range(0,MAX_N):
        for j in range(0,MAX_N):
            A[i][j] = np.random.uniform(1,512) + j 
            B[i][j] = np.random.uniform(1,512) + i


    mat = np.matrix(A)
    with open('A.txt', 'wb') as f:
        for line in mat:
            np.savetxt(f, line,fmt='%.2f')

    mat = np.matrix(B)
    with open('B.txt', 'wb') as f:
        for line in mat:
            np.savetxt(f, line, fmt='%.2f')


if __name__ == "__main__":

    if len(sys.argv) == 3:
        seed = int(sys.argv[1])
        tamMatrix = int(sys.argv[2])
        random.seed(seed)
        np.random.seed(seed)
        MAX_N = tamMatrix 
        initMatrix(MAX_N)
    else:
        print("ERROR: Número de parametros incorrectos!")
import numpy as np
import time
import sys
import os



def readMatrix(Nmax):
    # Matriz A
    A = np.loadtxt('A.txt', usecols=range(Nmax))
    # Matriz B
    B = np.loadtxt('B.txt', usecols=range(Nmax))

    return A, B

def MMmulti(N):
	A, B = readMatrix(N)
	C = np.zeros((N,N))

	#MM
	tic = time.perf_counter()

	for k in range(0,N):
		for i in range(0,N):
			for j in range(0,N):
				C[i][k] = C[i][k] + A[i][j] * B[j][k]; 

	toc = time.perf_counter()

	print(f"Running Time = {toc - tic:0.4f} seconds")
	result=np.sum(C)
	return result


if __name__ == '__main__':

	np.set_printoptions(linewidth=np.inf)
	np.set_printoptions(threshold=sys.maxsize)

	N=0
	if len(sys.argv) == 2:
		N = int(sys.argv[1])
	else:
		print("ERROR: Invalid number of parameters. Usage: mm_serie.py size")
		exit()

	result=MMmulti(N)
	print("result: ")
	print(result)


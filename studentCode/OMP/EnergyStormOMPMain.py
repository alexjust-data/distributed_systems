import sys
import numpy as np
from math import sqrt
import time
from numba import njit, prange
from numba import njit, set_num_threads, get_num_threads, prange
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
        exit(1)
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

    print(f"Number of threads configured: {get_num_threads()}")
    # 1.3. Intialize maximum levels to zero
    # 1.4. Allocate memory for the layer and initialize to zero
    layer = np.zeros(layer_size, dtype=float)
    layer_copy = np.zeros_like(layer)
    maximum = np.zeros(num_storms, dtype=float)
    positions = np.zeros(num_storms, dtype=int)

    # 2. Begin time measurement
    start_time = cp_wtime()

    maximum, positions, lastLayer= StudentTemplate.EnergyCore(num_storms, storms, layer_size, layer, layer_copy, maximum, positions, THRESHOLD)

    # 4. End time measurement
    end_time = cp_wtime()
    total_time = end_time - start_time


    totalEnergySum = float(np.sum(lastLayer))
    
    # 5. Results output
    
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
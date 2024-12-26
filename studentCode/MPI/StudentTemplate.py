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

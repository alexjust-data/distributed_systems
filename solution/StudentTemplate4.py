from mpi4py import MPI
import numpy as np
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

def EnergyCore(layer_size, layer, layer_copy, num_storms, storms, maximum, positions, rank, size, comm, THRESHOLD):
    # Calculate the range of cells each process is responsible for
    local_start = rank * (layer_size // size)
    local_end = (rank + 1) * (layer_size // size) if rank != size - 1 else layer_size
    local_layer = np.zeros(layer_size, dtype=float)

    # 3. Storms simulation
    for i in range(num_storms):
        # 3.1. Add impacts energies to layer cells
        for j in range(storms[i].size):
            # Get impact energy (expressed in thousandths)
            energy = round(float(storms[i].posval[2 * j + 1]) * 1000.0, 6)
            # Get impact position
            position = int(storms[i].posval[2 * j])
            # For each cell in the layer (only within the local range)
            for k in range(local_start, local_end):
                # Update the energy value for the cell
                update(local_layer, layer_size, k, position, energy, THRESHOLD)

        # Gather all updated layers from processes
        comm.Allreduce(local_layer, layer, op=MPI.SUM)

        # 3.2. Energy relaxation between storms
        layer_copy[:] = layer
        for k in range(local_start, local_end):
            if 1 <= k < layer_size - 1:
                layer[k] = (layer_copy[k - 1] + layer_copy[k] + layer_copy[k + 1]) / 3

        comm.Allreduce(MPI.IN_PLACE, layer, op=MPI.SUM)

        # 3.3. Locate the maximum value in the layer, and its position
        local_max = 0.0
        local_pos = -1
        for k in range(local_start, local_end):
            # Check it only if it is a local maximum
            if k > 0 and k < layer_size - 1 and layer[k] > layer[k - 1] and layer[k] > layer[k + 1]:
                if layer[k] > local_max:
                    local_max = layer[k]
                    local_pos = k

        # Reduce to find the global maximum and its position
        global_max = comm.allreduce(local_max, op=MPI.MAX)
        if local_max == global_max:
            global_pos = local_pos
        global_pos = comm.bcast(global_pos, root=0)

        if rank == 0:
            maximum[i] = global_max
            positions[i] = global_pos

    # END: Do NOT optimize/parallelize the code below this point
    lastLayer = layer
    return maximum, positions, lastLayer

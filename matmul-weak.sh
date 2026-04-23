#!/bin/bash -x

# Load required modules
module load xl_r spectrum-mpi cuda

# Weak scaling: total elements = number of MPI ranks * 2^24
# Compute exponent for the 1d-stencil program: exponent = log2(ranks) + 24
ranks=$SLURM_NPROCS
exponent=$(awk -v r=$ranks 'BEGIN { printf "%d", log(r)/log(2) + 24 }')

# Run the stencil with the dynamically computed size
mpirun --bind-to core --report-bindings -np $ranks \
    /gpfs/u/home/PCPG/PCPGcpps/scratch/mpi-cuda-exe $exponent

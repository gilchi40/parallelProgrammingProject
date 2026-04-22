#!/bin/bash -x
module load xl_r spectrum-mpi cuda
mpirun --bind-to core --report-bindings -np $SLURM_NPROCS /gpfs/u/home/PCPG/PCPGcpps/scratch/mpi-cuda-exe 10
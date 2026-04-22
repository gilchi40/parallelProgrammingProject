# parallelProgrammingProject

Run instructions:

cd scratch
module load xl_r spectrum-mpi cuda

nano mpi-cuda.c
nano mpi-cuda.cu
nano Makefile
make mpi-cuda
nano matmul-batch.sh 
chmod 700 matmul-batch.sh

sbatch -N 1 -n 32 -t 8 --partition=el8-rpi --gres=gpu:1 matmul-batch.sh
cat slurm-jobNum.out
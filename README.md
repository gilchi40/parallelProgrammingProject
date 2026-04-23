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

sbatch -N 4 -n 16 -t 30 --partition=el8 --gres=gpu:4 matmul-batch.sh
cat slurm-jobNum.out


nano matmul-weak.sh
chmod 700 matmul-weak.sh

Testsuit:

Strong:

sbatch -N 1 -n 4 -t 30 --partition=el8 --gres=gpu:4 ./matmul-batch.sh
cat slurm-jobNum.out
sbatch -N 1 -n 2 -t 30 --partition=el8 --gres=gpu:2 ./matmul-batch.sh
cat slurm-jobNum.out
sbatch -N 1 -n 4 -t 30 --partition=el8 --gres=gpu:4 ./matmul-batch.sh
cat slurm-jobNum.out
sbatch -N 2 -n 8 -t 30 --partition=el8 --gres=gpu:1 ./matmul-batch.sh
cat slurm-jobNum.out
sbatch -N 4 -n 16 -t 30 --partition=el8 --gres=gpu:4 ./matmul-batch.sh 
cat slurm-jobNum.out


Weak:

sbatch -N 1 -n 4 -t 30 --partition=el8 --gres=gpu:4 ./matmul-weak.sh
cat slurm-jobNum.out
sbatch -N 1 -n 2 -t 30 --partition=el8 --gres=gpu:2 ./matmul-weak.sh
cat slurm-jobNum.out
sbatch -N 1 -n 4 -t 30 --partition=el8 --gres=gpu:4 ./matmul-weak.sh
cat slurm-jobNum.out
sbatch -N 2 -n 8 -t 30 --partition=el8 --gres=gpu:1 ./matmul-weak.sh
cat slurm-jobNum.out
sbatch -N 4 -n 16 -t 30 --partition=el8 --gres=gpu:4 ./matmul-weak.sh 
cat slurm-jobNum.out
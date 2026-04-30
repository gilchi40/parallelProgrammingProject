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
Normal:

Strong:

sbatch -N 1 -n 2 -t 30 --partition=el8 --gres=gpu:2 ./matmul-batch.sh
cat slurm-4270534.out
sbatch -N 1 -n 4 -t 30 --partition=el8 --gres=gpu:4 ./matmul-batch.sh
cat slurm-4270535.out
sbatch -N 2 -n 8 -t 30 --partition=el8 --gres=gpu:4 ./matmul-batch.sh
cat slurm-4270596.out
sbatch -N 4 -n 16 -t 30 --partition=el8 --gres=gpu:4 ./matmul-batch.sh 
cat slurm-4270536.out

sbatch -N 4 -n 24 -t 30 --partition=el8 --gres=gpu:6 ./matmul-batch.sh 
cat slurm-4270537.out
sbatch -N 8 -n 32 -t 30 --partition=el8 --gres=gpu:4 ./matmul-batch.sh
cat slurm-4270538.out


Weak:

sbatch -N 1 -n 2 -t 30 --partition=el8 --gres=gpu:2 ./matmul-weak.sh
cat slurm-4270540.out
sbatch -N 1 -n 4 -t 30 --partition=el8 --gres=gpu:4 ./matmul-weak.sh
cat slurm-4270541.out
sbatch -N 2 -n 8 -t 30 --partition=el8 --gres=gpu:4 ./matmul-weak.sh
cat slurm-4270602.out
sbatch -N 4 -n 16 -t 30 --partition=el8 --gres=gpu:4 ./matmul-weak.sh 
cat slurm-4270543.out


Cannon:

Strong: n=15

sbatch -N 1 -n 2 -t 30 --partition=el8 --gres=gpu:2 ./matmul-batch.sh
cat slurm-4270616.out
sbatch -N 1 -n 4 -t 30 --partition=el8 --gres=gpu:4 ./matmul-batch.sh
cat slurm-4270617.out
sbatch -N 2 -n 8 -t 30 --partition=el8 --gres=gpu:4 ./matmul-batch.sh
cat slurm-4270618.out
sbatch -N 4 -n 16 -t 30 --partition=el8 --gres=gpu:4 ./matmul-batch.sh 
cat slurm-4270619.out

alt

sbatch -N 1 -n 4 -t 30 --partition=el8 --gres=gpu:2 ./matmul-batch.sh
cat slurm-4270630.out
sbatch -N 2 -n 16 -t 30 --partition=el8 --gres=gpu:4 ./matmul-batch.sh
cat slurm-4270631.out

Strong: n=17

sbatch -N 1 -n 2 -t 30 --partition=el8 --gres=gpu:2 ./matmul-lg-batch.sh
cat slurm-4270620.out
sbatch -N 1 -n 4 -t 30 --partition=el8 --gres=gpu:4 ./matmul-lg-batch.sh
cat slurm-4270621.out
sbatch -N 2 -n 8 -t 30 --partition=el8 --gres=gpu:4 ./matmul-lg-batch.sh
cat slurm-4270622.out
sbatch -N 4 -n 16 -t 30 --partition=el8 --gres=gpu:4 ./matmul-lg-batch.sh 
cat slurm-4270623.out

alt 
sbatch -N 1 -n 4 -t 30 --partition=el8 --gres=gpu:2 ./matmul-lg-batch.sh
cat slurm-4270632.out
sbatch -N 2 -n 16 -t 30 --partition=el8 --gres=gpu:4 ./matmul-lg-batch.sh
cat slurm-4270633.out



Weak:

sbatch -N 1 -n 2 -t 30 --partition=el8 --gres=gpu:2 ./matmul-weak.sh
cat slurm-4270625.out
sbatch -N 1 -n 4 -t 30 --partition=el8 --gres=gpu:4 ./matmul-weak.sh
cat slurm-4270626.out
sbatch -N 2 -n 8 -t 30 --partition=el8 --gres=gpu:4 ./matmul-weak.sh
cat slurm-4270627.out
sbatch -N 4 -n 16 -t 30 --partition=el8 --gres=gpu:4 ./matmul-weak.sh 
cat slurm-4270628.out

alt

sbatch -N 1 -n 4 -t 30 --partition=el8 --gres=gpu:2 ./matmul-weak.sh 
cat slurm-4270634.out
sbatch -N 2 -n 16 -t 30 --partition=el8 --gres=gpu:4 ./matmul-weak.sh 
cat slurm-4270635.out



Strong: n=16

sbatch -N 1 -n 2 -t 30 --partition=el8 --gres=gpu:2 ./matmul-lg-batch.sh
cat slurm-4270620.out
sbatch -N 1 -n 4 -t 30 --partition=el8 --gres=gpu:4 ./matmul-lg-batch.sh
cat slurm-4270621.out
sbatch -N 2 -n 8 -t 30 --partition=el8 --gres=gpu:4 ./matmul-lg-batch.sh
cat slurm-4270622.out
sbatch -N 4 -n 16 -t 30 --partition=el8 --gres=gpu:4 ./matmul-lg-batch.sh 
cat slurm-4270623.out

alt 
sbatch -N 1 -n 4 -t 30 --partition=el8 --gres=gpu:2 ./matmul-lg-batch.sh
cat slurm-4270632.out
sbatch -N 2 -n 16 -t 30 --partition=el8 --gres=gpu:4 ./matmul-lg-batch.sh
cat slurm-4270633.out





nano matmul-batch10.sh 
nano matmul-batch11.sh 
nano matmul-batch12.sh 
nano matmul-batch13.sh 
nano matmul-batch14.sh 

nano matmul-batch16.sh 


10
sbatch -N 1 -n 4 -t 10 --partition=el8 --gres=gpu:2 ./matmul-batch10.sh
sbatch -N 1 -n 4 -t 10 --partition=el8 --gres=gpu:4 ./matmul-batch10.sh
sbatch -N 2 -n 16 -t 10 --partition=el8 --gres=gpu:4 ./matmul-batch10.sh
sbatch -N 4 -n 16 -t 10 --partition=el8 --gres=gpu:4 ./matmul-batch10.sh 

cat slurm-4281497.out
cat slurm-4281498.out
cat slurm-4281499.out
cat slurm-4281500.out

11
sbatch -N 1 -n 4 -t 10 --partition=el8 --gres=gpu:2 ./matmul-batch11.sh
sbatch -N 1 -n 4 -t 10 --partition=el8 --gres=gpu:4 ./matmul-batch11.sh
sbatch -N 2 -n 16 -t 10 --partition=el8 --gres=gpu:4 ./matmul-batch11.sh
sbatch -N 4 -n 16 -t 10 --partition=el8 --gres=gpu:4 ./matmul-batch11.sh 

cat slurm-4281448.out
cat slurm-4281450.out 
cat slurm-4281451.out
cat slurm-4281452.out

12
sbatch -N 1 -n 4 -t 10 --partition=el8 --gres=gpu:2 ./matmul-batch12.sh
sbatch -N 1 -n 4 -t 10 --partition=el8 --gres=gpu:4 ./matmul-batch12.sh
sbatch -N 2 -n 16 -t 10 --partition=el8 --gres=gpu:4 ./matmul-batch12.sh
sbatch -N 4 -n 16 -t 10 --partition=el8 --gres=gpu:4 ./matmul-batch12.sh 

cat slurm-4281428.out
cat slurm-4281429.out
cat slurm-4281430.out
cat slurm-4281431.out

13
sbatch -N 1 -n 4 -t 10 --partition=el8 --gres=gpu:2 ./matmul-batch13.sh
sbatch -N 1 -n 4 -t 10 --partition=el8 --gres=gpu:4 ./matmul-batch13.sh
sbatch -N 2 -n 16 -t 10 --partition=el8 --gres=gpu:4 ./matmul-batch13.sh
sbatch -N 4 -n 16 -t 10 --partition=el8 --gres=gpu:4 ./matmul-batch13.sh 

cat slurm-4277690.out
cat slurm-4277691.out
cat slurm-4279323.out
cat slurm-4279324.out

14
sbatch -N 1 -n 4 -t 10 --partition=el8 --gres=gpu:2 ./matmul-batch14.sh
sbatch -N 1 -n 4 -t 10 --partition=el8 --gres=gpu:4 ./matmul-batch14.sh
sbatch -N 2 -n 16 -t 10 --partition=el8 --gres=gpu:4 ./matmul-batch14.sh
sbatch -N 4 -n 16 -t 10 --partition=el8 --gres=gpu:4 ./matmul-batch14.sh 

cat slurm-4276455.out
cat slurm-4276459.out
cat slurm-4279330.out
cat slurm-4279333.out

16
sbatch -N 1 -n 4 -t 30 --partition=el8 --gres=gpu:2 ./matmul-batch16.sh
sbatch -N 1 -n 4 -t 10 --partition=el8 --gres=gpu:4 ./matmul-batch16.sh
sbatch -N 2 -n 16 -t 10 --partition=el8 --gres=gpu:4 ./matmul-batch16.sh
sbatch -N 4 -n 16 -t 10 --partition=el8 --gres=gpu:4 ./matmul-batch16.sh 

cat slurm-4279272.out
cat slurm-4276415.out
cat slurm-4279339.out / cat slurm-4279552.out / cat slurm-4279596.out / cat slurm-4283862.out
cat slurm-4279381.out           / cat slurm-4279496.out


lg weak
sbatch -N 1 -n 4 -t 10 --partition=el8 --gres=gpu:2 ./matmul-weak.sh
sbatch -N 1 -n 4 -t 10 --partition=el8 --gres=gpu:4 ./matmul-weak.sh
sbatch -N 2 -n 16 -t 10 --partition=el8 --gres=gpu:4 ./matmul-weak.sh
sbatch -N 4 -n 16 -t 10 --partition=el8 --gres=gpu:4 ./matmul-weak.sh 

cat slurm-4276642.out
cat slurm-4276643.out
cat slurm-4281395.out
cat slurm-4281396.out


sbatch -N 32 -n 128 -t 10 --partition=el8 --gres=gpu:4 ./matmul-batch16.sh
cat slurm-4283195.out

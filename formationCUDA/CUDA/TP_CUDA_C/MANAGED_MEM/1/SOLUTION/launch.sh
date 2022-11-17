#!/bin/bash

#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --gres=gpu:2
#SBATCH --job-name=CUDA_TP_cepp_test
#SBATCH -C v10032g


#Setup env
module purge
source /software/load_me.sh
module load cuda/10.0

#Compilation

nvcc --version
nvcc --std=c++11 -Xcompiler " -fopenmp" vec_add.cu -o vec_add
nvcc --std=c++11 -Xcompiler " -fopenmp" vec_add_prefetch.cu -o vec_add_prefetch
nvcc --std=c++11 -Xcompiler " -fopenmp" vec_add_prefetch_cpu.cu -o vec_add_prefetch_cpu


#Execution
echo "Launching executable"
time ./vec_add
time ./vec_add_prefetch
time ./vec_add_prefetch_cpu
nvprof ./vec_add
nvprof ./vec_add_prefetch
nvprof ./vec_add_prefetch_cpu

echo "End of Script"

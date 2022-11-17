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
nvcc --std=c++11 -Xcompiler " -fopenmp" main.cu -o main

#Execution
for value in {1..3} 
do
echo Iteration $value
time ./main
done

nvprof --export-profile "prof_main.nvvp" ./main


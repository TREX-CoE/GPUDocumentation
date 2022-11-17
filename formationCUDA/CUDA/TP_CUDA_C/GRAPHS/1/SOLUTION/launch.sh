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
nvcc --std=c++11 -Xcompiler " -fopenmp" main_1.cu -o main_1
nvcc --std=c++11 -Xcompiler " -fopenmp" main_g.cu -o main_g

#Execution
for value in {1..3} 
do
echo Iteration $value
time ./main
time ./main_1
time ./main_g
done

#nvprof --export-profile "prof_main.nvvp" ./main
#nvprof --export-profile "prof_main_1.nvvp" ./main_1
nvprof --export-profile "prof_main_g.nvvp" ./main_g


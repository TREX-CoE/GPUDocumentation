#!/bin/bash
#SBATCH -J cuda_4_calmip_test
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --ntasks-per-node=16
#SBATCH --ntasks-per-core=1
#SBATCH --gres=gpu:1
#SBATCH --mem=20000
#SBATCH --time=00:10:00
#SBATCH --mail-user=paul.karlshoefer@atos.net

echo $SLURM_JOB_NODELIST

module load cuda/10.1.105

#g++ --std=c++11 -fopenmp -o nbody_omp.exe nBody_a_square_omp.cpp

nvcc --std=c++11 -Xcompiler -fopenmp -o nbody_cuda.exe nBody_a_square_omp.cu 

nvprof -o test_prof_v1.nvvp ./nbody_cuda.exe 1000 50

#for i in 1 2 4 8 16
#do 
#	export OMP_NUM_THREADS=$i
#	./nbody_omp.exe 1000 200
#done

./nbody_cuda.exe 8000 1000

echo "done"

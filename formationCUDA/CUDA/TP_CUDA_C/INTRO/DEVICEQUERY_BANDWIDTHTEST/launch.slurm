#!/bin/bash
#SBATCH -J cuda_workshop_intern
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -p CSL-6248_GPU_hdr100_192gb_2933
#SBATCH --ntasks-per-node=16
#SBATCH --ntasks-per-core=1
#SBATCH --gres=gpu:1
###SBATCH --mem=20000
#SBATCH --time=00:10:00
##SBATCH --mail-user=foo@bar.com

echo $SLURM_JOB_NODELIST


source /software/load_me.sh
module load cuda/10.2
export CUDA_HOME=/usr/local/cuda-10.2/

nvcc -I ${CUDA_HOME}/samples/common/inc  ./deviceQuery.cpp -o ./deviceQuery.exe
./deviceQuery.exe
#...

echo "done"

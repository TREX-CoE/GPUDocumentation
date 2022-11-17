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



nvprof -o prof_v2 ./main


echo "done"

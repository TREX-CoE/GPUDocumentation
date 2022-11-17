#!/bin/bash
#SBATCH -J form_xxx		#jobname
#SBATCH --nodes=1		#number of nodes
#SBATCH --ntasks=1		#nubmer of total tasks
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-core=1
#SBATCH --gres=gpu:3
#SBATCH --mem=40G
#SBATCH --time=00:15:00

module load nvidia/nvhpc/20.9

nvcc -g -G bad_sample.cu  

./a.out

cuda-memcheck --tool memcheck ./a.out


# Launch via:
# sbatch --reservation=volta1 mon_script.sh




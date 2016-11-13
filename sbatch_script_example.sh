#!/bin/bash -x
##Example Script for NaCl9k

#SBATCH --nodes=1
#SBATCH --output=chase.%j
#SBATCH --error=chase.err.%j
#SBATCH --time=01:00:00

module load intel-para Boost

export OMP_NUM_THREADS=24
export KMP_AFFINITY=verbose,scatter

srun --cpu_bind=none ./main.x --sequence true  --n 9273 --nev 256 --nex 40 --tol 1e-16 --bgn 2 --end 16 --deg 25 --opt N --path_in ~/../slai00/MATrix/NaCl/size9k/bin/ --path_eigp ~/../slai00/MATrix/NaCl/size9k/direct-solutions/ --mode A --name NaCl9k --legacy true

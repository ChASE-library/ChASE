#!/bin/bash -x
##Example Script for NaCl9k

#SBATCH --nodes=4
#SBATCH --ntasks-per-node=24
#SBATCH --output=chase.%j
#SBATCH --error=chase.err.%j
#SBATCH --time=08:00:00

#module load intel-para Boost
module load  Intel ParaStationMPI Boost CMake CUDA imkl
cd $HOME/build-intel

export OMP_NUM_THREADS=24
export KMP_AFFINITY=verbose,scatter

#srun --cpu_bind=none ./main.x --sequence true  --n 9273 --nev 256 --nex 40 --tol 1e-16 --bgn 2 --end 16 --deg 25 --opt N --path_in ~/../slai00/MATrix/NaCl/size9k/bin/ --path_eigp ~/../slai00/MATrix/NaCl/size9k/direct-solutions/ --mode A --name NaCl9k --legacy true
srun --cpu_bind=none ./elemental_driver --n 12455 --nev 1076 --nex 200 --path "/homeb/slai/slai00/MATrix/TiO2/size12k/bin/"  --bgn 1 --end 30

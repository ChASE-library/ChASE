#!/bin/bash -x
#SBATCH --account=zam
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --output=job-%j.out
#SBATCH --error=job-%j.err
#SBATCH --time=0:30:00
#SBATCH --partition=booster     
#SBATCH --gres=gpu:4

export CUDA_VISIBLE_DEVICES=0,1,2,3
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
export OMP_NUM_THREADS=${SRUN_CPUS_PER_TASK}

#srun flipSignGPUNCCLDistTest
#srun QuasiHEMMGPUDistTest
#srun QuasiLanczosGPUDistTest
#export CHASE_DISABLE_CHOLQR=0

#srun -n 1 ./tests/linalg/internal/cpu/SymOrHermCPUTest #ResidualsCPUTest
#srun -n 1 ./tests/linalg/internal/cuda/ResidGPUTest

ctest

#echo "==1=="
#ctest
#srun ./tests/linalg/internal/nccl/LanczosGPUNCCLDistTest

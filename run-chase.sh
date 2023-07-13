#!/bin/bash -x
#SBATCH --job-name="chase-fix-gpu"
#SBATCH --mail-user=
#SBATCH --mail-type=NONE
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=00:10:00
#SBATCH --output=job.nccl.out
#SBATCH --error=job.nccl.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4



export OMP_NUM_THREADS="32"

time -p srun --mpi=pmix_v3 build-nccl/examples/2_input_output/2_input_output_mgpu --isMatGen 1 --n 80000 --path_in=. --nev 2250 --nex 750 --complex 0 --tol 1e-10 --opt S --deg 20 --mode R --maxIter 20

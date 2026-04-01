#!/bin/bash -x
#SBATCH --account=zam
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=18
#SBATCH --output=job-%j.out
#SBATCH --error=job-%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=booster     
#SBATCH --gres=gpu:4

export CUDA_VISIBLE_DEVICES=0,1,2,3
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
export OMP_NUM_THREADS=${SRUN_CPUS_PER_TASK}

export NCCL_ALGO=allreduce:ring
export CHASE_LOG_LEVEL=debug

# 关键环境变量
export CHASE_QR_CHECK_ORTHO=1
export CHASE_LOG_LEVEL=debug
export OMP_NUM_THREADS=${SRUN_CPUS_PER_TASK}
export CHASE_DISABLE_CHOLQR=1
export CHASE_DISABLE_CHOLQR=1

# 测试参数
N=115000
NCOLS=8000
MB=64
NB=64

# A/B 测试运行
echo "Running Baseline (Low Precision/High Speed Ghost)..."
#srun -n 16 ./build_baseline/examples/5_bse_benchmark/herm_cyclic_dist_chase_run_gpu --path_in=/e/scratch/cjsc/wu7/matrices/herm/eigenvalue_hermitian_jenabse_bse_In2O3_115459_double.bin --n=115459 --nev=1200 --nex=800 --deg=20 --extraDeg=0 --maxDeg=36 --opt=S --tol_d=1e-9 --double=1 --numLanczos=10 --lanczosIter=20 --block_size=64 > baseline_chase.log 2>&1

srun -n 4 ./build_baseline/examples/6_householder_block_cyclic_benchmark/householder_block_cyclic_bench \
  --n $N --ncols $NCOLS --mb $MB --nb $NB --dtype z --iters 5 --warmup 1 > baseline.log 2>&1

echo "Running Strict + HIPREC (High Precision/True Performance)..."
#srun -n 16 ./build_strict/examples/5_bse_benchmark/herm_cyclic_dist_chase_run_gpu --path_in=/e/scratch/cjsc/wu7/matrices/herm/eigenvalue_hermitian_jenabse_bse_In2O3_115459_double.bin --n=115459 --nev=1200 --nex=800 --deg=20 --extraDeg=0 --maxDeg=36 --opt=S --tol_d=1e-9 --double=1 --numLanczos=10 --lanczosIter=20 --block_size=64 > strict_chase.log 2>&1

srun -n 4 ./build_strict/examples/6_householder_block_cyclic_benchmark/householder_block_cyclic_bench \
  --n $N --ncols $NCOLS --mb $MB --nb $NB --dtype z --iters 5 --warmup 1 > strict_hiprec.log 2>&1

# 结果对比汇总
echo "================================================================"
echo "RESULT SUMMARY (N=$N, NCOLS=$NCOLS)"
echo "----------------------------------------------------------------"
printf "%-20s | %-15s | %-15s\n" "MODE" "ORTHO_ERROR" "TIME_MS (FormQ)"
echo "----------------------------------------------------------------"


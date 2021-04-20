import sys
import os
import os.path
import argparse
import numpy as np

def get_mem_CPU(N, n_, m_, block_):

       tot_mem = float(n_*m_ + n_*block_ + m_*block_ + max(m_,n_)*block_ + 3 * N * block_)
       tot_mem *= 16
       tot_mem /= pow(1024,3)

       return tot_mem
	
def get_workspace_heevd(n):
    
    # HEEVD
    lwork_heevd = float(1 + 5*n + 2*pow(n,2))
    lwork_heevd *= 16
    lwork_heevd /= pow(1024,3)

    return lwork_heevd

def get_workspace_qr(n):
   
    # Optimal block size. On modern CPUs it is usually 64 or 128
    nb = 128

    # QR
    lwork_qr = float(n * nb)
    lwork_qr *= 16
    lwork_qr /= pow(1024,3)

    # GQR - xORNQR
    lwork_gqr = float(n * nb)
    lwork_gqr *= 16
    lwork_gqr /= pow(1024,3)

    return lwork_qr + lwork_gqr

def get_mpi_grid(n):

    idealSqrt = np.sqrt(n);
    divisors = [];
    currentDiv = 1;
    for currentDiv in range(n):
        if n % float(currentDiv + 1) == 0:
            divisors.append(currentDiv+1)

    # find the divisior closest to the sqrt(N)
    hIndex = min(range(len(divisors)), key=lambda i: abs(divisors[i]-idealSqrt))

    if (divisors[hIndex] == idealSqrt):
        wIndex = hIndex
    else:
        wIndex = hIndex + 1

    return divisors[hIndex], divisors[wIndex]

def get_mem_GPU(N, n_, m_, block_):
    """ 
        Receives the total matrix size (N) and the number of columns (n_) and rows (m_) of the GPU block
    """

    max_dim = max(m_, n_)

    # HEMM memory
    hemm_mem = float(3 * block_ * max_dim + m_ * n_)
    hemm_mem *= 16
    hemm_mem /= pow(1024,3)

    # QR memory
    qr_mem = float(N * block_)
    qr_mem *= 16
    qr_mem /= pow(1024,3)

    # RR memory
    rr_mem = float(N * block_ + block_ * block_)
    rr_mem *= 16
    rr_mem /= pow(1024,3)

    return hemm_mem + qr_mem + rr_mem


def main():

    # Parse input arguments.

    parser = argparse.ArgumentParser(description='The program computes the main memory requirements per MPI-rank and optionally GPU memory usage, based on the given input values: problem size, number of sough-after eigenpairs, extra search dimension, the number of MPI ranks and GPUs per MPI-rank')
    requiredNamed = parser.add_argument_group('Required arguments')
    requiredNamed.add_argument("--n", metavar="size", help='Size of the input matrix', type=int, required=True)
    requiredNamed.add_argument("--nev", metavar="nev", help='Number of eigenpairs', type=int, required=True)
    requiredNamed.add_argument("--nex", metavar="nex", help='Number of extra search space vectors', type=int, required=True)
    requiredNamed.add_argument("--mpi", metavar="mpi-ranks", help='Number of MPI processes', type=int, required=True)
    requiredNamed.add_argument("--gpus", metavar="num-gpus", help='Number of GPUs per MPI-rank', type=int, default='0')
    args = parser.parse_args()

    # Read inputs
    N = args.n
    nev = args.nev
    nex = args.nex
    comm_size = args.mpi
    gpus = args.gpus

    # Compute MPI grid
    [mpi_row, mpi_col] = get_mpi_grid(comm_size);

    # Compute per-MPI block dimension
    n_ = N / mpi_row
    m_ = N / mpi_col

    # Compute total amount of required memory per MPI rank
    tot_mem = get_mem_CPU(N, n_, m_, nev+nex)

    # Add heevd workspace (in both MPI-only and MPI+GPU heevd kernel is executed on the CPU)
    tot_mem += get_workspace_heevd(nev+nex)

    if (gpus):
        [gpu_row, gpu_col] = get_mpi_grid(gpus);
        gpu_n_ = n_ / gpu_row
        gpu_m_ = m_ / gpu_col
    
        tot_gpu = get_mem_GPU(N, gpu_n_, gpu_m_, nev+nex)
    else:
        # If not using GPU, then workspace arrays for QR and HEEVD are allocated in the main memory
        tot_mem += get_workspace_qr(nev+nex)


    print('\n')
    print('Problem size')
    print("-------------------------------")
    print('Matrix size:   ' + str(N))
    print('Eigenpairs:    ' + str(nev))
    print('Extra vectors: ' + str(nex))

    print("\nMPI configuration")
    print("-------------------------------")
    print('#MPI ranks:    ' + str(comm_size))
    print('MPI grid size: ' + str(mpi_row) + ' x ' + str(mpi_col))
    print('Block size:    ' + str(n_) + ' x ' + str(m_))
    
    if(gpus):
        print('\nGPU configuration per MPI-rank')
        print("-------------------------------")
        print('#GPUs:      ' + str(gpus))
        print('GPU grid:   ' + str(gpu_row) + ' x ' + str(gpu_col))
        print('Block size: ' + str(gpu_n_) + ' x ' + str(gpu_m_))


    print('\n')
    print('Main memory usage per MPI-rank: ' + str(format(tot_mem, '.3f')) + ' GB');
    print('Total main memory usage (' + str(comm_size) + ' ranks): ' + str(format(tot_mem * comm_size, '.3f')) + ' GB');

    if(gpus):
        print('\nMemory requirement per single GPU: ' + str(format(tot_gpu, '.3f')) + ' GB')
        print('GPU memory requirement per MPI-rank (' + str(gpus) + ' GPUs): ' + str(format(tot_gpu * gpus, '.3f')) + ' GB')
    print('\n')

if __name__ == '__main__':
    sys.exit(main())

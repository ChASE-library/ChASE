'''
This file is a part of ChASE.
Copyright (c) 2015-2021, Simulation and Data Laboratory Quantum Materials, 
  Forschungszentrum Juelich GmbH, Germany. All rights reserved.
License is 3-clause BSD:
https://github.com/ChASE-library/ChASE
'''

import sys
import os
import os.path
import argparse
import numpy as np

precision = { 'double': 8, 'float': 4, 'complex': 8, 'dcomplex': 16 }

def get_mem_CPU(N, n_, m_, block_, float_type, data_layout):

       tot_mem = float(n_*m_ + 2 * n_*block_ + 2 * m_*block_ + block_ * block_)

       if data_layout == "block-cyclic":
           tot_mem +=  N * 1

       tot_mem *= precision[float_type]
       tot_mem /= pow(1024,3)

       return tot_mem
	
def get_workspace_heevd(n, float_type):
    
    # HEEVD
    lwork_heevd = float(1 + 5*n + 2*pow(n,2))
    lwork_heevd *= precision[float_type]
    lwork_heevd /= pow(1024,3)

    return lwork_heevd

def get_workspace_qr(n, nb, float_type):
    '''
    Input arguments:
        n - number of columns
        nb - algorithmic block size
    '''

    # QR
    # For cholesky QR, no additional workspace required
    return 0

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

def get_mem_GPU(N, n_, m_, block_, float_type):
    """ 
        Receives the total matrix size (N) and the number of columns (n_) and rows (m_) of the GPU block
    """

    max_dim = max(m_, n_)

    # HEMM memory
    hemm_mem = float(n_*m_ + 2 * n_*block_ + 2 * m_*block_ + block_ * block_ )
    hemm_mem *= precision[float_type]
    hemm_mem /= pow(1024,3)

    # heevd memory
    lwork_heevd = float(1 + 5*block_ + 2*pow(block_,2))
    lwork_heevd *= precision[float_type]
    lwork_heevd /= pow(1024,3)


    return hemm_mem + lwork_heevd


def main():

    # Parse input arguments.

    parser = argparse.ArgumentParser(description='The program computes the main memory requirements per MPI-rank and optionally GPU memory usage, based on the given input values: problem size, number of sough-after eigenpairs, extra search dimension, the number of MPI ranks and GPUs per MPI-rank')
    requiredNamed = parser.add_argument_group('Required arguments')
    requiredNamed.add_argument("--n", metavar="size", help='Size of the input matrix', type=int, required=True)
    requiredNamed.add_argument("--nev", metavar="nev", help='Number of eigenpairs', type=int, required=True)
    requiredNamed.add_argument("--nex", metavar="nex", help='Number of extra search space vectors', type=int, required=True)
    requiredNamed.add_argument("--mpi", metavar="mpi-ranks", help='Number of MPI processes', type=int, required=True)
    parser.add_argument("--gpus", metavar="num gpus", help='Number of GPUs per MPI-rank', type=int, default='0')
    parser.add_argument("--nb", metavar="block size", help='Algoritmic block size for QR and ORMTR', type=int, default=128)
    parser.add_argument("--nrows", metavar="MPI rows", help='Row number of MPI proc grid', type=int, default=0)
    parser.add_argument("--ncols", metavar="MPI cols", help='Column number of MPI proc grid', type=int, default=0)
    parser.add_argument("--type", metavar="Type", help='Numerical type used in the calculations. Possible values: complex, dcomplex, float, double', default='double')
    parser.add_argument("--layout", metavar="Data Layout", help='The data layout of matrix across MPI grid. Possible values: block, block-cyclic', default='block')
    args = parser.parse_args()

    # Read inputs
    N = args.n
    nev = args.nev
    nex = args.nex
    comm_size = args.mpi
    gpus = args.gpus
    mpi_row = args.nrows
    mpi_col = args.ncols
    nb = args.nb
    float_type = args.type
    data_layout = args.layout 

    # Determine floating point type 
    if ( not float_type in precision):
        print('Missing or not supported floating point type: ' + str(float_type))
        exit()


    # Compute MPI grid
    # If row and col dimension is not set, then compute based on
    if (mpi_row == 0 and mpi_col == 0):
        [mpi_row, mpi_col] = get_mpi_grid(comm_size)
    else:
        if (mpi_row == 0 and mpi_col != 0):
            mpi_row = comm_size / mpi_col
        elif (mpi_row != 0 and mpi_col == 0):
            mpi_col = comm_size / mpi_row

    if (mpi_row * mpi_col != comm_size):
        print('The number of MPI row / columns (' + str(mpi_row) + 'x' + str(mpi_col) + ') does not matching the total MPI grid size (' + str(comm_size) +'). Exiting...')
        exit()

    # Compute per-MPI block dimension
    n_ = N / mpi_row
    m_ = N / mpi_col

    # Compute total amount of required memory per MPI rank
    tot_mem = get_mem_CPU(N, n_, m_, nev+nex, float_type, data_layout)

    # Add heevd workspace (in both MPI-only and MPI+GPU heevd kernel is executed on the CPU)
    tot_mem += get_workspace_heevd(nev+nex, float_type)

    if (gpus):
        [gpu_row, gpu_col] = get_mpi_grid(gpus)
        gpu_n_ = n_ / gpu_row
        gpu_m_ = m_ / gpu_col
    
        tot_gpu = get_mem_GPU(N, gpu_n_, gpu_m_, nev+nex, float_type)
    else:
        # If not using GPU, then workspace arrays for QR and HEEVD are allocated in the main memory
        tot_mem += get_workspace_qr(nev+nex, nb, float_type)


    print('\n')
    print('Problem size')
    print("-------------------------------")
    print('Matrix size:   ' + str(N))
    print('Eigenpairs:    ' + str(nev))
    print('Extra vectors: ' + str(nex))
    print('Precision:     ' + float_type + ' (' + str(precision[float_type]) + ' bytes)')

    print("\nMPI configuration")
    print("-------------------------------")
    print('#MPI ranks:    ' + str(comm_size))
    print('MPI grid size: ' + str(mpi_row) + ' x ' + str(mpi_col))
    print('Block size:    ' + str(n_) + ' x ' + str(m_))

    print("\nMatrix Distribution")
    print("-------------------------------")
    print('Data Layout:   ' + data_layout)

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
        print('\nMemory requirement per GPU: ' + str(format(tot_gpu, '.3f')) + ' GB')
        print('Total GPU memory per MPI-rank (' + str(gpus) + ' GPUs): ' + str(format(tot_gpu * gpus, '.3f')) + ' GB')
    print('\n')

if __name__ == '__main__':
    sys.exit(main())

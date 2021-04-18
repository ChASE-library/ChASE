import sys
import os
import os.path
import argparse
import numpy as np

def get_mem_CPU(N, n_, m_, block_):

       tot_mem = n_*m_ + n_*block_ + m_*block_ + max(m_,n_)*block_ + 3 * N * block_
       tot_mem *= 16
       tot_mem /= pow(1024,3)

       return tot_mem
	
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

def main():

    # Parse input arguments.

    parser = argparse.ArgumentParser(description='The program computes the main memory requirements per MPI-rank based on the given input values: problem size, number of sough-after eigenpairs, extra search dimension and the number of MPI ranks')
    requiredNamed = parser.add_argument_group('Required arguments')
    requiredNamed.add_argument("--n", metavar="size", help='Size of the input matrix', type=int, required=True)
    requiredNamed.add_argument("--nev", metavar="nev", help='Number of eigenpairs', type=int, required=True)
    requiredNamed.add_argument("--nex", metavar="nex", help='Number of extra search space vectors', type=int, required=True)
    requiredNamed.add_argument("--mpi", metavar="mpi-ranks", help='Number of MPI processes', type=int, required=True)
    args = parser.parse_args()

    # Read inputs
    N = args.n
    nev = args.nev
    nex = args.nex
    comm_size = args.mpi

    # Compute MPI grid
    [mpi_row, mpi_col] = get_mpi_grid(comm_size);

    print("\n\nGiven configuration:")
    print("-----------------------")
    print("Matrix size:   " + str(N))
    print("Eigenpairs:    " + str(nev))
    print("Extra vectors: " + str(nex))
    print("#MPI procs:    " + str(comm_size))
    print('MPI grid size: ' + str(mpi_row) + ' x ' + str(mpi_col))
    print("-----------------------")


    # Compute per-MPI block dimension
    n_ = N / mpi_row
    m_ = N / mpi_col

    # Compute total amount of required memory per MPI rank
    tot_mem = get_mem_CPU(N, n_, m_, nev+nex)

    print('\n')
    print('Memory usage per MPI-rank: ' + str(tot_mem) + ' GB');
    print('Total memory usage (' + str(comm_size) + ' ranks): ' + str(tot_mem * comm_size) + 'GB');
    print('\n')

if __name__ == '__main__':
    sys.exit(main())

#include <mpi.h>
#include <cuda_runtime.h>
#include <iostream>
#include <mpi-ext.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    bool is_cuda_aware = false;
    is_cuda_aware = (bool) MPIX_Query_cuda_support();
    MPI_Finalize();
}

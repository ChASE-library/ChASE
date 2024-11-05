#include <stdio.h>
#include <mpi.h>
#include <assert.h>
#include <vector>
#include <chrono>

#include <elpa/elpa.h>
#include "algorithm/types.hpp"
#include "grid/mpiGrid2D.hpp"
#include "linalg/distMatrix/distMatrix.hpp"

#define assert_elpa_ok(x) assert(x == ELPA_OK)
//using T = std::complex<double>;
using T = double;

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    MPI_Comm shm_comm;

    //shm_comm = MPI_COMM_WORLD;

    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shm_comm);

    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(MPI_COMM_WORLD);

    int *coords = mpi_grid->get_coords();

    std::size_t N = 10000;
    std::size_t nblk = 64;

    auto H_redundant = chase::distMatrix::RedundantMatrix<T, chase::platform::GPU>(N, N, mpi_grid);
    auto H = chase::distMatrix::BlockCyclicMatrix<T, chase::platform::CPU>(N, N, nblk, nblk, mpi_grid);

    std::vector<T> evecs_host(H.l_rows() * H.l_cols());
    auto evals = chase::distMatrix::RedundantMatrix<chase::Base<T>, chase::platform::CPU>(N, 1, mpi_grid);
    auto evecs = chase::distMatrix::BlockCyclicMatrix<T, chase::platform::GPU>(N, N, H.l_rows(), H.l_cols(), nblk, nblk, H.l_rows(), evecs_host.data(), mpi_grid);

    H_redundant.allocate_cpu_data();

    for(auto i = 0; i < N; i++)
    {   
        H_redundant.cpu_data()[i + i * H_redundant.l_ld()] = T(i);
    }

    std::chrono::high_resolution_clock::time_point start, end;
    std::chrono::duration<double> elapsed;

    start = std::chrono::high_resolution_clock::now();

    H_redundant.redistributeImpl_2(&H);
    
    end = std::chrono::high_resolution_clock::now();

    elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(
        end - start);

    if (mpi_grid->get_myRank() == 0) {
        printf("[%d MPI procs]: redistribution from redundant to blockcyclic in %f seconds! \n", mpi_grid->get_nprocs(), elapsed.count());
    }

    start = std::chrono::high_resolution_clock::now();

    if (elpa_init(ELPA_API_VERSION) != ELPA_OK) {
        fprintf(stderr, "Error: ELPA API version not supported");
        return 1;
    }

    elpa_t handle;
    int error_elpa, value;
    handle = elpa_allocate(&error_elpa);

    elpa_set(handle, "na", (int) N, &error_elpa);
    assert_elpa_ok(error_elpa);

    elpa_set(handle, "nev", (int) N, &error_elpa);
    assert_elpa_ok(error_elpa);

    elpa_set(handle, "local_nrows", (int) H.l_rows(), &error_elpa);
    assert_elpa_ok(error_elpa);

    elpa_set(handle, "local_ncols", (int) H.l_cols(), &error_elpa);
    assert_elpa_ok(error_elpa);

    elpa_set(handle, "nblk", (int) nblk, &error_elpa);
    assert_elpa_ok(error_elpa);

    elpa_set(handle, "mpi_comm_parent", (int)(MPI_Comm_c2f(MPI_COMM_WORLD)), &error_elpa);
    assert_elpa_ok(error_elpa);

    elpa_set(handle, "process_row", (int) coords[0], &error_elpa);
    //elpa_set(handle, "mpi_comm_rows", (int) (MPI_Comm_c2f(mpi_grid->get_col_comm())), &error_elpa);    
    assert_elpa_ok(error_elpa);

    elpa_set(handle, "process_col", (int) coords[1], &error_elpa);
    //elpa_set(handle, "mpi_comm_cols", (int) (MPI_Comm_c2f(mpi_grid->get_row_comm())), &error_elpa);    
    assert_elpa_ok(error_elpa);

    error_elpa = elpa_setup(handle);
    assert_elpa_ok(error_elpa);

    elpa_set(handle, "solver", ELPA_SOLVER_2STAGE, &error_elpa);
    assert_elpa_ok(error_elpa);

    elpa_set(handle, "nvidia-gpu", 1, &error_elpa);
    assert_elpa_ok(error_elpa);

    elpa_set(handle, "real_kernel", ELPA_2STAGE_REAL_NVIDIA_SM80_GPU, &error_elpa);
    //assert_elpa_ok(error_elpa);

    error_elpa = elpa_setup_gpu(handle);
    assert_elpa_ok(error_elpa);

    elpa_get(handle, "solver", &value, &error_elpa);
    assert_elpa_ok(error_elpa);

    int process_row = -1;
    elpa_get(handle, "process_row", &process_row, &error_elpa);
    int process_col = -1;
    elpa_get(handle, "process_col", &process_col, &error_elpa);

    int num_devices = -1;
    cudaGetDeviceCount(&num_devices);
    int shm_rank;
    MPI_Comm_rank(shm_comm, &shm_rank);
    cudaSetDevice(shm_rank);

    end = std::chrono::high_resolution_clock::now();

    elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(
        end - start);

    if (mpi_grid->get_myRank() == 0) {
        printf("[%d MPI procs]: ELPA initialized in %f seconds! \n", mpi_grid->get_nprocs(), elapsed.count());
    }

    start = std::chrono::high_resolution_clock::now();

    elpa_eigenvectors(handle, H.cpu_data(), evals.cpu_data(), evecs.cpu_data(), &error_elpa);
    assert_elpa_ok(error_elpa);

    end = std::chrono::high_resolution_clock::now();

    elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(
        end - start);

    if (mpi_grid->get_myRank() == 0) {
        printf("[%d MPI procs]: SOLVED matrix {na=%d, nev=%d, nblk = %d} with solver %d in %f seconds! \n", mpi_grid->get_nprocs(), (int)N, (int)N, (int)nblk, value, elapsed.count());
    }

    start = std::chrono::high_resolution_clock::now();

    evecs.H2D();
    evecs.redistributeImplAsync(&H_redundant);

    end = std::chrono::high_resolution_clock::now();

    elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(
        end - start);

    if (mpi_grid->get_myRank() == 0) {
        printf("[%d MPI procs]: redistribute from BlockCyclic to Redundant in %f seconds! \n", mpi_grid->get_nprocs(), elapsed.count());
    }

    if(mpi_grid->get_myRank() == 0){
    	printf("Finished Problem\n");
    	printf("Printing first %d eigenvalues and residuals\n", std::min(std::size_t(10), N));
    	printf("Index : Eigenvalues\n");
    	printf("----------------------\n");
    }

    for(auto i = 0; i < std::min(std::size_t(10), N); i++){
        if(mpi_grid->get_myRank() == 0) 
        {
            printf("%d    : %.6e\n",i, evals.cpu_data()[i]);
        }
    }

    MPI_Finalize();
}
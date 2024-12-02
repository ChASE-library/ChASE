#include <stdio.h>
#include <mpi.h>
#include <assert.h>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

#include "algorithm/types.hpp"
#include "grid/mpiGrid2D.hpp"
#include "linalg/matrix/matrix.hpp"
#include "external/cusolverpp/cusolverpp.hpp"
#include "Impl/chase_gpu/cuda_utils.hpp"
//using T = std::complex<double>;
using T = double;

int main(int argc, char** argv)
{
    cusolverDnHandle_t cusolverH = NULL;
    cudaStream_t stream = NULL;

    std::size_t N = 12000;
    std::vector<T> H_host(N * N);
    for(auto i = 0; i < N; i++)
    {   
        H_host.data()[i + i * N] = T(i);
    }

    chase::matrix::Matrix<T, chase::platform::GPU> H(N, N, N, H_host.data());
    chase::matrix::Matrix<chase::Base<T>, chase::platform::GPU> evals(N, 1);

    H.H2D();

    std::chrono::high_resolution_clock::time_point start, end;
    std::chrono::duration<double> elapsed;

    start = std::chrono::high_resolution_clock::now();

    CHECK_CUSOLVER_ERROR(cusolverDnCreate(&cusolverH));
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
    CHECK_CUSOLVER_ERROR(cusolverDnSetStream(cusolverH, stream));

    int lwork = 0;
    T *d_work;
    int *devInfo;
    int info = 0;

    CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void **>(&devInfo), sizeof(int)));

    CHECK_CUSOLVER_ERROR(chase::linalg::cusolverpp::cusolverDnTheevd_bufferSize(
                                                        cusolverH, 
                                                        CUSOLVER_EIG_MODE_VECTOR, 
                                                        CUBLAS_FILL_MODE_LOWER,
                                                        N, 
                                                        H.data(), 
                                                        H.ld(), 
                                                        evals.data(), 
                                                        &lwork));
    
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_work, sizeof(T) * lwork));

    end = std::chrono::high_resolution_clock::now();

    elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(
        end - start);

    printf("cusolver initialized in %f seconds! \n", elapsed.count());

    start = std::chrono::high_resolution_clock::now();

    CHECK_CUSOLVER_ERROR(chase::linalg::cusolverpp::cusolverDnTheevd(
                                    cusolverH, 
                                    CUSOLVER_EIG_MODE_VECTOR, 
                                    CUBLAS_FILL_MODE_LOWER, 
                                    N,
                                    H.data(),
                                    N,
                                    evals.data(),
                                    d_work, lwork, devInfo           
    ));

    end = std::chrono::high_resolution_clock::now();

    elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(
        end - start);

    printf("SOLVED matrix {na=%d, nev=%d} with cusolver in %f seconds! \n", (int)N, (int)N, elapsed.count());

    start = std::chrono::high_resolution_clock::now();

    CHECK_CUDA_ERROR(cudaMemcpy(&info, 
                                devInfo, 
                                1 * sizeof(int),
                                cudaMemcpyDeviceToHost));

    evals.D2H();

    end = std::chrono::high_resolution_clock::now();

    elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(
        end - start);

    printf("cusolver data post-processing in %f seconds! \n", elapsed.count());

    printf("Finished Problem\n");
    printf("Printing first %d eigenvalues and residuals\n", std::min(std::size_t(10), N));
    printf("Index : Eigenvalues\n");
    printf("----------------------\n");
    

    for(auto i = 0; i < std::min(std::size_t(10), N); i++){
        printf("%d    : %.6e\n",i, evals.cpu_data()[i]);
    }

    return 0;
}
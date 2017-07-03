//#pragma once
#ifndef _CHASE_GENERA_BLAS_CHASE_BLAS_FACTORY_
#define _CHASE_GENERA_BLAS_CHASE_BLAS_FACTORY_

#include <assert.h>
#include <mpi.h>

#include <memory>

#include "chase_blas.hpp"
#include "chase_blas_matrices.hpp"

#include "./skewedMatrixProperties.hpp"

#include "../../matrixFree/blas/matrixFreeBlas.hpp"
#include "../../matrixFree/blasInplace/matrixFreeBlasInplace.hpp"
#include "../../matrixFree/blasSkewed/matrixFreeBlas.hpp"
#include "../../matrixFree/cuda/matrixFreeCuda.hpp"
#include "../../matrixFree/cudaSkewed/matrixFreeCuda.hpp"
#include "../../matrixFree/mpi/matrixFreeMPI.hpp"

#include "chase_blas.hpp"
#include "template_wrapper.hpp"

#define MATRIX_FREE_IMPLEMENTATION_MPI_BLAS 1
#define MATRIX_FREE_IMPLEMENTATION_MPI_CUDA 2
#define MATRIX_FREE_IMPLEMENTATION_CUDA 3
#define MATRIX_FREE_IMPLEMENTATION_BLAS_INPLACE 4
#define MATRIX_FREE_IMPLEMENTATION_BLAS 5

#ifndef MATRIX_FREE_IMPLEMENTATION
#define MATRIX_FREE_IMPLEMENTATION MATRIX_FREE_IMPLEMENTATION_MPI_BLAS
#endif

template <class T>
class ChASEFactory {
 public:
  static std::unique_ptr<ChASE_Blas<T>> constructChASE(
      ChASE_Config config, T* V, Base<T>* ritzv,
      MPI_Comm comm = MPI_COMM_WORLD) {
    std::size_t N = config.getN();
    std::size_t max_block = config.getNev() + config.getNex();

    assert(V == NULL);

    auto matrices = ChASE_Blas_Matrices<T>(N, max_block);

#if MATRIX_FREE_IMPLEMENTATION == MATRIX_FREE_IMPLEMENTATION_MPI_CUDA
    auto properties = std::shared_ptr<SkewedMatrixProperties<T>>(
        new SkewedMatrixProperties<T>(N, max_block, comm));

    auto gemm_skewed = std::unique_ptr<MatrixFreeInterface<T>>(
        new MatrixFreeCudaSkewed<T>(properties));

    auto gemm = std::unique_ptr<MatrixFreeInterface<T>>(
        new MatrixFreeMPI<T>(properties, std::move(gemm_skewed)));

#elif MATRIX_FREE_IMPLEMENTATION == MATRIX_FREE_IMPLEMENTATION_MPI_BLAS
    auto properties = std::shared_ptr<SkewedMatrixProperties<T>>(
        new SkewedMatrixProperties<T>(N, max_block, comm));

    auto gemm_skewed = std::unique_ptr<MatrixFreeInterface<T>>(
        new MatrixFreeBlasSkewed<T>(properties));

    auto gemm = std::unique_ptr<MatrixFreeInterface<T>>(
        new MatrixFreeMPI<T>(properties, std::move(gemm_skewed)));

#elif MATRIX_FREE_IMPLEMENTATION == MATRIX_FREE_IMPLEMENTATION_CUDA
    auto gemm = std::unique_ptr<MatrixFreeInterface<T>>(
        new MatrixFreeCuda<T>(matrices.get_H(), N, max_block));

#elif MATRIX_FREE_IMPLEMENTATION == MATRIX_FREE_IMPLEMENTATION_BLAS_INPLACE
    auto gemm = std::unique_ptr<MatrixFreeInterface<T>>(
        new MatrixFreeBlasInplace<T>(matrices.get_H(), matrices.get_V1(),
                                     matrices.get_V2(), N, max_block));
#else
    // Try the simplest MatrixFreeBlas
    auto gemm = std::unique_ptr<MatrixFreeInterface<T>>(
        new MatrixFreeBlas<T>(matrices.get_H(), N, max_block));
#endif

    return std::unique_ptr<ChASE_Blas<T>>(
        new ChASE_Blas<T>(config, std::move(gemm), std::move(matrices)));
  }
};

#endif  // _CHASE_GENERA_BLAS_CHASE_BLAS_FACTORY_

/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*- */

#ifndef _CHASE_GENERA_BLAS_CHASE_BLAS_FACTORY_
#define _CHASE_GENERA_BLAS_CHASE_BLAS_FACTORY_

#include <assert.h>
#include <mpi.h>
#include <memory>

#include "algorithm/chase.h"

#include "genera/matrixfree/chase.h"

#define MATRIX_FREE_IMPLEMENTATION_MPI_BLAS 1
#define MATRIX_FREE_IMPLEMENTATION_MPI_CUDA 2
#define MATRIX_FREE_IMPLEMENTATION_CUDA 3
#define MATRIX_FREE_IMPLEMENTATION_BLAS_INPLACE 4
#define MATRIX_FREE_IMPLEMENTATION_BLAS 5
#define MATRIX_FREE_IMPLEMENTATION_DEBUG 6

#ifndef MATRIX_FREE_IMPLEMENTATION
#define MATRIX_FREE_IMPLEMENTATION MATRIX_FREE_IMPLEMENTATION_BLAS
#endif

#if MATRIX_FREE_IMPLEMENTATION == MATRIX_FREE_IMPLEMENTATION_MPI_CUDA
#include "genera/matrixfree/impl/cudaSkewed/matrixFreeCuda.hpp"
#include "genera/matrixfree/impl/mpi/matrixFreeMPI.hpp"

#elif MATRIX_FREE_IMPLEMENTATION == MATRIX_FREE_IMPLEMENTATION_MPI_BLAS
#include "genera/matrixfree/impl/blasSkewed/matrixFreeBlas.hpp"
#include "genera/matrixfree/impl/mpi/matrixFreeMPI.hpp"

#elif MATRIX_FREE_IMPLEMENTATION == MATRIX_FREE_IMPLEMENTATION_CUDA
#include "genera/matrixfree/impl/cuda/matrixFreeCuda.hpp"

#elif MATRIX_FREE_IMPLEMENTATION == MATRIX_FREE_IMPLEMENTATION_BLAS_INPLACE
#include "genera/matrixfree/impl/blasInplace/matrixFreeBlasInplace.hpp"

#elif MATRIX_FREE_IMPLEMENTATION == MATRIX_FREE_IMPLEMENTATION_BLAS
#include "genera/matrixfree/impl/blas/matrixFreeBlas.hpp"

#elif MATRIX_FREE_IMPLEMENTATION == MATRIX_FREE_IMPLEMENTATION_DEBUG
#include "genera/matrixfree/impl/debug/matrixFreeDebug.hpp"

#else

#endif

namespace chase {

using namespace chase::matrixfree;

template <typename T>
static std::unique_ptr<MatrixFreeChase<T>> constructChASE(
    ChaseConfig<T>& config, T *H, T *V, Base<T> *ritzv,
    MPI_Comm comm = MPI_COMM_WORLD) {
  std::size_t N = config.getN();
  std::size_t max_block = config.getNev() + config.getNex();

#if MATRIX_FREE_IMPLEMENTATION == MATRIX_FREE_IMPLEMENTATION_MPI_CUDA || \
    MATRIX_FREE_IMPLEMENTATION == MATRIX_FREE_IMPLEMENTATION_MPI_BLAS

  assert(H == nullptr);

  auto properties = std::shared_ptr<SkewedMatrixProperties<T>>(
      new SkewedMatrixProperties<T>(N, max_block, comm));

  auto matrices = ChASE_Blas_Matrices<T>(N, max_block, V, ritzv, properties);
#else
  auto matrices = ChASE_Blas_Matrices<T>(N, max_block, V, ritzv);
#endif

#if MATRIX_FREE_IMPLEMENTATION == MATRIX_FREE_IMPLEMENTATION_MPI_CUDA
  auto gemm_skewed = std::unique_ptr<MatrixFreeInterface<T>>(
      new MatrixFreeCudaSkewed<T>(properties));

  auto gemm = std::unique_ptr<MatrixFreeInterface<T>>(
      new MatrixFreeMPI<T>(properties, std::move(gemm_skewed)));

#elif MATRIX_FREE_IMPLEMENTATION == MATRIX_FREE_IMPLEMENTATION_MPI_BLAS
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

#elif MATRIX_FREE_IMPLEMENTATION == MATRIX_FREE_IMPLEMENTATION_DEBUG
  auto gemm = std::unique_ptr<MatrixFreeInterface<T>>(
      new MatrixFreeDebug<T>(matrices.get_H(), N, max_block));
#else
  // Try the simplest MatrixFreeBlas
  auto gemm = std::unique_ptr<MatrixFreeInterface<T>>(
      new MatrixFreeBlas<T>(matrices.get_H(), N, max_block));
#endif

  return std::unique_ptr<MatrixFreeChase<T>>(
      new MatrixFreeChase<T>(config, std::move(gemm), std::move(matrices)));
}
}
#endif  // _CHASE_GENERA_BLAS_CHASE_BLAS_FACTORY_

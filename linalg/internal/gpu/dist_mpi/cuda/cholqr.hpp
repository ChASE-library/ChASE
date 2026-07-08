// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "linalg/internal/gpu/dist_mpi/cuda/cuda_mpi_kernels.hpp"
#include "linalg/internal/gpu/dist_mpi/impl/cholqr_impl.hpp"
#include "linalg/internal/gpu/backend/types.hpp"

namespace chase
{
namespace linalg
{
namespace internal
{
template <typename T>
int cuda_mpi::cholQR1(cublasHandle_t blas_handle,
                      cusolverDnHandle_t solver_handle, std::size_t m,
                      std::size_t n, T* V, int ldv, MPI_Comm comm, T* workspace,
                      int lwork, T* A)
{
    return gpu_mpi::chol_qr1_impl<gpu_backend::cuda_backend,
                                        cublasHandle_t, cusolverDnHandle_t, T>(
        blas_handle, solver_handle, m, n, V, ldv, comm, workspace, lwork, A);
}

template <typename InputMultiVectorType>
int cuda_mpi::cholQR1(cublasHandle_t blas_handle,
                      cusolverDnHandle_t solver_handle,
                      InputMultiVectorType& V,
                      typename InputMultiVectorType::value_type* workspace,
                      int lwork, typename InputMultiVectorType::value_type* A)
{
    return gpu_mpi::chol_qr1_impl<gpu_backend::cuda_backend,
                                        cublasHandle_t, cusolverDnHandle_t,
                                        InputMultiVectorType>(
        blas_handle, solver_handle, V, workspace, lwork, A);
}

template <typename T>
int cuda_mpi::cholQR2(cublasHandle_t blas_handle,
                      cusolverDnHandle_t solver_handle, std::size_t m,
                      std::size_t n, T* V, int ldv, MPI_Comm comm, T* workspace,
                      int lwork, T* A)
{
    return gpu_mpi::chol_qr2_impl<gpu_backend::cuda_backend,
                                        cublasHandle_t, cusolverDnHandle_t, T>(
        blas_handle, solver_handle, m, n, V, ldv, comm, workspace, lwork, A);
}

template <typename InputMultiVectorType>
int cuda_mpi::cholQR2(cublasHandle_t blas_handle,
                      cusolverDnHandle_t solver_handle,
                      InputMultiVectorType& V,
                      typename InputMultiVectorType::value_type* workspace,
                      int lwork, typename InputMultiVectorType::value_type* A)
{
    return gpu_mpi::chol_qr2_impl<gpu_backend::cuda_backend,
                                        cublasHandle_t, cusolverDnHandle_t,
                                        InputMultiVectorType>(
        blas_handle, solver_handle, V, workspace, lwork, A);
}

template <typename T>
int cuda_mpi::shiftedcholQR2(cublasHandle_t blas_handle,
                             cusolverDnHandle_t solver_handle, std::size_t N,
                             std::size_t m, std::size_t n, T* V, int ldv,
                             MPI_Comm comm, T* workspace, int lwork, T* A)
{
    return gpu_mpi::shifted_chol_qr2_impl<gpu_backend::cuda_backend,
                                               cublasHandle_t,
                                               cusolverDnHandle_t, T>(
        blas_handle, solver_handle, N, m, n, V, ldv, comm, workspace, lwork,
        A);
}

template <typename T>
int cuda_mpi::modifiedGramSchmidtCholQR(
    cublasHandle_t blas_handle, cusolverDnHandle_t solver_handle, std::size_t m,
    std::size_t n, std::size_t locked, T* V, std::size_t ldv, MPI_Comm comm,
    T* workspace, int lwork, T* A)
{
    return gpu_mpi::modified_gram_schmidt_chol_qr_impl<
        gpu_backend::cuda_backend, cublasHandle_t, cusolverDnHandle_t, T>(
        blas_handle, solver_handle, m, n, locked, V, ldv, comm, workspace,
        lwork, A);
}

template <typename InputMultiVectorType>
void cuda_mpi::houseHoulderQR(InputMultiVectorType& V)
{
    gpu_mpi::householder_qr_impl(V);
}

} // namespace internal
} // namespace linalg
} // namespace chase

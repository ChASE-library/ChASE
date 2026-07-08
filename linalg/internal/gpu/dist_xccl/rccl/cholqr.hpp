// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "linalg/internal/gpu/dist_mpi/impl/cholqr_impl.hpp"
#include "linalg/internal/gpu/dist_xccl/impl/cholqr_impl.hpp"
#include "linalg/internal/gpu/backend/types.hpp"
#include "linalg/internal/gpu/dist_xccl/rccl/rccl_kernels.hpp"

namespace chase
{
namespace linalg
{
namespace internal
{

template <typename T>
int hip_rccl::cholQR1(rocblas_handle blas_handle,
                      rocblas_handle solver_handle, std::size_t m,
                      std::size_t n, T* V, int ldv, ncclComm_t comm,
                      T* workspace, int lwork, T* A)
{
    return gpu_xccl::chol_qr1_impl<gpu_backend::hip_backend, rocblas_handle,
                                   rocblas_handle, T>(
        blas_handle, solver_handle, m, n, V, ldv, comm, workspace, lwork, A);
}

template <typename InputMultiVectorType>
int hip_rccl::cholQR1(rocblas_handle blas_handle,
                      rocblas_handle solver_handle, InputMultiVectorType& V,
                      typename InputMultiVectorType::value_type* workspace,
                      int lwork,
                      typename InputMultiVectorType::value_type* A)
{
    return gpu_xccl::chol_qr1_impl<gpu_backend::hip_backend, rocblas_handle,
                                   rocblas_handle, InputMultiVectorType>(
        blas_handle, solver_handle, V, workspace, lwork, A);
}

template <typename T>
int hip_rccl::cholQR2(rocblas_handle blas_handle,
                      rocblas_handle solver_handle, std::size_t m,
                      std::size_t n, T* V, int ldv, ncclComm_t comm,
                      T* workspace, int lwork, T* A)
{
    return gpu_xccl::chol_qr2_impl<gpu_backend::hip_backend, rocblas_handle,
                                   rocblas_handle, T>(
        blas_handle, solver_handle, m, n, V, ldv, comm, workspace, lwork, A);
}

template <typename InputMultiVectorType>
int hip_rccl::cholQR2(rocblas_handle blas_handle,
                      rocblas_handle solver_handle, InputMultiVectorType& V,
                      typename InputMultiVectorType::value_type* workspace,
                      int lwork,
                      typename InputMultiVectorType::value_type* A)
{
    return gpu_xccl::chol_qr2_impl<gpu_backend::hip_backend, rocblas_handle,
                                   rocblas_handle, InputMultiVectorType>(
        blas_handle, solver_handle, V, workspace, lwork, A);
}

template <typename T>
int hip_rccl::shiftedcholQR2(rocblas_handle blas_handle,
                             rocblas_handle solver_handle, std::size_t N,
                             std::size_t m, std::size_t n, T* V, int ldv,
                             ncclComm_t comm, T* workspace, int lwork, T* A)
{
    return gpu_xccl::shifted_chol_qr2_impl<
        gpu_backend::hip_backend, rocblas_handle, rocblas_handle, T>(
        blas_handle, solver_handle, N, m, n, V, ldv, comm, workspace, lwork, A);
}

template <typename T>
int hip_rccl::modifiedGramSchmidtCholQR(
    rocblas_handle blas_handle, rocblas_handle solver_handle, std::size_t m,
    std::size_t n, std::size_t locked, T* V, std::size_t ldv, ncclComm_t comm,
    T* workspace, int lwork, T* A)
{
    return gpu_xccl::modified_gram_schmidt_chol_qr_impl<
        gpu_backend::hip_backend, rocblas_handle, rocblas_handle, T>(
        blas_handle, solver_handle, m, n, locked, V, ldv, comm, workspace,
        lwork, A);
}

template <typename InputMultiVectorType>
void hip_rccl::houseHoulderQR(InputMultiVectorType& V)
{
    gpu_mpi::householder_qr_impl(V);
}

} // namespace internal
} // namespace linalg
} // namespace chase

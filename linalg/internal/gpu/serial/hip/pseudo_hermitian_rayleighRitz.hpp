// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "linalg/internal/gpu/backend/types.hpp"
#include "linalg/internal/gpu/serial/impl/pseudo_hermitian_rayleigh_ritz.hpp"
#include "linalg/matrix/matrix.hpp"

#include <rocblas/rocblas.h>

namespace chase
{
namespace linalg
{
namespace internal
{
namespace hip
{

/**
 * @brief Pseudo-Hermitian Rayleigh–Ritz (POTRF + HEEVD v2 path).
 */
template <typename T>
void rayleighRitz_v2(
    rocblas_handle handle,
    ::chase::matrix::PseudoHermitianMatrix<T, ::chase::platform::GPU>* H,
    ::chase::matrix::Matrix<T, ::chase::platform::GPU>& V1,
    ::chase::matrix::Matrix<T, ::chase::platform::GPU>& V2,
    ::chase::matrix::Matrix<::chase::Base<T>, ::chase::platform::GPU>& ritzv,
    std::size_t offset, std::size_t subSize, rocblas_int* devInfo,
    ::chase::matrix::Matrix<T, ::chase::platform::GPU>* A = nullptr,
    T* workspace = nullptr, int lwork = 0)
{
    gpu_serial::impl::pseudo_hermitian_rayleighRitz_v2<gpu_backend::hip_backend,
                                                       T>(
        handle, handle, H, V1, V2, ritzv, offset, subSize, devInfo, A,
        workspace, lwork);
}

// ChASEGPU adapter overloads; HIP uses POTRF + HEEVD and ignores params/h_work.
template <typename T>
void rayleighRitz(
    rocblas_handle blas_handle, rocblas_handle solver_handle,
    rocblas_handle /*params*/,
    ::chase::matrix::PseudoHermitianMatrix<T, ::chase::platform::GPU>* H,
    ::chase::matrix::Matrix<T, ::chase::platform::GPU>& V1,
    ::chase::matrix::Matrix<T, ::chase::platform::GPU>& V2,
    ::chase::matrix::Matrix<::chase::Base<T>, ::chase::platform::GPU>& ritzv,
    std::size_t offset, std::size_t subSize, int* devInfo,
    T* d_workspace = nullptr, int d_lwork = 0, T* /*h_workspace*/ = nullptr,
    int /*h_lwork*/ = 0,
    ::chase::matrix::Matrix<T, ::chase::platform::GPU>* A = nullptr)
{
    gpu_serial::impl::pseudo_hermitian_rayleighRitz_v2<gpu_backend::hip_backend,
                                                       T>(
        blas_handle, solver_handle, H, V1, V2, ritzv, offset, subSize,
        reinterpret_cast<rocblas_int*>(devInfo), A, d_workspace, d_lwork);
}

template <typename T>
void rayleighRitz_v2(
    rocblas_handle blas_handle, rocblas_handle solver_handle,
    rocblas_handle /*params*/,
    ::chase::matrix::PseudoHermitianMatrix<T, ::chase::platform::GPU>* H,
    ::chase::matrix::Matrix<T, ::chase::platform::GPU>& V1,
    ::chase::matrix::Matrix<T, ::chase::platform::GPU>& V2,
    ::chase::matrix::Matrix<::chase::Base<T>, ::chase::platform::GPU>& ritzv,
    std::size_t offset, std::size_t subSize, int* devInfo,
    T* workspace = nullptr, int lwork = 0,
    ::chase::matrix::Matrix<T, ::chase::platform::GPU>* A = nullptr)
{
    gpu_serial::impl::pseudo_hermitian_rayleighRitz_v2<gpu_backend::hip_backend,
                                                       T>(
        blas_handle, solver_handle, H, V1, V2, ritzv, offset, subSize,
        reinterpret_cast<rocblas_int*>(devInfo), A, workspace, lwork);
}

} // namespace hip
} // namespace internal
} // namespace linalg
} // namespace chase

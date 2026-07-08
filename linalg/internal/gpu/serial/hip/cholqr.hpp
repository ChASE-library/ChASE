// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "linalg/internal/gpu/backend/types.hpp"
#include "linalg/internal/gpu/serial/impl/cholqr.hpp"
#include "linalg/matrix/matrix.hpp"

#include <cstddef>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

namespace chase
{
namespace linalg
{
namespace internal
{
namespace hip
{

template <typename T>
int cholQR1(rocblas_handle blas_handle, rocblas_handle solver_handle,
            chase::matrix::Matrix<T, chase::platform::GPU>& V,
            T* workspace = nullptr, int lwork = 0,
            chase::matrix::Matrix<T, chase::platform::GPU>* A = nullptr,
            int* external_devInfo = nullptr)
{
    return gpu_serial::impl::cholQR1<gpu_backend::hip_backend, T>(
        blas_handle, solver_handle, V, A,
        reinterpret_cast<rocblas_int*>(external_devInfo), workspace, lwork);
}

template <typename T>
int cholQR2(rocblas_handle blas_handle, rocblas_handle solver_handle,
            chase::matrix::Matrix<T, chase::platform::GPU>& V,
            T* workspace = nullptr, int lwork = 0,
            chase::matrix::Matrix<T, chase::platform::GPU>* A = nullptr,
            int* external_devInfo = nullptr)
{
    return gpu_serial::impl::cholQR2<gpu_backend::hip_backend, T>(
        blas_handle, solver_handle, V, A,
        reinterpret_cast<rocblas_int*>(external_devInfo), workspace, lwork);
}

template <typename T>
void houseHoulderQR(rocblas_handle solver_handle,
                    chase::matrix::Matrix<T, chase::platform::GPU>& V,
                    T* d_tau, int* devInfo, T* workspace = nullptr,
                    int lwork = 0)
{
    gpu_serial::impl::houseHoulderQR<gpu_backend::hip_backend, T>(
        solver_handle, V, d_tau, reinterpret_cast<rocblas_int*>(devInfo),
        workspace, lwork);
}

template <typename T>
int shiftedcholQR2(rocblas_handle blas_handle, rocblas_handle solver_handle,
                   chase::matrix::Matrix<T, chase::platform::GPU>& V,
                   T* workspace = nullptr, int lwork = 0,
                   chase::matrix::Matrix<T, chase::platform::GPU>* A = nullptr,
                   int* external_devInfo = nullptr)
{
    return gpu_serial::impl::shiftedcholQR2<gpu_backend::hip_backend, T>(
        blas_handle, solver_handle, V, A,
        reinterpret_cast<rocblas_int*>(external_devInfo), workspace, lwork);
}

} // namespace hip
} // namespace internal
} // namespace linalg
} // namespace chase

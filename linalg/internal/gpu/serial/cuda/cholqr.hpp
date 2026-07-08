// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "Impl/chase_gpu/nvtx.hpp"
#include "linalg/internal/gpu/backend/types.hpp"
#include "linalg/internal/gpu/serial/impl/cholqr.hpp"
#include "linalg/matrix/matrix.hpp"

#include <cublas_v2.h>
#include <cusolverDn.h>

namespace chase
{
namespace linalg
{
namespace internal
{
namespace cuda
{

template <typename T>
int cholQR1(cublasHandle_t cublas_handle, cusolverDnHandle_t cusolver_handle,
            chase::matrix::Matrix<T, chase::platform::GPU>& V,
            T* workspace = nullptr, int lwork = 0,
            chase::matrix::Matrix<T, chase::platform::GPU>* A = nullptr,
            int* external_devInfo = nullptr)
{
    SCOPED_NVTX_RANGE();
    return gpu_serial::impl::cholQR1<gpu_backend::cuda_backend, T>(
        cublas_handle, cusolver_handle, V, A, external_devInfo, workspace,
        lwork);
}

template <typename T>
int cholQR2(cublasHandle_t cublas_handle, cusolverDnHandle_t cusolver_handle,
            chase::matrix::Matrix<T, chase::platform::GPU>& V,
            T* workspace = nullptr, int lwork = 0,
            chase::matrix::Matrix<T, chase::platform::GPU>* A = nullptr,
            int* external_devInfo = nullptr)
{
    SCOPED_NVTX_RANGE();
    return gpu_serial::impl::cholQR2<gpu_backend::cuda_backend, T>(
        cublas_handle, cusolver_handle, V, A, external_devInfo, workspace,
        lwork);
}

template <typename T>
void houseHoulderQR(cusolverDnHandle_t cusolver_handle,
                    chase::matrix::Matrix<T, chase::platform::GPU>& V, T* d_tau,
                    int* devInfo, T* workspace = nullptr, int lwork = 0)
{
    SCOPED_NVTX_RANGE();
    gpu_serial::impl::houseHoulderQR<gpu_backend::cuda_backend, T>(
        cusolver_handle, V, d_tau, devInfo, workspace, lwork);
}

template <typename T>
int shiftedcholQR2(cublasHandle_t cublas_handle,
                   cusolverDnHandle_t cusolver_handle,
                   chase::matrix::Matrix<T, chase::platform::GPU>& V,
                   T* workspace = nullptr, int lwork = 0,
                   chase::matrix::Matrix<T, chase::platform::GPU>* A = nullptr,
                   int* external_devInfo = nullptr)
{
    SCOPED_NVTX_RANGE();
    return gpu_serial::impl::shiftedcholQR2<gpu_backend::cuda_backend, T>(
        cublas_handle, cusolver_handle, V, A, external_devInfo, workspace,
        lwork);
}

} // namespace cuda
} // namespace internal
} // namespace linalg
} // namespace chase

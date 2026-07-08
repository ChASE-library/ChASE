// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "Impl/chase_gpu/nvtx.hpp"
#include "linalg/internal/gpu/backend/types.hpp"
#include "linalg/internal/gpu/serial/impl/lanczos.hpp"
#include "linalg/matrix/matrix.hpp"

#include <cublas_v2.h>

namespace chase
{
namespace linalg
{
namespace internal
{
namespace cuda
{

template <typename T>
void lanczos(cublasHandle_t handle, std::size_t M, std::size_t numvec,
             chase::matrix::Matrix<T, chase::platform::GPU>* H,
             chase::matrix::Matrix<T, chase::platform::GPU>& V,
             chase::Base<T>* upperb, chase::Base<T>* ritzv, chase::Base<T>* Tau,
             chase::Base<T>* ritzV)
{
    SCOPED_NVTX_RANGE();
    gpu_serial::impl::lanczos<gpu_backend::cuda_backend, T>(
        handle, M, numvec, H, V, upperb, ritzv, Tau, ritzV);
}

template <typename T>
void lanczos(cublasHandle_t handle, std::size_t M,
             chase::matrix::Matrix<T, chase::platform::GPU>* H,
             chase::matrix::Matrix<T, chase::platform::GPU>& V,
             chase::Base<T>* upperb)
{
    SCOPED_NVTX_RANGE();
    gpu_serial::impl::lanczos<gpu_backend::cuda_backend, T>(handle, M, H, V,
                                                             upperb);
}

template <typename T>
void lanczos(cublasHandle_t handle, std::size_t M, std::size_t numvec,
             chase::matrix::PseudoHermitianMatrix<T, chase::platform::GPU>* H,
             chase::matrix::Matrix<T, chase::platform::GPU>& V,
             chase::Base<T>* upperb, chase::Base<T>* ritzv, chase::Base<T>* Tau,
             chase::Base<T>* ritzV)
{
    SCOPED_NVTX_RANGE();
    gpu_serial::impl::lanczos<gpu_backend::cuda_backend, T>(
        handle, M, numvec, H, V, upperb, ritzv, Tau, ritzV);
}

template <typename T>
void lanczos(cublasHandle_t handle, std::size_t M,
             chase::matrix::PseudoHermitianMatrix<T, chase::platform::GPU>* H,
             chase::matrix::Matrix<T, chase::platform::GPU>& V,
             chase::Base<T>* upperb)
{
    SCOPED_NVTX_RANGE();
    gpu_serial::impl::lanczos<gpu_backend::cuda_backend, T>(handle, M, H, V,
                                                             upperb);
}

} // namespace cuda
} // namespace internal
} // namespace linalg
} // namespace chase

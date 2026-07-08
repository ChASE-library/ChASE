// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "linalg/internal/gpu/dist_xccl/impl/lanczos_impl.hpp"
#include "linalg/internal/gpu/backend/types.hpp"
#include "linalg/internal/gpu/dist_xccl/nccl/nccl_kernels.hpp"

namespace chase
{
namespace linalg
{
namespace internal
{

template <typename MatrixType, typename InputMultiVectorType>
void cuda_nccl::lanczos_dispatch(
    cublasHandle_t blas_handle, std::size_t M, std::size_t numvec,
    MatrixType& H, InputMultiVectorType& V,
    chase::Base<typename MatrixType::value_type>* upperb,
    chase::Base<typename MatrixType::value_type>* ritzv,
    chase::Base<typename MatrixType::value_type>* Tau,
    chase::Base<typename MatrixType::value_type>* ritzV)
{
    gpu_xccl::lanczos_dispatch_impl<gpu_backend::cuda_backend,
                                          cublasHandle_t, cuda_nccl>(
        blas_handle, M, numvec, H, V, upperb, ritzv, Tau, ritzV);
}

template <typename MatrixType, typename InputMultiVectorType>
void cuda_nccl::lanczos_dispatch(
    cublasHandle_t blas_handle, std::size_t M, MatrixType& H,
    InputMultiVectorType& V,
    chase::Base<typename MatrixType::value_type>* upperb)
{
    gpu_xccl::lanczos_dispatch_impl<gpu_backend::cuda_backend,
                                          cublasHandle_t, cuda_nccl>(
        blas_handle, M, H, V, upperb);
}

template <typename MatrixType, typename InputMultiVectorType>
void cuda_nccl::lanczos(cublasHandle_t blas_handle, std::size_t M,
                        std::size_t numvec, MatrixType& H,
                        InputMultiVectorType& V,
                        chase::Base<typename MatrixType::value_type>* upperb,
                        chase::Base<typename MatrixType::value_type>* ritzv,
                        chase::Base<typename MatrixType::value_type>* Tau,
                        chase::Base<typename MatrixType::value_type>* ritzV)
{
    gpu_xccl::lanczos_impl<gpu_backend::cuda_backend, cublasHandle_t,
                                 cuda_nccl>(
        blas_handle, M, numvec, H, V, upperb, ritzv, Tau, ritzV);
}

template <typename MatrixType, typename InputMultiVectorType>
void cuda_nccl::lanczos(cublasHandle_t blas_handle, std::size_t M,
                        MatrixType& H, InputMultiVectorType& V,
                        chase::Base<typename MatrixType::value_type>* upperb)
{
    gpu_xccl::lanczos_impl<gpu_backend::cuda_backend, cublasHandle_t,
                                 cuda_nccl>(blas_handle, M, H, V, upperb);
}

} // namespace internal
} // namespace linalg
} // namespace chase

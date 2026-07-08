// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "linalg/internal/gpu/backend/types.hpp"
#include "linalg/internal/gpu/dist_xccl/impl/pseudo_hermitian_lanczos_impl.hpp"
#include "linalg/internal/gpu/dist_xccl/rccl/rccl_kernels.hpp"

namespace chase
{
namespace linalg
{
namespace internal
{
template <typename MatrixType, typename InputMultiVectorType>
void hip_rccl::pseudo_hermitian_lanczos(
    rocblas_handle blas_handle, std::size_t M, std::size_t numvec,
    MatrixType& H, InputMultiVectorType& V,
    chase::Base<typename MatrixType::value_type>* upperb,
    chase::Base<typename MatrixType::value_type>* ritzv,
    chase::Base<typename MatrixType::value_type>* Tau,
    chase::Base<typename MatrixType::value_type>* ritzV)
{
    gpu_xccl::pseudo_hermitian_lanczos_impl<
        gpu_backend::hip_backend, rocblas_handle, hip_rccl>(
        blas_handle, M, numvec, H, V, upperb, ritzv, Tau, ritzV);
}

template <typename MatrixType, typename InputMultiVectorType>
void hip_rccl::pseudo_hermitian_lanczos(
    rocblas_handle blas_handle, std::size_t M, MatrixType& H,
    InputMultiVectorType& V,
    chase::Base<typename MatrixType::value_type>* upperb)
{
    gpu_xccl::pseudo_hermitian_lanczos_impl<
        gpu_backend::hip_backend, rocblas_handle, hip_rccl>(
        blas_handle, M, H, V, upperb);
}

} // namespace internal
} // namespace linalg
} // namespace chase

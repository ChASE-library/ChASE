// This file is a part of ChASE.
// Copyright (c) 2015-2024, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "Impl/chase_gpu/nvtx.hpp"
#include "absTrace.cuh"
#include "algorithm/types.hpp"
#include "linalg/matrix/matrix.hpp"

namespace chase
{
namespace linalg
{
namespace internal
{
namespace cuda
{
/**
 * @brief Computes the absolute trace of a matrix on the GPU.
 *
 * This function computes the absolute trace of a matrix on the GPU. The trace
 * is calculated by summing the absolute values of the diagonal elements. It
 * utilizes the `absTrace_gpu` function to launch the corresponding CUDA kernel
 * based on the matrix type.
 *
 * @tparam T The type of elements in the matrix (e.g., float, double, complex
 * types).
 * @param[in] H The matrix whose absolute trace is to be computed. This matrix
 * should reside in GPU memory.
 * @param[out] absTrace The resulting absolute trace of the matrix.
 * @param[in] stream_ The optional CUDA stream to be used for kernel execution.
 * If not provided, the default stream is used.
 *
 * @note This function uses `SCOPED_NVTX_RANGE()` to mark the execution range
 * for profiling purposes using NVIDIA's NVTX.
 */
template <typename T>
void absTrace(chase::matrix::Matrix<T, chase::platform::GPU>& H,
              chase::Base<T>* absTrace, cudaStream_t* stream_ = nullptr)
{
    SCOPED_NVTX_RANGE();

    cudaStream_t usedStream = (stream_ == nullptr) ? 0 : *stream_;
    std::size_t n = std::min(H.rows(), H.cols());
    absTrace_gpu(H.data(), absTrace, n, H.ld(), usedStream);
}
} // namespace cuda
} // namespace internal
} // namespace linalg
} // namespace chase
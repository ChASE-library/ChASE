#pragma once

#include "absTrace.cuh"
#include "linalg/matrix/matrix.hpp"
#include "algorithm/types.hpp"
#include "Impl/cuda/nvtx.hpp"

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
     * This function computes the absolute trace of a matrix on the GPU. The trace is calculated by summing the
     * absolute values of the diagonal elements. It utilizes the `absTrace_gpu` function to launch the corresponding 
     * CUDA kernel based on the matrix type.
     *
     * @tparam T The type of elements in the matrix (e.g., float, double, complex types).
     * @param[in] H The matrix whose absolute trace is to be computed. This matrix should reside in GPU memory.
     * @param[out] absTrace The resulting absolute trace of the matrix.
     * @param[in] stream_ The optional CUDA stream to be used for kernel execution. If not provided, the default stream is used.
     *
     * @note This function uses `SCOPED_NVTX_RANGE()` to mark the execution range for profiling purposes using NVIDIA's NVTX.
     */    
    template<typename T>
    void absTrace(chase::matrix::Matrix<T, chase::platform::GPU>& H, chase::Base<T> *absTrace, cudaStream_t* stream_ = nullptr)
    {
        SCOPED_NVTX_RANGE();

        cudaStream_t usedStream = (stream_ == nullptr) ? 0 : *stream_;
        std::size_t n = std::min(H.rows(), H.cols());
        absTrace_gpu(H.data(), absTrace, n, H.ld(), usedStream);
    }
}
}
}
}
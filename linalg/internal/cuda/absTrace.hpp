#pragma once

#include "absTrace.cuh"
#include "linalg/matrix/matrix.hpp"
#include "algorithm/types.hpp"

namespace chase
{
namespace linalg
{
namespace internal
{
namespace cuda
{
    template<typename T>
    void absTrace(chase::matrix::MatrixGPU<T>& H, chase::Base<T> *absTrace, cudaStream_t* stream_ = nullptr)
    {
        cudaStream_t usedStream = (stream_ == nullptr) ? 0 : *stream_;
        std::size_t n = std::min(H.rows(), H.cols());
        absTrace_gpu(H.gpu_data(), absTrace, n, H.gpu_ld(), usedStream);
    }
}
}
}
}
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
#pragma once

#include "shiftDiagonal.cuh"
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
    void shiftDiagonal(chase::matrix::Matrix<T, chase::platform::GPU>& H, chase::Base<T> shift, cudaStream_t* stream_ = nullptr)
    {
        SCOPED_NVTX_RANGE();

        cudaStream_t usedStream = (stream_ == nullptr) ? 0 : *stream_;
        std::size_t n = std::min(H.rows(), H.cols());
        chase_shift_matrix(H.data(), n, H.ld(), shift, usedStream);
    }
}
}
}
}
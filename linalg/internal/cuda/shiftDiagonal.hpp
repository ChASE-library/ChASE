#pragma once

#include "shiftDiagonal.cuh"
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
    void shiftDiagonal(chase::matrix::MatrixGPU<T>& H, chase::Base<T> shift, cudaStream_t* stream_ = nullptr)
    {
        cudaStream_t usedStream = (stream_ == nullptr) ? 0 : *stream_;
        std::size_t n = std::min(H.rows(), H.cols());
        chase_shift_matrix(H.gpu_data(), n, H.gpu_ld(), shift, usedStream);
    }
}
}
}
}
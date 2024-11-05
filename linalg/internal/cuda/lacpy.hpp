#pragma once

#include "lacpy.cuh"
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
    void t_lacpy(char uplo, std::size_t m, std::size_t n, T *dA, std::size_t ldda, T *dB, std::size_t lddb, cudaStream_t *stream_ = nullptr )
    {
        SCOPED_NVTX_RANGE();

        cudaStream_t usedStream = (stream_ == nullptr) ? 0 : *stream_;
        t_lacpy_gpu(uplo, m, n, dA, ldda, dB, lddb, usedStream);
    }

    template<typename T>
    void extractUpperTriangular(T* d_matrix, std::size_t ld, T* d_upperTriangular, std::size_t n, cudaStream_t *stream_ = nullptr )
    {
        cudaStream_t usedStream = (stream_ == nullptr) ? 0 : *stream_;
        extractUpperTriangular(d_matrix, ld, d_upperTriangular, n, usedStream);
    }

    template<typename T>
    void unpackUpperTriangular(T* d_upperTriangular, std::size_t n, T* d_matrix, std::size_t ld, cudaStream_t *stream_ = nullptr )
    {
        cudaStream_t usedStream = (stream_ == nullptr) ? 0 : *stream_;
        unpackUpperTriangular(d_matrix, ld, d_upperTriangular, n, usedStream);
    }



}
}
}
}
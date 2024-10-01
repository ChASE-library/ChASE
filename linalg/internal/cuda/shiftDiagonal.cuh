#pragma once

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <iostream>
#include <complex>

namespace chase
{
namespace linalg
{
namespace internal
{
namespace cuda
{
    __global__ void sshift_matrix(float* A, std::size_t n, std::size_t lda, float shift);
    __global__ void dshift_matrix(double* A, std::size_t n, std::size_t lda, double shift);
    __global__ void cshift_matrix(cuComplex* A, std::size_t n, std::size_t lda, float shift);
    __global__ void zshift_matrix(cuDoubleComplex* A, std::size_t n, std::size_t lda, double shift);

    void chase_shift_matrix(float* A, std::size_t n, std::size_t lda, float shift, cudaStream_t stream_);
    void chase_shift_matrix(double* A, std::size_t n, std::size_t lda, double shift, cudaStream_t stream_);
    void chase_shift_matrix(std::complex<float>* A, std::size_t n, std::size_t lda, float shift,
                            cudaStream_t stream_);
    void chase_shift_matrix(std::complex<double>* A, std::size_t n, std::size_t lda, double shift,
                            cudaStream_t stream_);

}
}
}
}
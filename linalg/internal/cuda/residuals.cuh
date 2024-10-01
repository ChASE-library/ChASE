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
    template< int n, typename T >
    __device__ void cuda_sum_reduce(int i, T* x );
    __global__ void c_resids_kernel(std::size_t m, std::size_t n, const cuComplex *A, std::size_t lda, const cuComplex *B, 
                std::size_t ldb, float *ritzv, float *resids, bool is_sqrt );

    __global__ void z_resids_kernel(std::size_t m, std::size_t n, const cuDoubleComplex *A, std::size_t lda, const cuDoubleComplex *B,
                            std::size_t ldb, double *ritzv, double *resids, bool is_sqrt );

    __global__ void d_resids_kernel(std::size_t m, std::size_t n, const double *A, std::size_t lda, const double *B,
                            std::size_t ldb, double *ritzv, double *resids, bool is_sqrt );

    __global__ void s_resids_kernel(std::size_t m, std::size_t n, const float *A, std::size_t lda, const float *B,
                            std::size_t ldb, float *ritzv, float *resids, bool is_sqrt );


    void residual_gpu(std::size_t m, std::size_t n, std::complex<double> *dA, std::size_t lda, std::complex<double> *dB,
                            std::size_t ldb, double *d_ritzv, double *d_resids, bool is_sqrt, cudaStream_t stream_);

    void residual_gpu(std::size_t m, std::size_t n, std::complex<float> *dA, std::size_t lda, std::complex<float> *dB,
                            std::size_t ldb, float *d_ritzv, float *d_resids, bool is_sqrt, cudaStream_t stream_);

    void residual_gpu(std::size_t m, std::size_t n, double *dA, std::size_t lda, double *dB,
                            std::size_t ldb, double *d_ritzv, double *d_resids, bool is_sqrt, cudaStream_t stream_);

    void residual_gpu(std::size_t m, std::size_t n, float *dA, std::size_t lda, float *dB,
                            std::size_t ldb, float *d_ritzv, float *d_resids, bool is_sqrt, cudaStream_t stream_);

}
}
}
}
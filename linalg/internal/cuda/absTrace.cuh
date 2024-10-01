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
    template< std::size_t n, typename T >
    __device__ void cuda_sum_reduce(std::size_t i, T* x );        
    __global__ void s_absTraceKernel(float* d_matrix, float* d_trace, std::size_t n, std::size_t ld);
    __global__ void d_absTraceKernel(double* d_matrix, double* d_trace, std::size_t n, std::size_t ld);
    __global__ void c_absTraceKernel(cuComplex* d_matrix, float* d_trace, std::size_t n, std::size_t ld);
    __global__ void z_absTraceKernel(cuDoubleComplex* d_matrix, double* d_trace, std::size_t n, std::size_t ld);

    void absTrace_gpu(float* d_matrix, float* d_trace, std::size_t n, std::size_t ld, cudaStream_t stream_);
    void absTrace_gpu(double* d_matrix, double* d_trace, std::size_t n, std::size_t ld, cudaStream_t stream_);
    void absTrace_gpu(std::complex<float>* d_matrix, float* d_trace, std::size_t n, std::size_t ld, cudaStream_t stream_);
    void absTrace_gpu(std::complex<double>* d_matrix, double* d_trace, std::size_t n, std::size_t ld, cudaStream_t stream_);
}
}
}
}
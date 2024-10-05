#include "absTrace.cuh"

#define NB_X 256

namespace chase
{
namespace linalg
{
namespace internal
{
namespace cuda
{
    template< std::size_t n, typename T >
    __device__ void cuda_sum_reduce(std::size_t i, T* x )
    {
        __syncthreads();
        if ( n > 1024 ) { if ( i < 1024 && i + 1024 < n ) { x[i] += x[i+1024]; }  __syncthreads(); }
        if ( n >  512 ) { if ( i <  512 && i +  512 < n ) { x[i] += x[i+ 512]; }  __syncthreads(); }
        if ( n >  256 ) { if ( i <  256 && i +  256 < n ) { x[i] += x[i+ 256]; }  __syncthreads(); }
        if ( n >  128 ) { if ( i <  128 && i +  128 < n ) { x[i] += x[i+ 128]; }  __syncthreads(); }
        if ( n >   64 ) { if ( i <   64 && i +   64 < n ) { x[i] += x[i+  64]; }  __syncthreads(); }
        if ( n >   32 ) { if ( i <   32 && i +   32 < n ) { x[i] += x[i+  32]; }  __syncthreads(); }
        if ( n >   16 ) { if ( i <   16 && i +   16 < n ) { x[i] += x[i+  16]; }  __syncthreads(); }
        if ( n >    8 ) { if ( i <    8 && i +    8 < n ) { x[i] += x[i+   8]; }  __syncthreads(); }
        if ( n >    4 ) { if ( i <    4 && i +    4 < n ) { x[i] += x[i+   4]; }  __syncthreads(); }
        if ( n >    2 ) { if ( i <    2 && i +    2 < n ) { x[i] += x[i+   2]; }  __syncthreads(); }
        if ( n >    1 ) { if ( i <    1 && i +    1 < n ) { x[i] += x[i+   1]; }  __syncthreads(); }
    }

    __global__ void s_absTraceKernel(float* d_matrix, float* d_trace, std::size_t n, std::size_t ld)
    {
        __shared__ float partial_trace[NB_X];
        std::size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        partial_trace[threadIdx.x] = (idx < n) ? fabs(d_matrix[idx * ld + idx]) : 0.0f;
        cuda_sum_reduce<NB_X>(threadIdx.x, partial_trace);
        if (threadIdx.x == 0) {
            atomicAdd(d_trace, partial_trace[0]);
        }

    }

    __global__ void d_absTraceKernel(double* d_matrix, double* d_trace, std::size_t n, std::size_t ld) {
        __shared__ double partial_trace[NB_X];
        std::size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        partial_trace[threadIdx.x] = (idx < n) ? fabs(d_matrix[idx * ld + idx]) : 0.0;
        cuda_sum_reduce<NB_X>(threadIdx.x, partial_trace);
        if (threadIdx.x == 0) {
            atomicAdd(d_trace, partial_trace[0]);
        }
    }

    __global__ void c_absTraceKernel(cuComplex* d_matrix, float* d_trace, std::size_t n, std::size_t ld) {
        __shared__ float partial_trace[NB_X];
        std::size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        partial_trace[threadIdx.x] = (idx < n) ? cuCabsf(d_matrix[idx * ld + idx]) : 0.0f;
        cuda_sum_reduce<NB_X>(threadIdx.x, partial_trace);
        if (threadIdx.x == 0) {
            atomicAdd(d_trace, partial_trace[0]);
        }
    }

    __global__ void z_absTraceKernel(cuDoubleComplex* d_matrix, double* d_trace, std::size_t n, std::size_t ld) {
        __shared__ double partial_trace[NB_X];
        std::size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        partial_trace[threadIdx.x] = (idx < n) ? cuCabs(d_matrix[idx * ld + idx]) : 0.0;
        cuda_sum_reduce<NB_X>(threadIdx.x, partial_trace);
        if (threadIdx.x == 0) {
            atomicAdd(d_trace, partial_trace[0]);
        }
    }


    void absTrace_gpu(float* d_matrix, float* d_trace, std::size_t n, std::size_t ld, cudaStream_t stream_)
    {
        cudaMemset(d_trace, 0, sizeof(float));
        dim3 threads( NB_X);
        std::size_t gridSize = (n + NB_X - 1) / NB_X;
        dim3 grid( gridSize );

        s_absTraceKernel<<< grid, threads, 0, stream_ >>>(d_matrix, d_trace, n, ld);
    }

    void absTrace_gpu(double* d_matrix, double* d_trace, std::size_t n, std::size_t ld, cudaStream_t stream_)
    {
        cudaMemset(d_trace, 0, sizeof(double));
        dim3 threads( NB_X);
        std::size_t gridSize = (n + NB_X - 1) / NB_X;
        dim3 grid( gridSize );

        d_absTraceKernel<<< grid, threads, 0, stream_ >>>(d_matrix, d_trace, n, ld);
    }

    void absTrace_gpu(std::complex<float>* d_matrix, float* d_trace, std::size_t n, std::size_t ld, cudaStream_t stream_)
    {
        cudaMemset(d_trace, 0, sizeof(float));
        dim3 threads( NB_X);
        std::size_t gridSize = (n + NB_X - 1) / NB_X;
        dim3 grid( gridSize );

        c_absTraceKernel<<< grid, threads, 0, stream_ >>>(reinterpret_cast<cuComplex*>(d_matrix), d_trace, n, ld);
    }

    void absTrace_gpu(std::complex<double>* d_matrix, double* d_trace, std::size_t n, std::size_t ld, cudaStream_t stream_)
    {
        cudaMemset(d_trace, 0, sizeof(double));
        dim3 threads( NB_X);
        std::size_t gridSize = (n + NB_X - 1) / NB_X;
        dim3 grid( gridSize );

        z_absTraceKernel<<< grid, threads, 0, stream_ >>>(reinterpret_cast<cuDoubleComplex*>(d_matrix), d_trace, n, ld);
    }


}
}
}
}
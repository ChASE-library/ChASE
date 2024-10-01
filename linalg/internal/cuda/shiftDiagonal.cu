#include "shiftDiagonal.cuh"

#define blockSize 256

namespace chase
{
namespace linalg
{
namespace internal
{
namespace cuda
{
    __global__ void sshift_matrix(float* A, std::size_t n, std::size_t lda, float shift)
    {
        std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n)
            A[(idx) * lda + idx] += shift;
    }
    __global__ void dshift_matrix(double* A, std::size_t n, std::size_t lda, double shift)
    {
        std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n)
            A[(idx) * lda + idx] += shift;

    }
    __global__ void cshift_matrix(cuComplex* A, std::size_t n, std::size_t lda, float shift)
    {
        std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n)
            A[(idx) * lda + idx].x += shift;

    }
    __global__ void zshift_matrix(cuDoubleComplex* A, std::size_t n, std::size_t lda, double shift)
    {
        std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n)
            A[(idx) * lda + idx].x += shift;

    }

    void chase_shift_matrix(float* A, std::size_t n, std::size_t lda, float shift, cudaStream_t stream_)
    {
        std::size_t num_blocks = (n + (blockSize - 1)) / blockSize;
        sshift_matrix<<<num_blocks, blockSize, 0, stream_>>>(A, n, lda, shift);
    }
    void chase_shift_matrix(double* A, std::size_t n, std::size_t lda,  double shift, cudaStream_t stream_)
    {
        std::size_t num_blocks = (n + (blockSize - 1)) / blockSize;
        dshift_matrix<<<num_blocks, blockSize, 0, stream_>>>(A, n, lda, shift);
    }
    void chase_shift_matrix(std::complex<float>* A, std::size_t n, std::size_t lda, float shift,
                            cudaStream_t stream_)
    {
        std::size_t num_blocks = (n + (blockSize - 1)) / blockSize;
        cshift_matrix<<<num_blocks, blockSize, 0, stream_>>>(
            reinterpret_cast<cuComplex*>(A), n, lda, shift);

    }
    void chase_shift_matrix(std::complex<double>* A, std::size_t n, std::size_t lda, double shift,
                            cudaStream_t stream_)
    {
        std::size_t num_blocks = (n + (blockSize - 1)) / blockSize;
        zshift_matrix<<<num_blocks, blockSize, 0, stream_>>>(
            reinterpret_cast<cuDoubleComplex*>(A), n, lda, shift);

    }
}
}
}
}
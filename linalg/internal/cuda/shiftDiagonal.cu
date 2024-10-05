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


    __global__ void sshift_mgpu_matrix(float* A, std::size_t* off_m,
                                    std::size_t* off_n, std::size_t offsize,
                                    std::size_t ldH, float shift)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        std::size_t ind;
        if (i < offsize)
        {
            ind = off_n[i] * ldH + off_m[i];
            A[ind] += shift;
        }
    }

    __global__ void dshift_mgpu_matrix(double* A, std::size_t* off_m,
                                    std::size_t* off_n, std::size_t offsize,
                                    std::size_t ldH, double shift)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        std::size_t ind;
        if (i < offsize)
        {
            ind = off_n[i] * ldH + off_m[i];
            A[ind] += shift;
        }
    }

    __global__ void cshift_mgpu_matrix(cuComplex* A, std::size_t* off_m,
                                    std::size_t* off_n, std::size_t offsize,
                                    std::size_t ldH, float shift)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        std::size_t ind;
        if (i < offsize)
        {
            ind = off_n[i] * ldH + off_m[i];
            A[ind].x += shift;
        }
    }

    __global__ void zshift_mgpu_matrix(cuDoubleComplex* A, std::size_t* off_m,
                                    std::size_t* off_n, std::size_t offsize,
                                    std::size_t ldH, double shift)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        std::size_t ind;
        if (i < offsize)
        {
            ind = off_n[i] * ldH + off_m[i];
            A[ind].x += shift;
        }
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

    void chase_shift_mgpu_matrix(float* A, std::size_t* off_m, std::size_t* off_n,
                                std::size_t offsize, std::size_t ldH, float shift,
                                cudaStream_t stream_)
    {
        unsigned int grid = (offsize + blockSize - 1) / blockSize;
        if(grid == 0)
        {
            grid = 1;
        }
        dim3 threadsPerBlock(blockSize, 1);
        dim3 numBlocks(grid, 1);
        sshift_mgpu_matrix<<<numBlocks, threadsPerBlock, 0, stream_>>>( //
            A, off_m, off_n, offsize, ldH, shift);
    }

    void chase_shift_mgpu_matrix(double* A, std::size_t* off_m, std::size_t* off_n,
                                std::size_t offsize, std::size_t ldH, double shift,
                                cudaStream_t stream_)
    {
        unsigned int grid = (offsize + blockSize - 1) / blockSize;
        if(grid == 0)
        {
            grid = 1;
        }
        dim3 threadsPerBlock(blockSize, 1);
        dim3 numBlocks(grid, 1);
        dshift_mgpu_matrix<<<numBlocks, threadsPerBlock, 0, stream_>>>( //
            A, off_m, off_n, offsize, ldH, shift);
    }

    void chase_shift_mgpu_matrix(std::complex<float>* A, std::size_t* off_m,
                                std::size_t* off_n, std::size_t offsize,
                                std::size_t ldH, float shift, cudaStream_t stream_)
    {
        unsigned int grid = (offsize + blockSize - 1) / blockSize;
        if(grid == 0)
        {
            grid = 1;
        }
        dim3 threadsPerBlock(blockSize, 1);
        dim3 numBlocks(grid, 1);
        cshift_mgpu_matrix<<<numBlocks, threadsPerBlock, 0, stream_>>>( //
            reinterpret_cast<cuComplex*>(A), off_m, off_n,              //
            offsize, ldH, shift);
    }

    void chase_shift_mgpu_matrix(std::complex<double>* A, std::size_t* off_m,
                                std::size_t* off_n, std::size_t offsize,
                                std::size_t ldH, double shift,
                                cudaStream_t stream_)
    {
        unsigned int grid = (offsize + blockSize - 1) / blockSize;
        if(grid == 0)
        {
            grid = 1;
        }
        dim3 threadsPerBlock(blockSize, 1);
        dim3 numBlocks(grid, 1);
        zshift_mgpu_matrix<<<numBlocks, threadsPerBlock, 0, stream_>>>( //
            reinterpret_cast<cuDoubleComplex*>(A), off_m, off_n,        //
            offsize, ldH, shift);
    }

}
}
}
}
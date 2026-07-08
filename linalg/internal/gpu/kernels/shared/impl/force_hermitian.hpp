// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE


#define FORCE_HERM_BLOCK 256

namespace chase
{
namespace linalg
{
namespace internal
{
namespace CHASE_GPU_DEVICE_NAMESPACE
{
namespace
{
__device__ void decode_upper_tri_index(std::size_t k, std::size_t n,
                                       std::size_t& i, std::size_t& j)
{
    std::size_t rem = k;
    for (i = 0; i < n; ++i)
    {
        std::size_t row_len = n - i;
        if (rem < row_len)
        {
            j = i + rem;
            return;
        }
        rem -= row_len;
    }
    i = n;
    j = n;
}

__global__ void s_force_hermitian_kernel(float* A, std::size_t n,
                                         std::size_t lda)
{
    std::size_t k = blockIdx.x * blockDim.x + threadIdx.x;
    std::size_t total = n * (n + 1) / 2;
    if (k >= total)
        return;
    std::size_t i, j;
    decode_upper_tri_index(k, n, i, j);
    float val = 0.5f * (A[i + j * lda] + A[j + i * lda]);
    A[i + j * lda] = val;
    A[j + i * lda] = val;
}

__global__ void d_force_hermitian_kernel(double* A, std::size_t n,
                                         std::size_t lda)
{
    std::size_t k = blockIdx.x * blockDim.x + threadIdx.x;
    std::size_t total = n * (n + 1) / 2;
    if (k >= total)
        return;
    std::size_t i, j;
    decode_upper_tri_index(k, n, i, j);
    double val = 0.5 * (A[i + j * lda] + A[j + i * lda]);
    A[i + j * lda] = val;
    A[j + i * lda] = val;
}

__global__ void c_force_hermitian_kernel(CHASE_GPU_FLOAT_COMPLEX* A, std::size_t n,
                                         std::size_t lda)
{
    std::size_t k = blockIdx.x * blockDim.x + threadIdx.x;
    std::size_t total = n * (n + 1) / 2;
    if (k >= total)
        return;
    std::size_t i, j;
    decode_upper_tri_index(k, n, i, j);
    CHASE_GPU_FLOAT_COMPLEX aij = A[i + j * lda];
    CHASE_GPU_FLOAT_COMPLEX aji = A[j + i * lda];
    CHASE_GPU_FLOAT_COMPLEX sum = CHASE_GPU_CADD_FLOAT(aij, CHASE_GPU_CONJ_FLOAT(aji));
    CHASE_GPU_FLOAT_COMPLEX val = CHASE_GPU_MAKE_FLOAT_COMPLEX(0.5f * sum.x, 0.5f * sum.y);
    A[i + j * lda] = val;
    A[j + i * lda] = CHASE_GPU_CONJ_FLOAT(val);
}

__global__ void z_force_hermitian_kernel(CHASE_GPU_DOUBLE_COMPLEX* A, std::size_t n,
                                         std::size_t lda)
{
    std::size_t k = blockIdx.x * blockDim.x + threadIdx.x;
    std::size_t total = n * (n + 1) / 2;
    if (k >= total)
        return;
    std::size_t i, j;
    decode_upper_tri_index(k, n, i, j);
    CHASE_GPU_DOUBLE_COMPLEX aij = A[i + j * lda];
    CHASE_GPU_DOUBLE_COMPLEX aji = A[j + i * lda];
    CHASE_GPU_DOUBLE_COMPLEX sum = CHASE_GPU_CADD_DOUBLE(aij, CHASE_GPU_CONJ_DOUBLE(aji));
    CHASE_GPU_DOUBLE_COMPLEX val =
        CHASE_GPU_MAKE_DOUBLE_COMPLEX(0.5 * sum.x, 0.5 * sum.y);
    A[i + j * lda] = val;
    A[j + i * lda] = CHASE_GPU_CONJ_DOUBLE(val);
}
} // namespace

void force_hermitian_gpu(float* A, std::size_t n, std::size_t lda,
                         CHASE_GPU_STREAM_T stream)
{
    std::size_t total = n * (n + 1) / 2;
    std::size_t grid = (total + FORCE_HERM_BLOCK - 1) / FORCE_HERM_BLOCK;
    s_force_hermitian_kernel<<<grid, FORCE_HERM_BLOCK, 0, stream>>>(A, n, lda);
}

void force_hermitian_gpu(double* A, std::size_t n, std::size_t lda,
                         CHASE_GPU_STREAM_T stream)
{
    std::size_t total = n * (n + 1) / 2;
    std::size_t grid = (total + FORCE_HERM_BLOCK - 1) / FORCE_HERM_BLOCK;
    d_force_hermitian_kernel<<<grid, FORCE_HERM_BLOCK, 0, stream>>>(A, n, lda);
}

void force_hermitian_gpu(std::complex<float>* A, std::size_t n, std::size_t lda,
                         CHASE_GPU_STREAM_T stream)
{
    std::size_t total = n * (n + 1) / 2;
    std::size_t grid = (total + FORCE_HERM_BLOCK - 1) / FORCE_HERM_BLOCK;
    c_force_hermitian_kernel<<<grid, FORCE_HERM_BLOCK, 0, stream>>>(
        reinterpret_cast<CHASE_GPU_FLOAT_COMPLEX*>(A), n, lda);
}

void force_hermitian_gpu(std::complex<double>* A, std::size_t n,
                         std::size_t lda, CHASE_GPU_STREAM_T stream)
{
    std::size_t total = n * (n + 1) / 2;
    std::size_t grid = (total + FORCE_HERM_BLOCK - 1) / FORCE_HERM_BLOCK;
    z_force_hermitian_kernel<<<grid, FORCE_HERM_BLOCK, 0, stream>>>(
        reinterpret_cast<CHASE_GPU_DOUBLE_COMPLEX*>(A), n, lda);
}


#if defined(CHASE_GPU_DEVICE_BACKEND_HIP)
void forceHermitian(float* A, std::size_t n, std::size_t lda,
                    CHASE_GPU_STREAM_T stream)
{
    force_hermitian_gpu(A, n, lda, stream);
}

void forceHermitian(double* A, std::size_t n, std::size_t lda,
                    CHASE_GPU_STREAM_T stream)
{
    force_hermitian_gpu(A, n, lda, stream);
}

void forceHermitian(std::complex<float>* A, std::size_t n, std::size_t lda,
                    CHASE_GPU_STREAM_T stream)
{
    force_hermitian_gpu(A, n, lda, stream);
}

void forceHermitian(std::complex<double>* A, std::size_t n, std::size_t lda,
                    CHASE_GPU_STREAM_T stream)
{
    force_hermitian_gpu(A, n, lda, stream);
}
#endif

} // namespace CHASE_GPU_DEVICE_NAMESPACE
} // namespace internal
} // namespace linalg
} // namespace chase

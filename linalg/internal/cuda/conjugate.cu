// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include "conjugate.cuh"

namespace chase
{
namespace linalg
{
namespace internal
{
namespace cuda
{

const int conj_blockSize = 256;

// ----- In-place conjugation -----
__global__ void sconjugate_inplace(float* A, std::size_t m, std::size_t n,
                                   std::size_t lda)
{
    // Real: no-op (conjugate is identity)
    (void)A;
    (void)m;
    (void)n;
    (void)lda;
}

__global__ void dconjugate_inplace(double* A, std::size_t m, std::size_t n,
                                   std::size_t lda)
{
    (void)A;
    (void)m;
    (void)n;
    (void)lda;
}

__global__ void cconjugate_inplace(cuComplex* A, std::size_t m, std::size_t n,
                                   std::size_t lda)
{
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t col = idx / m;
    const std::size_t row = idx % m;

    if (col < n)
    {
        cuComplex z = A[row + lda * col];
        z.y = -z.y;
        A[row + lda * col] = z;
    }
}

__global__ void zconjugate_inplace(cuDoubleComplex* A, std::size_t m,
                                   std::size_t n, std::size_t lda)
{
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t col = idx / m;
    const std::size_t row = idx % m;

    if (col < n)
    {
        cuDoubleComplex z = A[row + lda * col];
        z.y = -z.y;
        A[row + lda * col] = z;
    }
}

// ----- Host launchers -----
void chase_conjugate_inplace(float* A, std::size_t m, std::size_t n,
                             std::size_t lda, cudaStream_t stream_)
{
    (void)A;
    (void)m;
    (void)n;
    (void)lda;
    (void)stream_;
    // Real: no-op, skip launch
}

void chase_conjugate_inplace(double* A, std::size_t m, std::size_t n,
                             std::size_t lda, cudaStream_t stream_)
{
    (void)A;
    (void)m;
    (void)n;
    (void)lda;
    (void)stream_;
}

void chase_conjugate_inplace(std::complex<float>* A, std::size_t m,
                             std::size_t n, std::size_t lda,
                             cudaStream_t stream_)
{
    std::size_t num_blocks =
        (m * n + (conj_blockSize - 1)) / conj_blockSize;
    cconjugate_inplace<<<num_blocks, conj_blockSize, 0, stream_>>>(
        reinterpret_cast<cuComplex*>(A), m, n, lda);
}

void chase_conjugate_inplace(std::complex<double>* A, std::size_t m,
                             std::size_t n, std::size_t lda,
                             cudaStream_t stream_)
{
    std::size_t num_blocks =
        (m * n + (conj_blockSize - 1)) / conj_blockSize;
    zconjugate_inplace<<<num_blocks, conj_blockSize, 0, stream_>>>(
        reinterpret_cast<cuDoubleComplex*>(A), m, n, lda);
}

} // namespace cuda
} // namespace internal
} // namespace linalg
} // namespace chase

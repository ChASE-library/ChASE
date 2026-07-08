// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE


namespace chase
{
namespace linalg
{
namespace internal
{
namespace CHASE_GPU_DEVICE_NAMESPACE
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

__global__ void cconjugate_inplace(CHASE_GPU_FLOAT_COMPLEX* A, std::size_t m, std::size_t n,
                                   std::size_t lda)
{
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t col = idx / m;
    const std::size_t row = idx % m;

    if (col < n)
    {
        CHASE_GPU_FLOAT_COMPLEX z = A[row + lda * col];
        z.y = -z.y;
        A[row + lda * col] = z;
    }
}

__global__ void zconjugate_inplace(CHASE_GPU_DOUBLE_COMPLEX* A, std::size_t m,
                                   std::size_t n, std::size_t lda)
{
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t col = idx / m;
    const std::size_t row = idx % m;

    if (col < n)
    {
        CHASE_GPU_DOUBLE_COMPLEX z = A[row + lda * col];
        z.y = -z.y;
        A[row + lda * col] = z;
    }
}

// ----- Host launchers -----
void chase_conjugate_inplace(float* A, std::size_t m, std::size_t n,
                             std::size_t lda, CHASE_GPU_STREAM_T stream_)
{
    (void)A;
    (void)m;
    (void)n;
    (void)lda;
    (void)stream_;
    // Real: no-op, skip launch
}

void chase_conjugate_inplace(double* A, std::size_t m, std::size_t n,
                             std::size_t lda, CHASE_GPU_STREAM_T stream_)
{
    (void)A;
    (void)m;
    (void)n;
    (void)lda;
    (void)stream_;
}

void chase_conjugate_inplace(std::complex<float>* A, std::size_t m,
                             std::size_t n, std::size_t lda,
                             CHASE_GPU_STREAM_T stream_)
{
    std::size_t num_blocks =
        (m * n + (conj_blockSize - 1)) / conj_blockSize;
    cconjugate_inplace<<<num_blocks, conj_blockSize, 0, stream_>>>(
        reinterpret_cast<CHASE_GPU_FLOAT_COMPLEX*>(A), m, n, lda);
}

void chase_conjugate_inplace(std::complex<double>* A, std::size_t m,
                             std::size_t n, std::size_t lda,
                             CHASE_GPU_STREAM_T stream_)
{
    std::size_t num_blocks =
        (m * n + (conj_blockSize - 1)) / conj_blockSize;
    zconjugate_inplace<<<num_blocks, conj_blockSize, 0, stream_>>>(
        reinterpret_cast<CHASE_GPU_DOUBLE_COMPLEX*>(A), m, n, lda);
}


#if defined(CHASE_GPU_DEVICE_BACKEND_HIP)
void conjugateInPlace(float* A, std::size_t m, std::size_t n, std::size_t lda,
                      CHASE_GPU_STREAM_T stream)
{
    chase_conjugate_inplace(A, m, n, lda, stream);
}

void conjugateInPlace(double* A, std::size_t m, std::size_t n, std::size_t lda,
                      CHASE_GPU_STREAM_T stream)
{
    chase_conjugate_inplace(A, m, n, lda, stream);
}

void conjugateInPlace(std::complex<float>* A, std::size_t m, std::size_t n,
                      std::size_t lda, CHASE_GPU_STREAM_T stream)
{
    chase_conjugate_inplace(A, m, n, lda, stream);
}

void conjugateInPlace(std::complex<double>* A, std::size_t m, std::size_t n,
                      std::size_t lda, CHASE_GPU_STREAM_T stream)
{
    chase_conjugate_inplace(A, m, n, lda, stream);
}
#endif

} // namespace CHASE_GPU_DEVICE_NAMESPACE
} // namespace internal
} // namespace linalg
} // namespace chase

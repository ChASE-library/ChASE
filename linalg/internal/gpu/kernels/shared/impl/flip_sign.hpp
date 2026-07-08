// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE


const int blockSize = 256;

namespace chase
{
namespace linalg
{
namespace internal
{
namespace CHASE_GPU_DEVICE_NAMESPACE
{
__global__ void sflipLowerHalfMatrixSign(float* A, std::size_t m, std::size_t n,
                                         std::size_t lda)
{
    const std::size_t half_m = m / 2;
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t col = idx / half_m;
    const std::size_t row_offset = idx % half_m;

    if (col < n)
    {
        const std::size_t row = half_m + row_offset;
        A[row + lda * col] = -A[row + lda * col];
    }
}
__global__ void dflipLowerHalfMatrixSign(double* A, std::size_t m,
                                         std::size_t n, std::size_t lda)
{
    const std::size_t half_m = m / 2;
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t col = idx / half_m;
    const std::size_t row_offset = idx % half_m;

    if (col < n)
    {
        const std::size_t row = half_m + row_offset;
        A[row + lda * col] = -A[row + lda * col];
    }
}
__global__ void cflipLowerHalfMatrixSign(CHASE_GPU_FLOAT_COMPLEX* A, std::size_t m,
                                         std::size_t n, std::size_t lda)
{
    const std::size_t half_m = m / 2;
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t col = idx / half_m;
    const std::size_t row_offset = idx % half_m;

    if (col < n)
    {
        const std::size_t row = half_m + row_offset;
        CHASE_GPU_FLOAT_COMPLEX& element = A[row + lda * col];
        element.x = -element.x;
        element.y = -element.y;
    }
}
__global__ void zflipLowerHalfMatrixSign(CHASE_GPU_DOUBLE_COMPLEX* A, std::size_t m,
                                         std::size_t n, std::size_t lda)
{
    const std::size_t half_m = m / 2;
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t col = idx / half_m;
    const std::size_t row_offset = idx % half_m;

    if (col < n)
    {
        const std::size_t row = half_m + row_offset;
        CHASE_GPU_DOUBLE_COMPLEX& element = A[row + lda * col];
        element.x = -element.x;
        element.y = -element.y;
    }
}

__global__ void sflipMatrixSign(float* A, std::size_t m, std::size_t n,
                                std::size_t lda)
{
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t col = idx / m;
    const std::size_t row = idx % m;

    if (col < n)
    {
        A[row + lda * col] = -A[row + lda * col];
    }
}
__global__ void dflipMatrixSign(double* A, std::size_t m, std::size_t n,
                                std::size_t lda)
{
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t col = idx / m;
    const std::size_t row = idx % m;

    if (col < n)
    {
        A[row + lda * col] = -A[row + lda * col];
    }
}
__global__ void cflipMatrixSign(CHASE_GPU_FLOAT_COMPLEX* A, std::size_t m, std::size_t n,
                                std::size_t lda)
{
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t col = idx / m;
    const std::size_t row = idx % m;

    if (col < n)
    {
        CHASE_GPU_FLOAT_COMPLEX& element = A[row + lda * col];
        element.x = -element.x;
        element.y = -element.y;
    }
}
__global__ void zflipMatrixSign(CHASE_GPU_DOUBLE_COMPLEX* A, std::size_t m,
                                std::size_t n, std::size_t lda)
{
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t col = idx / m;
    const std::size_t row = idx % m;

    if (col < n)
    {
        CHASE_GPU_DOUBLE_COMPLEX& element = A[row + lda * col];
        element.x = -element.x;
        element.y = -element.y;
    }
}

void chase_flipLowerHalfMatrixSign(float* A, std::size_t m, std::size_t n,
                                   std::size_t lda, CHASE_GPU_STREAM_T stream_)
{
    std::size_t num_blocks = (n * (m / 2) + (blockSize - 1)) / blockSize;
    sflipLowerHalfMatrixSign<<<num_blocks, blockSize, 0, stream_>>>(A, m, n,
                                                                    lda);
}
void chase_flipLowerHalfMatrixSign(double* A, std::size_t m, std::size_t n,
                                   std::size_t lda, CHASE_GPU_STREAM_T stream_)
{
    std::size_t num_blocks = (n * (m / 2) + (blockSize - 1)) / blockSize;
    dflipLowerHalfMatrixSign<<<num_blocks, blockSize, 0, stream_>>>(A, m, n,
                                                                    lda);
}
void chase_flipLowerHalfMatrixSign(std::complex<float>* A, std::size_t m,
                                   std::size_t n, std::size_t lda,
                                   CHASE_GPU_STREAM_T stream_)
{
    std::size_t num_blocks = (n * (m / 2) + (blockSize - 1)) / blockSize;
    cflipLowerHalfMatrixSign<<<num_blocks, blockSize, 0, stream_>>>(
        reinterpret_cast<CHASE_GPU_FLOAT_COMPLEX*>(A), m, n, lda);
}
void chase_flipLowerHalfMatrixSign(std::complex<double>* A, std::size_t m,
                                   std::size_t n, std::size_t lda,
                                   CHASE_GPU_STREAM_T stream_)
{
    std::size_t num_blocks = (n * (m / 2) + (blockSize - 1)) / blockSize;
    zflipLowerHalfMatrixSign<<<num_blocks, blockSize, 0, stream_>>>(
        reinterpret_cast<CHASE_GPU_DOUBLE_COMPLEX*>(A), m, n, lda);
}

void chase_flipMatrixSign(float* A, std::size_t m, std::size_t n,
                          std::size_t lda, CHASE_GPU_STREAM_T stream_)
{
    std::size_t num_blocks = (n * m + (blockSize - 1)) / blockSize;
    sflipMatrixSign<<<num_blocks, blockSize, 0, stream_>>>(A, m, n, lda);
}
void chase_flipMatrixSign(double* A, std::size_t m, std::size_t n,
                          std::size_t lda, CHASE_GPU_STREAM_T stream_)
{
    std::size_t num_blocks = (n * m + (blockSize - 1)) / blockSize;
    dflipMatrixSign<<<num_blocks, blockSize, 0, stream_>>>(A, m, n, lda);
}
void chase_flipMatrixSign(std::complex<float>* A, std::size_t m, std::size_t n,
                          std::size_t lda, CHASE_GPU_STREAM_T stream_)
{
    std::size_t num_blocks = (n * m + (blockSize - 1)) / blockSize;
    cflipMatrixSign<<<num_blocks, blockSize, 0, stream_>>>(
        reinterpret_cast<CHASE_GPU_FLOAT_COMPLEX*>(A), m, n, lda);
}
void chase_flipMatrixSign(std::complex<double>* A, std::size_t m, std::size_t n,
                          std::size_t lda, CHASE_GPU_STREAM_T stream_)
{
    std::size_t num_blocks = (n * m + (blockSize - 1)) / blockSize;
    zflipMatrixSign<<<num_blocks, blockSize, 0, stream_>>>(
        reinterpret_cast<CHASE_GPU_DOUBLE_COMPLEX*>(A), m, n, lda);
}

__global__ void sscaleLowerBlockRows(float* A, std::size_t lda,
                                     std::size_t row_start,
                                     std::size_t nrows_lower, std::size_t ncols,
                                     float scale)
{
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t col = idx / nrows_lower;
    const std::size_t row_offset = idx % nrows_lower;

    if (col < ncols)
    {
        const std::size_t row = row_start + row_offset;
        A[row + lda * col] *= scale;
    }
}

__global__ void dscaleLowerBlockRows(double* A, std::size_t lda,
                                     std::size_t row_start,
                                     std::size_t nrows_lower, std::size_t ncols,
                                     double scale)
{
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t col = idx / nrows_lower;
    const std::size_t row_offset = idx % nrows_lower;

    if (col < ncols)
    {
        const std::size_t row = row_start + row_offset;
        A[row + lda * col] *= scale;
    }
}

__global__ void cscaleLowerBlockRows(CHASE_GPU_FLOAT_COMPLEX* A, std::size_t lda,
                                     std::size_t row_start,
                                     std::size_t nrows_lower, std::size_t ncols,
                                     CHASE_GPU_FLOAT_COMPLEX scale)
{
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t col = idx / nrows_lower;
    const std::size_t row_offset = idx % nrows_lower;

    if (col < ncols)
    {
        const std::size_t row = row_start + row_offset;
        CHASE_GPU_FLOAT_COMPLEX& z = A[row + lda * col];
        z = CHASE_GPU_CMUL_FLOAT(z, scale);
    }
}

__global__ void zscaleLowerBlockRows(CHASE_GPU_DOUBLE_COMPLEX* A, std::size_t lda,
                                     std::size_t row_start,
                                     std::size_t nrows_lower, std::size_t ncols,
                                     CHASE_GPU_DOUBLE_COMPLEX scale)
{
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t col = idx / nrows_lower;
    const std::size_t row_offset = idx % nrows_lower;

    if (col < ncols)
    {
        const std::size_t row = row_start + row_offset;
        CHASE_GPU_DOUBLE_COMPLEX& z = A[row + lda * col];
        const double re = z.x * scale.x - z.y * scale.y;
        const double im = z.x * scale.y + z.y * scale.x;
        z.x = re;
        z.y = im;
    }
}

void chase_scaleLowerBlockRows(float* A, std::size_t lda, std::size_t row_start,
                               std::size_t nrows_lower, std::size_t ncols,
                               float scale, CHASE_GPU_STREAM_T stream_)
{
    if (nrows_lower == 0 || ncols == 0)
        return;
    const std::size_t num_blocks =
        (nrows_lower * ncols + (blockSize - 1)) / blockSize;
    sscaleLowerBlockRows<<<num_blocks, blockSize, 0, stream_>>>(
        A, lda, row_start, nrows_lower, ncols, scale);
}

void chase_scaleLowerBlockRows(double* A, std::size_t lda, std::size_t row_start,
                               std::size_t nrows_lower, std::size_t ncols,
                               double scale, CHASE_GPU_STREAM_T stream_)
{
    if (nrows_lower == 0 || ncols == 0)
        return;
    const std::size_t num_blocks =
        (nrows_lower * ncols + (blockSize - 1)) / blockSize;
    dscaleLowerBlockRows<<<num_blocks, blockSize, 0, stream_>>>(
        A, lda, row_start, nrows_lower, ncols, scale);
}

void chase_scaleLowerBlockRows(std::complex<float>* A, std::size_t lda,
                               std::size_t row_start, std::size_t nrows_lower,
                               std::size_t ncols, std::complex<float> scale,
                               CHASE_GPU_STREAM_T stream_)
{
    if (nrows_lower == 0 || ncols == 0)
        return;
    const std::size_t num_blocks =
        (nrows_lower * ncols + (blockSize - 1)) / blockSize;
    const CHASE_GPU_FLOAT_COMPLEX cs = {scale.real(), scale.imag()};
    cscaleLowerBlockRows<<<num_blocks, blockSize, 0, stream_>>>(
        reinterpret_cast<CHASE_GPU_FLOAT_COMPLEX*>(A), lda, row_start, nrows_lower, ncols,
        cs);
}

void chase_scaleLowerBlockRows(std::complex<double>* A, std::size_t lda,
                               std::size_t row_start, std::size_t nrows_lower,
                               std::size_t ncols, std::complex<double> scale,
                               CHASE_GPU_STREAM_T stream_)
{
    if (nrows_lower == 0 || ncols == 0)
        return;
    const std::size_t num_blocks =
        (nrows_lower * ncols + (blockSize - 1)) / blockSize;
    const CHASE_GPU_DOUBLE_COMPLEX zs = {scale.real(), scale.imag()};
    zscaleLowerBlockRows<<<num_blocks, blockSize, 0, stream_>>>(
        reinterpret_cast<CHASE_GPU_DOUBLE_COMPLEX*>(A), lda, row_start, nrows_lower,
        ncols, zs);
}

#if defined(CHASE_GPU_DEVICE_BACKEND_HIP)
void flipLowerHalfMatrixSign(float* A, std::size_t m, std::size_t n,
                             std::size_t lda, CHASE_GPU_STREAM_T stream)
{
    chase_flipLowerHalfMatrixSign(A, m, n, lda, stream);
}

void flipLowerHalfMatrixSign(double* A, std::size_t m, std::size_t n,
                             std::size_t lda, CHASE_GPU_STREAM_T stream)
{
    chase_flipLowerHalfMatrixSign(A, m, n, lda, stream);
}

void flipLowerHalfMatrixSign(std::complex<float>* A, std::size_t m,
                             std::size_t n, std::size_t lda,
                             CHASE_GPU_STREAM_T stream)
{
    chase_flipLowerHalfMatrixSign(A, m, n, lda, stream);
}

void flipLowerHalfMatrixSign(std::complex<double>* A, std::size_t m,
                             std::size_t n, std::size_t lda,
                             CHASE_GPU_STREAM_T stream)
{
    chase_flipLowerHalfMatrixSign(A, m, n, lda, stream);
}

void scaleLowerBlockRows(float* A, std::size_t lda, std::size_t row_start,
                         std::size_t nrows_lower, std::size_t ncols,
                         float scale, CHASE_GPU_STREAM_T stream)
{
    chase_scaleLowerBlockRows(A, lda, row_start, nrows_lower, ncols, scale,
                              stream);
}

void scaleLowerBlockRows(double* A, std::size_t lda, std::size_t row_start,
                         std::size_t nrows_lower, std::size_t ncols,
                         double scale, CHASE_GPU_STREAM_T stream)
{
    chase_scaleLowerBlockRows(A, lda, row_start, nrows_lower, ncols, scale,
                              stream);
}

void scaleLowerBlockRows(std::complex<float>* A, std::size_t lda,
                         std::size_t row_start, std::size_t nrows_lower,
                         std::size_t ncols, std::complex<float> scale,
                         CHASE_GPU_STREAM_T stream)
{
    chase_scaleLowerBlockRows(A, lda, row_start, nrows_lower, ncols, scale,
                              stream);
}

void scaleLowerBlockRows(std::complex<double>* A, std::size_t lda,
                         std::size_t row_start, std::size_t nrows_lower,
                         std::size_t ncols, std::complex<double> scale,
                         CHASE_GPU_STREAM_T stream)
{
    chase_scaleLowerBlockRows(A, lda, row_start, nrows_lower, ncols, scale,
                              stream);
}
#endif

} // namespace CHASE_GPU_DEVICE_NAMESPACE
} // namespace internal
} // namespace linalg
} // namespace chase

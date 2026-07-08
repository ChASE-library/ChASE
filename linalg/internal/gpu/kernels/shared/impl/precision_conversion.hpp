// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE


#define blockSize 256

namespace chase
{
namespace linalg
{
namespace internal
{
namespace CHASE_GPU_DEVICE_NAMESPACE
{
// Kernel to convert double to float
__global__ void convertDoubleToFloatKernel(double* input, float* output,
                                           std::size_t size)
{
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        output[idx] = CHASE_GPU_DOUBLE2FLOAT_RN(input[idx]);
    }
}

// Kernel to convert float to double
__global__ void convertFloatToDoubleKernel(float* input, double* output,
                                           std::size_t size)
{
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        output[idx] = static_cast<double>(input[idx]); // Direct cast
    }
}

// Kernel to convert std::complex<double> to std::complex<float>
__global__ void convertDoubleToFloatComplexKernel(CHASE_GPU_DOUBLE_COMPLEX* input,
                                                  CHASE_GPU_FLOAT_COMPLEX* output,
                                                  std::size_t size)
{
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        output[idx] = CHASE_GPU_MAKE_FLOAT_COMPLEX(
            CHASE_GPU_DOUBLE2FLOAT_RN(CHASE_GPU_CREAL_DOUBLE(input[idx])),
            CHASE_GPU_DOUBLE2FLOAT_RN(CHASE_GPU_CIMAG_DOUBLE(input[idx])));
    }
}

// Kernel to convert std::complex<float> to std::complex<double>
__global__ void convertFloatToDoubleComplexKernel(CHASE_GPU_FLOAT_COMPLEX* input,
                                                  CHASE_GPU_DOUBLE_COMPLEX* output,
                                                  std::size_t size)
{
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        output[idx] =
            CHASE_GPU_MAKE_DOUBLE_COMPLEX(static_cast<double>(CHASE_GPU_CREAL_FLOAT(input[idx])),
                                 static_cast<double>(CHASE_GPU_CIMAG_FLOAT(input[idx])));
    }
}

// Function to launch the conversion from double to float
#if defined(CHASE_GPU_DEVICE_BACKEND_CUDA)
void convert_DP_TO_SP_GPU(double* d_input, float* d_output, std::size_t size,
                          CHASE_GPU_STREAM_T* stream_)
{
    int numBlocks = (size + blockSize - 1) / blockSize;
    CHASE_GPU_STREAM_T usedStream = (stream_ == nullptr) ? 0 : *stream_;
    convertDoubleToFloatKernel<<<numBlocks, blockSize, 0, usedStream>>>(
        d_input, d_output, size);
}

// Function to launch the conversion from double to float complex
void convert_DP_TO_SP_GPU(std::complex<double>* d_input,
                          std::complex<float>* d_output, std::size_t size,
                          CHASE_GPU_STREAM_T* stream_)
{
    int numBlocks = (size + blockSize - 1) / blockSize;
    CHASE_GPU_STREAM_T usedStream = (stream_ == nullptr) ? 0 : *stream_;
    convertDoubleToFloatComplexKernel<<<numBlocks, blockSize, 0, usedStream>>>(
        reinterpret_cast<CHASE_GPU_DOUBLE_COMPLEX*>(d_input),
        reinterpret_cast<CHASE_GPU_FLOAT_COMPLEX*>(d_output), size);
}

// Function to launch the conversion from float to double
void convert_SP_TO_DP_GPU(float* d_input, double* d_output, std::size_t size,
                          CHASE_GPU_STREAM_T* stream_)
{
    int numBlocks = (size + blockSize - 1) / blockSize;
    CHASE_GPU_STREAM_T usedStream = (stream_ == nullptr) ? 0 : *stream_;
    convertFloatToDoubleKernel<<<numBlocks, blockSize, 0, usedStream>>>(
        d_input, d_output, size);
}

// Function to launch the conversion from float to double complex
void convert_SP_TO_DP_GPU(std::complex<float>* d_input,
                          std::complex<double>* d_output, std::size_t size,
                          CHASE_GPU_STREAM_T* stream_)
{
    int numBlocks = (size + blockSize - 1) / blockSize;
    CHASE_GPU_STREAM_T usedStream = (stream_ == nullptr) ? 0 : *stream_;
    convertFloatToDoubleComplexKernel<<<numBlocks, blockSize, 0, usedStream>>>(
        reinterpret_cast<CHASE_GPU_FLOAT_COMPLEX*>(d_input),
        reinterpret_cast<CHASE_GPU_DOUBLE_COMPLEX*>(d_output), size);
}
#elif defined(CHASE_GPU_DEVICE_BACKEND_HIP)
void convert_DP_TO_SP_GPU(double* d_input, float* d_output, std::size_t size,
                          CHASE_GPU_STREAM_T stream_)
{
    int numBlocks = (size + blockSize - 1) / blockSize;
    convertDoubleToFloatKernel<<<numBlocks, blockSize, 0, stream_>>>(
        d_input, d_output, size);
}

void convert_DP_TO_SP_GPU(std::complex<double>* d_input,
                          std::complex<float>* d_output, std::size_t size,
                          CHASE_GPU_STREAM_T stream_)
{
    int numBlocks = (size + blockSize - 1) / blockSize;
    convertDoubleToFloatComplexKernel<<<numBlocks, blockSize, 0, stream_>>>(
        reinterpret_cast<CHASE_GPU_DOUBLE_COMPLEX*>(d_input),
        reinterpret_cast<CHASE_GPU_FLOAT_COMPLEX*>(d_output), size);
}

void convert_SP_TO_DP_GPU(float* d_input, double* d_output, std::size_t size,
                          CHASE_GPU_STREAM_T stream_)
{
    int numBlocks = (size + blockSize - 1) / blockSize;
    convertFloatToDoubleKernel<<<numBlocks, blockSize, 0, stream_>>>(
        d_input, d_output, size);
}

void convert_SP_TO_DP_GPU(std::complex<float>* d_input,
                          std::complex<double>* d_output, std::size_t size,
                          CHASE_GPU_STREAM_T stream_)
{
    int numBlocks = (size + blockSize - 1) / blockSize;
    convertFloatToDoubleComplexKernel<<<numBlocks, blockSize, 0, stream_>>>(
        reinterpret_cast<CHASE_GPU_FLOAT_COMPLEX*>(d_input),
        reinterpret_cast<CHASE_GPU_DOUBLE_COMPLEX*>(d_output), size);
}
#endif

} // namespace CHASE_GPU_DEVICE_NAMESPACE
} // namespace internal
} // namespace linalg
} // namespace chase

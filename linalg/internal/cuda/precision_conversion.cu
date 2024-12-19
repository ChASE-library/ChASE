// This file is a part of ChASE.
// Copyright (c) 2015-2024, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include "precision_conversion.cuh"

#define blockSize 256

namespace chase
{
namespace linalg
{
namespace internal
{
namespace cuda
{
// Kernel to convert double to float
__global__ void convertDoubleToFloatKernel(double* input, float* output, std::size_t size) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __double2float_rn(input[idx]);  // Rounding to nearest even
    }
}

// Kernel to convert float to double
__global__ void convertFloatToDoubleKernel(float* input, double* output, std::size_t size) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = static_cast<double>(input[idx]);  // Direct cast
    }
}

// Kernel to convert std::complex<double> to std::complex<float>
__global__ void convertDoubleToFloatComplexKernel(cuDoubleComplex* input, cuComplex* output, std::size_t size) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = make_cuComplex(__double2float_rn(cuCreal(input[idx])), 
                                            __double2float_rn(cuCimag(input[idx])));
    }
}

// Kernel to convert std::complex<float> to std::complex<double>
__global__ void convertFloatToDoubleComplexKernel(cuComplex* input, cuDoubleComplex* output, std::size_t size) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = make_cuDoubleComplex(static_cast<double>(cuCrealf(input[idx])),
                                            static_cast<double>(cuCimagf(input[idx])));
    }
}

// Function to launch the conversion from double to float
void convert_DP_TO_SP_GPU(double* d_input, float* d_output, std::size_t size, cudaStream_t *stream_) {
    int numBlocks = (size + blockSize - 1) / blockSize;
    cudaStream_t usedStream = (stream_ == nullptr) ? 0 : *stream_;
    convertDoubleToFloatKernel<<<numBlocks, blockSize, 0, usedStream>>>(d_input, d_output, size);
}

// Function to launch the conversion from double to float complex
void convert_DP_TO_SP_GPU(std::complex<double>* d_input, std::complex<float>* d_output, std::size_t size, cudaStream_t *stream_) {
    int numBlocks = (size + blockSize - 1) / blockSize;
    cudaStream_t usedStream = (stream_ == nullptr) ? 0 : *stream_;
    convertDoubleToFloatComplexKernel<<<numBlocks, blockSize, 0, usedStream>>>(reinterpret_cast<cuDoubleComplex*>(d_input), reinterpret_cast<cuComplex*>(d_output), size);
}

// Function to launch the conversion from float to double
void convert_SP_TO_DP_GPU(float* d_input, double* d_output, std::size_t size, cudaStream_t *stream_) {
    int numBlocks = (size + blockSize - 1) / blockSize;
    cudaStream_t usedStream = (stream_ == nullptr) ? 0 : *stream_;
    convertFloatToDoubleKernel<<<numBlocks, blockSize, 0, usedStream>>>(d_input, d_output, size);
}

// Function to launch the conversion from float to double complex
void convert_SP_TO_DP_GPU(std::complex<float>* d_input, std::complex<double>* d_output, std::size_t size, cudaStream_t *stream_) {
    int numBlocks = (size + blockSize - 1) / blockSize;
    cudaStream_t usedStream = (stream_ == nullptr) ? 0 : *stream_;
    convertFloatToDoubleComplexKernel<<<numBlocks, blockSize, 0, usedStream>>>(reinterpret_cast<cuComplex*>(d_input), reinterpret_cast<cuDoubleComplex*>(d_output), size);
}

}
}
}
}
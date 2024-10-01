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
// Kernel to convert double to float
__global__ void convertDoubleToFloatKernel(double* input, float* output, std::size_t size);

// Kernel to convert float to double
__global__ void convertFloatToDoubleKernel(float* input, double* output, std::size_t size);

// Kernel to convert std::complex<double> to std::complex<float>
__global__ void convertDoubleToFloatComplexKernel(cuDoubleComplex* input, cuComplex* output, std::size_t size);

// Kernel to convert std::complex<float> to std::complex<double>
__global__ void convertFloatToDoubleComplexKernel(cuComplex* input, cuDoubleComplex* output, std::size_t size);

void convert_DP_TO_SP_GPU(double* d_input, float* d_output, std::size_t size, cudaStream_t *stream_ = nullptr);
void convert_DP_TO_SP_GPU(std::complex<double>* d_input, std::complex<float>* d_output, std::size_t size, cudaStream_t *stream_ = nullptr);

// Function to launch the conversion from float to double
void convert_SP_TO_DP_GPU(float* d_input, double* d_output, std::size_t size, cudaStream_t *stream_ = nullptr);
void convert_SP_TO_DP_GPU(std::complex<float>* d_input, std::complex<double>* d_output, std::size_t size, cudaStream_t *stream_ = nullptr);
}
}
}
}

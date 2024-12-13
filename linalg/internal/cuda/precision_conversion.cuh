// This file is a part of ChASE.
// Copyright (c) 2015-2024, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

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

/**
 * @brief Converts double precision data to single precision on the GPU.
 * 
 * This function performs the conversion of an array of double precision floating-point values
 * (`double`) to single precision floating-point values (`float`) on the GPU.
 * The operation is performed asynchronously with respect to the given CUDA stream, if provided.
 *
 * @param d_input Pointer to the input array of double precision floating-point values on the GPU.
 * @param d_output Pointer to the output array of single precision floating-point values on the GPU.
 * @param size The number of elements in the input/output arrays.
 * @param stream_ Optional CUDA stream for asynchronous execution. If `nullptr`, the default stream is used.
 */
void convert_DP_TO_SP_GPU(double* d_input, float* d_output, std::size_t size, cudaStream_t *stream_ = nullptr);
/**
 * @brief Converts an array of complex double precision data to complex single precision on the GPU.
 * 
 * This function performs the conversion of an array of complex double precision values (`std::complex<double>`)
 * to complex single precision values (`std::complex<float>`) on the GPU.
 * The operation is performed asynchronously with respect to the given CUDA stream, if provided.
 *
 * @param d_input Pointer to the input array of complex double precision values on the GPU.
 * @param d_output Pointer to the output array of complex single precision values on the GPU.
 * @param size The number of elements in the input/output arrays.
 * @param stream_ Optional CUDA stream for asynchronous execution. If `nullptr`, the default stream is used.
 */
void convert_DP_TO_SP_GPU(std::complex<double>* d_input, std::complex<float>* d_output, std::size_t size, cudaStream_t *stream_ = nullptr);

// Function to launch the conversion from float to double
/**
 * @brief Converts single precision data to double precision on the GPU.
 * 
 * This function performs the conversion of an array of single precision floating-point values
 * (`float`) to double precision floating-point values (`double`) on the GPU.
 * The operation is performed asynchronously with respect to the given CUDA stream, if provided.
 *
 * @param d_input Pointer to the input array of single precision floating-point values on the GPU.
 * @param d_output Pointer to the output array of double precision floating-point values on the GPU.
 * @param size The number of elements in the input/output arrays.
 * @param stream_ Optional CUDA stream for asynchronous execution. If `nullptr`, the default stream is used.
 */
void convert_SP_TO_DP_GPU(float* d_input, double* d_output, std::size_t size, cudaStream_t *stream_ = nullptr);
/**
 * @brief Converts an array of complex single precision data to complex double precision on the GPU.
 * 
 * This function performs the conversion of an array of complex single precision values (`std::complex<float>`)
 * to complex double precision values (`std::complex<double>`) on the GPU.
 * The operation is performed asynchronously with respect to the given CUDA stream, if provided.
 *
 * @param d_input Pointer to the input array of complex single precision values on the GPU.
 * @param d_output Pointer to the output array of complex double precision values on the GPU.
 * @param size The number of elements in the input/output arrays.
 * @param stream_ Optional CUDA stream for asynchronous execution. If `nullptr`, the default stream is used.
 */
void convert_SP_TO_DP_GPU(std::complex<float>* d_input, std::complex<double>* d_output, std::size_t size, cudaStream_t *stream_ = nullptr);
}
}
}
}

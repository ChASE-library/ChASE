// This file is a part of ChASE.
// Copyright (c) 2015-2024, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once
#include <cuda_runtime.h>
#include <iostream>

// Macro to check for CUDA errors
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, const char* const func, const char* const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

/**
 * @namespace chase::cuda::utils
 *
 * @brief Contains utility functions and structures for CUDA operations.
 *
 * This namespace includes custom utilities for managing GPU memory, such as a
 * custom deleter for freeing CUDA memory.
 */
namespace chase
{
namespace cuda
{
namespace utils
{
/**
 * @struct CudaDeleter
 *
 * @brief Custom deleter for freeing GPU memory managed by `std::unique_ptr`.
 *
 * This structure provides a custom `operator()` that is used to release GPU
 * memory using `cudaFree`. It is intended to be used with `std::unique_ptr` to
 * automatically free GPU memory when the pointer goes out of scope.
 */
struct CudaDeleter
{
    /**
     * @brief Frees the GPU memory pointed to by the provided pointer.
     *
     * This function is invoked when a `std::unique_ptr` with this custom
     * deleter goes out of scope. It ensures that the GPU memory is properly
     * freed using `cudaFree`.
     *
     * @tparam T The type of the pointer (e.g., `float*`, `int*`, etc.).
     * @param ptr A pointer to the memory that needs to be freed.
     */
    template <typename T>
    void operator()(T* ptr) const
    {
        cudaFree(static_cast<void*>(ptr)); // Cast to void* for cudaFree
        // std::cout << "cuda buffer managed by unique_ptr is freed" <<
        // std::endl;
    }
};

} // namespace utils
} // namespace cuda
} // namespace chase

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

namespace chase
{
namespace cuda
{
namespace utils
{
// Custom deleter for GPU memory
    struct CudaDeleter {
        template <typename T>
        void operator()(T* ptr) const {
            cudaFree(static_cast<void*>(ptr));  // Cast to void* for cudaFree
            //std::cout << "cuda buffer managed by unique_ptr is freed" << std::endl;
        }
    };

}
}
}

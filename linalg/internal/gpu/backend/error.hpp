// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#if defined(HAS_CUDA)
#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#endif

#if defined(HAS_HIP)
#include <hip/hip_runtime.h>

#include <stdexcept>
#include <string>
#endif

namespace chase
{
namespace linalg
{
namespace internal
{
namespace gpu_backend
{

#if defined(HAS_CUDA)
inline void check_cuda_error(cudaError_t err, const char* const expression,
                             const char* const file, int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << expression
                  << std::endl;
        std::exit(EXIT_FAILURE);
    }
}
#endif

#if defined(HAS_HIP)
inline void check_hip_error(hipError_t status, const char* expression,
                            const char* file, int line)
{
    if (status != hipSuccess)
    {
        throw std::runtime_error(std::string("HIP error in ") + expression +
                                 " at " + file + ":" + std::to_string(line) +
                                 ": " + hipGetErrorString(status));
    }
}
#endif

} // namespace gpu_backend
} // namespace internal
} // namespace linalg
} // namespace chase

#if defined(HAS_CUDA)
#define CHECK_CUDA_ERROR(expr)                                                 \
    ::chase::linalg::internal::gpu_backend::check_cuda_error(                  \
        (expr), #expr, __FILE__, __LINE__)
#endif

#if defined(HAS_HIP)
#define CHECK_HIP_ERROR(expr)                                                  \
    ::chase::linalg::internal::gpu_backend::check_hip_error(                   \
        (expr), #expr, __FILE__, __LINE__)
#endif

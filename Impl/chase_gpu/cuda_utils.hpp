// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "linalg/internal/gpu/backend/error.hpp"
#include <cuda_runtime.h>

/**
 * @namespace chase::cuda::utils
 *
 * @brief Contains utility functions and structures for CUDA operations.
 */
namespace chase
{
namespace cuda
{
namespace utils
{
struct CudaDeleter
{
    template <typename T>
    void operator()(T* ptr) const
    {
        cudaFree(static_cast<void*>(ptr));
    }
};

} // namespace utils
} // namespace cuda
} // namespace chase

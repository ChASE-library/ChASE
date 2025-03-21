// This file is a part of ChASE.
// Copyright (c) 2015-2024, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "flipSign.cuh"
#include "linalg/matrix/matrix.hpp"
#include "algorithm/types.hpp"
#include "Impl/chase_gpu/nvtx.hpp"

namespace chase
{
namespace linalg
{
namespace internal
{
namespace cuda
{
    /**
     * @brief Shifts the diagonal elements of a matrix by a specified value.
     * 
     * This function adds a scalar `shift` to the diagonal elements of the matrix `H` on the GPU. 
     * The operation is performed asynchronously using the provided CUDA stream, or the default stream if none is provided.
     * 
     * @tparam T The data type of the matrix elements (e.g., `float`, `double`).
     * 
     * @param H The matrix on which the diagonal elements will be shifted. It is a matrix on the GPU.
     * @param shift The value to be added to the diagonal elements of the matrix.
     * @param stream_ Optional CUDA stream for asynchronous execution. If `nullptr`, the default stream is used.
     * 
     * @note The function modifies the matrix `H` in-place. The number of diagonal elements processed is determined 
     *       by the minimum of the number of rows and columns of `H`.
     */    
    template<typename T>
    void flipLowerHalfMatrixSign(chase::matrix::Matrix<T, chase::platform::GPU> * H, cudaStream_t* stream_ = nullptr)
    {
        SCOPED_NVTX_RANGE();

        cudaStream_t usedStream = (stream_ == nullptr) ? 0 : *stream_;
        chase_flipLowerHalfMatrixSign(H->data(), H->cols(), H->ld(), usedStream);
    }
}
}
}
}

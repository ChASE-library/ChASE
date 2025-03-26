// This file is a part of ChASE.
// Copyright (c) 2015-2024, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "shiftDiagonal.cuh"
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
    void shiftDiagonal(chase::matrix::Matrix<T, chase::platform::GPU> * H, chase::Base<T> shift, cudaStream_t* stream_ = nullptr)
    {
        SCOPED_NVTX_RANGE();

        cudaStream_t usedStream = (stream_ == nullptr) ? 0 : *stream_;
        std::size_t n = std::min(H->rows(), H->cols());
        chase_shift_matrix(H->data(), n, H->ld(), shift, usedStream);
    }
    
    /**
     * @brief Set the diagonal elements of a matrix by a specified value.
     * 
     * This function replaces all the entries of the diagonal by a speciafied value. 
     * The operation is performed asynchronously using the provided CUDA stream, or the default stream if none is provided.
     * 
     * @tparam T The data type of the matrix elements (e.g., `float`, `double`).
     * 
     * @param H The matrix on which the diagonal elements will be shifted. It is a matrix on the GPU.
     * @param value The value for the diagonal entries of H.
     * @param stream_ Optional CUDA stream for asynchronous execution. If `nullptr`, the default stream is used.
     * 
     * @note The function modifies the matrix `H` in-place. The number of diagonal elements processed is determined 
     *       by the minimum of the number of rows and columns of `H`.
     */    
    template<typename T>
    void setDiagonal(chase::matrix::Matrix<T, chase::platform::GPU> * H, T value, cudaStream_t* stream_ = nullptr)
    {
        SCOPED_NVTX_RANGE();

        cudaStream_t usedStream = (stream_ == nullptr) ? 0 : *stream_;
        std::size_t n = std::min(H->rows(), H->cols());
        chase_set_diagonal(H->data(), n, H->ld(), value, usedStream);
    }
    
    /**
     * @brief Scale the rows of a matrix by spciefied real values
     * 
     * This function scales the rows of the matrix by the entries of values. The values should be already be in GPU memory.
     * The operation is performed asynchronously using the provided CUDA stream, or the default stream if none is provided.
     * 
     * @tparam T The data type of the matrix elements (e.g., `float`, `double`).
     * 
     * @param H The matrix on which the diagonal elements will be shifted. It is a matrix on the GPU.
     * @param values The real values for scaling the rows of H.
     * @param stream_ Optional CUDA stream for asynchronous execution. If `nullptr`, the default stream is used.
     * 
     * @note The function modifies the matrix `H` in-place. 
     */    
    template<typename T>
    void scaleMatrixRows(chase::matrix::Matrix<T, chase::platform::GPU> * H, chase::Base<T>* values, cudaStream_t* stream_ = nullptr)
    {
        SCOPED_NVTX_RANGE();

        cudaStream_t usedStream = (stream_ == nullptr) ? 0 : *stream_;
        chase_scale_rows_matrix(H->data(), H->rows(), H->cols(), H->ld(), values, usedStream);
    }
    
    /**
     * @brief Returns the inverse of a real coefficient subtracted by the real part of the diagonal of a matrix
     * 
     * This function computes the inverse entries of a given real value subtracted by the real part of the diagonal of a matrix.
     * The output vector, called new_diag, is a stored within the GPU memory.
     * The operation is performed asynchronously using the provided CUDA stream, or the default stream if none is provided.
     * 
     * @tparam T The data type of the matrix elements (e.g., `float`, `double`).
     * 
     * @param H The matrix on which the diagonal elements will be shifted. It is a matrix on the GPU.
     * @param values The real values for scaling the rows of H.
     * @param stream_ Optional CUDA stream for asynchronous execution. If `nullptr`, the default stream is used.
     * 
     * @note The function modifies the matrix `H` in-place. 
     */    
    template<typename T>
    void subtractInverseDiagonal(chase::matrix::Matrix<T, chase::platform::GPU> * H, chase::Base<T> coef, chase::Base<T>* new_diag, cudaStream_t* stream_ = nullptr)
    {
        SCOPED_NVTX_RANGE();

        cudaStream_t usedStream = (stream_ == nullptr) ? 0 : *stream_;
        chase_subtract_inverse_diagonal(H->data(), H->cols(), H->ld(), coef, new_diag, usedStream);
    }
}
}
}
}

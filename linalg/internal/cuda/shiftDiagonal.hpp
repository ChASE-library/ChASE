// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "Impl/chase_gpu/nvtx.hpp"
#include "algorithm/types.hpp"
#include "linalg/matrix/matrix.hpp"
#include "shiftDiagonal.cuh"

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
 * This function adds a scalar `shift` to the diagonal elements of the matrix
 * `H` on the GPU. The operation is performed asynchronously using the provided
 * CUDA stream, or the default stream if none is provided.
 *
 * @tparam T The data type of the matrix elements (e.g., `float`, `double`).
 *
 * @param H The matrix on which the diagonal elements will be shifted. It is a
 * matrix on the GPU.
 * @param shift The value to be added to the diagonal elements of the matrix.
 * @param stream_ Optional CUDA stream for asynchronous execution. If `nullptr`,
 * the default stream is used.
 *
 * @note The function modifies the matrix `H` in-place. The number of diagonal
 * elements processed is determined by the minimum of the number of rows and
 * columns of `H`.
 */
template <typename T>
void shiftDiagonal(chase::matrix::Matrix<T, chase::platform::GPU>* H,
                   chase::Base<T> shift, cudaStream_t* stream_ = nullptr)
{
    SCOPED_NVTX_RANGE();

    cudaStream_t usedStream = (stream_ == nullptr) ? 0 : *stream_;
    std::size_t n = std::min(H->rows(), H->cols());
    chase_shift_matrix(H->data(), n, H->ld(), shift, usedStream);
}

/** Add real \p shift to each diagonal entry (column-major: \c A[i+i*lda]).
 *  Complex: increments the real part only (same as \c chase_shift_matrix). */
template <typename T>
void add_diagonal_shift(T* A, std::size_t n, std::size_t lda,
                        chase::Base<T> shift, cudaStream_t stream_)
{
    SCOPED_NVTX_RANGE();
    chase_shift_matrix(A, n, lda, shift, stream_);
}

template <typename T>
void add_diagonal_shift(T* A, std::size_t n, chase::Base<T> shift,
                        cudaStream_t stream_)
{
    add_diagonal_shift(A, n, n, shift, stream_);
}

/**
 * @brief Shift diagonal using a device-resident scalar.
 *
 * Computes precision-dependent `H(ii) += (*d_shift) * scale` on GPU:
 * - single / complex<float>: `scale = 10 * epsilon(float)`
 * - double / complex<double>: `scale = sqrt(shift_scale_num_rows) * epsilon(double)`
 * The scale is evaluated inside the CUDA kernel so the full shift is
 * stream-ordered with `*d_shift` (no separate host-computed scale). Same
 * convention as shifted Cholesky QR in the MPI path: pass the global row count
 * of \f$V\f$ (not the Gram order) when applicable; for a square Gram-only scale
 * use `n`.
 *
 * @note Non-blocking streams: enqueue after the producer of `d_shift` (e.g.
 * absTrace) on the same stream.
 */
template <typename T>
void shiftDiagonalFromDeviceShift(chase::matrix::Matrix<T, chase::platform::GPU>* H,
                                  const chase::Base<T>* d_shift,
                                  cudaStream_t* stream_,
                                  std::size_t shift_scale_num_rows)
{
    SCOPED_NVTX_RANGE();
    cudaStream_t usedStream = (stream_ == nullptr) ? 0 : *stream_;
    std::size_t n = std::min(H->rows(), H->cols());
    chase_shift_matrix_from_device_shift(H->data(), n, H->ld(), d_shift,
                                         shift_scale_num_rows, usedStream);
}

/** @overload Uses `min(rows,cols)` of \p H for `sqrt` factor (legacy Gram-only scale). */
template <typename T>
void shiftDiagonalFromDeviceShift(chase::matrix::Matrix<T, chase::platform::GPU>* H,
                                  const chase::Base<T>* d_shift,
                                  cudaStream_t* stream_ = nullptr)
{
    std::size_t gram_n = std::min(H->rows(), H->cols());
    shiftDiagonalFromDeviceShift(H, d_shift, stream_, gram_n);
}

/**
 * @brief Set the diagonal elements of a matrix by a specified value.
 *
 * This function replaces all the entries of the diagonal by a speciafied value.
 * The operation is performed asynchronously using the provided CUDA stream, or
 * the default stream if none is provided.
 *
 * @tparam T The data type of the matrix elements (e.g., `float`, `double`).
 *
 * @param H The matrix on which the diagonal elements will be shifted. It is a
 * matrix on the GPU.
 * @param value The value for the diagonal entries of H.
 * @param stream_ Optional CUDA stream for asynchronous execution. If `nullptr`,
 * the default stream is used.
 *
 * @note The function modifies the matrix `H` in-place. The number of diagonal
 * elements processed is determined by the minimum of the number of rows and
 * columns of `H`.
 */
template <typename T>
void setDiagonal(chase::matrix::Matrix<T, chase::platform::GPU>* H, T value,
                 cudaStream_t* stream_ = nullptr)
{
    SCOPED_NVTX_RANGE();

    cudaStream_t usedStream = (stream_ == nullptr) ? 0 : *stream_;
    std::size_t n = std::min(H->rows(), H->cols());
    chase_set_diagonal(H->data(), n, H->ld(), value, usedStream);
}

/**
 * @brief Scale the rows of a matrix by spciefied real values
 *
 * This function scales the rows of the matrix by the entries of values. The
 * values should be already be in GPU memory. The operation is performed
 * asynchronously using the provided CUDA stream, or the default stream if none
 * is provided.
 *
 * @tparam T The data type of the matrix elements (e.g., `float`, `double`).
 *
 * @param H The matrix on which the diagonal elements will be shifted. It is a
 * matrix on the GPU.
 * @param values The real values for scaling the rows of H.
 * @param stream_ Optional CUDA stream for asynchronous execution. If `nullptr`,
 * the default stream is used.
 *
 * @note The function modifies the matrix `H` in-place.
 */
template <typename T>
void scaleMatrixRows(chase::matrix::Matrix<T, chase::platform::GPU>* H,
                     chase::Base<T>* values, cudaStream_t* stream_ = nullptr)
{
    SCOPED_NVTX_RANGE();

    cudaStream_t usedStream = (stream_ == nullptr) ? 0 : *stream_;
    chase_scale_rows_matrix(H->data(), H->rows(), H->cols(), H->ld(), values,
                            usedStream);
}

/**
 * @brief Returns the inverse of a real coefficient subtracted by the real part
 * of the diagonal of a matrix
 *
 * This function computes the inverse entries of a given real value subtracted
 * by the real part of the diagonal of a matrix. The output vector, called
 * new_diag, is a stored within the GPU memory. The operation is performed
 * asynchronously using the provided CUDA stream, or the default stream if none
 * is provided.
 *
 * @tparam T The data type of the matrix elements (e.g., `float`, `double`).
 *
 * @param H The matrix on which the diagonal elements will be shifted. It is a
 * matrix on the GPU.
 * @param values The real values for scaling the rows of H.
 * @param stream_ Optional CUDA stream for asynchronous execution. If `nullptr`,
 * the default stream is used.
 *
 * @note The function modifies the matrix `H` in-place.
 */
template <typename T>
void subtractInverseDiagonal(chase::matrix::Matrix<T, chase::platform::GPU>* H,
                             chase::Base<T> coef, chase::Base<T>* new_diag,
                             cudaStream_t* stream_ = nullptr)
{
    SCOPED_NVTX_RANGE();

    cudaStream_t usedStream = (stream_ == nullptr) ? 0 : *stream_;
    chase_subtract_inverse_diagonal(H->data(), H->cols(), H->ld(), coef,
                                    new_diag, usedStream);
}

/**
 * @brief Returns the inverse of a real coefficient added to the real part of
 * the diagonal of a matrix
 *
 * This function computes the inverse entries of a given real value added to the
 * real part of the diagonal of a matrix. The output vector, called new_diag, is
 * a stored within the GPU memory. The operation is performed asynchronously
 * using the provided CUDA stream, or the default stream if none is provided.
 *
 * @tparam T The data type of the matrix elements (e.g., `float`, `double`).
 *
 * @param H The matrix on which the diagonal elements will be shifted. It is a
 * matrix on the GPU.
 * @param values The real values for scaling the rows of H.
 * @param stream_ Optional CUDA stream for asynchronous execution. If `nullptr`,
 * the default stream is used.
 *
 * @note The function modifies the matrix `H` in-place.
 */
template <typename T>
void plusInverseDiagonal(chase::matrix::Matrix<T, chase::platform::GPU>* H,
                         chase::Base<T> coef, chase::Base<T>* new_diag,
                         cudaStream_t* stream_ = nullptr)
{
    SCOPED_NVTX_RANGE();

    cudaStream_t usedStream = (stream_ == nullptr) ? 0 : *stream_;
    chase_plus_inverse_diagonal(H->data(), H->cols(), H->ld(), coef, new_diag,
                                usedStream);
}

} // namespace cuda
} // namespace internal
} // namespace linalg
} // namespace chase

#pragma once

#include "lacpy.cuh"
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
     * @brief Copies a matrix from one location to another on the device.
     * 
     * This function performs the copy of matrix `A` to matrix `B` based on the specified triangular part.
     * 
     * @tparam T The data type of the matrix (e.g., float, double, complex types).
     * @param uplo The triangular part of the matrix to copy:
     *             - 'U' for the upper triangular part
     *             - 'L' for the lower triangular part.
     * @param m The number of rows of matrix A.
     * @param n The number of columns of matrix A.
     * @param dA Pointer to the input matrix `A` on the device.
     * @param ldda The leading dimension of matrix `A`.
     * @param dB Pointer to the output matrix `B` on the device.
     * @param lddb The leading dimension of matrix `B`.
     * @param stream_ Optional CUDA stream to execute the operation (default is `nullptr`).
     */    
    template<typename T>
    void t_lacpy(char uplo, std::size_t m, std::size_t n, T *dA, std::size_t ldda, T *dB, std::size_t lddb, cudaStream_t *stream_ = nullptr )
    {
        SCOPED_NVTX_RANGE();

        cudaStream_t usedStream = (stream_ == nullptr) ? 0 : *stream_;
        t_lacpy_gpu(uplo, m, n, dA, ldda, dB, lddb, usedStream);
    }
    /**
     * @brief Extracts the upper triangular part of a matrix and stores it in a separate matrix.
     * 
     * This function copies the upper triangular part of the input matrix `d_matrix` into `d_upperTriangular`.
     * 
     * @tparam T The data type of the matrix (e.g., float, double, complex types).
     * @param d_matrix Pointer to the input matrix on the device.
     * @param ld The leading dimension of the input matrix.
     * @param d_upperTriangular Pointer to the output matrix where the upper triangular part will be stored.
     * @param n The number of rows/columns in the matrix.
     * @param stream_ Optional CUDA stream to execute the operation (default is `nullptr`).
     */
    template<typename T>
    void extractUpperTriangular(T* d_matrix, std::size_t ld, T* d_upperTriangular, std::size_t n, cudaStream_t *stream_ = nullptr )
    {
        cudaStream_t usedStream = (stream_ == nullptr) ? 0 : *stream_;
        extractUpperTriangular(d_matrix, ld, d_upperTriangular, n, usedStream);
    }
    /**
     * @brief Unpacks an upper triangular matrix from a compressed storage format into a full matrix.
     * 
     * This function reconstructs a full matrix from its upper triangular part stored in `d_upperTriangular`.
     * 
     * @tparam T The data type of the matrix (e.g., float, double, complex types).
     * @param d_upperTriangular Pointer to the input upper triangular matrix on the device.
     * @param n The size of the matrix (number of rows/columns).
     * @param d_matrix Pointer to the output full matrix on the device.
     * @param ld The leading dimension of the output matrix.
     * @param stream_ Optional CUDA stream to execute the operation (default is `nullptr`).
     */
    template<typename T>
    void unpackUpperTriangular(T* d_upperTriangular, std::size_t n, T* d_matrix, std::size_t ld, cudaStream_t *stream_ = nullptr )
    {
        cudaStream_t usedStream = (stream_ == nullptr) ? 0 : *stream_;
        unpackUpperTriangular(d_matrix, ld, d_upperTriangular, n, usedStream);
    }



}
}
}
}
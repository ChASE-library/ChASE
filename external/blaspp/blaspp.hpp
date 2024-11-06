#pragma once 

#include <complex>
#include "algorithm/types.hpp"

/**
 * @page blaspp_namespace chase::linalg::blaspp Namespace
 * @brief A templated C++ interface to the BLAS library.
 *
 * This namespace contains templated functions that interface with the BLAS library to provide efficient linear algebra routines.
 * The functions are templated to work with different data types such as `float`, `double`, and `std::complex`.
 * These functions allow for easy and efficient numerical computations in scientific computing.
 */

/**
 * @defgroup BlasFunctions BLAS Routines
 * @brief Template functions that interface with BLAS routines.
 * 
 * These functions provide common linear algebra operations such as vector norms, dot products, matrix multiplications, etc.
 * They allow for operations on both real and complex numbers.
 */

namespace chase
{
namespace linalg
{
namespace blaspp
{

#define CBLAS_LAYOUT int
#define CblasConjTrans 1
#define CblasTrans 2
#define CblasNoTrans 3
#define CblasColMajor 1
#define LAPACK_COL_MAJOR 1
#define CBLAS_TRANSPOSE int
#define lapack_logical int
#define CBLAS_UPLO int
#define CBLAS_SIDE int
#define CblasLeft 1
#define CblasLower 1

/**
 * @ingroup BlasFunctions
 * @brief Compute the square root of the L2 norm of a vector.
 * 
 * This function computes the square root of the L2 norm (Euclidean norm) of a vector `x`. It is equivalent to the `BLAS` function `nrm2`.
 * 
 * @tparam T The data type of the vector elements. Supported types: `float`, `double`, `std::complex<float>`, `std::complex<double>`.
 * 
 * @param x The input vector.
 * 
 * @return The square root of the L2 norm of the vector.
 */
template <typename T>
Base<T> t_sqrt_norm(T x);

/**
 * @ingroup BlasFunctions
 * @brief Compute Euclidean distrance of a vector.
 * 
 * This function computes the Euclidean distrance of a vector `x`, which is the sum of the squared components.
 * 
 * @tparam T The data type of the vector elements. Supported types: `float`, `double`, `std::complex<float>`, `std::complex<double>`.
 * 
 * @param n The number of elements in the vector.
 * @param x The input vector.
 * 
 * @return The sum of the squared components of the vector.
 */
template <typename T>
Base<T> t_norm_p2(const std::size_t n, const T* x);

/**
 * @ingroup BlasFunctions
 * @brief Compute the L2 norm of a vector.
 * 
 * This function computes the L2 norm (Euclidean norm) of a vector `x`. It is equivalent to the `BLAS` function `nrm2`.
 * 
 * @tparam T The data type of the vector elements. Supported types: `float`, `double`, `std::complex<float>`, `std::complex<double>`.
 * 
 * @param n The number of elements in the vector.
 * @param x The input vector.
 * @param incx The stride for elements of vector `x`.
 * 
 * @return The L2 norm of the vector.
 */
template <typename T>
Base<T> t_nrm2(const std::size_t n, const T* x, const std::size_t incx);

/**
 * @ingroup BlasFunctions
 * @brief Compute the dot product of two vectors.
 * 
 * This function computes the dot product of two vectors `x` and `y`. It is equivalent to the `BLAS` function `dot`.
 * 
 * @tparam T The data type of the vector elements. Supported types: `float`, `double`, `std::complex<float>`, `std::complex<double>`.
 * 
 * @param n The number of elements in the vectors.
 * @param x The first input vector.
 * @param incx The stride for elements of vector `x`.
 * @param y The second input vector.
 * @param incy The stride for elements of vector `y`.
 * 
 * @return The dot product of the two vectors.
 */
template <typename T>
T t_dot(const std::size_t n, const T* x, const std::size_t incx, const T* y,
        const std::size_t incy);

/**
 * @ingroup BlasFunctions
 * @brief Scale a vector by a constant.
 * 
 * This function scales a vector `x` by a scalar `a`. It is equivalent to the `BLAS` function `scal`.
 * 
 * @tparam T The data type of the vector elements. Supported types: `float`, `double`, `std::complex<float>`, `std::complex<double>`.
 * 
 * @param n The number of elements in the vector.
 * @param a The scalar value to multiply each element of the vector.
 * @param x The input/output vector to be scaled.
 * @param incx The stride for elements of vector `x`.
 */        
template <typename T>
void t_scal(const std::size_t n, const T* a, T* x, const std::size_t incx);

/**
 * @ingroup BlasFunctions
 * @brief Compute the rank-1 update of a vector.
 * 
 * This function performs a rank-1 update of vector `y` by adding the scaled vector `a * x`. It is equivalent to the `BLAS` function `axpy`.
 * 
 * @tparam T The data type of the vector elements. Supported types: `float`, `double`, `std::complex<float>`, `std::complex<double>`.
 * 
 * @param n The number of elements in the vectors.
 * @param a The scalar multiplier for vector `x`.
 * @param x The input vector `x`.
 * @param incx The stride for elements of vector `x`.
 * @param y The input/output vector `y`.
 * @param incy The stride for elements of vector `y`.
 */
template <typename T>
void t_axpy(const std::size_t n, const T* a, const T* x, const std::size_t incx,
            T* y, const std::size_t incy);

/**
 * @ingroup BlasFunctions
 * @brief Perform a symmetric rank-k update.
 * 
 * This function performs the symmetric rank-k update of the form `A = alpha * x * x' + beta * A`.
 * It is equivalent to the `BLAS` function `syherk`.
 * 
 * @tparam T The data type of the matrix elements. Supported types: `float`, `double`, `std::complex<float>`, `std::complex<double>`.
 * 
 * @param uplo Specifies which part of the matrix to update ('U' for upper, 'L' for lower).
 * @param trans Specifies the type of transpose ('N' for no transpose, 'T' for transpose, 'C' for conjugate transpose).
 * @param n The order of the matrix `A`.
 * @param k The dimension of vector `x`.
 * @param alpha The scalar multiplier for the rank-k update.
 * @param a The matrix to be updated.
 * @param lda The leading dimension of `A`.
 * @param beta The scalar multiplier for the matrix `A`.
 * @param c The matrix to receive the update.
 * @param ldc The leading dimension of `C`.
 */
template <typename T>
void t_syherk(const char uplo, const char trans, const std::size_t n,
              const std::size_t k, const T* alpha, T* a, const std::size_t lda,
              const T* beta, T* c, const std::size_t ldc);

/**
 * @ingroup BlasFunctions
 * @brief Solve a triangular system.
 * 
 * This function solves a triangular system `A * X = B` or `X * A = B`, where `A` is a triangular matrix.
 * It is equivalent to the `BLAS` function `trsm`.
 * 
 * @tparam T The data type of the matrix elements. Supported types: `float`, `double`, `std::complex<float>`, `std::complex<double>`.
 * 
 * @param side Specifies whether `A` is on the left or right side of the equation.
 * @param uplo Specifies whether the matrix `A` is upper or lower triangular.
 * @param trans Specifies the type of transpose ('N' for no transpose, 'T' for transpose, 'C' for conjugate transpose).
 * @param diag Specifies whether `A` is unit triangular.
 * @param m The number of rows in matrix `B` and solution matrix `X`.
 * @param n The number of columns in matrix `B` and solution matrix `X`.
 * @param alpha The scalar multiplier.
 * @param a The triangular matrix `A`.
 * @param lda The leading dimension of `A`.
 * @param b The right-hand side matrix `B`.
 * @param ldb The leading dimension of `B`.
 */
template <typename T>
void t_trsm(const char side, const char uplo, const char trans, const char diag,
            const std::size_t m, const std::size_t n, const T* alpha,
            const T* a, const std::size_t lda, const T* b,
            const std::size_t ldb);

/**
 * @ingroup BlasFunctions
 * @brief Perform a matrix-matrix multiplication.
 * 
 * This function performs the matrix multiplication `C = alpha * A * B + beta * C`.
 * It is equivalent to the `BLAS` function `gemm`.
 * 
 * @tparam T The data type of the matrix elements. Supported types: `float`, `double`, `std::complex<float>`, `std::complex<double>`.
 * 
 * @param Layout Specifies the memory layout (row-major or column-major).
 * @param transa Specifies the transpose type for matrix `A`.
 * @param transb Specifies the transpose type for matrix `B`.
 * @param m The number of rows of matrix `A`.
 * @param n The number of columns of matrix `B`.
 * @param k The number of columns of matrix `A` (and rows of matrix `B`).
 * @param alpha The scalar multiplier for the product `A * B`.
 * @param a The first matrix `A`.
 * @param lda The leading dimension of `A`.
 * @param b The second matrix `B`.
 * @param ldb The leading dimension of `B`.
 * @param beta The scalar multiplier for matrix `C`.
 * @param c The matrix `C` to receive the result.
 * @param ldc The leading dimension of `C`.
 */
template <typename T>
void t_gemm(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE transa,
            const CBLAS_TRANSPOSE transb, const std::size_t m,
            const std::size_t n, const std::size_t k, const T* alpha,
            const T* a, const std::size_t lda, const T* b,
            const std::size_t ldb, const T* beta, T* c, const std::size_t ldc);

} //end of namespace blaspp
} //end of namespace linalg   
} //end of namespace chase

#include "blaspp.inc"
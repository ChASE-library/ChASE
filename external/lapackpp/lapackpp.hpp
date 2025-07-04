// This file is a part of ChASE.
// Copyright (c) 2015-2024, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include <complex>
#include "algorithm/types.hpp"
#include "external/blaspp/blaspp.hpp"

/**
 * @page lapackpp_namespace chase::linalg::lapackpp Namespace
 * @brief A templated C++ interface to the LAPACK library.
 *
 * This namespace `chase::linalg::lapackpp` contains templated functions that interface with the LAPACK library to provide efficient linear algebra routines.
 * The functions are templated to work with different data types such as `float`, `double`, and `std::complex`.
 * These functions allow for easy and efficient numerical computations in scientific computing.
 */

/**
 * @defgroup LapackFunctions LAPACK Routines
 * @brief Template functions that interface with LAPACK routines.
 * 
 * These functions provide common linear algebra operations such as QR factorization, etc.
 * They allow for operations on both real and complex numbers.
 */

namespace chase
{
namespace linalg
{
namespace lapackpp
{
/**
 * @ingroup LapackFunctions
 * @brief Copies a matrix from one location to another.
 * 
 * This function performs the operation of copying matrix `A` into matrix `B`.
 * It is equivalent to the `LAPACK` function `lacpy`.
 * 
 * @tparam T The data type of the matrix elements.
 * 
 * @param uplo Specifies which part of the matrix to copy:
 *             - 'U' for upper triangle,
 *             - 'L' for lower triangle,
 *             - otherwise, copies the entire matrix.
 * @param m The number of rows of matrix `A`.
 * @param n The number of columns of matrix `A`.
 * @param a The source matrix `A`.
 * @param lda The leading dimension of `A`.
 * @param b The destination matrix `B`.
 * @param ldb The leading dimension of `B`.
 */
template <typename T>
void t_lacpy(const char uplo, const std::size_t m, const std::size_t n,
             const T* a, const std::size_t lda, T* b, const std::size_t ldb);
/**
 * @ingroup LapackFunctions
 * @brief Computes the QR factorization of a matrix.
 * 
 * This function computes the QR factorization of a matrix `A`, storing the
 * resulting R matrix in `A` and the scalar factors of the elementary reflectors in `tau`.
 * 
 * @tparam T The data type of the matrix elements.
 * 
 * @param matrix_layout Specifies the memory layout (row-major or column-major).
 * @param m The number of rows of matrix `A`.
 * @param n The number of columns of matrix `A`.
 * @param a The matrix `A` to factorize. On exit, contains the R matrix and Householder reflectors.
 * @param lda The leading dimension of `A`.
 * @param tau Output array for scalar factors of the elementary reflectors.
 * @return The optimal block size.
 */
template <typename T>
std::size_t t_geqrf(int matrix_layout, std::size_t m, std::size_t n, T* a,
                    std::size_t lda, T* tau);
/**
 * @ingroup LapackFunctions
 * @brief Generates the orthogonal or unitary matrix Q from a QR factorization.
 * 
 * This function generates the matrix `Q` from a previously computed QR factorization.
 * 
 * @tparam T The data type of the matrix elements.
 * 
 * @param matrix_layout Specifies the memory layout (row-major or column-major).
 * @param m The number of rows of matrix `A`.
 * @param n The number of columns of matrix `A`.
 * @param k The number of elementary reflectors.
 * @param a The matrix `A` containing the elementary reflectors.
 * @param lda The leading dimension of `A`.
 * @param tau The scalar factors of the elementary reflectors.
 * @return The optimal block size.
 */
template <typename T>
std::size_t t_gqr(int matrix_layout, std::size_t m, std::size_t n,
                  std::size_t k, T* a, std::size_t lda, const T* tau);
/**
 * @ingroup LapackFunctions
 * @brief Computes the Cholesky factorization of a symmetric positive definite matrix.
 * 
 * This function computes the Cholesky factorization of a matrix `A`.
 * It is equivalent to the `LAPACK` function `potrf`.
 * 
 * @tparam T The data type of the matrix elements.
 * 
 * @param uplo Specifies which part of the matrix to use for factorization:
 *             - 'U' for upper triangle,
 *             - 'L' for lower triangle.
 * @param n The order of matrix `A`.
 * @param a The matrix `A` to factorize. On exit, contains the Cholesky factor.
 * @param lda The leading dimension of `A`.
 * @return 0 if successful, or a non-zero value if the matrix is not positive definite.
 */
template <typename T>
int t_potrf(const char uplo, const std::size_t n, T* a, const std::size_t lda);
/**
 * @ingroup LapackFunctions
 * @brief Computes eigenvalues and optionally eigenvectors of a symmetric tridiagonal matrix.
 * 
 * This function computes all eigenvalues or a subset of eigenvalues and, optionally,
 * eigenvectors of a symmetric tridiagonal matrix.
 * 
 * @tparam T The data type of the matrix elements.
 * 
 * @param matrix_layout Specifies the memory layout (row-major or column-major).
 * @param jobz Specifies whether to compute eigenvalues only or eigenvalues and eigenvectors:
 *             - 'N' for eigenvalues only,
 *             - 'V' for both eigenvalues and eigenvectors.
 * @param range Specifies the range of eigenvalues to find:
 *              - 'A' for all eigenvalues,
 *              - 'V' for eigenvalues in the half-open interval [vl, vu),
 *              - 'I' for eigenvalues with indices in the range [il, iu].
 * @param n The order of the matrix.
 * @param d The diagonal elements of the matrix.
 * @param e The off-diagonal elements of the matrix.
 * @param vl Lower bound of the interval (if `range` is 'V').
 * @param vu Upper bound of the interval (if `range` is 'V').
 * @param il Lower index of the interval (if `range` is 'I').
 * @param iu Upper index of the interval (if `range` is 'I').
 * @param m Output, the total number of eigenvalues found.
 * @param w Output, the eigenvalues.
 * @param z Output, the eigenvectors (if `jobz` is 'V').
 * @param ldz The leading dimension of `z`.
 * @param nzc The maximum number of columns in `z`.
 * @param isuppz Output, the indices of eigenvectors (if `jobz` is 'V').
 * @param tryrac Input, suggests whether to use faster processing.
 * @return The optimal block size.
 */
template <typename T>
std::size_t t_stemr(int matrix_layout, char jobz, char range, std::size_t n,
                    T* d, T* e, T vl, T vu, std::size_t il, std::size_t iu,
                    int* m, T* w, T* z, std::size_t ldz, std::size_t nzc,
                    int* isuppz, lapack_logical* tryrac);
/**
 * @ingroup LapackFunctions
 * @brief Computes all eigenvalues and, optionally, eigenvectors of a Hermitian matrix.
 * 
 * This function computes all eigenvalues and, optionally, eigenvectors of a Hermitian
 * matrix using a divide-and-conquer algorithm.
 * 
 * @tparam T The data type of the matrix elements.
 * 
 * @param matrix_layout Specifies the memory layout (row-major or column-major).
 * @param jobz Specifies whether to compute eigenvalues only or eigenvalues and eigenvectors:
 *             - 'N' for eigenvalues only,
 *             - 'V' for both eigenvalues and eigenvectors.
 * @param uplo Specifies which part of the matrix to use:
 *             - 'U' for upper triangle,
 *             - 'L' for lower triangle.
 * @param n The order of the matrix `A`.
 * @param a The matrix `A` to compute eigenvalues and eigenvectors for.
 * @param lda The leading dimension of `A`.
 * @param w Output, the eigenvalues.
 * @return The optimal block size.
 */
template <typename T>
std::size_t t_heevd(int matrix_layout, char jobz, char uplo, std::size_t n,
                    T* a, std::size_t lda, Base<T>* w);

/**
 * @ingroup LapackFunctions
 * @brief Computes all eigenvalues and, optionally, eigenvectors of a matrix.
 * 
 * This function computes all eigenvalues and, optionally, right eigenvectors 
 * of a Hermitian.
 * 
 * @tparam T The data type of the matrix elements.
 * 
 * @param matrix_layout Specifies the memory layout (row-major or column-major).
 * @param jobz Specifies whether to compute eigenvalues only or eigenvalues and eigenvectors:
 *             - 'N' for eigenvalues only,
 *             - 'V' for both eigenvalues and right eigenvectors.
 * @param n The order of the matrix `A`.
 * @param a The matrix `A` to compute eigenvalues and eigenvectors for.
 * @param lda The leading dimension of `A`.
 * @param wr Output, the real part of eigenvalues.
 * @param wi Output, the imag part of eigenvalues.
 * @param v Output, the eigenvectors.
 * @param ldv The leading dimension of `v`.
 * @return The optimal block size.
 */
template <typename T>
std::size_t t_geev(int matrix_layout, char jobz, std::size_t n,
                    T* a, std::size_t lda, Base<T>* wr, Base<T> * wi, T* v, std::size_t ldv);

/**
 * @ingroup LapackFunctions
 * @brief Computes the singular value decomposition of a matrix.
 * 
 * This function computes the singular value decomposition of a matrix `A`.
 * 
 * @tparam T The data type of the matrix elements.
 * 
 * @param jobu Specifies whether to compute singular values only or singular values and vectors:
 *             - 'N' for singular values only,
 *             - 'S' for singular values and left singular vectors,
 *             - 'O' for singular values and right singular vectors.    
 * @param jobvt Specifies whether to compute singular values only or singular values and vectors:
 *             - 'N' for singular values only,
 *             - 'S' for singular values and left singular vectors,
 *             - 'O' for singular values and right singular vectors.
 * @param m The number of rows of matrix `A`.
 * @param n The number of columns of matrix `A`.
 * @param a The matrix `A` to compute singular value decomposition for.
 * @param lda The leading dimension of `A`.
 * @param s Output, the singular values.
 * @param u Output, the left singular vectors.
 * @param ldu The leading dimension of `u`.
 * @param vt Output, the right singular vectors.
 * @param ldvt The leading dimension of `vt`.
 */
template <typename T>
void t_gesvd(const char jobu, const char jobvt, const std::size_t m,
             const std::size_t n, T* A, const std::size_t lda, Base<T>* S, T* U,
             const std::size_t ldu, T* Vt, const std::size_t ldvt);


} //end of namespace lapackpp
} //end of namespace linalg   
} //end of namespace chase

#include "lapackpp.inc"

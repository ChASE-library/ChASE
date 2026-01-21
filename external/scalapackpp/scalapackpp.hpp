// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include <complex>

#include "algorithm/types.hpp"

#ifdef HAS_SCALAPACK
/**
 * @page scalapackpp_namespace chase::linalg::scalapackpp Namespace
 * @brief A templated C++ interface to the ScaLAPACK library.
 *
 * This namespace `chase::linalg::scalapackpp` contains templated functions that
 * interface with the ScaLAPACK library to provide efficient distributed-memory
 * linear algebra routines. The functions are templated to work with different
 * data types such as `float`, `double`, and `std::complex`. These functions
 * allow for easy and efficient numerical computations in scientific computing.
 */

/**
 * @defgroup ScalapackFunctions ScaLAPACK Routines
 * @brief Template functions that interface with ScaLAPACK routines.
 *
 * These functions provide common linear algebra operations such as QR
 * factorization, etc. They allow for operations on both real and complex
 * numbers.
 */
namespace chase
{
namespace linalg
{
namespace scalapackpp
{
extern "C" void blacs_get_(int*, int*, int*);
extern "C" void blacs_pinfo_(int*, int*);
extern "C" void blacs_gridinit_(int*, char*, int*, int*);
extern "C" void blacs_gridinfo_(int*, int*, int*, int*, int*);
extern "C" void descinit_(int*, int*, int*, int*, int*, int*, int*, int*, int*,
                          int*);
extern "C" void blacs_gridexit_(int*);
extern "C" void blacs_gridmap_(int*, int*, int*, int*, int*);
extern "C" int numroc_(std::size_t*, std::size_t*, int*, int*, int*);

extern "C" void pdgeqrf_(int*, int*, double*, int*, int*, int*, double*,
                         double*, int*, int*);
extern "C" void psgeqrf_(int*, int*, float*, int*, int*, int*, float*, float*,
                         int*, int*);
extern "C" void pcgeqrf_(int*, int*, std::complex<float>*, int*, int*, int*,
                         std::complex<float>*, std::complex<float>*, int*,
                         int*);
extern "C" void pzgeqrf_(int*, int*, std::complex<double>*, int*, int*, int*,
                         std::complex<double>*, std::complex<double>*, int*,
                         int*);

extern "C" void pdorgqr_(int*, int*, int*, double*, int*, int*, int*, double*,
                         double*, int*, int*);
extern "C" void psorgqr_(int*, int*, int*, float*, int*, int*, int*, float*,
                         float*, int*, int*);
extern "C" void pzungqr_(int*, int*, int*, std::complex<double>*, int*, int*,
                         int*, std::complex<double>*, std::complex<double>*,
                         int*, int*);
extern "C" void pcungqr_(int*, int*, int*, std::complex<float>*, int*, int*,
                         int*, std::complex<float>*, std::complex<float>*, int*,
                         int*);

extern "C" void pstran_(int*, int*, float*, float*, int*, int*, int*, float*,
                        float*, int*, int*, int*);
extern "C" void pdtran_(int*, int*, double*, double*, int*, int*, int*, double*,
                        double*, int*, int*, int*);
extern "C" void pctranc_(int*, int*, std::complex<float>*, std::complex<float>*,
                         int*, int*, int*, std::complex<float>*,
                         std::complex<float>*, int*, int*, int*);
extern "C" void pztranc_(int*, int*, std::complex<double>*,
                         std::complex<double>*, int*, int*, int*,
                         std::complex<double>*, std::complex<double>*, int*,
                         int*, int*);

// Single precision (float)
extern "C" void psgemr2d_(int* m, int* n, float* A, int* ia, int* ja,
                          int* desca, float* B, int* ib, int* jb, int* descb,
                          int* ictxt);

// Double precision (double)
extern "C" void pdgemr2d_(int* m, int* n, double* A, int* ia, int* ja,
                          int* desca, double* B, int* ib, int* jb, int* descb,
                          int* ictxt);

// Single precision complex (std::complex<float>)
extern "C" void pcgemr2d_(int* m, int* n, std::complex<float>* A, int* ia,
                          int* ja, int* desca, std::complex<float>* B, int* ib,
                          int* jb, int* descb, int* ictxt);

// Double precision complex (std::complex<double>)
extern "C" void pzgemr2d_(int* m, int* n, std::complex<double>* A, int* ia,
                          int* ja, int* desca, std::complex<double>* B, int* ib,
                          int* jb, int* descb, int* ictxt);

extern "C" void pssyevd_(char*, char*, int*, float*, int*, int*, int*, float*,
                         float*, int*, int*, int*, float*, int*, int*, int*,
                         int*);
extern "C" void pdsyevd_(char*, char*, int*, double*, int*, int*, int*, double*,
                         double*, int*, int*, int*, double*, int*, int*, int*,
                         int*);
extern "C" void pcheevd_(char*, char*, int*, std::complex<float>*, int*, int*,
                         int*, float*, std::complex<float>*, int*, int*, int*,
                         std::complex<float>*, int*, float*, int*, int*, int*,
                         int*);
extern "C" void pzheevd_(char*, char*, int*, std::complex<double>*, int*, int*,
                         int*, double*, std::complex<double>*, int*, int*, int*,
                         std::complex<double>*, int*, double*, int*, int*, int*,
                         int*);

extern "C" void psgesvd_(char*, char*, int*, int*, float*, int*, int*, int*,
                         float*, float*, int*, int*, int*, float*, int*, int*,
                         int*, float*, int*, int*);
extern "C" void pdgesvd_(char*, char*, int*, int*, double*, int*, int*, int*,
                         double*, double*, int*, int*, int*, double*, int*,
                         int*, int*, double*, int*, int*);
extern "C" void pcgesvd_(char*, char*, int*, int*, std::complex<float>*, int*,
                         int*, int*, float*, std::complex<float>*, int*, int*,
                         int*, std::complex<float>*, int*, int*, int*,
                         std::complex<float>*, int*, float*, int*);
extern "C" void pzgesvd_(char*, char*, int*, int*, std::complex<double>*, int*,
                         int*, int*, double*, std::complex<double>*, int*, int*,
                         int*, std::complex<double>*, int*, int*, int*,
                         std::complex<double>*, int*, double*, int*);

/**
 * @ingroup ScalapackFunctions
 * @brief Computes the QR factorization of a distributed matrix.
 *
 * This function computes the QR factorization of a distributed matrix `A`.
 * It is equivalent to the `ScaLAPACK` function `pgeqrf`.
 *
 * @tparam T The data type of the matrix elements.
 *
 * @param m The number of rows in the matrix `A`.
 * @param n The number of columns in the matrix `A`.
 * @param A The matrix `A` to factorize. On exit, contains the R matrix and
 * Householder reflectors.
 * @param ia The row index in the global matrix `A` indicating the starting
 * position.
 * @param ja The column index in the global matrix `A` indicating the starting
 * position.
 * @param desc_a Array descriptor for the distributed matrix `A`.
 * @param tau Output array for scalar factors of the elementary reflectors.
 */
template <typename T>
void t_pgeqrf(std::size_t m, std::size_t n, T* A, int ia, int ja,
              std::size_t* desc_a, T* tau);
/**
 * @ingroup ScalapackFunctions
 * @brief Generates the orthogonal or unitary matrix Q from a QR factorization
 * of a distributed matrix.
 *
 * This function generates the matrix `Q` from a previously computed QR
 * factorization.
 *
 * @tparam T The data type of the matrix elements.
 *
 * @param m The number of rows in the matrix `A`.
 * @param n The number of columns in the matrix `A`.
 * @param k The number of elementary reflectors.
 * @param A The matrix `A` containing the elementary reflectors.
 * @param ia The row index in the global matrix `A` indicating the starting
 * position.
 * @param ja The column index in the global matrix `A` indicating the starting
 * position.
 * @param desc_a Array descriptor for the distributed matrix `A`.
 * @param tau The scalar factors of the elementary reflectors.
 */
template <typename T>
void t_pgqr(std::size_t m, std::size_t n, std::size_t k, T* A, int ia, int ja,
            std::size_t* desc_a, T* tau);

/**
 * @ingroup ScalapackFunctions
 * @brief Computes the conjugate transpose of a distributed matrix and scales
 * the result.
 *
 * This function computes the conjugate transpose of matrix `A`, scales it by
 * `alpha`, and adds it to matrix `C` scaled by `beta`.
 *
 * @tparam T The data type of the matrix elements.
 *
 * @param m The number of rows in the matrix `A`.
 * @param n The number of columns in the matrix `A`.
 * @param alpha Scalar multiplier for matrix `A`.
 * @param A The matrix `A` to transpose.
 * @param ia The row index in the global matrix `A` indicating the starting
 * position.
 * @param ja The column index in the global matrix `A` indicating the starting
 * position.
 * @param desc_a Array descriptor for the distributed matrix `A`.
 * @param beta Scalar multiplier for matrix `C`.
 * @param C The matrix `C` to accumulate the result.
 * @param ic The row index in the global matrix `C` indicating the starting
 * position.
 * @param jc The column index in the global matrix `C` indicating the starting
 * position.
 * @param desc_c Array descriptor for the distributed matrix `C`.
 */
template <typename T>
void t_ptranc(std::size_t m, std::size_t n, T alpha, T* A, int ia, int ja,
              std::size_t* desc_a, T beta, T* C, int ic, int jc,
              std::size_t* desc_c);

/**
 * @ingroup ScalapackFunctions
 * @brief Redistributes a distributed matrix from one process grid to another.
 *
 * This function performs a matrix redistribution between two distributed
 * matrices `A` and `B`.
 *
 * @tparam T The data type of the matrix elements.
 *
 * @param m The number of rows to redistribute.
 * @param n The number of columns to redistribute.
 * @param A The source matrix `A`.
 * @param ia The row index in the global matrix `A` indicating the starting
 * position.
 * @param ja The column index in the global matrix `A` indicating the starting
 * position.
 * @param desc_a Array descriptor for the source matrix `A`.
 * @param B The destination matrix `B`.
 * @param ib The row index in the global matrix `B` indicating the starting
 * position.
 * @param jb The column index in the global matrix `B` indicating the starting
 * position.
 * @param desc_b Array descriptor for the destination matrix `B`.
 * @param ictxt The context identifier for the source and destination matrices.
 */
template <typename T>
void t_pgemr2d(std::size_t m, std::size_t n, T* A, int ia, int ja,
               std::size_t* desc_a, T* B, int ib, int jb, std::size_t* desc_b,
               int ictxt);
/**
 * @ingroup ScalapackFunctions
 * @brief Computes all eigenvalues and, optionally, eigenvectors of a Hermitian
 * matrix.
 *
 * This function computes all eigenvalues and, optionally, eigenvectors of a
 * Hermitian matrix distributed across a process grid using the
 * divide-and-conquer algorithm.
 *
 * @tparam T The data type of the matrix elements.
 *
 * @param jobz Specifies whether to compute eigenvalues only or eigenvalues and
 * eigenvectors:
 *             - 'N' for eigenvalues only,
 *             - 'V' for both eigenvalues and eigenvectors.
 * @param uplo Specifies which part of the matrix to use:
 *             - 'U' for upper triangle,
 *             - 'L' for lower triangle.
 * @param N The order of the matrix `A`.
 * @param A The matrix `A` to compute eigenvalues and eigenvectors for.
 * @param desc_a Array descriptor for the matrix `A`.
 * @param W Output, the eigenvalues.
 * @param Z Output, the matrix `Z` containing eigenvectors if `jobz` is 'V'.
 * @param desc_z Array descriptor for the matrix `Z`.
 * @param info Output status information.
 */
template <typename T>
void t_pheevd(char jobz, char uplo, std::size_t N, T* A, std::size_t* desc_a,
              chase::Base<T>* W, T* Z, std::size_t* desc_z, int* info);

/**
 * @ingroup ScalapackFunctions
 * @brief Computes the singular value decomposition (SVD) of a distributed
 * matrix.
 *
 * This function computes the SVD of a distributed matrix `A` using the
 * divide-and-conquer algorithm. The SVD is: A = U * SIGMA * V^T, where U and
 * V^T are orthogonal matrices and SIGMA is diagonal.
 *
 * @tparam T The data type of the matrix elements.
 *
 * @param jobu Specifies options for computing left singular vectors:
 *             - 'N' for no left singular vectors,
 *             - 'V' for all left singular vectors in U.
 * @param jobvt Specifies options for computing right singular vectors:
 *              - 'N' for no right singular vectors,
 *              - 'V' for all right singular vectors in VT.
 * @param m The number of rows in the matrix `A`.
 * @param n The number of columns in the matrix `A`.
 * @param A The matrix `A` to decompose. On exit, the contents are destroyed.
 * @param ia The row index in the global matrix `A` indicating the starting
 * position.
 * @param ja The column index in the global matrix `A` indicating the starting
 * position.
 * @param desc_a Array descriptor for the distributed matrix `A`.
 * @param s Output array containing the singular values in descending order.
 * @param U Output matrix containing the left singular vectors if `jobu` is 'V'.
 * @param iu The row index in the global matrix `U` indicating the starting
 * position.
 * @param ju The column index in the global matrix `U` indicating the starting
 * position.
 * @param desc_u Array descriptor for the distributed matrix `U`.
 * @param VT Output matrix containing the right singular vectors (transposed) if
 * `jobvt` is 'V'.
 * @param ivt The row index in the global matrix `VT` indicating the starting
 * position.
 * @param jvt The column index in the global matrix `VT` indicating the starting
 * position.
 * @param desc_vt Array descriptor for the distributed matrix `VT`.
 * @param info Output status information.
 */
template <typename T>
void t_pgesvd(char jobu, char jobvt, std::size_t m, std::size_t n, T* A, int ia,
              int ja, std::size_t* desc_a, chase::Base<T>* s, T* U, int iu,
              int ju, std::size_t* desc_u, T* VT, int ivt, int jvt,
              std::size_t* desc_vt, int* info);

} // namespace scalapackpp
} // namespace linalg
} // namespace chase

#include "scalapackpp.inc"
#endif
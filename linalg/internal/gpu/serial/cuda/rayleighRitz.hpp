// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "Impl/chase_gpu/nvtx.hpp"
#include "linalg/internal/gpu/backend/types.hpp"
#include "linalg/internal/gpu/serial/impl/pseudo_hermitian_rayleigh_ritz.hpp"
#include "linalg/internal/gpu/serial/impl/pseudo_hermitian_rayleigh_ritz_v1.hpp"
#include "linalg/internal/gpu/serial/impl/rayleigh_ritz.hpp"
#include "linalg/matrix/matrix.hpp"

namespace chase
{
namespace linalg
{
namespace internal
{
namespace cuda
{
/**
 * @brief Perform the Rayleigh-Ritz procedure to compute eigenvalues and
 * eigenvectors of a matrix.
 *
 * The Rayleigh-Ritz method computes an approximation to the eigenvalues and
 * eigenvectors of a matrix by projecting the matrix onto a subspace defined by
 * a set of vectors (Q) and solving the eigenvalue problem for the reduced
 * matrix. The computed Ritz values are stored in the `ritzv` array, and the
 * resulting eigenvectors are stored in `W`.
 *
 * @tparam T Data type for the matrix (e.g., float, double, etc.).
 * @param[in] N The number of rows of the matrix H.
 * @param[in] H The input matrix (N x N).
 * @param[in] ldh The leading dimension of the matrix H.
 * @param[in] n The number of vectors in Q (subspace size).
 * @param[in] Q The input matrix of size (N x n), whose columns are the basis
 * vectors for the subspace.
 * @param[in] ldq The leading dimension of the matrix Q.
 * @param[out] W The output matrix (N x n), which will store the result of the
 * projection.
 * @param[in] ldw The leading dimension of the matrix W.
 * @param[out] ritzv The array of Ritz values, which contains the eigenvalue
 * approximations.
 * @param[in] A A temporary matrix used in intermediate calculations. If not
 * provided, it is allocated internally.
 *
 * The procedure performs the following steps:
 * 1. Computes the matrix-vector multiplication: W = H * Q.
 * 2. Computes A = W' * Q, where W' is the conjugate transpose of W.
 * 3. Solves the eigenvalue problem for A using LAPACK's `heevd` function,
 * computing the Ritz values in `ritzv`.
 * 4. Computes the final approximation to the eigenvectors by multiplying Q with
 * the computed eigenvectors.
 */
template <typename T>
void rayleighRitz(
    cublasHandle_t cublas_handle, cusolverDnHandle_t cusolver_handle,
    chase::matrix::Matrix<T, chase::platform::GPU>* H,
    chase::matrix::Matrix<T, chase::platform::GPU>& V1,
    chase::matrix::Matrix<T, chase::platform::GPU>& V2,
    chase::matrix::Matrix<chase::Base<T>, chase::platform::GPU>& ritzv,
    std::size_t offset, std::size_t subSize, int* devInfo,
    T* workspace = nullptr, int lwork_heevd = 0,
    chase::matrix::Matrix<T, chase::platform::GPU>* A = nullptr)
{
    SCOPED_NVTX_RANGE();
    gpu_serial::impl::rayleighRitz<gpu_backend::cuda_backend, T>(
        cublas_handle, cusolver_handle, H, V1, V2, ritzv, offset, subSize,
        devInfo, A, workspace, lwork_heevd);
}

/**
 * @brief Perform the Rayleigh-Ritz procedure to compute eigenvalues and
 * eigenvectors of a Pseudo-Hermitian matrix.
 *
 * The Rayleigh-Ritz method computes an approximation to the eigenvalues and
 * eigenvectors of a matrix by projecting the matrix onto a subspace defined by
 * a set of vectors (Q) and solving the eigenvalue problem for the reduced
 * matrix. The computed Ritz values are stored in the `ritzv` array, and the
 * resulting eigenvectors are stored in `W`.
 *
 * @tparam T Data type for the matrix (e.g., float, double, etc.).
 * @param[in] N The number of rows of the matrix H.
 * @param[in] H The input matrix (N x N).
 * @param[in] ldh The leading dimension of the matrix H.
 * @param[in] n The number of vectors in Q (subspace size).
 * @param[in] Q The input matrix of size (N x n), whose columns are the basis
 * vectors for the subspace.
 * @param[in] ldq The leading dimension of the matrix Q.
 * @param[out] W The output matrix (N x n), which will store the result of the
 * projection.
 * @param[in] ldw The leading dimension of the matrix W.
 * @param[out] ritzv The array of Ritz values, which contains the eigenvalue
 * approximations.
 * @param[in] A A temporary matrix used in intermediate calculations. If not
 * provided, it is allocated internally.
 *
 * The procedure performs the following steps:
 * 1. Computes the matrix-vector multiplication: W = H * Q.
 * 2. Computes A = W' * Q, where W' is the conjugate transpose of W.
 * 3. Solves the eigenvalue problem for A using LAPACK's `heevd` function,
 * computing the Ritz values in `ritzv`.
 * 4. Computes the final approximation to the eigenvectors by multiplying Q with
 * the computed eigenvectors.
 */

template <typename T>
void rayleighRitz(
    cublasHandle_t cublas_handle, cusolverDnHandle_t cusolver_handle,
    cusolverDnParams_t params,
    chase::matrix::PseudoHermitianMatrix<T, chase::platform::GPU>* H,
    chase::matrix::Matrix<T, chase::platform::GPU>& V1,
    chase::matrix::Matrix<T, chase::platform::GPU>& V2,
    chase::matrix::Matrix<chase::Base<T>, chase::platform::GPU>& ritzv,
    std::size_t offset, std::size_t subSize, int* devInfo,
    T* d_workspace = nullptr, int d_lwork = 0, T* h_workspace = nullptr,
    int h_lwork = 0,
    chase::matrix::Matrix<T, chase::platform::GPU>* A = nullptr)
{
    SCOPED_NVTX_RANGE();
    gpu_serial::impl::pseudo_hermitian_rayleighRitz_v1<T>(
        cublas_handle, cusolver_handle, params, H, V1, V2, ritzv, offset,
        subSize, devInfo, d_workspace, d_lwork, h_workspace, h_lwork, A);
}

/**
 * @brief Perform the Rayleigh-Ritz procedure to compute eigenvalues and
 * eigenvectors of a Pseudo-Hermitian matrix.
 *
 * The Rayleigh-Ritz method computes an approximation to the eigenvalues and
 * eigenvectors of a matrix by projecting the matrix onto a subspace defined by
 * a set of vectors (Q) and solving the eigenvalue problem for the reduced
 * matrix. The computed Ritz values are stored in the `ritzv` array, and the
 * resulting eigenvectors are stored in `W`.
 *
 * @tparam T Data type for the matrix (e.g., float, double, etc.).
 * @param[in] N The number of rows of the matrix H.
 * @param[in] H The input matrix (N x N).
 * @param[in] ldh The leading dimension of the matrix H.
 * @param[in] n The number of vectors in Q (subspace size).
 * @param[in] Q The input matrix of size (N x n), whose columns are the basis
 * vectors for the subspace.
 * @param[in] ldq The leading dimension of the matrix Q.
 * @param[out] W The output matrix (N x n), which will store the result of the
 * projection.
 * @param[in] ldw The leading dimension of the matrix W.
 * @param[out] ritzv The array of Ritz values, which contains the eigenvalue
 * approximations.
 * @param[in] A A temporary matrix used in intermediate calculations. If not
 * provided, it is allocated internally.
 *
 * The procedure performs the following steps:
 * 1. Computes the matrix-vector multiplication: W = H * Q.
 * 2. Computes A = W' * Q, where W' is the conjugate transpose of W.
 * 3. Solves the eigenvalue problem for A using LAPACK's `heevd` function,
 * computing the Ritz values in `ritzv`.
 * 4. Computes the final approximation to the eigenvectors by multiplying Q with
 * the computed eigenvectors.
 */

template <typename T>
void rayleighRitz_v2(
    cublasHandle_t cublas_handle, cusolverDnHandle_t cusolver_handle,
    cusolverDnParams_t /*params*/,
    chase::matrix::PseudoHermitianMatrix<T, chase::platform::GPU>* H,
    chase::matrix::Matrix<T, chase::platform::GPU>& V1,
    chase::matrix::Matrix<T, chase::platform::GPU>& V2,
    chase::matrix::Matrix<chase::Base<T>, chase::platform::GPU>& ritzv,
    std::size_t offset, std::size_t subSize, int* devInfo,
    T* workspace = nullptr, int lwork = 0,
    chase::matrix::Matrix<T, chase::platform::GPU>* A = nullptr)
{
    SCOPED_NVTX_RANGE();
    gpu_serial::impl::pseudo_hermitian_rayleighRitz_v2<gpu_backend::cuda_backend, T>(
        cublas_handle, cusolver_handle, H, V1, V2, ritzv, offset, subSize,
        devInfo, A, workspace, lwork);
}


} // namespace cuda
} // namespace internal
} // namespace linalg
} // namespace chase

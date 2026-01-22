// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "Impl/chase_gpu/cuda_utils.hpp"
#include "Impl/chase_gpu/nvtx.hpp"
#include "external/cublaspp/cublaspp.hpp"
#include "external/cusolverpp/cusolverpp.hpp"
#include "linalg/internal/cuda/flipSign.hpp"
#include "linalg/internal/cuda/lacpy.hpp"
#include "linalg/matrix/matrix.hpp"

using namespace chase::linalg;

namespace chase
{
namespace linalg
{
namespace internal
{
namespace cuda
{
/**
 * @brief Performs the Lanczos algorithm for eigenvalue computation on a GPU
 * using cuBLAS and CUDA.
 *
 * This function performs the Lanczos algorithm on a matrix `H` (of size `M`),
 * computing an orthonormal basis `V` and the tridiagonal matrix `T` for the
 * eigenvalue problem. It also computes Ritz values and estimates of the upper
 * bounds for the eigenvalues.
 *
 * @tparam T The data type of the matrix elements (e.g., float or double).
 * @param cublas_handle The cuBLAS handle to perform linear algebra operations
 * on the GPU.
 * @param M The number of Lanczos iterations.
 * @param numvec The number of runs of Lanczos.
 * @param H The matrix of size `M x M` that is the input for the Lanczos
 * algorithm.
 * @param V The matrix of size `M x numvec` that holds the orthonormal basis
 * vectors.
 * @param upperb Output parameter that will hold the upper bound for the Ritz
 * values.
 * @param ritzv Output array to hold the Ritz values computed during the Lanczos
 * algorithm.
 * @param Tau Output array to store the Tau values from the eigenvalue
 * decomposition.
 * @param ritzV Output array for storing the Ritz vectors.
 */
template <typename T>
void lanczos(cublasHandle_t cublas_handle, std::size_t M, std::size_t numvec,
             chase::matrix::Matrix<T, chase::platform::GPU>* H,
             chase::matrix::Matrix<T, chase::platform::GPU>& V,
             chase::Base<T>* upperb, chase::Base<T>* ritzv, chase::Base<T>* Tau,
             chase::Base<T>* ritzV)
{
    SCOPED_NVTX_RANGE();

    T One = T(1.0);
    T Zero = T(0.0);
    std::size_t N = H->rows();
    std::vector<chase::Base<T>> r_beta(numvec);

    std::vector<chase::Base<T>> d(M * numvec);
    std::vector<chase::Base<T>> e(M * numvec);

    std::vector<chase::Base<T>> real_alpha(numvec);
    std::vector<T> alpha(numvec, T(1.0));
    std::vector<T> beta(numvec, T(0.0));

    auto v_0 = chase::matrix::Matrix<T, chase::platform::GPU>(N, numvec);
    auto v_1 = chase::matrix::Matrix<T, chase::platform::GPU>(N, numvec);
    auto v_2 = chase::matrix::Matrix<T, chase::platform::GPU>(N, numvec);
    chase::linalg::internal::cuda::t_lacpy('A', N, numvec, V.data(), V.ld(),
                                           v_1.data(), v_1.ld());

    for (auto i = 0; i < numvec; i++)
    {
        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTnrm2(
            cublas_handle, v_1.rows(), v_1.data() + i * v_1.ld(), 1,
            &real_alpha[i]));
    }

    for (auto i = 0; i < numvec; i++)
    {
        alpha[i] = T(1 / real_alpha[i]);
    }

    for (auto i = 0; i < numvec; i++)
    {
        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTscal(
            cublas_handle, v_1.rows(), &alpha[i], v_1.data() + i * v_1.ld(),
            1));
    }

    for (std::size_t k = 0; k < M; k = k + 1)
    {
        for (auto i = 0; i < numvec; i++)
        {
            CHECK_CUDA_ERROR(
                cudaMemcpy(V.data() + k * V.ld(), v_1.data() + i * v_1.ld(),
                           v_1.rows() * sizeof(T), cudaMemcpyDeviceToDevice));
        }

        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(
            cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, H->rows(), numvec,
            H->cols(), &One, H->data(), H->ld(), v_1.data(), v_1.ld(), &Zero,
            v_2.data(), v_2.ld()));

        for (auto i = 0; i < numvec; i++)
        {
            CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTdot(
                cublas_handle, v_1.rows(), v_1.data() + i * v_1.ld(), 1,
                v_2.data() + i * v_2.ld(), 1, &alpha[i]));
        }

        for (auto i = 0; i < numvec; i++)
        {
            alpha[i] = -alpha[i];
        }

        for (auto i = 0; i < numvec; i++)
        {
            CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTaxpy(
                cublas_handle, v_1.rows(), &alpha[i], v_1.data() + i * v_1.ld(),
                1, v_2.data() + i * v_2.ld(), 1));
        }

        for (auto i = 0; i < numvec; i++)
        {
            alpha[i] = -alpha[i];
        }

        for (auto i = 0; i < numvec; i++)
        {
            d[k + M * i] = std::real(alpha[i]);
        }

        if (k > 0)
        {
            for (auto i = 0; i < numvec; i++)
            {
                beta[i] = T(-r_beta[i]);
            }
            for (auto i = 0; i < numvec; i++)
            {
                CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTaxpy(
                    cublas_handle, v_0.rows(), &beta[i],
                    v_0.data() + i * v_0.ld(), 1, v_2.data() + i * v_2.ld(),
                    1));
            }
        }

        for (auto i = 0; i < numvec; i++)
        {
            beta[i] = -beta[i];
        }

        for (auto i = 0; i < numvec; i++)
        {
            CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTnrm2(
                cublas_handle, v_2.rows(), v_2.data() + i * v_2.ld(), 1,
                &r_beta[i]));
        }

        for (auto i = 0; i < numvec; i++)
        {
            beta[i] = T(1 / r_beta[i]);
        }

        if (k == M - 1)
            break;

        for (auto i = 0; i < numvec; i++)
        {
            CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTscal(
                cublas_handle, v_2.rows(), &beta[i], v_2.data() + i * v_2.ld(),
                1));
        }

        for (auto i = 0; i < numvec; i++)
        {
            e[k + M * i] = r_beta[i];
        }

        v_1.swap(v_0);
        v_1.swap(v_2);
    }

    chase::linalg::internal::cuda::t_lacpy('A', N, numvec, v_1.data(), v_1.ld(),
                                           V.data(), V.ld());

    int notneeded_m;
    std::size_t vl = 0;
    std::size_t vu = 0;
    Base<T> ul = 0;
    Base<T> ll = 0;
    int tryrac = 0;
    std::vector<int> isuppz(2 * M);

    for (auto i = 0; i < numvec; i++)
    {
        lapackpp::t_stemr(LAPACK_COL_MAJOR, 'V', 'A', M, d.data() + i * M,
                          e.data() + i * M, ul, ll, vl, vu, &notneeded_m,
                          ritzv + M * i, ritzV, M, M, isuppz.data(), &tryrac);
        for (std::size_t k = 0; k < M; ++k)
        {
            Tau[k + i * M] = std::abs(ritzV[k * M]) * std::abs(ritzV[k * M]);
        }
    }

    Base<T> max;
    *upperb = std::max(std::abs(ritzv[0]), std::abs(ritzv[M - 1])) +
              std::abs(r_beta[0]);

    for (auto i = 1; i < numvec; i++)
    {
        max =
            std::max(std::abs(ritzv[i * M]), std::abs(ritzv[(i + 1) * M - 1])) +
            std::abs(r_beta[i]);
        *upperb = std::max(max, *upperb);
    }
}

/**
 * @brief Performs the Lanczos algorithm on matrix H to compute the tridiagonal
 * matrix and eigenvalues.
 *
 * This version of the Lanczos algorithm is a simplified version that computes
 * only the upper bound of the eigenvalue spectrum and does not compute
 * eigenvectors. It operates similarly to the full Lanczos algorithm but
 * omits the eigenvector computation step.
 *
 * @tparam T The data type of the matrix elements (e.g., float, double).
 *
 * @param cublas_handle cuBLAS handle used to perform matrix operations.
 * @param M Number of Lanczos iterations.
 * @param H The input matrix \( H \) (square matrix).
 * @param V The input matrix \( V \) (Lanczos starting vector).
 * @param upperb Pointer to store the upper bound of the computed eigenvalues.
 *
 * @note The function modifies the input matrices `H` and `V` during the
 * computation. It computes the tridiagonal matrix and updates the eigenvalues
 * in `ritzv`, then stores the upper bound of the eigenvalues in the `upperb`
 * pointer.
 */
template <typename T>
void lanczos(cublasHandle_t cublas_handle, std::size_t M,
             chase::matrix::Matrix<T, chase::platform::GPU>* H,
             chase::matrix::Matrix<T, chase::platform::GPU>& V,
             chase::Base<T>* upperb)
{
    SCOPED_NVTX_RANGE();

    T One = T(1.0);
    T Zero = T(0.0);
    chase::Base<T> r_beta;
    std::size_t N = H->rows();

    std::vector<Base<T>> d(M);
    std::vector<Base<T>> e(M);

    chase::Base<T> real_alpha;
    T alpha = T(1.0);
    T beta = T(0.0);

    auto v_0 = chase::matrix::Matrix<T, chase::platform::GPU>(N, 1);
    auto v_1 = chase::matrix::Matrix<T, chase::platform::GPU>(N, 1);
    auto v_2 = chase::matrix::Matrix<T, chase::platform::GPU>(N, 1);

    chase::linalg::internal::cuda::t_lacpy('A', N, 1, V.data(), V.ld(),
                                           v_1.data(), v_1.ld());

    CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTnrm2(
        cublas_handle, v_1.rows(), v_1.data(), 1, &real_alpha));
    alpha = T(1 / real_alpha);

    CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTscal(
        cublas_handle, v_1.rows(), &alpha, v_1.data(), 1));
    for (std::size_t k = 0; k < M; k = k + 1)
    {
        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(
            cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, H->rows(), 1, H->cols(),
            &One, H->data(), H->ld(), v_1.data(), v_1.ld(), &Zero, v_2.data(),
            v_2.ld()));

        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTdot(
            cublas_handle, v_1.rows(), v_1.data(), 1, v_2.data(), 1, &alpha));
        alpha = -alpha;

        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTaxpy(
            cublas_handle, v_1.rows(), &alpha, v_1.data(), 1, v_2.data(), 1));
        alpha = -alpha;

        d[k] = std::real(alpha);

        if (k > 0)
        {
            beta = T(-r_beta);
            CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTaxpy(
                cublas_handle, v_0.rows(), &beta, v_0.data(), 1, v_2.data(),
                1));
        }

        beta = -beta;

        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTnrm2(
            cublas_handle, v_2.rows(), v_2.data(), 1, &r_beta));
        beta = T(1 / r_beta);

        if (k == M - 1)
            break;

        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTscal(
            cublas_handle, v_2.rows(), &beta, v_2.data(), 1));
        e[k] = r_beta;

        v_1.swap(v_0);
        v_1.swap(v_2);
    }

    int notneeded_m;
    std::size_t vl = 0;
    std::size_t vu = 0;
    Base<T> ul = 0;
    Base<T> ll = 0;
    int tryrac = 0;
    std::vector<int> isuppz(2 * M);
    std::vector<Base<T>> ritzv(M);

    lapackpp::t_stemr<Base<T>>(
        LAPACK_COL_MAJOR, 'N', 'A', M, d.data(), e.data(), ul, ll, vl, vu,
        &notneeded_m, ritzv.data(), NULL, M, M, isuppz.data(), &tryrac);

    *upperb =
        std::max(std::abs(ritzv[0]), std::abs(ritzv[M - 1])) + std::abs(r_beta);
}

/**
 * @brief Lanczos algorithm for eigenvalue computation of Pseudo-Hermitian
 * matrices.
 *
 * This function performs the Lanczos algorithm, which is used to estimate
 * the upper bound of spectra of the pseudo-Hermitian matrix H, i.e., where H is
 * pseudo-Hermitian and SH is Hermitian Positive Definite. The algorithm is
 * iteratively applied to the matrix H, where the input matrix `H` is a square
 * matrix of size `N x N`. The Lanczos algorithm builds an orthonormal basis of
 * the Krylov subspace, and the resulting tridiagonal matrix is diagonalized
 * using the `t_stemr` function. This pseudo-code of this implementation can be
 * found in `https://doi.org/10.1016/j.commatsci.2011.02.021`
 *
 * @tparam T The data type for the matrix elements (e.g., float, double).
 * @param cublas_handle cuBLAS handle used to perform matrix operations.
 * @param M The number of Lanczos iterations.
 * @param numvec The number of runs of Lanczos.
 * @param N The size of the input matrix `H`.
 * @param H The input matrix for the Lanczos algorithm (of size `N x N`).
 * @param V The input matrix used for storing vectors (of size `N x numvec`).
 * @param upperb A pointer to the upper bound of the eigenvalue spectrum.
 * @param ritzv A pointer to store the Ritz eigenvalues.
 * @param Tau A pointer to store the computed Tau values.
 * @param ritzV A pointer to store the Ritz eigenvectors.
 */

template <typename T>
void lanczos(cublasHandle_t cublas_handle, std::size_t M, std::size_t numvec,
             chase::matrix::PseudoHermitianMatrix<T, chase::platform::GPU>* H,
             chase::matrix::Matrix<T, chase::platform::GPU>& V,
             chase::Base<T>* upperb, chase::Base<T>* ritzv, chase::Base<T>* Tau,
             chase::Base<T>* ritzV)
{
    SCOPED_NVTX_RANGE();

    T One = T(1.0);
    T Zero = T(0.0);

    std::size_t N = H->rows();
    std::vector<chase::Base<T>> r_beta(numvec);

    std::vector<chase::Base<T>> d(M * numvec);
    std::vector<chase::Base<T>> e(M * numvec);

    std::vector<chase::Base<T>> real_alpha(numvec);
    std::vector<T> alpha(numvec, T(1.0));
    std::vector<T> beta(numvec, T(0.0));

    auto v_0 = chase::matrix::Matrix<T, chase::platform::GPU>(N, numvec);
    auto v_1 = chase::matrix::Matrix<T, chase::platform::GPU>(N, numvec);
    auto v_2 = chase::matrix::Matrix<T, chase::platform::GPU>(N, numvec);
    auto Sv = chase::matrix::Matrix<T, chase::platform::GPU>(N, numvec);

    chase::linalg::internal::cuda::t_lacpy('A', N, numvec, V.data(), V.ld(),
                                           v_1.data(), v_1.ld());

    CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(
        cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, H->rows(), numvec, H->cols(),
        &One, H->data(), H->ld(), v_1.data(), v_1.ld(), &Zero, v_2.data(),
        v_2.ld()));

    chase::linalg::internal::cuda::t_lacpy('A', N, numvec, v_2.data(), v_2.ld(),
                                           Sv.data(), Sv.ld());

    chase::linalg::internal::cuda::flipLowerHalfMatrixSign(N, numvec, Sv.data(),
                                                           N);

    for (auto i = 0; i < numvec; i++)
    {
        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTdot(
            cublas_handle, v_1.rows(), v_1.data() + i * v_1.ld(), 1,
            Sv.data() + i * Sv.ld(), 1, &beta[i]));
    }

    for (auto i = 0; i < numvec; i++)
    {
        beta[i] = One / sqrt(beta[i]);
    }

    for (auto i = 0; i < numvec; i++)
    {
        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTscal(
            cublas_handle, v_1.rows(), &beta[i], v_1.data() + i * v_1.ld(), 1));
    }

    for (auto i = 0; i < numvec; i++)
    {
        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTscal(
            cublas_handle, v_2.rows(), &beta[i], v_2.data() + i * v_2.ld(), 1));
    }

    for (std::size_t k = 0; k < M; k = k + 1)
    {
        for (auto i = 0; i < numvec; i++)
        {
            CHECK_CUDA_ERROR(
                cudaMemcpy(V.data() + k * V.ld(), v_1.data() + i * v_1.ld(),
                           v_1.rows() * sizeof(T), cudaMemcpyDeviceToDevice));
        }

        for (auto i = 0; i < numvec; i++)
        {
            CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTdot(
                cublas_handle, v_2.rows(), v_2.data() + i * v_2.ld(), 1,
                Sv.data() + i * Sv.ld(), 1, &alpha[i]));
        }

        for (auto i = 0; i < numvec; i++)
        {
            alpha[i] = -alpha[i] * beta[i];
        }

        for (auto i = 0; i < numvec; i++)
        {

            CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTaxpy(
                cublas_handle, v_1.rows(), &alpha[i], v_1.data() + i * v_1.ld(),
                1, v_2.data() + i * v_2.ld(), 1));
        }

        for (auto i = 0; i < numvec; i++)
        {
            alpha[i] = -alpha[i];
        }

        for (auto i = 0; i < numvec; i++)
        {
            d[k + M * i] = std::real(alpha[i]);
        }

        if (k == M - 1)
            break;

        for (auto i = 0; i < numvec; i++)
        {
            beta[i] = -One / beta[i];
        }

        for (auto i = 0; i < numvec; i++)
        {
            CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTaxpy(
                cublas_handle, v_0.rows(), &beta[i], v_0.data() + i * v_0.ld(),
                1, v_2.data() + i * v_2.ld(), 1));
        }

        for (auto i = 0; i < numvec; i++)
        {
            beta[i] = -beta[i];
        }

        v_1.swap(v_0);
        v_1.swap(v_2);

        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(
            cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, H->rows(), numvec,
            H->cols(), &One, H->data(), H->ld(), v_1.data(), v_1.ld(), &Zero,
            v_2.data(), v_2.ld()));

        chase::linalg::internal::cuda::t_lacpy('A', N, numvec, v_2.data(),
                                               v_2.ld(), Sv.data(), Sv.ld());

        chase::linalg::internal::cuda::flipLowerHalfMatrixSign(N, numvec,
                                                               Sv.data(), N);

        for (auto i = 0; i < numvec; i++)
        {
            CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTdot(
                cublas_handle, v_1.rows(), v_1.data() + i * v_1.ld(), 1,
                Sv.data() + i * Sv.ld(), 1, &beta[i]));
        }

        for (auto i = 0; i < numvec; i++)
        {
            beta[i] = sqrt(beta[i]);
        }

        for (auto i = 0; i < numvec; i++)
        {
            r_beta[i] = std::real(beta[i]);
        }

        for (auto i = 0; i < numvec; i++)
        {
            e[k + M * i] = r_beta[i];
        }

        for (auto i = 0; i < numvec; i++)
        {
            beta[i] = One / beta[i];
        }

        for (auto i = 0; i < numvec; i++)
        {
            CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTscal(
                cublas_handle, v_1.rows(), &beta[i], v_1.data() + i * v_1.ld(),
                1));
        }

        for (auto i = 0; i < numvec; i++)
        {
            CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTscal(
                cublas_handle, v_2.rows(), &beta[i], v_2.data() + i * v_2.ld(),
                1));
        }
    }

    chase::linalg::internal::cuda::t_lacpy('A', N, numvec, v_1.data(), v_1.ld(),
                                           V.data(), V.ld());

    int notneeded_m;
    std::size_t vl = 0;
    std::size_t vu = 0;
    Base<T> ul = 0;
    Base<T> ll = 0;
    int tryrac = 0;
    std::vector<int> isuppz(2 * M);

    for (auto i = 0; i < numvec; i++)
    {
        lapackpp::t_stemr(LAPACK_COL_MAJOR, 'V', 'A', M, d.data() + i * M,
                          e.data() + i * M, ul, ll, vl, vu, &notneeded_m,
                          ritzv + M * i, ritzV, M, M, isuppz.data(), &tryrac);
        for (std::size_t k = 0; k < M; ++k)
        {
            Tau[k + i * M] = std::abs(ritzV[k * M]) * std::abs(ritzV[k * M]);
        }
    }

    *upperb = ritzv[M - 1];
}

/**
 * @brief Lanczos algorithm for eigenvalue computation of pseudo-hermitian
 * matrices (simplified version).
 *
 * This version of the Lanczos algorithm is a simplified version that computes
 * only the upper bound of the eigenvalue spectrum of pseudo-hermitian matrices
 * and does not computei eigenvectors. It operates similarly to the full
 * Lanczos algorithm but omits the eigenvector computation step.
 *
 * @tparam T The data type for the matrix elements (e.g., float, double).
 * @param cublas_handle cuBLAS handle used to perform matrix operations.
 * @param M The number of Lanczos iterations.
 * @param H The input pseudo-hermitian matrix for the Lanczos algorithm (of size
 * `N x N`).
 * @param V The input matrix used for storing vectors (of size `N x 1`).
 * @param upperb A pointer to the upper bound of the eigenvalue spectrum.
 */
template <typename T>
void lanczos(cublasHandle_t cublas_handle, std::size_t M,
             chase::matrix::PseudoHermitianMatrix<T, chase::platform::GPU>* H,
             chase::matrix::Matrix<T, chase::platform::GPU>& V,
             chase::Base<T>* upperb)
{
    SCOPED_NVTX_RANGE();

    T One = T(1.0);
    T Zero = T(0.0);

    std::size_t N = H->rows();
    chase::Base<T> r_beta;

    std::vector<chase::Base<T>> d(M);
    std::vector<chase::Base<T>> e(M);

    chase::Base<T> real_alpha;
    T alpha, beta;

    auto v_0 = chase::matrix::Matrix<T, chase::platform::GPU>(N, 1);
    auto v_1 = chase::matrix::Matrix<T, chase::platform::GPU>(N, 1);
    auto v_2 = chase::matrix::Matrix<T, chase::platform::GPU>(N, 1);
    auto Sv = chase::matrix::Matrix<T, chase::platform::GPU>(N, 1);

    chase::linalg::internal::cuda::t_lacpy('A', N, 1, V.data(), V.ld(),
                                           v_1.data(), v_1.ld());

    CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(
        cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, H->rows(), 1, H->cols(), &One,
        H->data(), H->ld(), v_1.data(), v_1.ld(), &Zero, v_2.data(), v_2.ld()));

    chase::linalg::internal::cuda::t_lacpy('A', N, 1, v_2.data(), v_2.ld(),
                                           Sv.data(), Sv.ld());

    chase::linalg::internal::cuda::flipLowerHalfMatrixSign(N, 1, Sv.data(), N);

    CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTdot(
        cublas_handle, v_1.rows(), v_1.data(), 1, Sv.data(), 1, &beta));

    beta = One / sqrt(beta);

    CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTscal(
        cublas_handle, v_1.rows(), &beta, v_1.data(), 1));
    CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTscal(
        cublas_handle, v_2.rows(), &beta, v_2.data(), 1));

    for (std::size_t k = 0; k < M; k = k + 1)
    {

        CHECK_CUDA_ERROR(cudaMemcpy(V.data() + k * V.ld(), v_1.data(),
                                    v_1.rows() * sizeof(T),
                                    cudaMemcpyDeviceToDevice));

        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTdot(
            cublas_handle, v_2.rows(), v_2.data(), 1, Sv.data(), 1, &alpha));
        alpha = -alpha * beta;

        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTaxpy(
            cublas_handle, v_1.rows(), &alpha, v_1.data(), 1, v_2.data(), 1));

        alpha = -alpha;

        d[k] = std::real(alpha);

        if (k == M - 1)
            break;

        beta = -One / beta;

        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTaxpy(
            cublas_handle, v_0.rows(), &beta, v_0.data(), 1, v_2.data(), 1));
        beta = -beta;

        v_1.swap(v_0);
        v_1.swap(v_2);

        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(
            cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, H->rows(), 1, H->cols(),
            &One, H->data(), H->ld(), v_1.data(), v_1.ld(), &Zero, v_2.data(),
            v_2.ld()));

        chase::linalg::internal::cuda::t_lacpy('A', N, 1, v_2.data(), v_2.ld(),
                                               Sv.data(), Sv.ld());

        chase::linalg::internal::cuda::flipLowerHalfMatrixSign(N, 1, Sv.data(),
                                                               N);

        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTdot(
            cublas_handle, v_1.rows(), v_1.data(), 1, Sv.data(), 1, &beta));
        beta = sqrt(beta);

        r_beta = std::real(beta);

        e[k] = r_beta;

        beta = One / beta;

        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTscal(
            cublas_handle, v_1.rows(), &beta, v_1.data(), 1));

        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTscal(
            cublas_handle, v_2.rows(), &beta, v_2.data(), 1));
    }

    chase::linalg::internal::cuda::t_lacpy('A', N, 1, v_1.data(), v_1.ld(),
                                           V.data(), V.ld());

    int notneeded_m;
    std::size_t vl = 0;
    std::size_t vu = 0;
    Base<T> ul = 0;
    Base<T> ll = 0;
    int tryrac = 0;
    std::vector<int> isuppz(2 * M);
    std::vector<Base<T>> ritzv(M);

    lapackpp::t_stemr<Base<T>>(
        LAPACK_COL_MAJOR, 'N', 'A', M, d.data(), e.data(), ul, ll, vl, vu,
        &notneeded_m, ritzv.data(), NULL, M, M, isuppz.data(), &tryrac);

    *upperb = ritzv.data()[M - 1];
}
} // namespace cuda
} // namespace internal
} // namespace linalg
} // namespace chase

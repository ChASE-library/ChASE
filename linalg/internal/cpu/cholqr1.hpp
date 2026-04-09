// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include <sstream>

#include "algorithm/logger.hpp"
#include "external/blaspp/blaspp.hpp"
#include "external/lapackpp/lapackpp.hpp"
#include "linalg/internal/cpu/utils.hpp"

using namespace chase::linalg;

namespace chase
{
namespace linalg
{
namespace internal
{
namespace cpu
{
/**
 * \brief Performs Cholesky QR factorization (degree 1).
 *
 * This function performs Cholesky QR factorization on the matrix V.
 * It computes \( A = V^T V \) and then solves \( A X = V \).
 *
 * \param m The number of rows of matrix V.
 * \param n The number of columns of matrix V.
 * \param V The matrix on which the factorization is performed.
 * \param ldv The leading dimension of V.
 * \param A The output matrix that stores the result of the Cholesky
 * factorization (optional, will be allocated if null).
 * \return 0 if successful, non-zero value otherwise.
 */
template <typename T>
int cholQR1(std::size_t m, std::size_t n, T* V, int ldv, T* A = nullptr)
{
    T one = T(1.0);
    T zero = T(0.0);
    int info = 1;

    std::unique_ptr<T[]> ptr;

    if (A == nullptr)
    {
        ptr = std::unique_ptr<T[]>{new T[n * n]};
        A = ptr.get();
    }

    blaspp::t_syherk('U', 'C', n, m, &one, V, ldv, &zero, A, n);
    info = lapackpp::t_potrf('U', n, A, n);

    if (info != 0)
    {
        return info;
    }
    else
    {
        blaspp::t_trsm('R', 'U', 'N', 'N', m, n, &one, A, n, V, ldv);
#ifdef CHASE_OUTPUT
        chase::GetLogger().Log(chase::LogLevel::Info, "linalg",
            "choldegree: 1\n", 0);
#endif
        return info;
    }
}

/**
 * \brief Performs Cholesky QR factorization (degree 2).
 *
 * This function performs Cholesky QR factorization on the matrix V.
 * It applies two iterations of Cholesky QR factorization.
 *
 * \param m The number of rows of matrix V.
 * \param n The number of columns of matrix V.
 * \param V The matrix on which the factorization is performed.
 * \param ldv The leading dimension of V.
 * \param A The output matrix that stores the result of the Cholesky
 * factorization (optional, will be allocated if null).
 * \return 0 if successful, non-zero value otherwise.
 */
template <typename T>
int cholQR2(std::size_t m, std::size_t n, T* V, int ldv, T* A = nullptr)
{
    T one = T(1.0);
    T zero = T(0.0);
    int info = 1;

    std::unique_ptr<T[]> ptr;

    if (A == nullptr)
    {
        ptr = std::unique_ptr<T[]>{new T[n * n]};
        A = ptr.get();
    }

    blaspp::t_syherk('U', 'C', n, m, &one, V, ldv, &zero, A, n);
    info = lapackpp::t_potrf('U', n, A, n);
    if (info != 0)
    {
        return info;
    }
    else
    {
        blaspp::t_trsm('R', 'U', 'N', 'N', m, n, &one, A, n, V, ldv);
        blaspp::t_syherk('U', 'C', n, m, &one, V, ldv, &zero, A, n);
        info = lapackpp::t_potrf('U', n, A, n);
        blaspp::t_trsm('R', 'U', 'N', 'N', m, n, &one, A, n, V, ldv);
#ifdef CHASE_OUTPUT
        chase::GetLogger().Log(chase::LogLevel::Info, "linalg",
            "choldegree: 2", 0);
#endif
        return info;
    }
}

/**
 * \brief Performs Cholesky QR factorization with shifting (degree 2).
 *
 * This function performs Cholesky QR factorization on the matrix V, with a
 * shift applied to the matrix diagonal. It applies two iterations of Cholesky
 * QR factorization with a diagonal shift.
 *
 * \param m The number of rows of matrix V.
 * \param n The number of columns of matrix V.
 * \param V The matrix on which the factorization is performed.
 * \param ldv The leading dimension of V.
 * \param A The output matrix that stores the result of the Cholesky
 * factorization (optional, will be allocated if null).
 * \return 0 if successful, non-zero value otherwise.
 */
template <typename T>
int shiftedcholQR2(std::size_t m, std::size_t n, T* V, int ldv, T* A = nullptr)
{
    Base<T> shift;
    T one = T(1.0);
    T zero = T(0.0);
    int info = 1;

    std::unique_ptr<T[]> ptr;

    if (A == nullptr)
    {
        ptr = std::unique_ptr<T[]>{new T[n * n]};
        A = ptr.get();
    }

    blaspp::t_syherk('U', 'C', n, m, &one, V, ldv, &zero, A, n);
    Base<T> nrmf = 0.0;
    computeDiagonalAbsSum(n, n, A, n, &nrmf);
    if constexpr (std::is_same<Base<T>, float>::value)
    {
        shift = static_cast<Base<T>>(10.0) * nrmf *
                std::numeric_limits<Base<T>>::epsilon();
    }
    else
    {
        shift = std::sqrt(static_cast<double>(m)) * nrmf *
                std::numeric_limits<Base<T>>::epsilon();
    }
    shiftMatrixDiagonal(n, n, A, n, (T)shift);
    info = lapackpp::t_potrf('U', n, A, n);
    if (info != 0)
    {
        return info;
    }
    else
    {
        blaspp::t_trsm('R', 'U', 'N', 'N', m, n, &one, A, n, V, ldv);
        blaspp::t_syherk('U', 'C', n, m, &one, V, ldv, &zero, A, n);
        info = lapackpp::t_potrf('U', n, A, n);
        blaspp::t_trsm('R', 'U', 'N', 'N', m, n, &one, A, n, V, ldv);
        blaspp::t_syherk('U', 'C', n, m, &one, V, ldv, &zero, A, n);
        info = lapackpp::t_potrf('U', n, A, n);
        blaspp::t_trsm('R', 'U', 'N', 'N', m, n, &one, A, n, V, ldv);

#ifdef CHASE_OUTPUT
        std::ostringstream oss;
        oss << "choldegree: 2, shift = " << shift;
        chase::GetLogger().Log(chase::LogLevel::Info, "linalg", oss.str(), 0);
#endif
        return info;
    }
}

/**
 * \brief Performs Householder QR factorization.
 *
 * This function computes the QR factorization of matrix V using the Householder
 * transformation.
 *
 * \param m The number of rows of matrix V.
 * \param n The number of columns of matrix V.
 * \param V The matrix on which the factorization is performed.
 * \param ldv The leading dimension of V.
 */
template <typename T>
void houseHoulderQR(std::size_t m, std::size_t n, T* V, std::size_t ldv)
{
    std::unique_ptr<T[]> tau(new T[n]);

    chase::linalg::lapackpp::t_geqrf(LAPACK_COL_MAJOR, m, n, V, ldv, tau.get());
    chase::linalg::lapackpp::t_gqr(LAPACK_COL_MAJOR, m, n, n, V, ldv,
                                   tau.get());
}

template <typename T>
chase::Base<T> computeConditionNumber(std::size_t m, std::size_t n, T* V,
                                      std::size_t ldv)
{
    std::vector<chase::Base<T>> S(n);
    T* U;
    std::size_t ld = 1;
    T* Vt;
    std::size_t min_mn = std::min(m, n);

    // Basic parameter validation
    if (m == 0 || n == 0)
    {
        std::ostringstream oss;
        oss << "Error: Invalid matrix dimensions m=" << m << ", n=" << n;
        chase::GetLogger().Log(chase::LogLevel::Error, "linalg", oss.str(), 0);
        return std::numeric_limits<chase::Base<T>>::infinity();
    }

    // Call Lapack SVD function
    chase::linalg::lapackpp::t_gesvd('N', 'N', m, n, V, ldv, S.data(), U, ld,
                                     Vt, ld);

#ifdef CHASE_OUTPUT
    // Debug: Print singular values to understand what's happening
    if (min_mn > 0)
    {
        std::ostringstream oss;
        oss << "SVD Debug: Matrix size " << m << "x" << n << ", min_mn=" << min_mn << "\nFirst 5 singular values: ";
        for (int i = 0; i < std::min(5, (int)min_mn); i++)
            oss << S[i] << " ";
        if (min_mn > 5)
        {
            oss << "\nLast 5 singular values: ";
            for (int i = std::max(0, (int)min_mn - 5); i < (int)min_mn; i++)
                oss << S[i] << " ";
        }
        chase::GetLogger().Log(chase::LogLevel::Debug, "linalg", oss.str(), 0);
    }
#endif
    chase::Base<T> cond_num = std::numeric_limits<chase::Base<T>>::infinity();
    if (min_mn > 0)
    {
        chase::Base<T> sigma_max =
            S[0]; // Singular values are sorted in descending order
        chase::Base<T> sigma_min = S[min_mn - 1];
        // Check for rank deficiency
        const chase::Base<T> eps =
            std::numeric_limits<chase::Base<T>>::epsilon();
        const chase::Base<T> tolerance = std::max(m, n) * sigma_max * eps;
#ifdef CHASE_OUTPUT
        std::ostringstream oss_cond;
        oss_cond << "Condition number debug: sigma_max=" << sigma_max
                 << ", sigma_min=" << sigma_min << ", tolerance=" << tolerance
                 << ", eps=" << eps;
        chase::GetLogger().Log(chase::LogLevel::Debug, "linalg",
            oss_cond.str(), 0);
#endif
        if (sigma_min > tolerance && sigma_min > 0)
        {
            cond_num = sigma_max / sigma_min;
        }
    }

    return cond_num;
}
} // namespace cpu
} // namespace internal
} // end of namespace linalg
} // end of namespace chase
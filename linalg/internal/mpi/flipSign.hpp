// This file is a part of ChASE.
// Copyright (c) 2015-2024, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"
#include "linalg/internal/mpi/mpi_kernels.hpp"
#include <omp.h>

namespace chase
{
namespace linalg
{
namespace internal
{
/**
 * @brief Flip the sign of a BlockBlockMatrix lower part.
 *
 * This function flips the sign of the lower part of the BlockBlockMatrix `H`
 *
 * @tparam T The data type of the matrix elements (e.g., `float`, `double`).
 * @param H The BlockBlockMatrix whose diagonal elements will be shifted.
 */
template <typename T>
void cpu_mpi::flipLowerHalfMatrixSign(
    chase::distMatrix::BlockBlockMatrix<T, chase::platform::CPU>& H)
{
    T alpha = -T(1.0);

    if (H.l_half() < H.l_rows())
    {
        if (H.l_half() == 0)
        {
            chase::linalg::blaspp::t_scal(H.l_rows() * H.l_cols(), &alpha,
                                          H.l_data(), 1);
        }
        else
        {
            for (auto i = 0; i < H.l_cols(); i++)
            {
                chase::linalg::blaspp::t_scal(
                    H.l_rows() - H.l_half(), &alpha,
                    H.l_data() + H.l_half() + i * H.l_ld(), 1);
            }
        }
    }
}

/**
 * @brief Flip the sign of a BlockCyclicMatrix lower part.
 *
 * This function flips the sign of the lower part of the BlockCyclicMatrix `H`
 *
 * @tparam T The data type of the matrix elements (e.g., `float`, `double`).
 * @param H The BlockCyclicMatrix whose lower part will be flipped
 */
template <typename T>
void cpu_mpi::flipLowerHalfMatrixSign(
    chase::distMatrix::BlockCyclicMatrix<T, chase::platform::CPU>& H)
{
    T alpha = -T(1.0);

    if (H.l_half() < H.l_rows())
    {
        if (H.l_half() == 0)
        {
            chase::linalg::blaspp::t_scal(H.l_rows() * H.l_cols(), &alpha,
                                          H.l_data(), 1);
        }
        else
        {
            for (auto i = 0; i < H.l_cols(); i++)
            {
                chase::linalg::blaspp::t_scal(
                    H.l_rows() - H.l_half(), &alpha,
                    H.l_data() + H.l_half() + i * H.l_ld(), 1);
            }
        }
    }
}

/**
 * @brief Flip the sign of the lower part of a subset of a distMultiVector.
 *
 * This function flips the sign of the lower part of a subset of the
 * distMultiVector `V`
 *
 * @tparam T The data type of the matrix elements (e.g., `float`, `double`).
 * @param V The distMultiVector whose the sign of the lower part will be
 * flipped.
 * @param subSize The number of columns to be flipped
 */
template <typename T, chase::distMultiVector::CommunicatorType comm_type>
void cpu_mpi::flipLowerHalfMatrixSign(
    chase::distMultiVector::DistMultiVector1D<T, comm_type,
                                              chase::platform::CPU>& V,
    std::size_t offset, std::size_t subSize)
{
    T alpha = -T(1.0);

    if (V.l_half() < V.l_rows())
    {
        if (V.l_half() == 0)
        {
            chase::linalg::blaspp::t_scal(V.l_rows() * subSize, &alpha,
                                          V.l_data() + offset * V.l_ld(), 1);
        }
        else
        {
            for (auto i = offset; i < subSize + offset; i++)
            {
                chase::linalg::blaspp::t_scal(
                    V.l_rows() - V.l_half(), &alpha,
                    V.l_data() + V.l_half() + i * V.l_ld(), 1);
            }
        }
    }
}

/**
 * @brief Flip the sign of the whole lower part of a distMultiVector.
 *
 * This function flips the sign of the lower part of the distMultiVector `V`
 *
 * @tparam T The data type of the matrix elements (e.g., `float`, `double`).
 * @param V The distMultiVector whose the sign of the lower part will be
 * flipped.
 */
template <typename T, chase::distMultiVector::CommunicatorType comm_type>
void cpu_mpi::flipLowerHalfMatrixSign(chase::distMultiVector::DistMultiVector1D<
                                      T, comm_type, chase::platform::CPU>& V)
{
    cpu_mpi::flipLowerHalfMatrixSign(V, 0, V.l_cols());
}

/**
 * @brief Flip the sign of the lower part of a subset of a
 * distMultiVectorBlockCyclic.
 *
 * This function flips the sign of the lower part of a subset of the
 * distMultiVectorBlockCyclic `V`
 *
 * @tparam T The data type of the matrix elements (e.g., `float`, `double`).
 * @param V The distMultiVectorBlockCyclic whose the sign of the lower part will
 * be flipped.
 * @param subSize The number of columns to be flipped
 */
template <typename T, chase::distMultiVector::CommunicatorType comm_type>
void cpu_mpi::flipLowerHalfMatrixSign(
    chase::distMultiVector::DistMultiVectorBlockCyclic1D<
        T, comm_type, chase::platform::CPU>& V,
    std::size_t offset, std::size_t subSize)
{
    T alpha = -T(1.0);

    if (V.l_half() < V.l_rows())
    {
        if (V.l_half() == 0)
        {
            chase::linalg::blaspp::t_scal(V.l_rows() * subSize, &alpha,
                                          V.l_data() + offset * V.l_ld(), 1);
        }
        else
        {
            for (auto i = offset; i < subSize + offset; i++)
            {
                chase::linalg::blaspp::t_scal(
                    V.l_rows() - V.l_half(), &alpha,
                    V.l_data() + V.l_half() + i * V.l_ld(), 1);
            }
        }
    }
}

/**
 * @brief Flip the sign of the whole lower part of a distMultiVectorBlockCyclic.
 *
 * This function flips the sign of the lower part of the
 * distMultiVectorBlockCyclic `V`
 *
 * @tparam T The data type of the matrix elements (e.g., `float`, `double`).
 * @param V The distMultiVectorBlockCyclic whose the sign of the lower part will
 * be flipped.
 */
template <typename T, chase::distMultiVector::CommunicatorType comm_type>
void cpu_mpi::flipLowerHalfMatrixSign(
    chase::distMultiVector::DistMultiVectorBlockCyclic1D<
        T, comm_type, chase::platform::CPU>& V)
{
    cpu_mpi::flipLowerHalfMatrixSign(V, 0, V.l_cols());
}

} // namespace internal
} // namespace linalg
} // namespace chase

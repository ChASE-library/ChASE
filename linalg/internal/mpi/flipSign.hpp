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
    std::size_t xlen = H.l_rows();
    std::size_t ylen = H.l_cols();
    std::size_t* g_offs = H.g_offs();
    std::size_t h_ld = H.l_ld();

    T alpha = -T(1.0);

    if (g_offs[0] >= H.g_rows() / 2)
    {

        chase::linalg::blaspp::t_scal(xlen * ylen, &alpha, H.l_data(), 1);
    }
    else if (g_offs[0] + xlen > H.g_rows() / 2)
    {

        auto shift = g_offs[0] + xlen - H.g_rows() / 2;

        for (auto i = 0; i < ylen; i++)
        {
            chase::linalg::blaspp::t_scal(xlen - shift, &alpha,
                                          H.l_data() + shift + i * h_ld, 1);
        }
    }
}

/**
 * @brief Flip the sign of a BlockBlockMatrix lower part.
 *
 * This function flips the sign of the lower part of the BlockBlockMatrix `V`
 *
 * @tparam T The data type of the matrix elements (e.g., `float`, `double`).
 * @param V The BlockBlockMatrix whose diagonal elements will be shifted.
 */
template <typename T, chase::distMultiVector::CommunicatorType comm_type>
void cpu_mpi::flipLowerHalfMatrixSign(chase::distMultiVector::DistMultiVector1D<
                                      T, comm_type, chase::platform::CPU>& V)
{
    std::size_t xlen = V.l_rows();
    std::size_t ylen = V.l_cols();
    std::size_t g_off = V.g_off();
    std::size_t v_ld = V.l_ld();

    T alpha = -T(1.0);

    if (g_off >= V.g_rows() / 2)
    {

        chase::linalg::blaspp::t_scal(xlen * ylen, &alpha, V.l_data(), 1);
    }
    else if (g_off + xlen > V.g_rows() / 2)
    {

        auto shift = g_off + xlen - V.g_rows() / 2;

        for (auto i = 0; i < ylen; i++)
        {
            chase::linalg::blaspp::t_scal(xlen - shift, &alpha,
                                          V.l_data() + shift + i * v_ld, 1);
        }
    }
}
} // namespace internal
} // namespace linalg
} // namespace chase

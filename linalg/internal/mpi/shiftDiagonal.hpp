// This file is a part of ChASE.
// Copyright (c) 2015-2024, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include <omp.h>
#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/internal/mpi/mpi_kernels.hpp"

namespace chase
{
namespace linalg
{
namespace internal
{
    /**
     * @brief Applies a shift to the diagonal elements of a BlockBlockMatrix.
     *
     * This function iterates over the local rows and columns of the distributed matrix `H`
     * and adds a given shift value to the diagonal elements (where row index equals column index).
     *
     * @tparam T The data type of the matrix elements (e.g., `float`, `double`).
     * @param H The BlockBlockMatrix whose diagonal elements will be shifted.
     * @param shift The value to add to each diagonal element.
     */    
    template<typename T>
    void cpu_mpi::shiftDiagonal(chase::distMatrix::BlockBlockMatrix<T, chase::platform::CPU>& H, T shift)
    {
        std::size_t xlen = H.l_rows();
        std::size_t ylen = H.l_cols();
        std::size_t *g_offs = H.g_offs();
        std::size_t h_ld = H.l_ld();

        #pragma omp parallel for
        for(auto j = 0; j < ylen; j++)
        {
            for(auto i = 0; i < xlen; i++)
            {

                if(g_offs[0] + i == g_offs[1] + j)
                {
                    H.l_data()[i + j * h_ld ] += shift;
                }
            }
        }
    }

    /**
     * @brief Applies a shift to the diagonal elements of a BlockCyclicMatrix.
     *
     * This function iterates over the matrix blocks, identifying diagonal elements based on 
     * global and local offsets, and adds a specified shift value to these elements.
     *
     * @tparam T The data type of the matrix elements (e.g., `float`, `double`).
     * @param H The BlockCyclicMatrix whose diagonal elements will be shifted.
     * @param shift The value to add to each diagonal element.
     */
    template<typename T>
    void cpu_mpi::shiftDiagonal(chase::distMatrix::BlockCyclicMatrix<T, chase::platform::CPU>& H, T shift)
    {
        auto m_contiguous_global_offs = H.m_contiguous_global_offs();
        auto n_contiguous_global_offs = H.n_contiguous_global_offs();
        auto m_contiguous_local_offs = H.m_contiguous_local_offs();
        auto n_contiguous_local_offs = H.n_contiguous_local_offs();
        auto m_contiguous_lens = H.m_contiguous_lens();
        auto n_contiguous_lens = H.n_contiguous_lens();
        auto mblocks = H.mblocks();
        auto nblocks = H.nblocks();

        for (std::size_t j = 0; j < nblocks; j++)
        {
            for (std::size_t i = 0; i < mblocks; i++)
            {
                for (std::size_t q = 0; q < n_contiguous_lens[j]; q++)
                {
                    for (std::size_t p = 0; p < m_contiguous_lens[i]; p++)
                    {
                        if (q + n_contiguous_global_offs[j] == p + m_contiguous_global_offs[i])
                        {
                             H.l_data()[(q + n_contiguous_local_offs[j]) * H.l_ld() + p + m_contiguous_local_offs[i]] +=
                                shift;
                        }
                    }
                }
            }
        }

    }    

}
}
}
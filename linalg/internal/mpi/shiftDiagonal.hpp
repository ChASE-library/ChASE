#pragma once

#include <omp.h>
#include "linalg/distMatrix/distMatrix.hpp"

namespace chase
{
namespace linalg
{
namespace internal
{
namespace mpi
{
    template<typename T>
    void shiftDiagonal(chase::distMatrix::BlockBlockMatrix<T, chase::platform::CPU>& H, T shift)
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

    template<typename T>
    void shiftDiagonal(chase::distMatrix::BlockCyclicMatrix<T, chase::platform::CPU>& H, T shift)
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
}
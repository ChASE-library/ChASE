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

}
}
}
}
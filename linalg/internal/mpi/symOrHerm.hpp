#pragma once

#include <vector>
#include <random>
#include "algorithm/types.hpp"
#include "linalg/blaspp/blaspp.hpp"
#include "linalg/lapackpp/lapackpp.hpp"
#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"
#include "linalg/internal/mpi/hemm.hpp"

namespace chase
{
namespace linalg
{
namespace internal
{
namespace mpi
{

    template<typename T>
    bool checkSymmetryEasy(chase::distMatrix::BlockBlockMatrix<T>& H) 
    {
        T One = T(1.0);
        T Zero = T(0.0);

        std::size_t N = H.g_rows();
        auto v = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::column>(N, 1, H.getMpiGrid_shared_ptr());
        auto u = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::row>(N, 1, H.getMpiGrid_shared_ptr());
        auto uT = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::column>(N, 1, H.getMpiGrid_shared_ptr());
        auto v_2 = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::row>(N, 1, H.getMpiGrid_shared_ptr());

        int *coords = H.getMpiGrid()->get_coords();

        std::mt19937 gen(1337.0 + coords[0]);
        std::normal_distribution<> d;
        
        for (auto i = 0; i < v.l_ld() * v.l_cols(); i++)
        {
            v.l_data()[i] = getRandomT<T>([&]() { return d(gen); });
        }

        v.redistributeImpl(&v_2);

        MatrixMultiplyMultiVectors(&One, 
                                       H,
                                       v,
                                       &Zero,
                                       u);

        MatrixMultiplyMultiVectors(&One, 
                                       H,
                                       v_2,
                                       &Zero,
                                       uT);

        u.redistributeImpl(&v);

        bool is_sym = true;

        for(auto i = 0; i < v.l_rows(); i++)
        {
            if(std::abs(v.l_data()[i] - uT.l_data()[i]) > 1e-10)
            {
                is_sym = false;
                break;
            }
        }

        MPI_Allreduce(MPI_IN_PLACE, 
                      &is_sym, 
                      1, 
                      MPI_CXX_BOOL, 
                      MPI_LAND, 
                      H.getMpiGrid()->get_col_comm());

        return is_sym;

    }

    template<typename T>
    void symOrHermMatrix(char uplo, chase::distMatrix::BlockBlockMatrix<T>& H) 
    {
#ifdef HAS_SCALAPACK
        std::size_t *desc = H.scalapack_descriptor_init();

        std::size_t xlen = H.l_rows();
        std::size_t ylen = H.l_cols();
        std::size_t *g_offs = H.g_offs();
        std::size_t h_ld = H.l_ld();

        if(uplo == 'U')
        {
            #pragma omp parallel for
            for(auto j = 0; j < ylen; j++)
            {
                for(auto i = 0; i < xlen; i++)
                {
                    if(g_offs[0] + i == g_offs[1] + j)
                    {
                        H.l_data()[i + j * h_ld ] *= T(0.5);
                    }
                    else if(g_offs[0] + i > g_offs[1] + j)
                    {
                        H.l_data()[i + j * h_ld ] = T(0.0);
                    }
                }
            }
        }
        else
        {
            #pragma omp parallel for
            for(auto j = 0; j < ylen; j++)
            {
                for(auto i = 0; i < xlen; i++)
                {
                    if(g_offs[0] + i == g_offs[1] + j)
                    {
                        H.l_data()[i + j * h_ld ] *= T(0.5);
                    }
                    else if(g_offs[0] + i < g_offs[1] + j)
                    {
                        H.l_data()[i + j * h_ld ] = T(0.0);
                    }
                }
            }
        }

        T One = T(1.0);
        T Zero = T(0.0);
        int zero = 0;
        int one = 1;
        std::vector<T> tmp(H.l_rows() * (H.l_cols() + 2*H.l_cols()));
        chase::linalg::scalapackpp::t_ptranc(H.g_rows(), 
                                             H.g_cols(), 
                                             One, 
                                             H.l_data(), 
                                             one, one, desc, Zero, tmp.data(), one, one, desc);
        #pragma omp parallel for
        for(auto j = 0; j < H.l_cols(); j++)
        {
            for(auto i = 0; i < H.l_rows(); i++)
            {
                H.l_data()[i + j * H.l_ld()] += tmp[i + j * H.l_rows()];
            }
        }      
#else
        std::runtime_error("For ChASE-MPI, symOrHermMatrix requires ScaLAPACK, which is not detected\n");
#endif
    }

}
}
}
}
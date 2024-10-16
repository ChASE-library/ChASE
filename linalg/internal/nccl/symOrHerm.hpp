#pragma once

#include <vector>
#include <random>
#include "algorithm/types.hpp"
#include "linalg/blaspp/blaspp.hpp"
#include "linalg/lapackpp/lapackpp.hpp"
#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"
#include "linalg/internal/nccl/hemm.hpp"
#include "linalg/internal/cuda/random_normal_distribution.cuh"

namespace chase
{
namespace linalg
{
namespace internal
{
namespace nccl
{
    //this function now assumes the hermitian matrix is provided on CPU
    template<typename T>
    bool checkSymmetryEasy(cublasHandle_t cublas_handle, chase::distMatrix::BlockBlockMatrix<T, chase::platform::GPU>& H) 
    {
        T One = T(1.0);
        T Zero = T(0.0);

        std::size_t N = H.g_rows();
        auto v = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::column, chase::platform::GPU>(N, 1, H.getMpiGrid_shared_ptr());
        auto u = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::row, chase::platform::GPU>(N, 1, H.getMpiGrid_shared_ptr());
        auto uT = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::column, chase::platform::GPU>(N, 1, H.getMpiGrid_shared_ptr());
        auto v_2 = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::row, chase::platform::GPU>(N, 1, H.getMpiGrid_shared_ptr());

        int *coords = H.getMpiGrid()->get_coords();
        unsigned long long seed = 1337 + coords[0];

        curandStatePhilox4_32_10_t* states_ = NULL;
        CHECK_CUDA_ERROR(cudaMalloc((void**)&states_,
                             sizeof(curandStatePhilox4_32_10_t) * (256 * 32)));

        chase::linalg::internal::cuda::chase_rand_normal(seed, states_, v.l_data(), v.l_ld() * v.l_cols(),
                        (cudaStream_t)0);

        v.redistributeImpl(&v_2);

        MatrixMultiplyMultiVectors(cublas_handle,
                                       &One, 
                                       H,
                                       v,
                                       &Zero,
                                       u);

        MatrixMultiplyMultiVectors(cublas_handle,
                                       &One, 
                                       H,
                                       v_2,
                                       &Zero,
                                       uT);

        u.redistributeImpl(&v);

        v.allocate_cpu_data();
        uT.allocate_cpu_data();

        v.D2H();
        uT.D2H();

        bool is_sym = true;

        for(auto i = 0; i < v.l_rows(); i++)
        {
            if(std::abs(v.cpu_data()[i] - uT.cpu_data()[i]) > 1e-10)
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
        
        if (states_)
            CHECK_CUDA_ERROR(cudaFree(states_));   

        return is_sym;
    }

    //this function now assumes the hermitian matrix is provided on CPU
    template<typename T>
    void symOrHermMatrix(char uplo, chase::distMatrix::BlockBlockMatrix<T, chase::platform::GPU>& H) 
    {
#ifdef HAS_SCALAPACK
        std::size_t *desc = H.scalapack_descriptor_init();

        std::size_t xlen = H.l_rows();
        std::size_t ylen = H.l_cols();
        std::size_t *g_offs = H.g_offs();
        std::size_t h_ld = H.cpu_ld();

        if(uplo == 'U')
        {
            #pragma omp parallel for
            for(auto j = 0; j < ylen; j++)
            {
                for(auto i = 0; i < xlen; i++)
                {
                    if(g_offs[0] + i == g_offs[1] + j)
                    {
                        H.cpu_data()[i + j * h_ld ] *= T(0.5);
                    }
                    else if(g_offs[0] + i > g_offs[1] + j)
                    {
                        H.cpu_data()[i + j * h_ld ] = T(0.0);
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
                        H.cpu_data()[i + j * h_ld ] *= T(0.5);
                    }
                    else if(g_offs[0] + i < g_offs[1] + j)
                    {
                        H.cpu_data()[i + j * h_ld ] = T(0.0);
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
                                             H.cpu_data(), 
                                             one, one, desc, Zero, tmp.data(), one, one, desc);
        #pragma omp parallel for
        for(auto j = 0; j < H.l_cols(); j++)
        {
            for(auto i = 0; i < H.l_rows(); i++)
            {
                H.cpu_data()[i + j * H.l_ld()] += tmp[i + j * H.l_rows()];
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
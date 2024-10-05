#pragma once

#include <omp.h>
#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/internal/cuda/shiftDiagonal.cuh"
namespace chase
{
namespace linalg
{
namespace internal
{
namespace nccl
{
    
    template<typename T>
    void shiftDiagonal(chase::distMatrix::BlockBlockMatrix<T, chase::platform::GPU>& H, 
                       std::size_t* d_off_m, 
                       std::size_t* d_off_n, 
                       std::size_t offsize, 
                       chase::Base<T> shift)
    {
        chase::linalg::internal::cuda::chase_shift_mgpu_matrix(H.l_data(), 
                                                               d_off_m, 
                                                               d_off_n,
                                                               offsize, 
                                                               H.l_ld(), 
                                                               shift,
                                                               (cudaStream_t)0);
    }
}
}
}
}
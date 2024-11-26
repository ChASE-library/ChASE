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
namespace cuda_aware_mpi
{
    
    template<typename MatrixType>
    void shiftDiagonal(MatrixType& H, 
                       std::size_t* d_off_m, 
                       std::size_t* d_off_n, 
                       std::size_t offsize, 
                       chase::Base<typename MatrixType::value_type> shift)
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
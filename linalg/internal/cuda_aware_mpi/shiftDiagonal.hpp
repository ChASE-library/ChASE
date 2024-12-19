// This file is a part of ChASE.
// Copyright (c) 2015-2024, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include <omp.h>
#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/internal/cuda/shiftDiagonal.cuh"
#include "linalg/internal/cuda_aware_mpi/cuda_mpi_kernels.hpp"

namespace chase
{
namespace linalg
{
namespace internal
{    
    template<typename MatrixType>
    void cuda_mpi::shiftDiagonal(MatrixType& H, 
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
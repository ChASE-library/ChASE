#pragma once

/**
 * \defgroup cuda_aware_mpi_kernels chase::linalg::internal::cuda_aware_mpi Namespace
 * \brief The `chase::linalg::internal::cuda_aware_mpi` namespace contains 
 * kernels required by ChASE for the distributed-memory GPU using cuda-aware-mpi
 * for communications.
 * @{
 */
#include "linalg/internal/cuda_aware_mpi/cholqr.hpp"
#include "linalg/internal/cuda_aware_mpi/hemm.hpp"
#include "linalg/internal/cuda_aware_mpi/lanczos.hpp"
#include "linalg/internal/cuda_aware_mpi/rayleighRitz.hpp"
#include "linalg/internal/cuda_aware_mpi/residuals.hpp"
#include "linalg/internal/cuda_aware_mpi/shiftDiagonal.hpp"
#include "linalg/internal/cuda_aware_mpi/symOrHerm.hpp"
/** @} */
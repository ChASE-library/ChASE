#pragma once

/**
 * \defgroup mpi_kernels chase::linalg::internal::mpi Namespace
 * \brief The `chase::linalg::internal::mpi` namespace contains 
 * kernels required by ChASE for the distributed-memory CPU
 * @{
 */
#include "linalg/internal/mpi/cholqr.hpp"
#include "linalg/internal/mpi/hemm.hpp"
#include "linalg/internal/mpi/lanczos.hpp"
#include "linalg/internal/mpi/rayleighRitz.hpp"
#include "linalg/internal/mpi/residuals.hpp"
#include "linalg/internal/mpi/shiftDiagonal.hpp"
#include "linalg/internal/mpi/symOrHerm.hpp"
/** @} */
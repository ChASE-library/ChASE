#pragma once

/**
 * \defgroup nccl_kernels chase::linalg::internal::nccl Namespace
 * \brief The `chase::linalg::internal::nccl` namespace contains 
 * kernels required by ChASE for the distributed-memory GPU using NCCL
 * for communications.
 * @{
 */
#include "linalg/internal/nccl/cholqr.hpp"
#include "linalg/internal/nccl/hemm.hpp"
#include "linalg/internal/nccl/lanczos.hpp"
#include "linalg/internal/nccl/rayleighRitz.hpp"
#include "linalg/internal/nccl/residuals.hpp"
#include "linalg/internal/nccl/shiftDiagonal.hpp"
#include "linalg/internal/nccl/symOrHerm.hpp"
/** @} */
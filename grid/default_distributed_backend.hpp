// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "grid/mpiGrid2D.hpp"

namespace chase
{
namespace grid
{
namespace backend
{

// Single source of truth for the default distributed GPU collective backend.
//
// Policy:
//   - NVIDIA (CUDA + NCCL): default to NCCL. This is the long-standing,
//     well-behaved default and must be preserved.
//   - AMD (HIP + RCCL): default to RCCL (NCCL-compatible API). GPU-aware MPI
//     remains available via CHASE_DISTRIBUTED_GPU_BACKEND_MPI (CMake
//     -DCHASE_HIP_DISTRIBUTED_COLLECTIVES=MPI).
//   - Any build can force the NCCL/RCCL path with
//     CHASE_DISTRIBUTED_GPU_BACKEND_NCCL, or MPI with
//     CHASE_DISTRIBUTED_GPU_BACKEND_MPI.
//   - CUDA callers may opt out of NCCL with CHASE_INTERFACE_DISABLE_NCCL.
#if defined(CHASE_DISTRIBUTED_GPU_BACKEND_MPI)
using DefaultDistributedGpu = MPI;
#elif defined(CHASE_DISTRIBUTED_GPU_BACKEND_NCCL)
using DefaultDistributedGpu = NCCL;
#elif defined(HAS_CUDA) && defined(HAS_NCCL) && !defined(CHASE_INTERFACE_DISABLE_NCCL)
using DefaultDistributedGpu = NCCL;
#elif defined(HAS_HIP) && defined(HAS_RCCL)
using DefaultDistributedGpu = NCCL;
#else
using DefaultDistributedGpu = MPI;
#endif

} // namespace backend
} // namespace grid
} // namespace chase

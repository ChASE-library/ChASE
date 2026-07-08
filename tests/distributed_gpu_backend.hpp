// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "grid/mpiTypes.hpp"
#include "grid/default_distributed_backend.hpp"

// Selects the distributed GPU collective backend for pChASEGPU integration tests.
//
// This defers to the single source of truth in
// grid/default_distributed_backend.hpp:
//   CUDA builds: NCCL when available, else MPI.
//   HIP builds:  RCCL when available (default); GPU-aware MPI when
//                CHASE_DISTRIBUTED_GPU_BACKEND_MPI is defined at compile time.
//
// Configure -DCHASE_HIP_DISTRIBUTED_COLLECTIVES=MPI to opt out of RCCL on HIP.
// ChaseDistributedSolveGPUNCCLTest still forces the explicit NCCL/RCCL path.

namespace chase
{
namespace test
{

using DistributedGpuBackend = chase::grid::backend::DefaultDistributedGpu;

} // namespace test
} // namespace chase

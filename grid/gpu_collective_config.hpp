// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

// GPU collective backends: NVIDIA NCCL (CUDA) or AMD RCCL (HIP).
// RCCL exposes an NCCL-compatible API (ncclComm_t, ncclAllReduce, ...).
#if defined(HAS_NCCL) || defined(HAS_RCCL)
#define CHASE_HAS_GPU_COLLECTIVE 1
#endif

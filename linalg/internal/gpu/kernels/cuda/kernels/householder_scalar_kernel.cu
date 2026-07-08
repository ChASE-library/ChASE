// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#define CHASE_GPU_DEVICE_BACKEND_CUDA
#define CHASE_CUDA_HOUSEHOLDER_DEFINE_KERNELS
#include "householder_scalar_kernel.cuh"
#include "linalg/internal/gpu/kernels/shared/impl/backend_config.hpp"
#include "linalg/internal/gpu/kernels/shared/impl/householder_scalar_kernel.hpp"

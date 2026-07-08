// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

/**
 * \defgroup gpu_kernels chase::linalg::internal::cuda Namespace
 * \brief The `chase::linalg::internal::cuda` namespace contains
 * kernels required by ChASE for the single GPU
 * @{
 */
#include "linalg/internal/gpu/kernels/cuda/launchers/abs_trace.cuh"
#include "linalg/internal/gpu/serial/cuda/cholqr.hpp"
#include "linalg/internal/gpu/kernels/cuda/launchers/conjugate.cuh"
#include "linalg/internal/gpu/kernels/cuda/launchers/flip_sign.cuh"
#include "linalg/internal/gpu/kernels/cuda/launchers/lacpy.cuh"
#include "linalg/internal/gpu/serial/cuda/lanczos.hpp"
#include "linalg/internal/gpu/kernels/cuda/launchers/precision_conversion.cuh"
#include "linalg/internal/gpu/kernels/cuda/launchers/random_normal_distribution.cuh"
#include "linalg/internal/gpu/serial/cuda/rayleighRitz.hpp"
#include "linalg/internal/gpu/kernels/cuda/launchers/residuals.cuh"
#include "linalg/internal/gpu/kernels/cuda/launchers/shift_diagonal.cuh"
/** @} */

// This file is a part of ChASE.
// Copyright (c) 2015-2024, Simulation and Data Laboratory Quantum Materials,
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
#include "linalg/internal/cuda/absTrace.cuh"
#include "linalg/internal/cuda/absTrace.hpp"
#include "linalg/internal/cuda/cholqr.hpp"
#include "linalg/internal/cuda/lacpy.cuh"
#include "linalg/internal/cuda/lacpy.hpp"
#include "linalg/internal/cuda/lanczos.hpp"
#include "linalg/internal/cuda/precision_conversion.cuh"
#include "linalg/internal/cuda/random_normal_distribution.cuh"
#include "linalg/internal/cuda/random_normal_distribution.hpp"
#include "linalg/internal/cuda/rayleighRitz.hpp"
#include "linalg/internal/cuda/residuals.cuh"
#include "linalg/internal/cuda/residuals.hpp"
#include "linalg/internal/cuda/shiftDiagonal.cuh"
#include "linalg/internal/cuda/shiftDiagonal.hpp"
/** @} */
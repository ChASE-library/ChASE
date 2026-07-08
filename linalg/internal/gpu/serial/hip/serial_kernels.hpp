// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

/**
 * \defgroup hip_kernels chase::linalg::internal::hip Namespace
 * \brief HIP kernels and serial ChASE GPU algorithm entry points.
 * @{
 */
#include "linalg/internal/gpu/serial/hip/cholqr.hpp"
#include "linalg/internal/gpu/kernels/hip/launchers/abs_trace.hpp"
#include "linalg/internal/gpu/kernels/hip/launchers/conjugate.hpp"
#include "linalg/internal/gpu/kernels/hip/launchers/flip_sign.hpp"
#include "linalg/internal/gpu/kernels/hip/launchers/matrix_diag.hpp"
#include "linalg/internal/gpu/kernels/hip/launchers/force_hermitian.hpp"
#include "linalg/internal/gpu/kernels/hip/launchers/householder_scalar_kernel.hpp"
#include "linalg/internal/gpu/kernels/hip/launchers/lacpy.hpp"
#include "linalg/internal/gpu/serial/hip/lanczos.hpp"
#include "linalg/internal/gpu/kernels/hip/launchers/orthog_launchers.hpp"
#include "linalg/internal/gpu/kernels/hip/launchers/precision_conversion.hpp"
#include "linalg/internal/gpu/kernels/hip/launchers/random_normal_distribution.hpp"
#include "linalg/internal/gpu/kernels/hip/launchers/residuals.hpp"
#include "linalg/internal/gpu/kernels/hip/launchers/shift_diagonal.hpp"
#include "linalg/internal/gpu/serial/hip/rayleighRitz.hpp"
#include "linalg/internal/gpu/serial/hip/pseudo_hermitian_rayleighRitz.hpp"
/** @} */

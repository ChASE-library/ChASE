// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

/**
 * \defgroup cpu_kernels chase::linalg::internal::cpu Namespace
 * \brief The `chase::linalg::internal::cpu` namespace contains k
 * ernels required by ChASE for the shared-memory CPU architectures
 * @{
 */
#include "linalg/internal/cpu/cholqr1.hpp"
#include "linalg/internal/cpu/lanczos.hpp"
#include "linalg/internal/cpu/rayleighRitz.hpp"
#include "linalg/internal/cpu/residuals.hpp"
#include "linalg/internal/cpu/symOrHerm.hpp"
#include "linalg/internal/cpu/utils.hpp"
/** @} */
// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "Impl/chase_gpu/nvtx.hpp"
#include "forceHermitian.cuh"
#include <complex>

namespace chase
{
namespace linalg
{
namespace internal
{
namespace cuda
{

/** In-place Hermitian symmetrization: \f$ A \leftarrow \tfrac12(A + A^H)\f$.
 *  Real types: symmetric average. Reduces asymmetric noise before \c heevd. */
template <typename T>
void force_hermitian(T* A, std::size_t n, std::size_t lda, cudaStream_t stream)
{
    SCOPED_NVTX_RANGE();
    force_hermitian_gpu(A, n, lda, stream);
}

template <typename T>
void force_hermitian(T* A, std::size_t n, cudaStream_t stream)
{
    SCOPED_NVTX_RANGE();
    force_hermitian_gpu(A, n, n, stream);
}

} // namespace cuda
} // namespace internal
} // namespace linalg
} // namespace chase

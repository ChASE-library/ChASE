// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "Impl/chase_gpu/nvtx.hpp"
#include "conjugate.cuh"
#include "linalg/matrix/matrix.hpp"
#include <cstddef>

namespace chase
{
namespace linalg
{
namespace internal
{
namespace cuda
{
/**
 * @brief In-place conjugation of a matrix block on the GPU (no transposition).
 * For complex types, replaces each element by its conjugate; for real types, no-op.
 *
 * @tparam T Element type (float, double, std::complex<float>, std::complex<double>).
 * @param A Device pointer to the matrix block.
 * @param m Number of rows.
 * @param n Number of columns.
 * @param lda Leading dimension of A.
 * @param stream_ CUDA stream (0 for default).
 */
template <typename T>
void conjugate_inplace(T* A, std::size_t m, std::size_t n, std::size_t lda,
                       cudaStream_t stream_ = 0)
{
    SCOPED_NVTX_RANGE();
    chase_conjugate_inplace(A, m, n, lda, stream_);
}

} // namespace cuda
} // namespace internal
} // namespace linalg
} // namespace chase

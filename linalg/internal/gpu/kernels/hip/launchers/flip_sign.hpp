// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include <cstddef>
#include <complex>
#include <hip/hip_runtime.h>

namespace chase
{
namespace linalg
{
namespace internal
{
namespace hip
{

void flipLowerHalfMatrixSign(float* A, std::size_t m, std::size_t n,
                             std::size_t lda, hipStream_t stream = nullptr);

void flipLowerHalfMatrixSign(double* A, std::size_t m, std::size_t n,
                             std::size_t lda, hipStream_t stream = nullptr);

void flipLowerHalfMatrixSign(std::complex<float>* A, std::size_t m,
                             std::size_t n, std::size_t lda,
                             hipStream_t stream = nullptr);

void flipLowerHalfMatrixSign(std::complex<double>* A, std::size_t m,
                             std::size_t n, std::size_t lda,
                             hipStream_t stream = nullptr);

void scaleLowerBlockRows(float* A, std::size_t lda, std::size_t row_start,
                         std::size_t nrows_lower, std::size_t ncols, float scale,
                         hipStream_t stream = nullptr);

void scaleLowerBlockRows(double* A, std::size_t lda, std::size_t row_start,
                         std::size_t nrows_lower, std::size_t ncols,
                         double scale, hipStream_t stream = nullptr);

void scaleLowerBlockRows(std::complex<float>* A, std::size_t lda,
                         std::size_t row_start, std::size_t nrows_lower,
                         std::size_t ncols, std::complex<float> scale,
                         hipStream_t stream = nullptr);

void scaleLowerBlockRows(std::complex<double>* A, std::size_t lda,
                         std::size_t row_start, std::size_t nrows_lower,
                         std::size_t ncols, std::complex<double> scale,
                         hipStream_t stream = nullptr);

} // namespace hip
} // namespace internal
} // namespace linalg
} // namespace chase

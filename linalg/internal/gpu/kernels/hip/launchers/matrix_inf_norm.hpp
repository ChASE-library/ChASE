// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include <complex>
#include <cstddef>
#include <hip/hip_runtime.h>

namespace chase
{
namespace linalg
{
namespace internal
{
namespace hip
{
void matrixInfNorm_gpu(const float* d_matrix, std::size_t rows,
                       std::size_t cols, std::size_t ld, float* d_norm,
                       hipStream_t stream_);
void matrixInfNorm_gpu(const double* d_matrix, std::size_t rows,
                       std::size_t cols, std::size_t ld, double* d_norm,
                       hipStream_t stream_);
void matrixInfNorm_gpu(const std::complex<float>* d_matrix, std::size_t rows,
                       std::size_t cols, std::size_t ld, float* d_norm,
                       hipStream_t stream_);
void matrixInfNorm_gpu(const std::complex<double>* d_matrix, std::size_t rows,
                       std::size_t cols, std::size_t ld, double* d_norm,
                       hipStream_t stream_);

void matrixRowAbsSums_gpu(const float* d_matrix, std::size_t rows,
                          std::size_t cols, std::size_t ld, float* d_row_sums,
                          hipStream_t stream_);
void matrixRowAbsSums_gpu(const double* d_matrix, std::size_t rows,
                          std::size_t cols, std::size_t ld, double* d_row_sums,
                          hipStream_t stream_);
void matrixRowAbsSums_gpu(const std::complex<float>* d_matrix, std::size_t rows,
                          std::size_t cols, std::size_t ld, float* d_row_sums,
                          hipStream_t stream_);
void matrixRowAbsSums_gpu(const std::complex<double>* d_matrix,
                          std::size_t rows, std::size_t cols, std::size_t ld,
                          double* d_row_sums, hipStream_t stream_);

void matrixRowSumsMax_gpu(const float* d_row_sums, std::size_t rows,
                          float* d_result, hipStream_t stream_);
void matrixRowSumsMax_gpu(const double* d_row_sums, std::size_t rows,
                          double* d_result, hipStream_t stream_);
} // namespace hip
} // namespace internal
} // namespace linalg
} // namespace chase

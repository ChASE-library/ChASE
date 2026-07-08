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

void shiftDiagonal(float* A, std::size_t n, std::size_t lda, float shift,
                   hipStream_t stream = nullptr);

void shiftDiagonal(double* A, std::size_t n, std::size_t lda, double shift,
                   hipStream_t stream = nullptr);

void shiftDiagonal(std::complex<float>* A, std::size_t n, std::size_t lda,
                   float shift, hipStream_t stream = nullptr);

void shiftDiagonal(std::complex<double>* A, std::size_t n, std::size_t lda,
                   double shift, hipStream_t stream = nullptr);

void shift_matrix_from_device_shift(float* A, std::size_t n, std::size_t lda,
                                    const float* d_shift,
                                    std::size_t shift_scale_num_rows,
                                    hipStream_t stream = nullptr);

void shift_matrix_from_device_shift(double* A, std::size_t n, std::size_t lda,
                                    const double* d_shift,
                                    std::size_t shift_scale_num_rows,
                                    hipStream_t stream = nullptr);

void shift_matrix_from_device_shift(std::complex<float>* A, std::size_t n,
                                    std::size_t lda, const float* d_shift,
                                    std::size_t shift_scale_num_rows,
                                    hipStream_t stream = nullptr);

void shift_matrix_from_device_shift(std::complex<double>* A, std::size_t n,
                                    std::size_t lda, const double* d_shift,
                                    std::size_t shift_scale_num_rows,
                                    hipStream_t stream = nullptr);

void shift_mgpu_matrix(float* A, std::size_t* off_m, std::size_t* off_n,
                       std::size_t offsize, std::size_t ldH, float shift,
                       hipStream_t stream = nullptr);

void shift_mgpu_matrix(double* A, std::size_t* off_m, std::size_t* off_n,
                       std::size_t offsize, std::size_t ldH, double shift,
                       hipStream_t stream = nullptr);

void shift_mgpu_matrix(std::complex<float>* A, std::size_t* off_m,
                       std::size_t* off_n, std::size_t offsize, std::size_t ldH,
                       float shift, hipStream_t stream = nullptr);

void shift_mgpu_matrix(std::complex<double>* A, std::size_t* off_m,
                       std::size_t* off_n, std::size_t offsize, std::size_t ldH,
                       double shift, hipStream_t stream = nullptr);

} // namespace hip
} // namespace internal
} // namespace linalg
} // namespace chase

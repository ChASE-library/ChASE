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

void chase_set_diagonal(float* A, std::size_t n, std::size_t lda, float coef,
                        hipStream_t stream = nullptr);

void chase_set_diagonal(double* A, std::size_t n, std::size_t lda, double coef,
                        hipStream_t stream = nullptr);

void chase_set_diagonal(std::complex<float>* A, std::size_t n, std::size_t lda,
                        std::complex<float> coef,
                        hipStream_t stream = nullptr);

void chase_set_diagonal(std::complex<double>* A, std::size_t n, std::size_t lda,
                        std::complex<double> coef,
                        hipStream_t stream = nullptr);

void chase_inverse_entries(float* vector, std::size_t n,
                           hipStream_t stream = nullptr);

void chase_inverse_entries(double* vector, std::size_t n,
                           hipStream_t stream = nullptr);

void chase_plus_inverse_diagonal(float* A, std::size_t n, std::size_t lda,
                                 float coef, float* new_diag,
                                 hipStream_t stream = nullptr);

void chase_plus_inverse_diagonal(double* A, std::size_t n, std::size_t lda,
                                 double coef, double* new_diag,
                                 hipStream_t stream = nullptr);

void chase_plus_inverse_diagonal(std::complex<float>* A, std::size_t n,
                                 std::size_t lda, float coef,
                                 float* new_diag,
                                 hipStream_t stream = nullptr);

void chase_plus_inverse_diagonal(std::complex<double>* A, std::size_t n,
                                 std::size_t lda, double coef,
                                 double* new_diag,
                                 hipStream_t stream = nullptr);

void subtract_inverse_diagonal(float* A, std::size_t n, std::size_t lda,
                               float coef, float* new_diag,
                               hipStream_t stream = nullptr);

void subtract_inverse_diagonal(double* A, std::size_t n, std::size_t lda,
                               double coef, double* new_diag,
                               hipStream_t stream = nullptr);

void subtract_inverse_diagonal(std::complex<float>* A, std::size_t n,
                               std::size_t lda, float coef, float* new_diag,
                               hipStream_t stream = nullptr);

void subtract_inverse_diagonal(std::complex<double>* A, std::size_t n,
                               std::size_t lda, double coef, double* new_diag,
                               hipStream_t stream = nullptr);

void chase_scale_rows_matrix(float* A, std::size_t m, std::size_t n,
                             std::size_t lda, float* coef,
                             hipStream_t stream = nullptr);

void chase_scale_rows_matrix(double* A, std::size_t m, std::size_t n,
                             std::size_t lda, double* coef,
                             hipStream_t stream = nullptr);

void chase_scale_rows_matrix(std::complex<float>* A, std::size_t m,
                             std::size_t n, std::size_t lda, float* coef,
                             hipStream_t stream = nullptr);

void chase_scale_rows_matrix(std::complex<double>* A, std::size_t m,
                             std::size_t n, std::size_t lda, double* coef,
                             hipStream_t stream = nullptr);

} // namespace hip
} // namespace internal
} // namespace linalg
} // namespace chase

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

void t_lacpy_gpu(char uplo, std::size_t m, std::size_t n, float* dA,
                 std::size_t ldda, float* dB, std::size_t lddb,
                 hipStream_t stream = nullptr);

void t_lacpy_gpu(char uplo, std::size_t m, std::size_t n, double* dA,
                 std::size_t ldda, double* dB, std::size_t lddb,
                 hipStream_t stream = nullptr);

void t_lacpy_gpu(char uplo, std::size_t m, std::size_t n,
                 std::complex<float>* dA, std::size_t ldda,
                 std::complex<float>* dB, std::size_t lddb,
                 hipStream_t stream = nullptr);

void t_lacpy_gpu(char uplo, std::size_t m, std::size_t n,
                 std::complex<double>* dA, std::size_t ldda,
                 std::complex<double>* dB, std::size_t lddb,
                 hipStream_t stream = nullptr);

void extractUpperTriangular(float* d_matrix, std::size_t ld,
                            float* d_upperTriangular, std::size_t n,
                            hipStream_t stream = nullptr);

void extractUpperTriangular(double* d_matrix, std::size_t ld,
                            double* d_upperTriangular, std::size_t n,
                            hipStream_t stream = nullptr);

void extractUpperTriangular(std::complex<float>* d_matrix, std::size_t ld,
                            std::complex<float>* d_upperTriangular,
                            std::size_t n, hipStream_t stream = nullptr);

void extractUpperTriangular(std::complex<double>* d_matrix, std::size_t ld,
                            std::complex<double>* d_upperTriangular,
                            std::size_t n, hipStream_t stream = nullptr);

void unpackUpperTriangular(float* d_matrix, std::size_t ld,
                           const float* d_upperTriangular, std::size_t n,
                           hipStream_t stream = nullptr);

void unpackUpperTriangular(double* d_matrix, std::size_t ld,
                           const double* d_upperTriangular, std::size_t n,
                           hipStream_t stream = nullptr);

void unpackUpperTriangular(std::complex<float>* d_matrix, std::size_t ld,
                           const std::complex<float>* d_upperTriangular,
                           std::size_t n, hipStream_t stream = nullptr);

void unpackUpperTriangular(std::complex<double>* d_matrix, std::size_t ld,
                           const std::complex<double>* d_upperTriangular,
                           std::size_t n, hipStream_t stream = nullptr);

} // namespace hip
} // namespace internal
} // namespace linalg
} // namespace chase

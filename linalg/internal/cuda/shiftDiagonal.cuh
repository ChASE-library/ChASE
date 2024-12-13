// This file is a part of ChASE.
// Copyright (c) 2015-2024, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <iostream>
#include <complex>

namespace chase
{
namespace linalg
{
namespace internal
{
namespace cuda
{
    __global__ void sshift_matrix(float* A, std::size_t n, std::size_t lda, float shift);
    __global__ void dshift_matrix(double* A, std::size_t n, std::size_t lda, double shift);
    __global__ void cshift_matrix(cuComplex* A, std::size_t n, std::size_t lda, float shift);
    __global__ void zshift_matrix(cuDoubleComplex* A, std::size_t n, std::size_t lda, double shift);

    __global__ void sshift_mgpu_matrix(float* A, std::size_t* off_m,
                                    std::size_t* off_n, std::size_t offsize,
                                    std::size_t ldH, float shift);
    __global__ void dshift_mgpu_matrix(double* A, std::size_t* off_m,
                                    std::size_t* off_n, std::size_t offsize,
                                    std::size_t ldH, double shift);
    __global__ void cshift_mgpu_matrix(cuComplex* A, std::size_t* off_m,
                                    std::size_t* off_n, std::size_t offsize,
                                    std::size_t ldH, float shift);
     __global__ void zshift_mgpu_matrix(cuDoubleComplex* A, std::size_t* off_m,
                                    std::size_t* off_n, std::size_t offsize,
                                    std::size_t ldH, double shift);

    void chase_shift_matrix(float* A, std::size_t n, std::size_t lda, float shift, cudaStream_t stream_);
    void chase_shift_matrix(double* A, std::size_t n, std::size_t lda, double shift, cudaStream_t stream_);
    void chase_shift_matrix(std::complex<float>* A, std::size_t n, std::size_t lda, float shift,
                            cudaStream_t stream_);
    void chase_shift_matrix(std::complex<double>* A, std::size_t n, std::size_t lda, double shift,
                            cudaStream_t stream_);


    void chase_shift_mgpu_matrix(float* A, std::size_t* off_m, std::size_t* off_n,
                                std::size_t offsize, std::size_t ldH, float shift,
                                cudaStream_t stream_);
    void chase_shift_mgpu_matrix(double* A, std::size_t* off_m, std::size_t* off_n,
                                std::size_t offsize, std::size_t ldH, double shift,
                                cudaStream_t stream_);
    void chase_shift_mgpu_matrix(std::complex<float>* A, std::size_t* off_m,
                                std::size_t* off_n, std::size_t offsize,
                                std::size_t ldH, float shift, cudaStream_t stream_);
    void chase_shift_mgpu_matrix(std::complex<double>* A, std::size_t* off_m,
                                std::size_t* off_n, std::size_t offsize,
                                std::size_t ldH, double shift,
                                cudaStream_t stream_);
}
}
}
}
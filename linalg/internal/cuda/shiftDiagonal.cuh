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
    
    __global__ void ssubtract_inverse_diagonal(float* A, std::size_t n, std::size_t lda, float coef, float* new_diag);
    __global__ void dsubtract_inverse_diagonal(double* A, std::size_t n, std::size_t lda, double coef, double* new_diag);
    __global__ void csubtract_inverse_diagonal(cuComplex* A, std::size_t n, std::size_t lda, float coef, double* new_diag);
    __global__ void zsubtract_inverse_diagonal(cuDoubleComplex* A, std::size_t n, std::size_t lda, double coef, double* new_diag);
    
    void chase_subtract_inverse_diagonal(float* A, std::size_t n, std::size_t lda, float coef, float* new_diag, cudaStream_t stream_);
    void chase_subtract_inverse_diagonal(double* A, std::size_t n, std::size_t lda, double coef, double* new_diag, cudaStream_t stream_);
    void chase_subtract_inverse_diagonal(std::complex<float>* A, std::size_t n, std::size_t lda, float coef, float* new_diag, cudaStream_t stream_);
    void chase_subtract_inverse_diagonal(std::complex<double>* A, std::size_t n, std::size_t lda, double coef, double* new_diag, cudaStream_t stream_);
    
    __global__ void splus_inverse_diagonal(float* A, std::size_t n, std::size_t lda, float coef, float* new_diag);
    __global__ void dplus_inverse_diagonal(double* A, std::size_t n, std::size_t lda, double coef, double* new_diag);
    __global__ void cplus_inverse_diagonal(cuComplex* A, std::size_t n, std::size_t lda, float coef, double* new_diag);
    __global__ void zplus_inverse_diagonal(cuDoubleComplex* A, std::size_t n, std::size_t lda, double coef, double* new_diag);
    
    void chase_plus_inverse_diagonal(float* A, std::size_t n, std::size_t lda, float coef, float* new_diag, cudaStream_t stream_);
    void chase_plus_inverse_diagonal(double* A, std::size_t n, std::size_t lda, double coef, double* new_diag, cudaStream_t stream_);
    void chase_plus_inverse_diagonal(std::complex<float>* A, std::size_t n, std::size_t lda, float coef, float* new_diag, cudaStream_t stream_);
    void chase_plus_inverse_diagonal(std::complex<double>* A, std::size_t n, std::size_t lda, double coef, double* new_diag, cudaStream_t stream_);

    __global__ void sset_diagonal(float* A, std::size_t n, std::size_t lda, float coef);
    __global__ void dset_diagonal(double* A, std::size_t n, std::size_t lda, double coef);
    __global__ void cset_diagonal(cuComplex* A, std::size_t n, std::size_t lda, cuComplex coef);
    __global__ void zset_diagonal(cuDoubleComplex* A, std::size_t n, std::size_t lda, cuDoubleComplex coef);

    void chase_set_diagonal(float* A, std::size_t n, std::size_t lda, float coef, cudaStream_t stream_);
    void chase_set_diagonal(double* A, std::size_t n, std::size_t lda, double coef, cudaStream_t stream_);
    void chase_set_diagonal(std::complex<float>* A, std::size_t n, std::size_t lda, std::complex<float> coef, cudaStream_t stream_);
    void chase_set_diagonal(std::complex<double>* A, std::size_t n, std::size_t lda, std::complex<double> coef, cudaStream_t stream_);
    
    __global__ void sscale_rows_matrix(float* A, std::size_t m, std::size_t n, std::size_t lda, float* coef);
    __global__ void dscale_rows_matrix(double* A, std::size_t m, std::size_t n, std::size_t lda, double* coef);
    __global__ void cscale_rows_matrix(cuComplex* A, std::size_t m, std::size_t n, std::size_t lda, float* coef);
    __global__ void zscale_rows_matrix(cuDoubleComplex* A, std::size_t m, std::size_t n, std::size_t lda, double* coef); // Can remain real probably?

    void chase_scale_rows_matrix(float* A, std::size_t m, std::size_t n, std::size_t lda, float* coef, cudaStream_t stream_);
    void chase_scale_rows_matrix(double* A, std::size_t m, std::size_t n, std::size_t lda, double* coef, cudaStream_t stream_);
    void chase_scale_rows_matrix(std::complex<float>* A, std::size_t m, std::size_t n, std::size_t lda, float* coef, cudaStream_t stream_);
    void chase_scale_rows_matrix(std::complex<double>* A, std::size_t m, std::size_t n, std::size_t lda, double* coef, cudaStream_t stream_);
    

}
}
}
}

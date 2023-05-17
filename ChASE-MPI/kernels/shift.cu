/* -*- Mode: C++; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
// This file is a part of ChASE.
// Copyright (c) 2015-2023, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include "cublas_v2.h"
#include <complex>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <omp.h>

#define BLOCKDIM 256
#define GRIDDIM 32

#define BLK_X 64
#define BLK_Y BLK_X

const int max_blocks = 65535;

static __device__ void dlacpy_full_device(
    int m, int n,
    const double *dA, int ldda,
    double       *dB, int lddb )
{
    int ind = blockIdx.x*BLK_X + threadIdx.x;
    int iby = blockIdx.y*BLK_Y;
    bool full = (iby + BLK_Y <= n);
    if ( ind < m ) {
        dA += ind + iby*ldda;
        dB += ind + iby*lddb;
        if ( full ) {
            #pragma unroll
            for( int j=0; j < BLK_Y; ++j ) {
                dB[j*lddb] = dA[j*ldda];
            }
        }
        else {
            for( int j=0; j < BLK_Y && iby+j < n; ++j ) {
                dB[j*lddb] = dA[j*ldda];
            }
        }
    }
}

__global__ void dlacpy_full_kernel(
    int m, int n,
    const double *dA, int ldda,
    double       *dB, int lddb )
{
    dlacpy_full_device(m, n, dA, ldda, dB, lddb);
}


static __device__ void slacpy_full_device(
    int m, int n,
    const float *dA, int ldda,
    float       *dB, int lddb )
{
    int ind = blockIdx.x*BLK_X + threadIdx.x;
    int iby = blockIdx.y*BLK_Y;
    bool full = (iby + BLK_Y <= n);
    if ( ind < m ) {
        dA += ind + iby*ldda;
        dB += ind + iby*lddb;
        if ( full ) {
            #pragma unroll
            for( int j=0; j < BLK_Y; ++j ) {
                dB[j*lddb] = dA[j*ldda];
            }
        }
        else {
            for( int j=0; j < BLK_Y && iby+j < n; ++j ) {
                dB[j*lddb] = dA[j*ldda];
            }
        }
    }
}

__global__ void slacpy_full_kernel(
    int m, int n,
    const float *dA, int ldda,
    float       *dB, int lddb )
{
    slacpy_full_device(m, n, dA, ldda, dB, lddb);
}


static __device__ void zlacpy_full_device(
    int m, int n,
    const cuDoubleComplex *dA, int ldda,
    cuDoubleComplex       *dB, int lddb )
{
    int ind = blockIdx.x*BLK_X + threadIdx.x;
    int iby = blockIdx.y*BLK_Y;
    bool full = (iby + BLK_Y <= n);
    if ( ind < m ) {
        dA += ind + iby*ldda;
        dB += ind + iby*lddb;
        if ( full ) {
            #pragma unroll
            for( int j=0; j < BLK_Y; ++j ) {
                dB[j*lddb] = dA[j*ldda];
            }
        }
        else {
            for( int j=0; j < BLK_Y && iby+j < n; ++j ) {
                dB[j*lddb] = dA[j*ldda];
            }
        }
    }
}

__global__
void zlacpy_full_kernel(
    int m, int n,
    const cuDoubleComplex *dA, int ldda,
    cuDoubleComplex       *dB, int lddb )
{
    zlacpy_full_device(m, n, dA, ldda, dB, lddb);
}

static __device__ void clacpy_full_device(
    int m, int n,
    const cuComplex *dA, int ldda,
    cuComplex       *dB, int lddb )
{
    int ind = blockIdx.x*BLK_X + threadIdx.x;
    int iby = blockIdx.y*BLK_Y;
    bool full = (iby + BLK_Y <= n);
    if ( ind < m ) {
        dA += ind + iby*ldda;
        dB += ind + iby*lddb;
        if ( full ) {
            #pragma unroll
            for( int j=0; j < BLK_Y; ++j ) {
                dB[j*lddb] = dA[j*ldda];
            }
        }
        else {
            for( int j=0; j < BLK_Y && iby+j < n; ++j ) {
                dB[j*lddb] = dA[j*ldda];
            }
        }
    }
}

__global__ void clacpy_full_kernel(
    int m, int n,
    const cuComplex *dA, int ldda,
    cuComplex       *dB, int lddb )
{
    clacpy_full_device(m, n, dA, ldda, dB, lddb);
}


// generate `n` random float numbers on GPU
__global__ void s_normal_kernel(unsigned long long seed, curandStatePhilox4_32_10_t* states,
                                float* v, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandStatePhilox4_32_10_t* state = states + tid;
    curand_init(seed, tid, 0, state);

    int i;
    int nthreads = gridDim.x * blockDim.x;

    for (i = tid; i < n; i += nthreads)
    {
        v[i] = curand_normal(state);
    }
}

// generate `n` random double numbers on GPU
__global__ void d_normal_kernel(unsigned long long seed, curandStatePhilox4_32_10_t* states,
                                double* v, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandStatePhilox4_32_10_t* state = states + tid;
    curand_init(seed, tid, 0, state);

    int i;
    int nthreads = gridDim.x * blockDim.x;

    for (i = tid; i < n; i += nthreads)
    {
        v[i] = curand_normal_double(state);
    }
}
// generate `n` random complex single numbers on GPU
__global__ void c_normal_kernel(unsigned long long seed, curandStatePhilox4_32_10_t* states,
                                cuComplex* v, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandStatePhilox4_32_10_t* state = states + tid;
    curand_init(seed, tid, 0, state);

    int i;
    int nthreads = gridDim.x * blockDim.x;

    for (i = tid; i < n; i += nthreads)
    {
        float rnd = curand_normal(state);
        v[i].x = rnd;
        v[i].y = rnd;
    }
}

// generate `n` random complex double numbers on GPU
__global__ void z_normal_kernel(unsigned long long seed, curandStatePhilox4_32_10_t* states,
                                cuDoubleComplex* v, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandStatePhilox4_32_10_t* state = states + tid;
    curand_init(seed, tid, 0, state);

    int i;
    int nthreads = gridDim.x * blockDim.x;

    for (i = tid; i < n; i += nthreads)
    {
        double rnd = curand_normal_double(state);
        v[i].x = rnd;
        v[i].y = rnd;
    }
}

__global__ void sshift_matrix(float* A, int n, float shift)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        A[(idx)*n + idx] += shift;
}

__global__ void dshift_matrix(double* A, int n, double shift)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        A[(idx)*n + idx] += shift;
}

__global__ void cshift_matrix(cuComplex* A, int n, float shift)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        A[(idx)*n + idx].x += shift;
}

__global__ void zshift_matrix(cuDoubleComplex* A, int n, double shift)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        A[(idx)*n + idx].x += shift;
}

__global__ void sshift_mgpu_matrix(float* A, std::size_t* off_m,
                                   std::size_t* off_n, std::size_t offsize,
                                   std::size_t ldH, float shift)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    std::size_t ind;
    if (i < offsize)
    {
        ind = off_n[i] * ldH + off_m[i];
        A[ind] += shift;
    }
}

__global__ void dshift_mgpu_matrix(double* A, std::size_t* off_m,
                                   std::size_t* off_n, std::size_t offsize,
                                   std::size_t ldH, double shift)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    std::size_t ind;
    if (i < offsize)
    {
        ind = off_n[i] * ldH + off_m[i];
        A[ind] += shift;
    }
}

__global__ void cshift_mgpu_matrix(cuComplex* A, std::size_t* off_m,
                                   std::size_t* off_n, std::size_t offsize,
                                   std::size_t ldH, float shift)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    std::size_t ind;
    if (i < offsize)
    {
        ind = off_n[i] * ldH + off_m[i];
        A[ind].x += shift;
    }
}

__global__ void zshift_mgpu_matrix(cuDoubleComplex* A, std::size_t* off_m,
                                   std::size_t* off_n, std::size_t offsize,
                                   std::size_t ldH, double shift)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    std::size_t ind;
    if (i < offsize)
    {
        ind = off_n[i] * ldH + off_m[i];
        A[ind].x += shift;
    }
}
//only full copy is support right now
void t_lacpy_gpu(char uplo, int m, int n, float *dA, int ldda, float *dB, int lddb, cudaStream_t stream_ )
{
    #define dA(i_, j_) (dA + (i_) + (j_)*ldda)
    #define dB(i_, j_) (dB + (i_) + (j_)*lddb)
    int super_NB = max_blocks*BLK_X;
    dim3 super_grid(  (m + super_NB - 1) / super_NB,  (n + super_NB - 1) / super_NB );

    dim3 threads( BLK_X, 1 );
    dim3 grid;

    int mm, nn;
    for( unsigned int i=0; i < super_grid.x; ++i ) {
        mm = (i == super_grid.x-1 ? m % super_NB : super_NB);
        grid.x = ( mm + BLK_X - 1) / BLK_X;
        for( unsigned int j=0; j < super_grid.y; ++j ) {  // full row
            nn = (j == super_grid.y-1 ? n % super_NB : super_NB);
            grid.y = ( nn + BLK_X - 1) / BLK_Y;;
            slacpy_full_kernel <<< grid, threads, 0, stream_ >>>
                ( mm, nn, dA(i*super_NB, j*super_NB), ldda, dB(i*super_NB, j*super_NB), lddb );
        }
    }
}

void t_lacpy_gpu(char uplo, int m, int n, double *dA, int ldda, double *dB, int lddb, cudaStream_t stream_ )
{
    #define dA(i_, j_) (dA + (i_) + (j_)*ldda)
    #define dB(i_, j_) (dB + (i_) + (j_)*lddb)	
    int super_NB = max_blocks*BLK_X;
    dim3 super_grid(  (m + super_NB - 1) / super_NB,  (n + super_NB - 1) / super_NB );     

    dim3 threads( BLK_X, 1 );
    dim3 grid;

    int mm, nn;
    for( unsigned int i=0; i < super_grid.x; ++i ) {
        mm = (i == super_grid.x-1 ? m % super_NB : super_NB);
        grid.x = ( mm + BLK_X - 1) / BLK_X;
        for( unsigned int j=0; j < super_grid.y; ++j ) {  // full row
            nn = (j == super_grid.y-1 ? n % super_NB : super_NB);
            grid.y = ( nn + BLK_X - 1) / BLK_Y;;
            dlacpy_full_kernel <<< grid, threads, 0, stream_ >>>
                ( mm, nn, dA(i*super_NB, j*super_NB), ldda, dB(i*super_NB, j*super_NB), lddb );
        }
    }
}	

void t_lacpy_gpu(char uplo, int m, int n, std::complex<double> *ddA, int ldda, std::complex<double> *ddB, int lddb, cudaStream_t stream_ )
{
    cuDoubleComplex *dA = reinterpret_cast<cuDoubleComplex*>(ddA);
    cuDoubleComplex *dB = reinterpret_cast<cuDoubleComplex*>(ddB);
    #define dA(i_, j_) (dA + (i_) + (j_)*ldda)
    #define dB(i_, j_) (dB + (i_) + (j_)*lddb)
    int super_NB = max_blocks*BLK_X;
    dim3 super_grid(  (m + super_NB - 1) / super_NB,  (n + super_NB - 1) / super_NB );

    dim3 threads( BLK_X, 1 );
    dim3 grid;

    int mm, nn;
    for( unsigned int i=0; i < super_grid.x; ++i ) {
        mm = (i == super_grid.x-1 ? m % super_NB : super_NB);
        grid.x = ( mm + BLK_X - 1) / BLK_X;
        for( unsigned int j=0; j < super_grid.y; ++j ) {  // full row
            nn = (j == super_grid.y-1 ? n % super_NB : super_NB);
            grid.y = ( nn + BLK_X - 1) / BLK_Y;;
            zlacpy_full_kernel <<< grid, threads, 0, stream_ >>>
                ( mm, nn, dA(i*super_NB, j*super_NB), ldda, dB(i*super_NB, j*super_NB), lddb );
        }
    }
}

void t_lacpy_gpu(char uplo, int m, int n, std::complex<float> *ddA, int ldda, std::complex<float> *ddB, int lddb, cudaStream_t stream_ )
{
    cuComplex *dA = reinterpret_cast<cuComplex*>(ddA);
    cuComplex *dB = reinterpret_cast<cuComplex*>(ddB);
    #define dA(i_, j_) (dA + (i_) + (j_)*ldda)
    #define dB(i_, j_) (dB + (i_) + (j_)*lddb)
    int super_NB = max_blocks*BLK_X;
    dim3 super_grid(  (m + super_NB - 1) / super_NB,  (n + super_NB - 1) / super_NB );

    dim3 threads( BLK_X, 1 );
    dim3 grid;

    int mm, nn;
    for( unsigned int i=0; i < super_grid.x; ++i ) {
        mm = (i == super_grid.x-1 ? m % super_NB : super_NB);
        grid.x = ( mm + BLK_X - 1) / BLK_X;
        for( unsigned int j=0; j < super_grid.y; ++j ) {  // full row
            nn = (j == super_grid.y-1 ? n % super_NB : super_NB);
            grid.y = ( nn + BLK_X - 1) / BLK_Y;;
            clacpy_full_kernel <<< grid, threads, 0, stream_ >>>
                ( mm, nn, dA(i*super_NB, j*super_NB), ldda, dB(i*super_NB, j*super_NB), lddb );
        }
    }
}

void chase_rand_normal(unsigned long long seed, curandStatePhilox4_32_10_t* states, float* v,
                       int n, cudaStream_t stream_)
{
    s_normal_kernel<<<GRIDDIM, BLOCKDIM, 0, stream_>>>(seed, states, v, n);
}

void chase_rand_normal(unsigned long long seed, curandStatePhilox4_32_10_t* states, double* v,
                       int n, cudaStream_t stream_)
{
    d_normal_kernel<<<GRIDDIM, BLOCKDIM, 0, stream_>>>(seed, states, v, n);
}

void chase_rand_normal(unsigned long long seed, curandStatePhilox4_32_10_t* states,
                       std::complex<float>* v, int n, cudaStream_t stream_)
{
    c_normal_kernel<<<GRIDDIM, BLOCKDIM, 0, stream_>>>(
        seed, states, reinterpret_cast<cuComplex*>(v), n);
}

void chase_rand_normal(unsigned long long seed, curandStatePhilox4_32_10_t* states,
                       std::complex<double>* v, int n, cudaStream_t stream_)
{
    z_normal_kernel<<<GRIDDIM, BLOCKDIM, 0, stream_>>>(
        seed, states, reinterpret_cast<cuDoubleComplex*>(v), n);
}

void chase_shift_matrix(float* A, int n, float shift, cudaStream_t* stream_)
{
    int num_blocks = (n + (BLOCKDIM - 1)) / BLOCKDIM;
    sshift_matrix<<<num_blocks, BLOCKDIM, 0, *stream_>>>(A, n, shift);
}

void chase_shift_matrix(double* A, int n, double shift, cudaStream_t* stream_)
{
    int num_blocks = (n + (BLOCKDIM - 1)) / BLOCKDIM;
    dshift_matrix<<<num_blocks, BLOCKDIM, 0, *stream_>>>(A, n, shift);
}

void chase_shift_matrix(std::complex<float>* A, int n, float shift,
                        cudaStream_t* stream_)
{
    int num_blocks = (n + (BLOCKDIM - 1)) / BLOCKDIM;
    cshift_matrix<<<num_blocks, BLOCKDIM, 0, *stream_>>>(
        reinterpret_cast<cuComplex*>(A), n, shift);
}

void chase_shift_matrix(std::complex<double>* A, int n, double shift,
                        cudaStream_t* stream_)
{
    int num_blocks = (n + (BLOCKDIM - 1)) / BLOCKDIM;
    zshift_matrix<<<num_blocks, BLOCKDIM, 0, *stream_>>>(
        reinterpret_cast<cuDoubleComplex*>(A), n, shift);
}

void chase_shift_mgpu_matrix(float* A, std::size_t* off_m, std::size_t* off_n,
                             std::size_t offsize, std::size_t ldH, float shift,
                             cudaStream_t stream_)
{

    unsigned int grid = (offsize + 256 - 1) / 256;
    dim3 threadsPerBlock(256, 1);
    dim3 numBlocks(grid, 1);
    sshift_mgpu_matrix<<<numBlocks, threadsPerBlock, 0, stream_>>>( //
        A, off_m, off_n, offsize, ldH, shift);
}

void chase_shift_mgpu_matrix(double* A, std::size_t* off_m, std::size_t* off_n,
                             std::size_t offsize, std::size_t ldH, double shift,
                             cudaStream_t stream_)
{

    unsigned int grid = (offsize + 256 - 1) / 256;
    dim3 threadsPerBlock(256, 1);
    dim3 numBlocks(grid, 1);
    dshift_mgpu_matrix<<<numBlocks, threadsPerBlock, 0, stream_>>>( //
        A, off_m, off_n, offsize, ldH, shift);
}

void chase_shift_mgpu_matrix(std::complex<float>* A, std::size_t* off_m,
                             std::size_t* off_n, std::size_t offsize,
                             std::size_t ldH, float shift, cudaStream_t stream_)
{

    unsigned int grid = (offsize + 256 - 1) / 256;
    dim3 threadsPerBlock(256, 1);
    dim3 numBlocks(grid, 1);
    cshift_mgpu_matrix<<<numBlocks, threadsPerBlock, 0, stream_>>>( //
        reinterpret_cast<cuComplex*>(A), off_m, off_n,              //
        offsize, ldH, shift);
}

void chase_shift_mgpu_matrix(std::complex<double>* A, std::size_t* off_m,
                             std::size_t* off_n, std::size_t offsize,
                             std::size_t ldH, double shift,
                             cudaStream_t stream_)
{

    unsigned int grid = (offsize + 256 - 1) / 256;
    dim3 threadsPerBlock(256, 1);
    dim3 numBlocks(grid, 1);
    zshift_mgpu_matrix<<<numBlocks, threadsPerBlock, 0, stream_>>>( //
        reinterpret_cast<cuDoubleComplex*>(A), off_m, off_n,        //
        offsize, ldH, shift);
}

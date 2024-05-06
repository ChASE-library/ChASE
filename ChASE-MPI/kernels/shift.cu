/* -*- Mode: C++; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
// This file is a part of ChASE.
// Copyright (c) 2015-2023, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include <cfloat>
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
#define NB_X 256

const int max_blocks = 65535;

template< int n, typename T >
__device__ void cuda_sum_reduce(int i, T* x )
{
    __syncthreads();
    if ( n > 1024 ) { if ( i < 1024 && i + 1024 < n ) { x[i] += x[i+1024]; }  __syncthreads(); }
    if ( n >  512 ) { if ( i <  512 && i +  512 < n ) { x[i] += x[i+ 512]; }  __syncthreads(); }
    if ( n >  256 ) { if ( i <  256 && i +  256 < n ) { x[i] += x[i+ 256]; }  __syncthreads(); }
    if ( n >  128 ) { if ( i <  128 && i +  128 < n ) { x[i] += x[i+ 128]; }  __syncthreads(); }
    if ( n >   64 ) { if ( i <   64 && i +   64 < n ) { x[i] += x[i+  64]; }  __syncthreads(); }
    if ( n >   32 ) { if ( i <   32 && i +   32 < n ) { x[i] += x[i+  32]; }  __syncthreads(); }
    if ( n >   16 ) { if ( i <   16 && i +   16 < n ) { x[i] += x[i+  16]; }  __syncthreads(); }
    if ( n >    8 ) { if ( i <    8 && i +    8 < n ) { x[i] += x[i+   8]; }  __syncthreads(); }
    if ( n >    4 ) { if ( i <    4 && i +    4 < n ) { x[i] += x[i+   4]; }  __syncthreads(); }
    if ( n >    2 ) { if ( i <    2 && i +    2 < n ) { x[i] += x[i+   2]; }  __syncthreads(); }
    if ( n >    1 ) { if ( i <    1 && i +    1 < n ) { x[i] += x[i+   1]; }  __syncthreads(); }
}

template<int n, typename T>
__device__ void cuda_max_reduce(int i, T* x) {
    __syncthreads();
    if (n > 1024) { if (i < 1024 && i + 1024 < n) { x[i] = max(x[i], x[i + 1024]); } __syncthreads(); }
    if (n >  512) { if (i <  512 && i +  512 < n) { x[i] = max(x[i], x[i +  512]); } __syncthreads(); }
    if (n >  256) { if (i <  256 && i +  256 < n) { x[i] = max(x[i], x[i +  256]); } __syncthreads(); }
    if (n >  128) { if (i <  128 && i +  128 < n) { x[i] = max(x[i], x[i +  128]); } __syncthreads(); }
    if (n >   64) { if (i <   64 && i +   64 < n) { x[i] = max(x[i], x[i +   64]); } __syncthreads(); }
    if (n >   32) { if (i <   32 && i +   32 < n) { x[i] = max(x[i], x[i +   32]); } __syncthreads(); }
    if (n >   16) { if (i <   16 && i +   16 < n) { x[i] = max(x[i], x[i +   16]); } __syncthreads(); }
    if (n >    8) { if (i <    8 && i +    8 < n) { x[i] = max(x[i], x[i +    8]); } __syncthreads(); }
    if (n >    4) { if (i <    4 && i +    4 < n) { x[i] = max(x[i], x[i +    4]); } __syncthreads(); }
    if (n >    2) { if (i <    2 && i +    2 < n) { x[i] = max(x[i], x[i +    2]); } __syncthreads(); }
    if (n >    1) { if (i <    1 && i +    1 < n) { x[i] = max(x[i], x[i +    1]); } __syncthreads(); }
}

__device__ double atomicMax(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
            __double_as_longlong(fmax(val, __longlong_as_double(assumed))));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__device__ float atomicMax(float* address, float val)
{
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
            __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__ void s_computeNegative(float* d, float* neg) {
    *neg = -(*d);
}

__global__ void d_computeNegative(double* d, double* neg) {
    *neg = -(*d);
}

__global__ void c_computeNegative(cuComplex* d, cuComplex* neg) {
    *neg = make_cuComplex(-cuCrealf(*d), -cuCimagf(*d));
}

__global__ void z_computeNegative(cuDoubleComplex* d, cuDoubleComplex* neg) {
    *neg = make_cuDoubleComplex(-cuCreal(*d), -cuCimag(*d));
}

__global__ void s_InverseAndConvert(float *d, float *inv)
{
    *inv = 1.0 / (*d);
}

__global__ void d_InverseAndConvert(double *d, double *inv)
{
    *inv = 1.0 / (*d);
}

__global__ void c_InverseAndConvert(float *d, cuComplex *inv)
{
    float inv_1 = 1.0 / (*d);
    *inv = make_cuComplex(inv_1, 0.0);
}

__global__ void z_InverseAndConvert(double *d, cuDoubleComplex *inv)
{
    double inv_1 = 1.0 / (*d);
    *inv = make_cuDoubleComplex(inv_1, 0.0);
}

__global__ void s_convert2Real(float *a, float *b)
{
    *b = *a;
}

__global__ void d_convert2Real(double *a, double *b)
{
    *b = *a;
}

__global__ void c_convert2Real(cuComplex *a, float *b)
{
    *b = cuCrealf(*a);
}

__global__ void z_convert2Real(cuDoubleComplex *a, double *b)
{
    *b = cuCreal(*a);
}

__global__ void s_convert2Complex(float *a, float *b)
{
    *b = *a;
}

__global__ void d_convert2Complex(double *a, double *b)
{
    *b = *a;
}

__global__ void c_convert2Complex(float *a, cuComplex *b)
{
    *b = make_cuComplex(*a, 0.0);
}

__global__ void z_convert2Complex(double *a, cuDoubleComplex *b)
{
    *b = make_cuDoubleComplex(*a, 0.0);
}

__global__ void s_absTraceKernel(float* d_matrix, float* d_trace, int n, int ld) {
    __shared__ float partial_trace[NB_X];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    partial_trace[threadIdx.x] = (idx < n) ? fabs(d_matrix[idx * ld + idx]) : 0.0f;
    cuda_sum_reduce<NB_X>(threadIdx.x, partial_trace);
    if (threadIdx.x == 0) {
        atomicAdd(d_trace, partial_trace[0]);
    }
}

__global__ void d_absTraceKernel(double* d_matrix, double* d_trace, int n, int ld) {
    __shared__ double partial_trace[NB_X];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    partial_trace[threadIdx.x] = (idx < n) ? fabs(d_matrix[idx * ld + idx]) : 0.0;
    cuda_sum_reduce<NB_X>(threadIdx.x, partial_trace);
    if (threadIdx.x == 0) {
        atomicAdd(d_trace, partial_trace[0]);
    }
}

__global__ void c_absTraceKernel(cuComplex* d_matrix, float* d_trace, int n, int ld) {
    __shared__ float partial_trace[NB_X];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    partial_trace[threadIdx.x] = (idx < n) ? cuCabsf(d_matrix[idx * ld + idx]) : 0.0f;
    cuda_sum_reduce<NB_X>(threadIdx.x, partial_trace);
    if (threadIdx.x == 0) {
        atomicAdd(d_trace, partial_trace[0]);
    }
}

__global__ void z_absTraceKernel(cuDoubleComplex* d_matrix, double* d_trace, int n, int ld) {
    __shared__ double partial_trace[NB_X];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    partial_trace[threadIdx.x] = (idx < n) ? cuCabs(d_matrix[idx * ld + idx]) : 0.0;
    cuda_sum_reduce<NB_X>(threadIdx.x, partial_trace);
    if (threadIdx.x == 0) {
        atomicAdd(d_trace, partial_trace[0]);
    }
}

__global__ void s_find_max_and_add_abs_scalar(float *array, int size, float *scalar, float *max_value)
{
    __shared__ float shared_array[NB_X];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    shared_array[threadIdx.x] = (idx < size) ? array[idx]:  -FLT_MAX; 

    __syncthreads();
    cuda_max_reduce<NB_X>(threadIdx.x, shared_array);
    if (threadIdx.x == 0) {
        atomicMax(max_value, shared_array[0] + fabs(*scalar));
    }    
}

__global__ void d_find_max_and_add_abs_scalar(double *array, int size, double *scalar, double *max_value)
{
    __shared__ double shared_array[NB_X];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    shared_array[threadIdx.x] = (idx < size) ? array[idx]: -DBL_MAX;

    __syncthreads();
    cuda_max_reduce<NB_X>(threadIdx.x, shared_array);
    if (threadIdx.x == 0) {
        atomicMax(max_value, shared_array[0] + fabs(*scalar));
    }    
}

__global__ void c_find_max_and_add_abs_scalar(float *array, int size, cuComplex *scalar, float *max_value)
{
    __shared__ float shared_array[NB_X];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    shared_array[threadIdx.x] = (idx < size) ? array[idx]:  -FLT_MAX; 

    __syncthreads();
    cuda_max_reduce<NB_X>(threadIdx.x, shared_array);
    if (threadIdx.x == 0) {
        atomicMax(max_value, shared_array[0] + cuCabsf(*scalar));
    }    
}

__global__ void z_find_max_and_add_abs_scalar(double *array, int size, cuDoubleComplex *scalar, double *max_value)
{
    __shared__ double shared_array[NB_X];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    shared_array[threadIdx.x] = (idx < size) ? array[idx]: -DBL_MAX;

    __syncthreads();
    cuda_max_reduce<NB_X>(threadIdx.x, shared_array);
    if (threadIdx.x == 0) {
        atomicMax(max_value, shared_array[0] + cuCabs(*scalar));
    }    
}

__global__ void c_resids_kernel(int m, int n, const cuComplex *A, int lda, const cuComplex *B, 
			 int ldb, float *ritzv, float *resids, bool is_sqrt )
{
    __shared__ float ssum[NB_X];
    int tx = threadIdx.x;
    A += blockIdx.x*lda;
    B += blockIdx.x*lda;
    
    ssum[tx] = 0;
    for(int i = tx; i < m; i += NB_X)
    {
        cuComplex alpha;
       	alpha.x = ritzv[blockIdx.x];
	alpha.y = 0.0;
	cuComplex a = cuCmulf(alpha, B[i]);
	cuComplex b = cuCsubf(A[i], a);
    	float nrm = cuCabsf(b);
	ssum[tx] += nrm * nrm;
    }

    cuda_sum_reduce<NB_X>(tx, ssum);
    if ( tx == 0 ) {
        if(is_sqrt)
	{
	    resids[ blockIdx.x ] = sqrtf(ssum[0]);	
	}
	else{
	    resids[ blockIdx.x ] = ssum[0];
    	}
    }
    
}

__global__ void z_resids_kernel(int m, int n, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B,
                         int ldb, double *ritzv, double *resids, bool is_sqrt )
{
    __shared__ double ssum[NB_X];
    int tx = threadIdx.x;
    A += blockIdx.x*lda;
    B += blockIdx.x*lda;

    ssum[tx] = 0;
    for(int i = tx; i < m; i += NB_X)
    {
        cuDoubleComplex alpha;
        alpha.x = ritzv[blockIdx.x];
	alpha.y = 0.0;
        cuDoubleComplex a = cuCmul(alpha, B[i]);
        cuDoubleComplex b = cuCsub(A[i], a);
        double nrm = cuCabs(b);
	ssum[tx] += nrm * nrm;
    }

    cuda_sum_reduce<NB_X>(tx, ssum);
    if ( tx == 0 ) {
        if(is_sqrt)
        {
            resids[ blockIdx.x ] = sqrt(ssum[0]);
        }
        else{
            resids[ blockIdx.x ] = ssum[0];
        }
    }
}

__global__ void d_resids_kernel(int m, int n, const double *A, int lda, const double *B,
                         int ldb, double *ritzv, double *resids, bool is_sqrt )
{
    __shared__ double ssum[NB_X];
    int tx = threadIdx.x;
    A += blockIdx.x*lda;
    B += blockIdx.x*lda;

    ssum[tx] = 0;
    for(int i = tx; i < m; i += NB_X)
    {
        double alpha;
        alpha = ritzv[blockIdx.x];
        double a = alpha * B[i];
        double b = A[i] - a;
        ssum[tx] += b * b;
    }

    cuda_sum_reduce<NB_X>(tx, ssum);
    if ( tx == 0 ) {
        if(is_sqrt)
        {
            resids[ blockIdx.x ] = sqrt(ssum[0]);
        }
        else{
            resids[ blockIdx.x ] = ssum[0];
        }
    }
}

__global__ void s_resids_kernel(int m, int n, const float *A, int lda, const float *B,
                         int ldb, float *ritzv, float *resids, bool is_sqrt )
{
    __shared__ float ssum[NB_X];
    int tx = threadIdx.x;
    A += blockIdx.x*lda;
    B += blockIdx.x*lda;

    ssum[tx] = 0;
    for(int i = tx; i < m; i += NB_X)
    {
        float alpha;
        alpha = ritzv[blockIdx.x];
        float a = alpha * B[i];
        float b = A[i] - a;
        ssum[tx] += b * b;
    }

    cuda_sum_reduce<NB_X>(tx, ssum);
    if ( tx == 0 ) {
        if(is_sqrt)
        {
            resids[ blockIdx.x ] = sqrtf(ssum[0]);
        }
        else{
            resids[ blockIdx.x ] = ssum[0];
        }
    }
}

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

static __device__ void dlacpy_upper_device(
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
        int copyLimit = min(ind - iby + 1, BLK_Y);
        if ( full ) {
            #pragma unroll
            for( int j=0; j < BLK_Y; ++j ) {
                dB[j*lddb] = (j < copyLimit) ?  dA[j*ldda] : 0.0;
            }
        }
        else {
            for( int j=0; j < BLK_Y && iby+j < n; ++j ) {
                dB[j*lddb] =  (j < copyLimit) ?  dA[j*ldda] : 0.0;
            }
        }
    }
}

__global__ void dlacpy_upper_kernel(
    int m, int n,
    const double *dA, int ldda,
    double       *dB, int lddb )
{
    dlacpy_upper_device(m, n, dA, ldda, dB, lddb);
}

static __device__ void slacpy_upper_device(
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
        int copyLimit = min(ind - iby + 1, BLK_Y);
        if ( full ) {
            #pragma unroll
            for( int j=0; j < BLK_Y; ++j ) {
                dB[j*lddb] = (j < copyLimit) ?  dA[j*ldda] : 0.0;
            }
        }
        else {
            for( int j=0; j < BLK_Y && iby+j < n; ++j ) {
                dB[j*lddb] =  (j < copyLimit) ?  dA[j*ldda] : 0.0;
            }
        }
    }
}

__global__ void slacpy_upper_kernel(
    int m, int n,
    const float *dA, int ldda,
    float       *dB, int lddb )
{
    slacpy_upper_device(m, n, dA, ldda, dB, lddb);
}

static __device__ void clacpy_upper_device(
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
        int copyLimit = min(ind - iby + 1, BLK_Y);
        if ( full ) {
            #pragma unroll
            for( int j=0; j < BLK_Y; ++j ) {
                dB[j*lddb] = (j < copyLimit) ?  dA[j*ldda] : make_cuFloatComplex(0.0f, 0.0f);
            }
        }
        else {
            for( int j=0; j < BLK_Y && iby+j < n; ++j ) {
                dB[j*lddb] =  (j < copyLimit) ?  dA[j*ldda] : make_cuFloatComplex(0.0f, 0.0f);
            }
        }
    }
}

__global__ void clacpy_upper_kernel(
    int m, int n,
    const cuComplex *dA, int ldda,
    cuComplex       *dB, int lddb )
{
    clacpy_upper_device(m, n, dA, ldda, dB, lddb);
}

static __device__ void zlacpy_upper_device(
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
        int copyLimit = min(ind - iby + 1, BLK_Y);
        if ( full ) {
            #pragma unroll
            for( int j=0; j < BLK_Y; ++j ) {
                dB[j*lddb] = (j < copyLimit) ?  dA[j*ldda] : make_cuDoubleComplex(0.0, 0.0);
            }
        }
        else {
            for( int j=0; j < BLK_Y && iby+j < n; ++j ) {
                dB[j*lddb] =  (j < copyLimit) ?  dA[j*ldda] : make_cuDoubleComplex(0.0, 0.0);
            }
        }
    }
}

__global__ void zlacpy_upper_kernel(
    int m, int n,
    const cuDoubleComplex *dA, int ldda,
    cuDoubleComplex       *dB, int lddb )
{
    zlacpy_upper_device(m, n, dA, ldda, dB, lddb);
}


static __device__ void dlacpy_lower_device(
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
        int copyStart = max(ind - iby, 0);
        if ( full ) {
            #pragma unroll
            for( int j=0; j < BLK_Y; ++j ) {
                dB[j*lddb] = (j >= copyStart) ?  dA[j*ldda] : 0.0;
            }
        }
        else {
            for( int j=0; j < BLK_Y && iby+j < n; ++j ) {
                dB[j*lddb] =  (j >= copyStart) ?  dA[j*ldda] : 0.0;
            }
        }
    }
}

__global__ void dlacpy_lower_kernel(
    int m, int n,
    const double *dA, int ldda,
    double       *dB, int lddb )
{
    dlacpy_lower_device(m, n, dA, ldda, dB, lddb);
}

static __device__ void slacpy_lower_device(
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
        int copyStart = max(ind - iby, 0);
        if ( full ) {
            #pragma unroll
            for( int j=0; j < BLK_Y; ++j ) {
                dB[j*lddb] = (j >= copyStart) ?  dA[j*ldda] : 0.0;
            }
        }
        else {
            for( int j=0; j < BLK_Y && iby+j < n; ++j ) {
                dB[j*lddb] =  (j >= copyStart) ?  dA[j*ldda] : 0.0;
            }
        }
    }
}

__global__ void slacpy_lower_kernel(
    int m, int n,
    const float *dA, int ldda,
    float       *dB, int lddb )
{
    slacpy_lower_device(m, n, dA, ldda, dB, lddb);
}

static __device__ void clacpy_lower_device(
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
        int copyStart = max(ind - iby, 0);
        if ( full ) {
            #pragma unroll
            for( int j=0; j < BLK_Y; ++j ) {
                dB[j*lddb] = (j >= copyStart) ?  dA[j*ldda] : make_cuFloatComplex(0.0f, 0.0f);
            }
        }
        else {
            for( int j=0; j < BLK_Y && iby+j < n; ++j ) {
                dB[j*lddb] =  (j >= copyStart) ?  dA[j*ldda] : make_cuFloatComplex(0.0f, 0.0f);
            }
        }
    }
}

__global__ void clacpy_lower_kernel(
    int m, int n,
    const cuComplex *dA, int ldda,
    cuComplex       *dB, int lddb )
{
    clacpy_lower_device(m, n, dA, ldda, dB, lddb);
}

static __device__ void zlacpy_lower_device(
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
        int copyStart = max(ind - iby, 0);
        if ( full ) {
            #pragma unroll
            for( int j=0; j < BLK_Y; ++j ) {
                dB[j*lddb] = (j >= copyStart) ?  dA[j*ldda] : make_cuDoubleComplex(0.0, 0.0);
            }
        }
        else {
            for( int j=0; j < BLK_Y && iby+j < n; ++j ) {
                dB[j*lddb] =  (j >= copyStart) ?  dA[j*ldda] : make_cuDoubleComplex(0.0, 0.0);
            }
        }
    }
}

__global__ void zlacpy_lower_kernel(
    int m, int n,
    const cuDoubleComplex *dA, int ldda,
    cuDoubleComplex       *dB, int lddb )
{
    zlacpy_lower_device(m, n, dA, ldda, dB, lddb);
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

void computeNegative_gpu(float *d, float *neg, cudaStream_t stream_)
{
    s_computeNegative<<<1,1,0,stream_>>>(d, neg);
}

void computeNegative_gpu(double *d, double *neg, cudaStream_t stream_)
{
    d_computeNegative<<<1,1,0,stream_>>>(d, neg);
}

void computeNegative_gpu(std::complex<float> *d, std::complex<float> *neg, cudaStream_t stream_)
{
    c_computeNegative<<<1,1,0,stream_>>>(reinterpret_cast<cuComplex*>(d), reinterpret_cast<cuComplex*>(neg) );
}

void computeNegative_gpu(std::complex<double> *d, std::complex<double> *neg, cudaStream_t stream_)
{
    z_computeNegative<<<1,1,0,stream_>>>(reinterpret_cast<cuDoubleComplex*>(d), reinterpret_cast<cuDoubleComplex*>(neg) );
}

void InverseAndConvert_gpu(float *d, float *inv, cudaStream_t stream_)
{
    s_InverseAndConvert<<<1,1,0,stream_>>>(d, inv);
}

void InverseAndConvert_gpu(double *d, double *inv, cudaStream_t stream_)
{
    d_InverseAndConvert<<<1,1,0,stream_>>>(d, inv);
}

void InverseAndConvert_gpu(float *d, std::complex<float> *inv, cudaStream_t stream_)
{
    c_InverseAndConvert<<<1,1,0,stream_>>>(d, reinterpret_cast<cuComplex*>(inv));
}

void InverseAndConvert_gpu(double *d, std::complex<double> *inv, cudaStream_t stream_)
{
    z_InverseAndConvert<<<1,1,0,stream_>>>(d, reinterpret_cast<cuDoubleComplex*>(inv));
}

void convert2Real_gpu(float *a, float *b,  cudaStream_t stream_)
{
    s_convert2Real<<<1,1,0,stream_>>>(a, b);
}

void convert2Real_gpu(double *a, double *b,  cudaStream_t stream_)
{
    d_convert2Real<<<1,1,0,stream_>>>(a, b);
}

void convert2Real_gpu(std::complex<double> *a, double *b,  cudaStream_t stream_)
{
    z_convert2Real<<<1,1,0,stream_>>>(reinterpret_cast<cuDoubleComplex*>(a), b);
}

void convert2Real_gpu(std::complex<float> *a, float *b,  cudaStream_t stream_)
{
    c_convert2Real<<<1,1,0,stream_>>>(reinterpret_cast<cuComplex*>(a), b);
}


void convert2Complex_gpu(float *a, float *b,  cudaStream_t stream_)
{
    s_convert2Complex<<<1,1,0,stream_>>>(a, b);
}

void convert2Complex_gpu(double *a, double *b,  cudaStream_t stream_)
{
    d_convert2Complex<<<1,1,0,stream_>>>(a, b);
}

void convert2Complex_gpu(float *a, std::complex<float> *b,  cudaStream_t stream_)
{
    c_convert2Complex<<<1,1,0,stream_>>>(a, reinterpret_cast<cuComplex*>(b));
}

void convert2Complex_gpu(double *a, std::complex<double> *b,  cudaStream_t stream_)
{
    z_convert2Complex<<<1,1,0,stream_>>>(a, reinterpret_cast<cuDoubleComplex*>(b));
}

void absTrace_gpu(float* d_matrix, float* d_trace, int n, int ld, cudaStream_t stream_)
{
    dim3 threads( NB_X);
    int gridSize = (n + NB_X - 1) / NB_X;
    dim3 grid( gridSize );

    s_absTraceKernel<<< grid, threads, 0, stream_ >>>(d_matrix, d_trace, n, ld);
}

void absTrace_gpu(double* d_matrix, double* d_trace, int n, int ld, cudaStream_t stream_)
{
    dim3 threads( NB_X);
    int gridSize = (n + NB_X - 1) / NB_X;
    dim3 grid( gridSize );

    d_absTraceKernel<<< grid, threads, 0, stream_ >>>(d_matrix, d_trace, n, ld);
}

void absTrace_gpu(std::complex<float>* d_matrix, float* d_trace, int n, int ld, cudaStream_t stream_)
{
    dim3 threads( NB_X);
    int gridSize = (n + NB_X - 1) / NB_X;
    dim3 grid( gridSize );

    c_absTraceKernel<<< grid, threads, 0, stream_ >>>(reinterpret_cast<cuComplex*>(d_matrix), d_trace, n, ld);
}

void absTrace_gpu(std::complex<double>* d_matrix, double* d_trace, int n, int ld, cudaStream_t stream_)
{
    dim3 threads( NB_X);
    int gridSize = (n + NB_X - 1) / NB_X;
    dim3 grid( gridSize );

    z_absTraceKernel<<< grid, threads, 0, stream_ >>>(reinterpret_cast<cuDoubleComplex*>(d_matrix), d_trace, n, ld);
}

void find_max_and_add_abs_scalar_gpu(float *array, int size, float *scalar, float *max, cudaStream_t stream_)
{
    dim3 threads( NB_X);
    int gridSize = (size + NB_X - 1) / NB_X;
    dim3 grid( gridSize ); 
    s_find_max_and_add_abs_scalar<<< grid, threads, 0, stream_ >>>(array, size, scalar, max);
}

void find_max_and_add_abs_scalar_gpu(double *array, int size, double *scalar, double *max, cudaStream_t stream_)
{
    dim3 threads( NB_X);
    int gridSize = (size + NB_X - 1) / NB_X;
    dim3 grid( gridSize ); 
    d_find_max_and_add_abs_scalar<<< grid, threads, 0, stream_ >>>(array, size, scalar, max);
}

void find_max_and_add_abs_scalar_gpu(float *array, int size, std::complex<float> *scalar, float *max, cudaStream_t stream_)
{
    dim3 threads( NB_X);
    int gridSize = (size + NB_X - 1) / NB_X;
    dim3 grid( gridSize ); 
    c_find_max_and_add_abs_scalar<<< grid, threads, 0, stream_ >>>(array, size, reinterpret_cast<cuComplex*>(scalar), max);
}

void find_max_and_add_abs_scalar_gpu(double *array, int size, std::complex<double> *scalar, double *max, cudaStream_t stream_)
{
    dim3 threads( NB_X);
    int gridSize = (size + NB_X - 1) / NB_X;
    dim3 grid( gridSize ); 
    z_find_max_and_add_abs_scalar<<< grid, threads, 0, stream_ >>>(array, size, reinterpret_cast<cuDoubleComplex*>(scalar), max);
}

void residual_gpu(int m, int n, std::complex<double> *dA, int lda, std::complex<double> *dB,
                         int ldb, double *d_ritzv, double *d_resids, bool is_sqrt, cudaStream_t stream_)
{
    dim3 threads( NB_X);	
    dim3 grid( n );
    z_resids_kernel<<< grid, threads, 0, stream_ >>>( m, n, reinterpret_cast<cuDoubleComplex*>(dA), lda, reinterpret_cast<cuDoubleComplex*>(dB), ldb, d_ritzv, d_resids,is_sqrt);
}

void residual_gpu(int m, int n, std::complex<float> *dA, int lda, std::complex<float> *dB,
                         int ldb, float *d_ritzv, float *d_resids, bool is_sqrt, cudaStream_t stream_)
{
    dim3 threads( NB_X);
    dim3 grid( n );
    c_resids_kernel<<< grid, threads, 0, stream_ >>>( m, n, reinterpret_cast<cuComplex*>(dA), lda, reinterpret_cast<cuComplex*>(dB), ldb, d_ritzv, d_resids,is_sqrt);
}

void residual_gpu(int m, int n, double *dA, int lda, double *dB,
                         int ldb, double *d_ritzv, double *d_resids, bool is_sqrt, cudaStream_t stream_)
{
    dim3 threads( NB_X);
    dim3 grid( n );
    d_resids_kernel<<< grid, threads, 0, stream_ >>>( m, n, dA, lda, dB, ldb, d_ritzv, d_resids,is_sqrt);
}

void residual_gpu(int m, int n, float *dA, int lda, float *dB,
                         int ldb, float *d_ritzv, float *d_resids, bool is_sqrt, cudaStream_t stream_)
{
    dim3 threads( NB_X);
    dim3 grid( n );
    s_resids_kernel<<< grid, threads, 0, stream_ >>>( m, n, dA, lda, dB, ldb, d_ritzv, d_resids,is_sqrt);
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
    if ( uplo == 'U' ) 
    {
        for( unsigned int i=0; i < super_grid.x; ++i ) {
            mm = (i == super_grid.x-1 ? m % super_NB : super_NB);
            grid.x = ( mm + BLK_X - 1) / BLK_X;
            for( unsigned int j=0; j < super_grid.y; ++j ) {  // full row
                nn = (j == super_grid.y-1 ? n % super_NB : super_NB);
                grid.y = ( nn + BLK_X - 1) / BLK_Y;
                if ( i == j ) {  // diagonal super block
                    slacpy_upper_kernel <<< grid, threads, 0, stream_ >>>
                        ( mm, nn, dA(i*super_NB, j*super_NB), ldda, dB(i*super_NB, j*super_NB), lddb );
                }
                else // off diagonal super block
                {   
                    slacpy_full_kernel <<< grid, threads, 0, stream_ >>>
                        ( mm, nn, dA(i*super_NB, j*super_NB), ldda, dB(i*super_NB, j*super_NB), lddb );
                }
            }
        }    
    }
    else if(uplo == 'L')
    {
        for( unsigned int i=0; i < super_grid.x; ++i ) {
            mm = (i == super_grid.x-1 ? m % super_NB : super_NB);
            grid.x = ( mm + BLK_X - 1) / BLK_X;
            for( unsigned int j=0; j < super_grid.y; ++j ) {  // full row
                nn = (j == super_grid.y-1 ? n % super_NB : super_NB);
                grid.y = ( nn + BLK_X - 1) / BLK_Y;
                if ( i == j ) {  // diagonal super block
                    slacpy_lower_kernel <<< grid, threads, 0, stream_ >>>
                        ( mm, nn, dA(i*super_NB, j*super_NB), ldda, dB(i*super_NB, j*super_NB), lddb );
                }
                else // off diagonal super block
                {   
                    slacpy_full_kernel <<< grid, threads, 0, stream_ >>>
                        ( mm, nn, dA(i*super_NB, j*super_NB), ldda, dB(i*super_NB, j*super_NB), lddb );
                }
            }
        }
    }
    else
    {
        for( unsigned int i=0; i < super_grid.x; ++i ) {
            mm = (i == super_grid.x-1 ? m % super_NB : super_NB);
            grid.x = ( mm + BLK_X - 1) / BLK_X;
            for( unsigned int j=0; j < super_grid.y; ++j ) {  // full row
                nn = (j == super_grid.y-1 ? n % super_NB : super_NB);
                grid.y = ( nn + BLK_X - 1) / BLK_Y;
                slacpy_full_kernel <<< grid, threads, 0, stream_ >>>
                    ( mm, nn, dA(i*super_NB, j*super_NB), ldda, dB(i*super_NB, j*super_NB), lddb );
            }
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
    if ( uplo == 'U' ) 
    {
        for( unsigned int i=0; i < super_grid.x; ++i ) {
            mm = (i == super_grid.x-1 ? m % super_NB : super_NB);
            grid.x = ( mm + BLK_X - 1) / BLK_X;
            for( unsigned int j=0; j < super_grid.y; ++j ) {  // full row
                nn = (j == super_grid.y-1 ? n % super_NB : super_NB);
                grid.y = ( nn + BLK_X - 1) / BLK_Y;
                if ( i == j ) {  // diagonal super block
                    dlacpy_upper_kernel <<< grid, threads, 0, stream_ >>>
                        ( mm, nn, dA(i*super_NB, j*super_NB), ldda, dB(i*super_NB, j*super_NB), lddb );
                }
                else // off diagonal super block
                {   
                    dlacpy_full_kernel <<< grid, threads, 0, stream_ >>>
                        ( mm, nn, dA(i*super_NB, j*super_NB), ldda, dB(i*super_NB, j*super_NB), lddb );
                }
            }
        }    
    }
    else if(uplo == 'L')
    {
        for( unsigned int i=0; i < super_grid.x; ++i ) {
            mm = (i == super_grid.x-1 ? m % super_NB : super_NB);
            grid.x = ( mm + BLK_X - 1) / BLK_X;
            for( unsigned int j=0; j < super_grid.y; ++j ) {  // full row
                nn = (j == super_grid.y-1 ? n % super_NB : super_NB);
                grid.y = ( nn + BLK_X - 1) / BLK_Y;
                if ( i == j ) {  // diagonal super block
                    dlacpy_lower_kernel <<< grid, threads, 0, stream_ >>>
                        ( mm, nn, dA(i*super_NB, j*super_NB), ldda, dB(i*super_NB, j*super_NB), lddb );
                }
                else // off diagonal super block
                {   
                    dlacpy_full_kernel <<< grid, threads, 0, stream_ >>>
                        ( mm, nn, dA(i*super_NB, j*super_NB), ldda, dB(i*super_NB, j*super_NB), lddb );
                }
            }
        }
    }
    else
    {
        for( unsigned int i=0; i < super_grid.x; ++i ) {
            mm = (i == super_grid.x-1 ? m % super_NB : super_NB);
            grid.x = ( mm + BLK_X - 1) / BLK_X;
            for( unsigned int j=0; j < super_grid.y; ++j ) {  // full row
                nn = (j == super_grid.y-1 ? n % super_NB : super_NB);
                grid.y = ( nn + BLK_X - 1) / BLK_Y;
                dlacpy_full_kernel <<< grid, threads, 0, stream_ >>>
                    ( mm, nn, dA(i*super_NB, j*super_NB), ldda, dB(i*super_NB, j*super_NB), lddb );
            }
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
    if ( uplo == 'U' ) 
    {
        for( unsigned int i=0; i < super_grid.x; ++i ) {
            mm = (i == super_grid.x-1 ? m % super_NB : super_NB);
            grid.x = ( mm + BLK_X - 1) / BLK_X;
            for( unsigned int j=0; j < super_grid.y; ++j ) {  // full row
                nn = (j == super_grid.y-1 ? n % super_NB : super_NB);
                grid.y = ( nn + BLK_X - 1) / BLK_Y;
                if ( i == j ) {  // diagonal super block
                    zlacpy_upper_kernel <<< grid, threads, 0, stream_ >>>
                        ( mm, nn, dA(i*super_NB, j*super_NB), ldda, dB(i*super_NB, j*super_NB), lddb );
                }
                else // off diagonal super block
                {   
                    zlacpy_full_kernel <<< grid, threads, 0, stream_ >>>
                        ( mm, nn, dA(i*super_NB, j*super_NB), ldda, dB(i*super_NB, j*super_NB), lddb );
                }
            }
        }    
    }
    else if(uplo == 'L')
    {
        for( unsigned int i=0; i < super_grid.x; ++i ) {
            mm = (i == super_grid.x-1 ? m % super_NB : super_NB);
            grid.x = ( mm + BLK_X - 1) / BLK_X;
            for( unsigned int j=0; j < super_grid.y; ++j ) {  // full row
                nn = (j == super_grid.y-1 ? n % super_NB : super_NB);
                grid.y = ( nn + BLK_X - 1) / BLK_Y;
                if ( i == j ) {  // diagonal super block
                    zlacpy_lower_kernel <<< grid, threads, 0, stream_ >>>
                        ( mm, nn, dA(i*super_NB, j*super_NB), ldda, dB(i*super_NB, j*super_NB), lddb );
                }
                else // off diagonal super block
                {   
                    zlacpy_full_kernel <<< grid, threads, 0, stream_ >>>
                        ( mm, nn, dA(i*super_NB, j*super_NB), ldda, dB(i*super_NB, j*super_NB), lddb );
                }
            }
        }
    }
    else
    {
        for( unsigned int i=0; i < super_grid.x; ++i ) {
            mm = (i == super_grid.x-1 ? m % super_NB : super_NB);
            grid.x = ( mm + BLK_X - 1) / BLK_X;
            for( unsigned int j=0; j < super_grid.y; ++j ) {  // full row
                nn = (j == super_grid.y-1 ? n % super_NB : super_NB);
                grid.y = ( nn + BLK_X - 1) / BLK_Y;
                zlacpy_full_kernel <<< grid, threads, 0, stream_ >>>
                    ( mm, nn, dA(i*super_NB, j*super_NB), ldda, dB(i*super_NB, j*super_NB), lddb );
            }
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
    if ( uplo == 'U' ) 
    {
        for( unsigned int i=0; i < super_grid.x; ++i ) {
            mm = (i == super_grid.x-1 ? m % super_NB : super_NB);
            grid.x = ( mm + BLK_X - 1) / BLK_X;
            for( unsigned int j=0; j < super_grid.y; ++j ) {  // full row
                nn = (j == super_grid.y-1 ? n % super_NB : super_NB);
                grid.y = ( nn + BLK_X - 1) / BLK_Y;
                if ( i == j ) {  // diagonal super block
                    clacpy_upper_kernel <<< grid, threads, 0, stream_ >>>
                        ( mm, nn, dA(i*super_NB, j*super_NB), ldda, dB(i*super_NB, j*super_NB), lddb );
                }
                else // off diagonal super block
                {   
                    clacpy_full_kernel <<< grid, threads, 0, stream_ >>>
                        ( mm, nn, dA(i*super_NB, j*super_NB), ldda, dB(i*super_NB, j*super_NB), lddb );
                }
            }
        }    
    }
    else if(uplo == 'L')
    {
        for( unsigned int i=0; i < super_grid.x; ++i ) {
            mm = (i == super_grid.x-1 ? m % super_NB : super_NB);
            grid.x = ( mm + BLK_X - 1) / BLK_X;
            for( unsigned int j=0; j < super_grid.y; ++j ) {  // full row
                nn = (j == super_grid.y-1 ? n % super_NB : super_NB);
                grid.y = ( nn + BLK_X - 1) / BLK_Y;
                if ( i == j ) {  // diagonal super block
                    clacpy_lower_kernel <<< grid, threads, 0, stream_ >>>
                        ( mm, nn, dA(i*super_NB, j*super_NB), ldda, dB(i*super_NB, j*super_NB), lddb );
                }
                else // off diagonal super block
                {   
                    clacpy_full_kernel <<< grid, threads, 0, stream_ >>>
                        ( mm, nn, dA(i*super_NB, j*super_NB), ldda, dB(i*super_NB, j*super_NB), lddb );
                }
            }
        }
    }
    else
    {
        for( unsigned int i=0; i < super_grid.x; ++i ) {
            mm = (i == super_grid.x-1 ? m % super_NB : super_NB);
            grid.x = ( mm + BLK_X - 1) / BLK_X;
            for( unsigned int j=0; j < super_grid.y; ++j ) {  // full row
                nn = (j == super_grid.y-1 ? n % super_NB : super_NB);
                grid.y = ( nn + BLK_X - 1) / BLK_Y;
                clacpy_full_kernel <<< grid, threads, 0, stream_ >>>
                    ( mm, nn, dA(i*super_NB, j*super_NB), ldda, dB(i*super_NB, j*super_NB), lddb );
            }
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
    unsigned int grid = (offsize + BLOCKDIM - 1) / BLOCKDIM;
    if(grid == 0)
    {
        grid = 1;
    }
    dim3 threadsPerBlock(BLOCKDIM, 1);
    dim3 numBlocks(grid, 1);
    sshift_mgpu_matrix<<<numBlocks, threadsPerBlock, 0, stream_>>>( //
        A, off_m, off_n, offsize, ldH, shift);
}

void chase_shift_mgpu_matrix(double* A, std::size_t* off_m, std::size_t* off_n,
                             std::size_t offsize, std::size_t ldH, double shift,
                             cudaStream_t stream_)
{
    unsigned int grid = (offsize + BLOCKDIM - 1) / BLOCKDIM;
    if(grid == 0)
    {
        grid = 1;
    }
    dim3 threadsPerBlock(BLOCKDIM, 1);
    dim3 numBlocks(grid, 1);
    dshift_mgpu_matrix<<<numBlocks, threadsPerBlock, 0, stream_>>>( //
        A, off_m, off_n, offsize, ldH, shift);
}

void chase_shift_mgpu_matrix(std::complex<float>* A, std::size_t* off_m,
                             std::size_t* off_n, std::size_t offsize,
                             std::size_t ldH, float shift, cudaStream_t stream_)
{
    unsigned int grid = (offsize + BLOCKDIM - 1) / BLOCKDIM;
    if(grid == 0)
    {
        grid = 1;
    }
    dim3 threadsPerBlock(BLOCKDIM, 1);
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
    unsigned int grid = (offsize + BLOCKDIM - 1) / BLOCKDIM;
    if(grid == 0)
    {
        grid = 1;
    }
    dim3 threadsPerBlock(BLOCKDIM, 1);
    dim3 numBlocks(grid, 1);
    zshift_mgpu_matrix<<<numBlocks, threadsPerBlock, 0, stream_>>>( //
        reinterpret_cast<cuDoubleComplex*>(A), off_m, off_n,        //
        offsize, ldH, shift);
}

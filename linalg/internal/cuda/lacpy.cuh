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
static __device__ void dlacpy_full_device(
    std::size_t m, std::size_t n,
    const double *dA, std::size_t ldda,
    double       *dB, std::size_t lddb );

__global__ void dlacpy_full_kernel(
    std::size_t m, std::size_t n,
    const double *dA, std::size_t ldda,
    double       *dB, std::size_t lddb );


static __device__ void slacpy_full_device(
    std::size_t m, std::size_t n,
    const float *dA, std::size_t ldda,
    float       *dB, std::size_t lddb );

__global__ void slacpy_full_kernel(
    std::size_t m, std::size_t n,
    const float *dA, std::size_t ldda,
    float       *dB, std::size_t lddb );


static __device__ void zlacpy_full_device(
    std::size_t m, std::size_t n,
    const cuDoubleComplex *dA, std::size_t ldda,
    cuDoubleComplex       *dB, std::size_t lddb );

__global__
void zlacpy_full_kernel(
    std::size_t m, std::size_t n,
    const cuDoubleComplex *dA, std::size_t ldda,
    cuDoubleComplex       *dB, std::size_t lddb );

static __device__ void clacpy_full_device(
    std::size_t m, std::size_t n,
    const cuComplex *dA, std::size_t ldda,
    cuComplex       *dB, std::size_t lddb );

__global__ void clacpy_full_kernel(
    std::size_t m, std::size_t n,
    const cuComplex *dA, std::size_t ldda,
    cuComplex       *dB, std::size_t lddb );

static __device__ void dlacpy_upper_device(
    std::size_t m, std::size_t n,
    const double *dA, std::size_t ldda,
    double       *dB, std::size_t lddb );

__global__ void dlacpy_upper_kernel(
    std::size_t m, std::size_t n,
    const double *dA, std::size_t ldda,
    double       *dB, std::size_t lddb );

static __device__ void slacpy_upper_device(
    std::size_t m, std::size_t n,
    const float *dA, std::size_t ldda,
    float       *dB, std::size_t lddb );

__global__ void slacpy_upper_kernel(
    std::size_t m, std::size_t n,
    const float *dA, std::size_t ldda,
    float       *dB, std::size_t lddb );

static __device__ void clacpy_upper_device(
    std::size_t m, std::size_t n,
    const cuComplex *dA, std::size_t ldda,
    cuComplex       *dB, std::size_t lddb );

__global__ void clacpy_upper_kernel(
    std::size_t m, std::size_t n,
    const cuComplex *dA, std::size_t ldda,
    cuComplex       *dB, std::size_t lddb );

static __device__ void zlacpy_upper_device(
    std::size_t m, std::size_t n,
    const cuDoubleComplex *dA, std::size_t ldda,
    cuDoubleComplex       *dB, std::size_t lddb );

__global__ void zlacpy_upper_kernel(
    std::size_t m, std::size_t n,
    const cuDoubleComplex *dA, std::size_t ldda,
    cuDoubleComplex       *dB, std::size_t lddb );


static __device__ void dlacpy_lower_device(
    std::size_t m, std::size_t n,
    const double *dA, std::size_t ldda,
    double       *dB, std::size_t lddb );

__global__ void dlacpy_lower_kernel(
    std::size_t m, std::size_t n,
    const double *dA, std::size_t ldda,
    double       *dB, std::size_t lddb );

static __device__ void slacpy_lower_device(
    std::size_t m, std::size_t n,
    const float *dA, std::size_t ldda,
    float       *dB, std::size_t lddb );

__global__ void slacpy_lower_kernel(
    std::size_t m, std::size_t n,
    const float *dA, std::size_t ldda,
    float       *dB, std::size_t lddb );

static __device__ void clacpy_lower_device(
    std::size_t m, std::size_t n,
    const cuComplex *dA, std::size_t ldda,
    cuComplex       *dB, std::size_t lddb );

__global__ void clacpy_lower_kernel(
    std::size_t m, std::size_t n,
    const cuComplex *dA, std::size_t ldda,
    cuComplex       *dB, std::size_t lddb );

static __device__ void zlacpy_lower_device(
    std::size_t m, std::size_t n,
    const cuDoubleComplex *dA, std::size_t ldda,
    cuDoubleComplex       *dB, std::size_t lddb );

__global__ void zlacpy_lower_kernel(
    std::size_t m, std::size_t n,
    const cuDoubleComplex *dA, std::size_t ldda,
    cuDoubleComplex       *dB, std::size_t lddb );

void t_lacpy_gpu(char uplo, std::size_t m, std::size_t n, float *dA, std::size_t ldda, float *dB, std::size_t lddb, cudaStream_t stream_ );
void t_lacpy_gpu(char uplo, std::size_t m, std::size_t n, double *dA, std::size_t ldda, double *dB, std::size_t lddb, cudaStream_t stream_);
void t_lacpy_gpu(char uplo, std::size_t m, std::size_t n, std::complex<double> *ddA, std::size_t ldda, std::complex<double> *ddB, std::size_t lddb, cudaStream_t stream_ );
void t_lacpy_gpu(char uplo, std::size_t m, std::size_t n, std::complex<float> *ddA, std::size_t ldda, std::complex<float> *ddB, std::size_t lddb, cudaStream_t stream_ );

}
}
}
}
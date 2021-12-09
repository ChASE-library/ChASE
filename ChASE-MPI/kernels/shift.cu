/* -*- Mode: C++; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
// This file is a part of ChASE.
// Copyright (c) 2015-2021, Simulation and Data Laboratory Quantum Materials, 
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include <cuComplex.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <complex>
#include "cublas_v2.h"

#define BLOCKDIM 256

__global__ void sshift_matrix(float* A, int n, float shift) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) A[(idx)*n + idx] += shift;
}

__global__ void dshift_matrix(double* A, int n, double shift) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) A[(idx)*n + idx] += shift;
}

__global__ void cshift_matrix(cuComplex* A, int n, float shift) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) A[(idx)*n + idx].x += shift;
}

__global__ void zshift_matrix(cuDoubleComplex* A, int n, double shift) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) A[(idx)*n + idx].x += shift;
}

__global__ void sshift_mgpu_matrix(float* A, std::size_t* off_m, std::size_t* off_n,
                                  std::size_t offsize, std::size_t ldH, float shift) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t ind;
  if(i < offsize){
        ind = off_n[i] * ldH + off_m[i];
        A[ind] += shift;
  }
}

__global__ void dshift_mgpu_matrix(double* A, std::size_t* off_m, std::size_t* off_n,
                                  std::size_t offsize, std::size_t ldH, double shift) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t ind;
  if(i < offsize){
        ind = off_n[i] * ldH + off_m[i];
        A[ind] += shift;
  }
}

__global__ void cshift_mgpu_matrix(cuComplex* A, std::size_t* off_m, std::size_t* off_n,
                                  std::size_t offsize, std::size_t ldH, float shift) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t ind;
  if(i < offsize){
        ind = off_n[i] * ldH + off_m[i];
        A[ind].x += shift;
  }
}

__global__ void zshift_mgpu_matrix(cuDoubleComplex* A, std::size_t* off_m, std::size_t* off_n,
                                  std::size_t offsize, std::size_t ldH, double shift) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t ind;
  if(i < offsize){
  	ind = off_n[i] * ldH + off_m[i];
	A[ind].x += shift;
  }
}

void chase_shift_matrix(float* A, int n, float shift,
                         cudaStream_t* stream_) {
  int num_blocks = (n + (BLOCKDIM - 1)) / BLOCKDIM;
  sshift_matrix<<<num_blocks, BLOCKDIM, 0, *stream_>>>(
      A, n, shift);
}

void chase_shift_matrix(double* A, int n, double shift,
                         cudaStream_t* stream_) {
  int num_blocks = (n + (BLOCKDIM - 1)) / BLOCKDIM;
  dshift_matrix<<<num_blocks, BLOCKDIM, 0, *stream_>>>(
      A, n, shift);
}

void chase_shift_matrix(std::complex<float>* A, int n, float shift,
                         cudaStream_t* stream_) {
  int num_blocks = (n + (BLOCKDIM - 1)) / BLOCKDIM;
  cshift_matrix<<<num_blocks, BLOCKDIM, 0, *stream_>>>(
      reinterpret_cast<cuComplex*>(A), n, shift);
}

void chase_shift_matrix(std::complex<double>* A, int n, double shift,
                         cudaStream_t* stream_) {
  int num_blocks = (n + (BLOCKDIM - 1)) / BLOCKDIM;
  zshift_matrix<<<num_blocks, BLOCKDIM, 0, *stream_>>>(
      reinterpret_cast<cuDoubleComplex*>(A), n, shift);
}

void chase_shift_mgpu_matrix(float* A, std::size_t* off_m, std::size_t* off_n,
                            std::size_t offsize, std::size_t ldH, float shift,
                             cudaStream_t stream_) {

  unsigned int grid = (offsize + 256 - 1) / 256;
  dim3 threadsPerBlock(256, 1);
  dim3 numBlocks(grid, 1);
  sshift_mgpu_matrix<<<numBlocks, threadsPerBlock, 0, stream_>>>(  //
      A, off_m, off_n, offsize, ldH, shift);

}


void chase_shift_mgpu_matrix(double* A, std::size_t* off_m, std::size_t* off_n,
                            std::size_t offsize, std::size_t ldH, double shift,
                             cudaStream_t stream_) {

  unsigned int grid = (offsize + 256 - 1) / 256;
  dim3 threadsPerBlock(256, 1);
  dim3 numBlocks(grid, 1);
  dshift_mgpu_matrix<<<numBlocks, threadsPerBlock, 0, stream_>>>(  //
      A, off_m, off_n, offsize, ldH, shift);

}

void chase_shift_mgpu_matrix(std::complex<float>* A, std::size_t* off_m, std::size_t* off_n,
                            std::size_t offsize, std::size_t ldH, float shift,
                             cudaStream_t stream_) {

  unsigned int grid = (offsize + 256 - 1) / 256;
  dim3 threadsPerBlock(256, 1);
  dim3 numBlocks(grid, 1);
  cshift_mgpu_matrix<<<numBlocks, threadsPerBlock, 0, stream_>>>(  //
      reinterpret_cast<cuComplex*>(A), off_m, off_n, //
                                offsize, ldH, shift);

}


void chase_shift_mgpu_matrix(std::complex<double>* A, std::size_t* off_m, std::size_t* off_n,
                            std::size_t offsize, std::size_t ldH, double shift,
                             cudaStream_t stream_) {
  
  unsigned int grid = (offsize + 256 - 1) / 256;
  dim3 threadsPerBlock(256, 1);
  dim3 numBlocks(grid, 1);
  zshift_mgpu_matrix<<<numBlocks, threadsPerBlock, 0, stream_>>>(  //
      reinterpret_cast<cuDoubleComplex*>(A), off_m, off_n, //
      				offsize, ldH, shift);

}


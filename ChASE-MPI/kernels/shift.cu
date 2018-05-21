#include <cuComplex.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <complex>
#include "cublas_v2.h"

#define BLOCKDIM 256

__global__ void zshift_matrix(cuDoubleComplex* A, int n, double shift) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) A[(idx)*n + idx].x += shift;
}

__global__ void zshift_mpi_matrix(cuDoubleComplex* A, std::size_t* off,
                                  std::size_t n, std::size_t m, double shift) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (off[0] + j == (i + off[1]) && i < n && j < m) {
    A[i * m + j].x += shift;
  }
}

void chase_zshift_matrix(std::complex<double>* A, int n, double shift,
                         cudaStream_t* stream_) {
  int num_blocks = (n + (BLOCKDIM - 1)) / BLOCKDIM;
  zshift_matrix<<<num_blocks, BLOCKDIM, 0, *stream_>>>(
      reinterpret_cast<cuDoubleComplex*>(A), n, shift);
}

void chase_zshift_mpi_matrix(std::complex<double>* A, std::size_t* off,
                             std::size_t n, std::size_t m, double shift,
                             cudaStream_t* stream_) {
  // x  ^= i \in [0,n_]
  // y  ^= j \in [0,m_]
  //dim3 threadsPerBlock(16, 16);
  // dim3 numBlocks(n + (threadsPerBlock.x - 1) / threadsPerBlock.x,
  //                m + (threadsPerBlock.y - 1) / threadsPerBlock.y);

  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks(1, 1);
  zshift_mpi_matrix<<<numBlocks, threadsPerBlock, 0, *stream_>>>(  //
      reinterpret_cast<cuDoubleComplex*>(A), off, n, m, shift);
}

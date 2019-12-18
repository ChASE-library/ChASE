/**
 * The program tests the mgpu_cudaHemm class for computing
 *
 * 			W = op(H) * V + W
 *
 * on multiple GPUs, where A is square matrix and B and C are tall and skinny.
 * Matrix A is divided into tiles and each tile is distributed to one GPU. 
 * The matrix A cannot be larger than a total aggregated memory of all availble GPUs.
 * Matrices B and C are divided in a column-tiles
 */

#include <cstdio>
#include <cstdlib>
#include <complex>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cuda_profiler_api.h>

#define MKL_Complex16 std::complex<double>
#include "mkl.h"

#include "ChASE-MPI/impl/mgpu_cudaHemm.hpp"

using T = std::complex<double>;
using namespace chase;
using namespace chase::mpi;

typedef mgpu_cudaHemm<T> MGPU;

void print(T *A, int ldA, int m, int n);

int main (int argc, char *argv[]) {

	/// Matrix dimension
	int m;
	int n; 
	int blockDim;
 	
	/// Matrices
	T *H = nullptr;
	T *V = nullptr;
	T *W = nullptr;
	T *GPU_OUT = nullptr;

	// Random generator seeds
	int iseed1[] = {1,11,7,1};
	int iseed2[] = {3,7,13,13};
	int iseed3[] = {3,11,13,1};

	/// Auxiliary variables
	int num_elem;
	int two = 2;

	/// MGPU object
	MGPU *M = nullptr;

	/// Hemm parameters
	//T alpha(1.5, 1.0);
	//T beta(0.0, 0.0);
	double alpha = 1.5;
	double beta = 0.0;	
	char transa = 'N';	

	/// Read matrix size
	m = atoi(argv[1]);
    n = atoi(argv[2]);
	blockDim = atoi(argv[3]);

	// Allocate arrays for A, B, and C on the host
	cudaMallocHost((void**)&H, m*n*sizeof(T));
	cudaMallocHost((void**)&V, std::max(m,n)*blockDim*sizeof(T));
	cudaMallocHost((void**)&W, std::max(m,n)*blockDim*sizeof(T));
	cudaMallocHost((void**)&GPU_OUT, std::max(m,n)*blockDim*sizeof(T));

	// Fill matrices with random values
	num_elem = std::max(m,n) * blockDim;
	zlarnv_(&two, iseed1, &num_elem, V);
	zlarnv(&two, iseed2, &num_elem, W);
	num_elem = m * n;
	zlarnv(&two, iseed3, &num_elem, H);

	std::cout << std::endl << "====== CPU PART ====== " << std::endl;
 	/// Compute CPU version
	char side = CblasLeft;
	char uplo = CblasUpper;
	cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, blockDim, n, &alpha, H, m, V, std::max(m,n), &beta, W, std::max(m,n));

	std::cout << "CPU output: " << std::endl;
	print(W, blockDim, std::max(m,n), blockDim);

	std::cout << std::endl << "====== GPU PART ====== " << std::endl;
	// Construct a new MGPU object
	M = new MGPU(m, n, blockDim);

	// Copy H to GPUs
    M->distribute_H(H);
	
	// Copy V and W to GPUs
	M->distribute_V(V, n, blockDim);	

	// Run on GPUs
	M->computeHemm(blockDim, alpha, beta);
	
	// Collect results from GPUs
	M->return_V(GPU_OUT, m, blockDim);

	std::cout << std::endl << "GPU output: " << std::endl;
	print(GPU_OUT, blockDim, std::max(m,n), blockDim);
	
	// Compare CPU and GPUs results
	alpha = -1.0;
	cblas_zaxpy(std::max(m,n)*blockDim, &alpha, GPU_OUT, 1, W, 1);

	char norm = 'F';
	int rows = std::max(m,n);
	double *tmp = nullptr;
	double error = zlange(&norm, &rows, &blockDim, W, &rows, tmp);

	// Print output
	std::cout << "Absolute error = " << std::scientific << error << std::endl;

	delete M;	
	cudaFreeHost(H);
	cudaFreeHost(V);
	cudaFreeHost(W);
	cudaFreeHost(GPU_OUT);
 
	return EXIT_SUCCESS;
}

void print(T *A, int ldA, int m, int n) {

	for (int i=0; i<m; i++) {
		for (int j=0; j<n; j++) {
			std::cout << real(A[i*ldA + j]) << " ";
		}
		std::cout << std::endl;
	}
	
/*	for (int i=0; i<m*n; i++) {
		std::cout << real(A[i]) << " ";
	}
	std::cout << std::endl;
*/
}

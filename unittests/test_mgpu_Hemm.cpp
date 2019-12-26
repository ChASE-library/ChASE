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
#include <iomanip>
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
	std::complex<double> alpha(1.5, 0.0);
	std::complex<double> beta(0.0, 0.0);	
	char transa = 'N';
	char transb = 'N';

	/// Read matrix size
	m = atoi(argv[1]);
	n = atoi(argv[2]);
	blockDim = atoi(argv[3]);

	int ldH = m;
	int ldV = n;
	int ldW = m;

	// Allocate arrays for A, B, and C on the host
	cudaMallocHost((void**)&H, ldH*n*sizeof(T));
	cudaMallocHost((void**)&V, ldV*blockDim*sizeof(T));
	cudaMallocHost((void**)&W, ldW*blockDim*sizeof(T));
	cudaMallocHost((void**)&GPU_OUT, ldW*blockDim*sizeof(T));

	// Fill matrices with random values
	num_elem = ldV * blockDim;
	zlarnv(&two, iseed1, &num_elem, V);
	num_elem = ldW * blockDim;
	zlarnv(&two, iseed2, &num_elem, W);
	num_elem = ldH * n;
	zlarnv(&two, iseed3, &num_elem, H);

#if 0
    std::cout << "H = " << std::endl;
	print(H, ldH, m, n);

	std::cout << "V = " << std::endl;
	print(V, ldV, m, blockDim); 
#endif
	std::cout << std::endl << "====== CPU PART ====== " << std::endl;
 	/// Compute CPU version

	//cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, blockDim, n, &alpha, H, ldH, V, ldV, &beta, W, ldW);
	zgemm(&transa, &transb, &m, &blockDim, &n, &alpha, H, &ldH, V, &ldV, &beta, W, &ldW);

	std::cout << std::endl << "====== GPU PART ====== " << std::endl;
	// Construct a new MGPU object
	M = new MGPU(m, n, blockDim);

	// Copy H to GPUs
	M->distribute_H(H, ldH);
	
	// Copy V and W to GPUs
	M->distribute_V(V, ldV, blockDim);	

	// Run on GPUs
	M->computeHemm(blockDim, alpha, beta);
	
	// Collect results from GPUs
	M->return_V(GPU_OUT, ldW, blockDim);

	//M->synchronizeAll();

#if 0
	std::cout << "CPU output: " << std::endl;
	print(W, ldW, m, blockDim);

	std::cout << std::endl << "GPU output: " << std::endl;
	print(GPU_OUT, ldW, m, blockDim);
#endif	

#if 0
	double err = 10e-12;
	std::cout << "Error on positions: " << std::endl;
	for(int i=0; i<m; i++){
		for(int j=0; j<blockDim; j++){
			if(real(GPU_OUT[j*ldW+i]) - real(W[j*ldW+i]) > err && imag(GPU_OUT[j*ldW+i]) - imag(W[j*ldW+i]) > err) std::cout<<"("<<i<< ","<<j<<"), ";
		}
	}
	std::cout << std::endl;
#endif
	// Compare CPU and GPUs results
	std::complex<double> zalpha(-1.0, 0.0);
	cblas_zaxpy(m*blockDim, &zalpha, GPU_OUT, 1, W, 1);

#if 0
	std::cout << "Difference after zaxpy:" << std::endl;
	for(int i=0; i<m*blockDim; i++){
		if (real(W[i]) > err || imag(W[i]) > err) std::cout << i << ", ";
	}
	std::cout << std::endl;
#endif
	char norm = 'M';
	int rows = m;
	double *tmp = nullptr;
	double error = zlange(&norm, &rows, &blockDim, W, &ldW, tmp);

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
			//std::cout << std::fixed << std::setprecision(6) << std::setw(10) << real(A[i*ldA + j]) << " ";
			std::cout << std::fixed << std::setprecision(6) << std::setw(25) << A[j*ldA + i] << " ";
		}
		std::cout << std::endl;
	}
	
/*	for (int i=0; i<m*n; i++) {
		std::cout << real(A[i]) << " ";
	}
	std::cout << std::endl;
*/
}

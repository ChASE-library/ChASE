// This file is a part of ChASE.
// Copyright (c) 2015-2024, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include "external/cublaspp/cublaspp.hpp"
#include "external/cusolverpp/cusolverpp.hpp"
#include "Impl/chase_gpu/cuda_utils.hpp"
#include "linalg/matrix/matrix.hpp"
#include "linalg/internal/cuda/flipSign.hpp"
#include "linalg/internal/cuda/shiftDiagonal.cuh"
#include "Impl/chase_gpu/nvtx.hpp"

using namespace chase::linalg;

namespace chase
{
namespace linalg
{
namespace internal
{
namespace cuda
{
    /**
    * @brief Perform the Rayleigh-Ritz procedure to compute eigenvalues and eigenvectors of a matrix.
    *
    * The Rayleigh-Ritz method computes an approximation to the eigenvalues and eigenvectors of a matrix
    * by projecting the matrix onto a subspace defined by a set of vectors (Q) and solving the eigenvalue
    * problem for the reduced matrix. The computed Ritz values are stored in the `ritzv` array, and the 
    * resulting eigenvectors are stored in `W`.
    *
    * @tparam T Data type for the matrix (e.g., float, double, etc.).
    * @param[in] N The number of rows of the matrix H.
    * @param[in] H The input matrix (N x N).
    * @param[in] ldh The leading dimension of the matrix H.
    * @param[in] n The number of vectors in Q (subspace size).
    * @param[in] Q The input matrix of size (N x n), whose columns are the basis vectors for the subspace.
    * @param[in] ldq The leading dimension of the matrix Q.
    * @param[out] W The output matrix (N x n), which will store the result of the projection.
    * @param[in] ldw The leading dimension of the matrix W.
    * @param[out] ritzv The array of Ritz values, which contains the eigenvalue approximations.
    * @param[in] A A temporary matrix used in intermediate calculations. If not provided, it is allocated internally.
    *
    * The procedure performs the following steps:
    * 1. Computes the matrix-vector multiplication: W = H * Q.
    * 2. Computes A = W' * Q, where W' is the conjugate transpose of W.
    * 3. Solves the eigenvalue problem for A using LAPACK's `heevd` function, computing the Ritz values in `ritzv`.
    * 4. Computes the final approximation to the eigenvectors by multiplying Q with the computed eigenvectors.
    */      
    template<typename T>
    void rayleighRitz(cublasHandle_t cublas_handle, 
                      cusolverDnHandle_t cusolver_handle,
                      chase::matrix::Matrix<T, chase::platform::GPU> * H,
                      chase::matrix::Matrix<T, chase::platform::GPU>& V1,
                      chase::matrix::Matrix<T, chase::platform::GPU>& V2,
                      chase::matrix::Matrix<chase::Base<T>, chase::platform::GPU>& ritzv,
                      std::size_t offset,
                      std::size_t subSize,
                      int* devInfo,
                      T *workspace = nullptr,
                      int lwork_heevd = 0,
                      chase::matrix::Matrix<T, chase::platform::GPU> *A = nullptr)
    {
        SCOPED_NVTX_RANGE();

        if(A == nullptr)
        {
            A = new chase::matrix::Matrix<T, chase::platform::GPU>(subSize, subSize);
        }

        if(workspace == nullptr)
        {
            lwork_heevd = 0;
            CHECK_CUSOLVER_ERROR(chase::linalg::cusolverpp::cusolverDnTheevd_bufferSize(
                                    cusolver_handle, 
                                    CUSOLVER_EIG_MODE_VECTOR, 
                                    CUBLAS_FILL_MODE_LOWER, 
                                    subSize, 
                                    A->data(), 
                                    subSize, 
                                    ritzv.data(), 
                                    &lwork_heevd));
            CHECK_CUDA_ERROR(cudaMalloc((void**)&workspace, sizeof(T) * lwork_heevd));
        }

        T One = T(1.0);
        T Zero = T(0.0);

        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(cublas_handle, 
                                       CUBLAS_OP_C, 
                                       CUBLAS_OP_N, 
                                       H->rows(), 
                                       subSize, 
                                       H->cols(), 
                                       &One, 
                                       H->data(),
                                       H->ld(),
                                       V1.data() + offset * V1.ld(),
                                       V1.ld(),
                                       &Zero,
                                       V2.data() + offset * V2.ld(),
                                       V2.ld()));

        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(cublas_handle, 
                                       CUBLAS_OP_C, 
                                       CUBLAS_OP_N, 
                                       subSize, 
                                       subSize, 
                                       V2.rows(),
                                       &One, 
                                       V2.data() + offset * V2.ld(),
                                       V2.ld(), 
                                       V1.data() + offset * V1.ld(),
                                       V1.ld(),
                                       &Zero, 
                                       A->data(),
                                       subSize));
        
        CHECK_CUSOLVER_ERROR(chase::linalg::cusolverpp::cusolverDnTheevd(
                                       cusolver_handle, 
                                       CUSOLVER_EIG_MODE_VECTOR, 
                                       CUBLAS_FILL_MODE_LOWER, 
                                       subSize,
                                       A->data(),
                                       subSize,
                                       ritzv.data() + offset,
                                       workspace, lwork_heevd, devInfo           
        ));

        int info;
        CHECK_CUDA_ERROR(cudaMemcpy(&info, 
                                    devInfo, 
                                    1 * sizeof(int),
                                    cudaMemcpyDeviceToHost));

        if(info != 0)
        {
            throw std::runtime_error("cusolver HEEVD failed in RayleighRitz");
        }        

        CHECK_CUDA_ERROR(cudaMemcpy(ritzv.cpu_data() + offset, 
                                    ritzv.data() + offset, 
                                    subSize * sizeof(chase::Base<T>),
                                    cudaMemcpyDeviceToHost));

        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(cublas_handle, 
                                       CUBLAS_OP_N, 
                                       CUBLAS_OP_N, 
                                       V1.rows(), 
                                       subSize, 
                                       subSize,
                                       &One, 
                                       V1.data() + offset * V1.ld(),
                                       V1.ld(), 
                                       A->data(),
                                       subSize, 
                                       &Zero,
                                       V2.data() + offset * V2.ld(),
                                       V2.ld()));


    }

    /**
    * @brief Perform the Rayleigh-Ritz procedure to compute eigenvalues and eigenvectors of a Quasi-Hermitian matrix.
    *
    * The Rayleigh-Ritz method computes an approximation to the eigenvalues and eigenvectors of a matrix
    * by projecting the matrix onto a subspace defined by a set of vectors (Q) and solving the eigenvalue
    * problem for the reduced matrix. The computed Ritz values are stored in the `ritzv` array, and the 
    * resulting eigenvectors are stored in `W`.
    *
    * @tparam T Data type for the matrix (e.g., float, double, etc.).
    * @param[in] N The number of rows of the matrix H.
    * @param[in] H The input matrix (N x N).
    * @param[in] ldh The leading dimension of the matrix H.
    * @param[in] n The number of vectors in Q (subspace size).
    * @param[in] Q The input matrix of size (N x n), whose columns are the basis vectors for the subspace.
    * @param[in] ldq The leading dimension of the matrix Q.
    * @param[out] W The output matrix (N x n), which will store the result of the projection.
    * @param[in] ldw The leading dimension of the matrix W.
    * @param[out] ritzv The array of Ritz values, which contains the eigenvalue approximations.
    * @param[in] A A temporary matrix used in intermediate calculations. If not provided, it is allocated internally.
    *
    * The procedure performs the following steps:
    * 1. Computes the matrix-vector multiplication: W = H * Q.
    * 2. Computes A = W' * Q, where W' is the conjugate transpose of W.
    * 3. Solves the eigenvalue problem for A using LAPACK's `heevd` function, computing the Ritz values in `ritzv`.
    * 4. Computes the final approximation to the eigenvectors by multiplying Q with the computed eigenvectors.
    */ 

    template<typename T>
    void rayleighRitz(cublasHandle_t cublas_handle, 
                      cusolverDnHandle_t cusolver_handle,
		              cusolverDnParams_t params,
                      chase::matrix::QuasiHermitianMatrix<T, chase::platform::GPU> * H,
                      chase::matrix::Matrix<T, chase::platform::GPU>& V1,
                      chase::matrix::Matrix<T, chase::platform::GPU>& V2,
                      chase::matrix::Matrix<chase::Base<T>, chase::platform::GPU>& ritzv,
                      std::size_t offset,
                      std::size_t subSize,
                      int* devInfo,
                      T *d_workspace = nullptr,
                      int d_lwork = 0,
                      T *h_workspace = nullptr,
                      int h_lwork = 0,
                      chase::matrix::Matrix<T, chase::platform::GPU> *A = nullptr,
                      chase::matrix::Matrix<T, chase::platform::GPU> *half_Q = nullptr)
    {
        SCOPED_NVTX_RANGE();

	std::size_t N = H->rows();
	std::size_t ldh = H->ld();
	std::size_t k = N/2;
	std::size_t n = subSize;

	cudaStream_t usedStream;
        CHECK_CUBLAS_ERROR(cublasGetStream(cublas_handle, &usedStream));

        if(A == nullptr)
        {
            A = new chase::matrix::Matrix<T, chase::platform::GPU>(n, n);
        }

        if(half_Q == nullptr)
        {
            half_Q = new chase::matrix::Matrix<T, chase::platform::GPU>(k, n);
        }

    T *halfQ = half_Q->data();

	//chase::matrix::Matrix<T, chase::platform::GPU> ritzv_complex;
    std::vector<T> ritzvs_cmplex_cpu(n);

    //use halfQ as a workspace for the complex ritz values
    //it should valid for complex valued types
    //as ritzv_complex is used only after the lifetime of halfQ is over.
    //for real valued types, halfQ row number should be at least 2.
    //Thus N should be at least 4.
    T *ritzv_complex = half_Q->data();

	//Allocating workspace memory
        if(d_workspace == nullptr || h_workspace == nullptr) //To update once Xgeev is plugged in
        {
	    std::size_t temp_d_lwork = 0;
	    std::size_t temp_h_lwork = 0;

	    CHECK_CUSOLVER_ERROR(chase::linalg::cusolverpp::cusolverDnTgeev_bufferSize(
                                                            cusolver_handle,
                                                            params,
                                                            CUSOLVER_EIG_MODE_NOVECTOR,
                                                            CUSOLVER_EIG_MODE_VECTOR,
                                                            n,
                                                            A->data(),
                                                            n,
                                                            ritzv_complex,
                                                            NULL,1,
                                                            V1.data() + offset * V1.ld(),V1.ld(),
                                                            &temp_d_lwork,
                                                            &temp_h_lwork));
	    if(d_workspace == nullptr){

		d_lwork = temp_d_lwork;

            	CHECK_CUDA_ERROR(cudaMalloc((void**)&d_workspace, sizeof(T) * d_lwork));
	    }

	    if(h_workspace == nullptr){
		
		h_lwork = temp_h_lwork;
		
		h_workspace = new T[h_lwork]();
	    }
        }

        T One   = T(1.0);
        T Zero  = T(0.0);
	T alpha = T(2.0);

	chase::Base<T> real_One = chase::Base<T>(1.0);

	chase::matrix::Matrix<chase::Base<T>, chase::platform::GPU> diag(n,1);

	//Performs Q_2^T Q_2 for the construction of the dual basis, Q_2 is the lower part of Q
	//std::cout << "Performs Q_2^T Q_2 for the construction of the dual basis, Q_2 is the lower part of Q" << std::endl;
	CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(cublas_handle, 
                         	       CUBLAS_OP_C, 
                                       CUBLAS_OP_N, 
                                       n, 
                                       n, 
                                       k, 
                                       &alpha, 
                                       V1.data() + offset * V1.ld() + k,
                                       V1.ld(),
                                       V1.data() + offset * V1.ld() + k,
                                       V1.ld(),
                                       &Zero,
                                       A->data(),
                                       n));

       	//Pre-compute the scaling weights such that diag = Ql^T Qr
	//std::cout << "Pre-compute the scaling weights such that diag = Ql^T Qr" << std::endl;
	chase::linalg::internal::cuda::chase_subtract_inverse_diagonal(A->data(),
				      n,
				      n,
				      real_One,
				      diag.data(), usedStream);

	//The Matrix A now contains the data for creating the upper part of Ql
	//std::cout << "The Matrix A now contains the data for creating the upper part of Ql" << std::endl;
	chase::linalg::internal::cuda::chase_set_diagonal(A->data(),
				       n,
				       n,
				       One, usedStream);

	//Compute the upper part of Ql
	//std::cout << "Compute the upper part of Ql" << std::endl;
	CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(cublas_handle, 
                         	       CUBLAS_OP_N, 
                                       CUBLAS_OP_N, 
                                       k, 
                                       n, 
                                       n, 
                                       &One, 
                                       V1.data() + offset * V1.ld(),
                                       V1.ld(),
                                       A->data(),
                                       n,
                                       &Zero,
                                       halfQ,
                                       k));

	//Performs the multiplication of the first k cols of H with the upper part of Ql
	//std::cout << "Performs the multiplication of the first k cols of H with the upper part of Ql" << std::endl;
        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(cublas_handle, 
                         	       CUBLAS_OP_N, 
                                       CUBLAS_OP_N, 
                                       N, 
                                       n, 
                                       k, 
                                       &One, 
                                       H->data(),
                                       ldh,
                                       halfQ,
                                       k,
                                       &Zero,
                                       V2.data() + offset * V2.ld(),
                                       V2.ld()));

	alpha = -One;
		
	//The Matrix A now contains the data for creating the upper part of Ql
	//std::cout << "The Matrix A now contains the data for creating the upper part of Ql" << std::endl;
	chase::linalg::internal::cuda::chase_set_diagonal(A->data(),
				       n,
				       n,
				       alpha, usedStream);
	
	//Compute the negative of the lower part of Ql
	//std::cout << "Compute the negative of the lower part of Ql" << std::endl;
	CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(cublas_handle, 
                         	       CUBLAS_OP_N, 
                                       CUBLAS_OP_N, 
                                       k, 
                                       n, 
                                       n, 
                                       &alpha, 
                                       V1.data() + offset * V1.ld() + k,
                                       V1.ld(),
                                       A->data(),
                                       n,
                                       &Zero,
                                       halfQ,
                                       k));

	//Performs the multiplication of the last k cols of H with the negative lower part of Ql
	//std::cout << "Performs the multiplication of the last k cols of H with the negative lower part of Ql" << std::endl;
        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(cublas_handle, 
                         	       CUBLAS_OP_N, 
                                       CUBLAS_OP_N, 
                                       N, 
                                       n, 
                                       k, 
                                       &One, 
                                       H->data() + k * ldh,
                                       ldh,
                                       halfQ,
                                       k,
                                       &One,
                                       V2.data() + offset * V2.ld(),
                                       V2.ld()));

	//Flip the sign of the lower part of V to emulate the multiplication H' * Ql
	//std::cout << "Flip the sign of the lower part of V to emulate the multiplication H' * Ql" << std::endl;
	chase::linalg::internal::cuda::flipLowerHalfMatrixSign(N,n,V2.data() + offset * V2.ld(),V2.ld(), &usedStream);
	
	//Last GEMM for the construction of the rayleigh Quotient : (H' * Ql)' * Qr
	//std::cout << "Last GEMM for the construction of the rayleigh Quotient : (H' * Ql)' * Qr" << std::endl;
        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(cublas_handle, 
                                       CUBLAS_OP_C, 
                                       CUBLAS_OP_N, 
                                       n, 
                                       n, 
                                       N,
                                       &One, 
                                       V2.data() + offset * V2.ld(),
                                       V2.ld(), 
                                       V1.data() + offset * V1.ld(),
                                       V1.ld(),
                                       &Zero, 
                                       A->data(),
                                       n));
       
	//Scale the rows because Ql' * Qr = diag =/= I
	//std::cout << "Scale the rows because Ql' * Qr = diag =/= I" << std::endl;

	chase::linalg::internal::cuda::chase_scale_rows_matrix(A->data(),n,n,n,diag.data(),usedStream);

    T *W = V2.data() + offset * V2.ld();

	//Compute the eigenpairs of the non-hermitian rayleigh quotient	
	//std::cout << "Compute the eigenpairs of the non-hermitian rayleigh quotient" << std::endl;
	CHECK_CUSOLVER_ERROR(chase::linalg::cusolverpp::cusolverDnTgeev(cusolver_handle,
					params,
					CUSOLVER_EIG_MODE_NOVECTOR,
					CUSOLVER_EIG_MODE_VECTOR,
					n,
					A->data(),
					n,
					ritzv_complex,
					NULL,
					1,
					W,
					n,
					d_workspace,
					d_lwork,
					h_workspace,
					h_lwork,
					devInfo));

        int info;
        CHECK_CUDA_ERROR(cudaMemcpy(&info, 
                                    devInfo, 
                                    1 * sizeof(int),
                                    cudaMemcpyDeviceToHost));

        if(info != 0)
        {
            throw std::runtime_error("cusolver HEEVD failed in RayleighRitz");
        }       


	//std::cout << "Copying the complex ritz values back to cpu" << std::endl;
	//thrust::device_vector<int> indices(n); //Does not compile, returns unimplemented on this system...
	chase::Base<T> *ptx = ritzv.cpu_data() + offset;
	if constexpr (std::is_same<T, std::complex<chase::Base<T>>>::value)
	{
        cudaMemcpy(ritzvs_cmplex_cpu.data(), ritzv_complex, n * sizeof(T), cudaMemcpyDeviceToHost);
		for(auto i = 0; i < n; i++){
			ptx[i] = std::real(ritzvs_cmplex_cpu[i]);
		}
	}
	else
	{
		CHECK_CUDA_ERROR(cudaMemcpy(ptx, 
                                    ritzv_complex, 
                                    n * sizeof(chase::Base<T>),
                                    cudaMemcpyDeviceToHost));
	}

	std::vector<std::size_t> indices(n);
	std::iota(indices.begin(), indices.end(), 0); // Fill with 0, 1, ..., n-1
	std::sort(indices.begin(), indices.end(),
			[&ptx](std::size_t i1, std::size_t i2) { return ptx[i1] < ptx[i2]; });

	std::vector<Base<T>> sorted_ritzv(n);
        T *d_sorted_W = A->data();
	// Reorder eigenvalues and eigenvectors
	
	for (std::size_t i = 0; i < n; ++i) {
		sorted_ritzv[i] = ptx[indices[i]];
        	CHECK_CUDA_ERROR(cudaMemcpy(d_sorted_W + i * n, 
                                    W + indices[i] * n, 
                                    n * sizeof(T),
                                    cudaMemcpyDeviceToDevice));
	}

	std::copy(sorted_ritzv.begin(), sorted_ritzv.end(), ptx);

	ritzv.H2D();

        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(cublas_handle, 
                                       CUBLAS_OP_N, 
                                       CUBLAS_OP_N, 
                                       V2.rows(), 
                                       n, 
                                       n,
                                       &One, 
                                       V1.data() + offset * V1.ld(),
                                       V1.ld(),
                                       d_sorted_W,
                                       n,
                                       &Zero,
                                       V2.data() + offset * V2.ld(),
                                       V2.ld()));
    } 



}
}
}
}

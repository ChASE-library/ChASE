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
    * @brief Perform the Rayleigh-Ritz procedure to compute eigenvalues and eigenvectors of a Pseudo-Hermitian matrix.
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
                      chase::matrix::PseudoHermitianMatrix<T, chase::platform::GPU> * H,
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
                      chase::matrix::Matrix<T, chase::platform::GPU> *A = nullptr)
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
            A = new chase::matrix::Matrix<T, chase::platform::GPU>(3 * n, n);
        }

        std::vector<T> ritzvs_cmplex_cpu(n);

        T *ritzv_complex = A->data() + n * n;

#ifdef XGEEV_EXISTS
        //Allocating workspace memory for Xgeev
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
#endif

        T One   = T(1.0);
        T Zero  = T(0.0);
	T alpha = T(2.0);
        T NegOne = T(-1.0);
        T NegativeTwo = T(-2.0);

	    chase::Base<T> real_One = chase::Base<T>(1.0);

	    chase::matrix::Matrix<chase::Base<T>, chase::platform::GPU> diag(n,1);

        T *M = A->data() + n * n;
        T *W = A->data() + 2 * n * n;

        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(cublas_handle, 
                                        CUBLAS_OP_N, 
                                        CUBLAS_OP_N, 
                                        N, 
                                        n, 
                                        N, 
                                        &One, 
                                        H->data(), ldh, 
                                        V1.data() + offset * V1.ld(),
                                        V1.ld(), 
                                        &Zero, 
                                        V2.data() + offset * V2.ld(),
                                        V2.ld()));  //T = AQr

        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(cublas_handle, 
                                        CUBLAS_OP_C, 
                                        CUBLAS_OP_N, 
                                        n, 
                                        n, 
                                        N, 
                                        &One, 
                                        V1.data() + offset * V1.ld(), V1.ld(), V2.data() + offset * V2.ld(), V2.ld(), &Zero, W, n));  //A = Qr^* T

	CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(cublas_handle, 
                                        CUBLAS_OP_C, 
                                        CUBLAS_OP_N, 
                                        n, 
                                        n, 
                                        k, 
                                        &NegativeTwo,
                                        V1.data() + offset * V1.ld() + k, 
                                        V1.ld(), 
                                        V1.data() + offset * V1.ld() + k, 
                                        V1.ld(), 
                                        &Zero, 
                                        M, 
                                        n)); //M = -2 Qr_2^* Qr_2

    	chase::linalg::internal::cuda::chase_plus_inverse_diagonal(M,
				      n,
				      n,
				      real_One,
				      diag.data(), usedStream);

        chase::linalg::internal::cuda::chase_set_diagonal(M,
                        n,
                        n,
                        Zero, usedStream);

        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(cublas_handle, 
                                        CUBLAS_OP_N, 
                                        CUBLAS_OP_N, 
                                        n, 
                                        n, 
                                        n, 
                                        &NegOne,
                                        M, n, W, n, &Zero, A->data(), n)); //A = (Diag(M) - M) * A

        chase::linalg::internal::cuda::flipLowerHalfMatrixSign(N,n,V2.data() + offset * V2.ld(),V2.ld(), &usedStream);

	    //Last GEMM for the construction of the rayleigh Quotient : (H' * Ql)' * Qr
        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(cublas_handle, 
                                        CUBLAS_OP_C, 
                                        CUBLAS_OP_N, 
                                        n, 
                                        n, 
                                        N,
                                        &One, 
                                        V1.data() + offset * V1.ld(),
                                        V1.ld(), 
                                        V2.data() + offset * V2.ld(),
                                        V2.ld(), 
                                        &One, 
                                        A->data(), n)); 


    	chase::linalg::internal::cuda::chase_scale_rows_matrix(A->data(),n,n,n,diag.data(),usedStream);

	chase::Base<T> * ptx = ritzv.cpu_data() + offset;

#ifdef XGEEV_EXISTS
        
        //Compute the eigenpairs of the non-hermitian rayleigh quotient on GPU	
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

#else	
#ifdef CHASE_OUTPUT
	std::cout << "WARNING! XGeev not found in cuda. Compute Geev on CPU with Lapack..." << std::endl;
#endif
	std::vector<chase::Base<T>> ptx_imag = std::vector<chase::Base<T>>(n,chase::Base<T>(0.0));

	A->allocate_cpu_data();

	A->D2H();

	T * W_cpu = A->cpu_data() + 2*n*n;
	
	//Compute the eigenpairs of the non-hermitian rayleigh quotient on the CPU
        lapackpp::t_geev(LAPACK_COL_MAJOR, 'V', n, A->cpu_data(), n, ptx, ptx_imag.data(), W_cpu, n);

	A->H2D();
#endif

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

    /**
    * @brief Perform the Rayleigh-Ritz procedure to compute eigenvalues and eigenvectors of a Pseudo-Hermitian matrix.
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
    void rayleighRitz_v2(cublasHandle_t cublas_handle, 
                      cusolverDnHandle_t cusolver_handle,
		      cusolverDnParams_t params,
                      chase::matrix::PseudoHermitianMatrix<T, chase::platform::GPU> * H,
                      chase::matrix::Matrix<T, chase::platform::GPU>& V1,
                      chase::matrix::Matrix<T, chase::platform::GPU>& V2,
                      chase::matrix::Matrix<chase::Base<T>, chase::platform::GPU>& ritzv,
                      std::size_t offset,
                      std::size_t subSize,
                      int* devInfo,
                      T *workspace = nullptr,
                      int lwork = 0,
                      chase::matrix::Matrix<T, chase::platform::GPU> *A = nullptr)
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
            A = new chase::matrix::Matrix<T, chase::platform::GPU>(2 * n, n);
        }

        //Allocating workspace memory for Xgeev
        if(workspace == nullptr) //To update once Xgeev is plugged in
        {
            lwork = 0;
            CHECK_CUSOLVER_ERROR(chase::linalg::cusolverpp::cusolverDnTheevd_bufferSize(
                                    cusolver_handle, 
                                    CUSOLVER_EIG_MODE_VECTOR, 
                                    CUBLAS_FILL_MODE_LOWER, 
                                    subSize, 
                                    A->data(), 
                                    subSize, 
                                    ritzv.data(), 
                                    &lwork));

	    int lwork_potrf = 0;
            CHECK_CUSOLVER_ERROR(chase::linalg::cusolverpp::cusolverDnTpotrf_bufferSize(
                                                                cusolver_handle,
                                                                CUBLAS_FILL_MODE_LOWER,
                                                                n,
                                                                A->data(),
                                                                n,
                                                                &lwork_potrf));

	    if(lwork < lwork_potrf) lwork = lwork_potrf;

            CHECK_CUDA_ERROR(cudaMalloc((void**)&workspace, sizeof(T) * lwork));
        }

        T One   = T(1.0);
        T Zero  = T(0.0);
        T NegativeTwo = T(-2.0);

	chase::Base<T> real_One = chase::Base<T>(1.0);

        T *M = A->data() + n * n;

        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(cublas_handle, 
                                        CUBLAS_OP_N, 
                                        CUBLAS_OP_N, 
                                        N, 
                                        n, 
                                        N, 
                                        &One, 
                                        H->data(), ldh, 
                                        V1.data() + offset * V1.ld(),
                                        V1.ld(), 
                                        &Zero, 
                                        V2.data() + offset * V2.ld(),
                                        V2.ld()));  //T = AQr

        chase::linalg::internal::cuda::flipLowerHalfMatrixSign(N,n,V2.data() + offset * V2.ld(),V2.ld(), &usedStream);

        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(cublas_handle, 
                                        CUBLAS_OP_C, 
                                        CUBLAS_OP_N, 
                                        n, 
                                        n, 
                                        N, 
                                        &One, 
                                        V1.data() + offset * V1.ld(), V1.ld(), V2.data() + offset * V2.ld(), V2.ld(), &Zero, A->data(), n));  //A = Qr^* T

	CHECK_CUSOLVER_ERROR(chase::linalg::cusolverpp::cusolverDnTpotrf(cusolver_handle,
                                                                         CUBLAS_FILL_MODE_LOWER,
                                                                         n,
                                                                         A->data(),
                                                                         n,
                                                                         workspace,
                                                                         lwork,
                                                                         devInfo));

	int info;
        CHECK_CUDA_ERROR(cudaMemcpy(&info, devInfo, 1 * sizeof(int), cudaMemcpyDeviceToHost));

        if(info != 0)
        {
            throw std::runtime_error("cusolver POTRF Failed in Pseudo-Hermitian RayleighRitz");
        }        
        
	cudaMemset(M, 0, sizeof(T)*n*n);

	chase::linalg::internal::cuda::chase_set_diagonal(M,
                        n,
                        n,
                        One, usedStream);
	
	CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(cublas_handle, 
                                        CUBLAS_OP_C, 
                                        CUBLAS_OP_N, 
                                        n, 
                                        n, 
                                        k, 
                                        &NegativeTwo,
                                        V1.data() + offset * V1.ld() + k, 
                                        V1.ld(), 
                                        V1.data() + offset * V1.ld() + k, 
                                        V1.ld(), 
                                        &One, 
                                        M, 
                                        n)); //M = I - 2 Qr_2^* Qr_2
            
	CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTtrsm(cublas_handle,
                                                                    CUBLAS_SIDE_LEFT,
                                                                    CUBLAS_FILL_MODE_LOWER,
                                                                    CUBLAS_OP_N,
                                                                    CUBLAS_DIAG_NON_UNIT,
                                                                    n,
                                                                    n,
                                                                    &One,
                                                                    A->data(),
                                                                    n,
                                                                    M,
                                                                    n));

	CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTtrsm(cublas_handle,
                                                                    CUBLAS_SIDE_RIGHT,
                                                                    CUBLAS_FILL_MODE_LOWER,
                                                                    CUBLAS_OP_C,
                                                                    CUBLAS_DIAG_NON_UNIT,
                                                                    n,
                                                                    n,
                                                                    &One,
                                                                    A->data(),
                                                                    n,
                                                                    M,
                                                                    n));

        CHECK_CUSOLVER_ERROR(chase::linalg::cusolverpp::cusolverDnTheevd(
                                       cusolver_handle, 
                                       CUSOLVER_EIG_MODE_VECTOR, 
                                       CUBLAS_FILL_MODE_LOWER, 
                                       n,
                                       M,
                                       n,
                                       ritzv.data() + offset,
                                       workspace, lwork, devInfo));

        CHECK_CUDA_ERROR(cudaMemcpy(&info, devInfo, 1 * sizeof(int), cudaMemcpyDeviceToHost));

        if(info != 0)
        {
            throw std::runtime_error("cusolver HEEVD failed in Pseudo-Hermitian RayleighRitz");
        }        
	
	CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTtrsm(cublas_handle,
                                                                    CUBLAS_SIDE_LEFT,
                                                                    CUBLAS_FILL_MODE_LOWER,
                                                                    CUBLAS_OP_C,
                                                                    CUBLAS_DIAG_NON_UNIT,
                                                                    n,
                                                                    n,
                                                                    &One,
                                                                    A->data(),
                                                                    n,
                                                                    M,
                                                                    n));
	chase::linalg::internal::cuda::chase_inverse_entries(ritzv.data()+offset,subSize,usedStream);

        CHECK_CUDA_ERROR(cudaMemcpy(ritzv.cpu_data() + offset,
                                    ritzv.data() + offset,
                                    subSize * sizeof(chase::Base<T>),
                                    cudaMemcpyDeviceToHost));

        std::vector<chase::Base<T>> vectorNorms(subSize);
        std::vector<T> norm_divider(subSize);

        for(auto idx = 0; idx < subSize; idx++)
        {
                CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTnrm2(cublas_handle,
                                                                    subSize,
                                                                    M + idx * subSize,
                                                                    1,
                                                                    &vectorNorms[idx]));
        }

        for(auto idx = 0; idx < subSize; idx++)
        {
            norm_divider[idx] = T(1 / vectorNorms[idx]);
        }

        for(auto idx = 0; idx < subSize; idx++)
        {
              CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTscal(cublas_handle,
                                                                      subSize,
                                                                      &norm_divider[idx],
                                                                      M + idx * subSize,
                                                                      1));
        }

        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(cublas_handle, 
                                       CUBLAS_OP_N, 
                                       CUBLAS_OP_N, 
                                       V1.rows(), 
                                       n, 
                                       n,
                                       &One, 
                                       V1.data() + offset * V1.ld(),
                                       V1.ld(), 
                                       M,
                                       n, 
                                       &Zero,
                                       V2.data() + offset * V2.ld(),
                                       V2.ld()));


    } 

}
}
}
}

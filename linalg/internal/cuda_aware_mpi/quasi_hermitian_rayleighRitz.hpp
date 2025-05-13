// This file is a part of ChASE.
// Copyright (c) 2015-2024, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once
#include <numeric> //std::iota

#include "mpi.h"
#include "grid/mpiTypes.hpp"
#include "external/blaspp/blaspp.hpp"
#include "external/lapackpp/lapackpp.hpp"
#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"
#include "linalg/internal/cuda_aware_mpi/cuda_mpi_kernels.hpp"
#include "external/cublaspp/cublaspp.hpp"
#include "external/cusolverpp/cusolverpp.hpp"

namespace chase
{
namespace linalg
{
namespace internal
{        
    /**
     * @brief Performs the Rayleigh-Ritz procedure, reducing the matrix H using the basis vectors
     * in V1 and V2, and storing the results in W1, W2, and the eigenvalues in ritzv.
     * 
     * This function computes the distributed matrix multiplication and reduction required
     * for the Rayleigh-Ritz procedure. If matrix A is provided, it uses that as a workspace.
     * Otherwise, a temporary matrix is created.
     *
     * @tparam MatrixType Type of the input matrix H.
     * @tparam InputMultiVectorType Type of the input multivector V1 and V2.
     * @param H The input matrix representing the system to be reduced.
     * @param V1 The input multivector representing the orthonormal basis.
     * @param V2 duplication of V1.
     * @param W1 Multivector sotring H*V1.
     * @param W2 Resulting multivector by redistribute V2 from column communicator based to row communicator based.
     * @param ritzv Pointer to an array where computed eigenvalues of the reduced matrix will be stored.
     * @param offset The starting offset in V1, V2, W1, and W2 for submatrix processing.
     * @param subSize Size of the submatrix used in reduction.
     * @param A Optional workspace matrix for storing intermediate results; if nullptr, a temporary matrix is created.
     * 
     * @throws std::invalid_argument if ritzv is a nullptr.
     */                     
    template <typename MatrixType, typename InputMultiVectorType>
    void cuda_mpi::quasi_hermitian_rayleighRitz(cublasHandle_t cublas_handle,
                      cusolverDnHandle_t cusolver_handle,
		      cusolverDnParams_t params,
                      MatrixType& H,
                      InputMultiVectorType& V1,
                      InputMultiVectorType& V2,
                      typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type& W1,
                      typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type& W2,
                      chase::distMatrix::RedundantMatrix<chase::Base<typename MatrixType::value_type>, chase::platform::GPU>& ritzv,
                      std::size_t offset,
                      std::size_t subSize,
                      int* devInfo,
		      typename MatrixType::value_type *d_workspace,
                      int d_lwork,
                      typename MatrixType::value_type *h_workspace,
                      int h_lwork,
                      chase::distMatrix::RedundantMatrix<typename MatrixType::value_type, chase::platform::GPU>* A) 
    {
        using T = typename MatrixType::value_type;

        cudaStream_t usedStream;
        CHECK_CUBLAS_ERROR(cublasGetStream(cublas_handle, &usedStream));

        std::unique_ptr<chase::distMatrix::RedundantMatrix<T, chase::platform::GPU>> A_ptr;

        if (A == nullptr) {
            // Allocate A if not provided
            A_ptr = std::make_unique<chase::distMatrix::RedundantMatrix<T, chase::platform::GPU>>(3*subSize, subSize, V1.getMpiGrid_shared_ptr());
            A = A_ptr.get();
        }
        
	T One    = T(1.0);
        T Zero   = T(0.0);
        T NegOne = -T(1.0);
	chase::Base<T> real_Zero = chase::Base<T>(0.0);

	T * M = A->l_data() + subSize * subSize; 
	T * W = A->l_data() + 2 * subSize * subSize;
        T * ritzv_complex = A->l_data() + subSize * subSize;
        
	std::vector<T> ritzvs_cmplex_cpu(subSize);

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
                                                            subSize,
                                                            A->l_data(),
                                                            subSize,
                                                            ritzv_complex,
                                                            NULL,1,
                                                            V1.l_data() + offset * V1.l_ld(),V1.l_ld(),
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
#else
        if (A == nullptr) {
		A->allocate_cpu_data();
	}
#endif

        //Allocate the space for scaling weights. Can be Base<T> since reals?	
	chase::distMatrix::RedundantMatrix<chase::Base<typename MatrixType::value_type>, chase::platform::GPU> diag(1,subSize,V1.getMpiGrid_shared_ptr());

        //Allocate the space for the imaginary parts of ritz values
        std::vector<Base<T>> ritzvi(subSize, Base<T>(0.0));

        // Perform the distributed matrix-matrix multiplication
        chase::linalg::internal::cuda_mpi::MatrixMultiplyMultiVectorsAndRedistribute(
			cublas_handle,
                        H, 
                        V1, 
                        W1, 
                        V2, 
                        W2,
                        offset,
                        subSize);

        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(cublas_handle, 
                                       CUBLAS_OP_C, 
                                       CUBLAS_OP_N, 
                                       subSize, 
                                       subSize, 
                                       W2.l_rows(),
                                       &One, 
                                       W2.l_data() + offset * W2.l_ld(),
                                       W2.l_ld(), 
                                       W1.l_data() + offset * W1.l_ld(), 
                                       W1.l_ld(),
                                       &Zero, 
                                       W,
                                       subSize));

        // Perform the MPI_Allreduce to sum up the results
	
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        MPI_Allreduce(MPI_IN_PLACE,
                      W,
                      subSize*subSize,
                      chase::mpi::getMPI_Type<T>(),
                      MPI_SUM,
                      A->getMpiGrid()->get_row_comm());

	chase::linalg::internal::cuda_mpi::flipLowerHalfMatrixSign(V2, offset, subSize);
	
        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(cublas_handle, 
                                       CUBLAS_OP_C, 
                                       CUBLAS_OP_N, 
                                       subSize, 
                                       subSize, 
                                       V2.l_rows(),
                                       &One, 
                                       V2.l_data() + offset * V2.l_ld(),
                                       V2.l_ld(), 
                                       V1.l_data() + offset * V1.l_ld(), 
                                       V1.l_ld(),
                                       &Zero, 
                                       M,
                                       subSize));
	
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        MPI_Allreduce(MPI_IN_PLACE,
                      M,
                      subSize*subSize,
                      chase::mpi::getMPI_Type<T>(),
                      MPI_SUM,
                      A->getMpiGrid()->get_col_comm());

        chase::linalg::internal::cuda::chase_plus_inverse_diagonal(M,
                                      subSize,
                                      subSize,
                                      real_Zero,
                                      diag.l_data(), usedStream);

        chase::linalg::internal::cuda::chase_set_diagonal(M,
                        subSize,
                        subSize,
                        Zero, usedStream);
	
        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(cublas_handle,
                                        CUBLAS_OP_N,
                                        CUBLAS_OP_N,
                                        subSize,
                                        subSize,
                                        subSize,
                                        &NegOne,
                                        M, subSize, W, subSize, &Zero, A->l_data(), subSize)); //A = (Diag(M) - M) * A
	
	chase::linalg::internal::cuda_mpi::flipLowerHalfMatrixSign(W1, offset, subSize);

        //Last GEMM for the construction of the rayleigh Quotient : (H' * Ql)' * Qr
        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(cublas_handle,
                                        CUBLAS_OP_C,
                                        CUBLAS_OP_N,
                                        subSize,
                                        subSize,
                                        W2.l_rows(),
                                        &One,
                                        W2.l_data() + offset * W2.l_ld(),
                                        W2.l_ld(),
                                        W1.l_data() + offset * W1.l_ld(),
                                        W1.l_ld(),
                                        &Zero,
                                        M, subSize));
        
	
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        MPI_Allreduce(MPI_IN_PLACE,
                      M,
                      subSize*subSize,
                      chase::mpi::getMPI_Type<T>(),
                      MPI_SUM,
                      A->getMpiGrid()->get_row_comm());

        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTaxpy(cublas_handle,
                                                                subSize * subSize,
                                                                &One,
                                                                M,
                                                                1,
                                                                A->l_data(),
                                                                1));

        chase::linalg::internal::cuda::chase_scale_rows_matrix(A->l_data(),subSize,subSize,subSize,diag.l_data(),usedStream);
	
        chase::Base<T> * ptx = ritzv.cpu_data() + offset;

#ifdef XGEEV_EXISTS

        //Compute the eigenpairs of the non-hermitian rayleigh quotient on GPU  
        //std::cout << "Compute the eigenpairs of the non-hermitian rayleigh quotient" << std::endl;
        CHECK_CUSOLVER_ERROR(chase::linalg::cusolverpp::cusolverDnTgeev(cusolver_handle,
                        params,
                        CUSOLVER_EIG_MODE_NOVECTOR,
                        CUSOLVER_EIG_MODE_VECTOR,
                        subSize,
                        A->l_data(),
                        subSize,
                        ritzv_complex,
                        NULL,
                        1,
                        W,
                        subSize,
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
            cudaMemcpy(ritzvs_cmplex_cpu.data(), ritzv_complex, subSize * sizeof(T), cudaMemcpyDeviceToHost);
            for(auto i = 0; i < subSize; i++){
                ptx[i] = std::real(ritzvs_cmplex_cpu[i]);
            }
        }
        else
        {
            CHECK_CUDA_ERROR(cudaMemcpy(ptx,
                                        ritzv_complex,
                                        subSize * sizeof(chase::Base<T>),
                                        cudaMemcpyDeviceToHost));
        }

#else
#ifdef CHASE_OUTPUT
	if (H.grank() == 0){
        	std::cout << "WARNING! XGeev not found in cuda. Compute Geev on CPU with Lapack..." << std::endl;
	}
#endif
        std::vector<chase::Base<T>> ptx_imag = std::vector<chase::Base<T>>(subSize,chase::Base<T>(0.0));

        A->D2H();

        T * W_cpu = A->cpu_data() + 2*subSize*subSize;

        //Compute the eigenpairs of the non-hermitian rayleigh quotient on the CPU
        lapackpp::t_geev(LAPACK_COL_MAJOR, 'V', subSize, A->cpu_data(), subSize, ptx, ptx_imag.data(), W_cpu, subSize);

        A->H2D();
#endif

        std::vector<std::size_t> indices(subSize);
        std::iota(indices.begin(), indices.end(), 0); // Fill with 0, 1, ..., n-1
        std::sort(indices.begin(), indices.end(),
                [&ptx](std::size_t i1, std::size_t i2) { return ptx[i1] < ptx[i2]; });

        std::vector<Base<T>> sorted_ritzv(subSize);
        T *d_sorted_W = A->l_data();
        // Reorder eigenvalues and eigenvectors

        for (std::size_t i = 0; i < subSize; ++i) {
            sorted_ritzv[i] = ptx[indices[i]];
            CHECK_CUDA_ERROR(cudaMemcpy(d_sorted_W + i * subSize,
                                        W + indices[i] * subSize,
                                        subSize * sizeof(T),
                                        cudaMemcpyDeviceToDevice));
        }

        std::copy(sorted_ritzv.begin(), sorted_ritzv.end(), ptx);

        ritzv.H2D();

        chase::linalg::internal::cuda_mpi::flipLowerHalfMatrixSign(V2, offset, subSize);

        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(cublas_handle,
                                    CUBLAS_OP_N,
                                    CUBLAS_OP_N,
                                    V2.l_rows(),
                                    subSize,
                                    subSize,
                                    &One,
                                    V2.l_data() + offset * V2.l_ld(),
                                    V2.l_ld(),
                                    d_sorted_W,
                                    subSize,
                                    &Zero,
                                    V1.l_data() + offset * V1.l_ld(),
                                    V1.l_ld()));
    
    }	

}
}
}

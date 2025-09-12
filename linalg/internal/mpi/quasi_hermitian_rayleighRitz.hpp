// This file is a part of ChASE.
// Copyright (c) 2015-2024, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once
#include <numeric> //std::iota

#include "mpi.h"
#include "grid/mpiTypes.hpp"
#include "linalg/internal/cpu/utils.hpp"
#include "external/blaspp/blaspp.hpp"
#include "external/lapackpp/lapackpp.hpp"
#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"
#include "linalg/internal/mpi/mpi_kernels.hpp"
#include "external/scalapackpp/scalapackpp.hpp"

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
     void cpu_mpi::quasi_hermitian_rayleighRitz(MatrixType& H,
                       InputMultiVectorType& V1,
                       InputMultiVectorType& V2,
                       typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type& W1,
                       typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type& W2,
                       chase::Base<typename MatrixType::value_type>* ritzv,
                       std::size_t offset,
                       std::size_t subSize,
                       chase::distMatrix::RedundantMatrix<typename MatrixType::value_type, chase::platform::CPU>* A)                
    {
        using T = typename MatrixType::value_type;

        if (ritzv == nullptr) {
            throw std::invalid_argument("ritzv cannot be a nullptr.");
        }

        std::unique_ptr<chase::distMatrix::RedundantMatrix<T, chase::platform::CPU>> A_ptr;
        
	if (A == nullptr) {
            // Allocate A if not provided
            A_ptr = std::make_unique<chase::distMatrix::RedundantMatrix<T, chase::platform::CPU>>(subSize, 3 * subSize, V1.getMpiGrid_shared_ptr());
            A = A_ptr.get();
        }

        T One    = T(1.0);
        T Zero   = T(0.0);
        T NegOne = -T(1.0);

	T * M = A->l_data() + subSize * subSize; 
	T * W = A->l_data() + 2 * subSize * subSize;

        //Allocate the space for scaling weights. Can be Base<T> since reals?
        std::vector<T> diag(subSize, T(0.0)); 

        //Allocate the space for the imaginary parts of ritz values
        std::vector<Base<T>> ritzvi(subSize, Base<T>(0.0));
    
        // Perform the distributed matrix-matrix multiplication
        chase::linalg::internal::cpu_mpi::MatrixMultiplyMultiVectorsAndRedistributeAsync(
                        H, 
                        V1, 
                        W1, 
                        V2, 
                        W2,
                        offset,
                        subSize);

        chase::linalg::blaspp::t_gemm(CblasColMajor, 
                                     CblasConjTrans, 
                                     CblasNoTrans, 
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
                                     subSize);

        // Perform the MPI_Allreduce to sum up the results
        MPI_Allreduce(MPI_IN_PLACE, 
                    W, 
                    subSize * subSize, 
                    chase::mpi::getMPI_Type<T>(), 
                    MPI_SUM, 
                    A->getMpiGrid()->get_row_comm());
		
	chase::linalg::internal::cpu_mpi::flipLowerHalfMatrixSign(V2, offset, subSize);
	
	chase::linalg::blaspp::t_gemm(CblasColMajor, 
                                     CblasConjTrans, 
                                     CblasNoTrans, 
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
                                     subSize);

        // Perform the MPI_Allreduce to sum up the results
        MPI_Allreduce(MPI_IN_PLACE, 
                    M, 
                    subSize * subSize, 
                    chase::mpi::getMPI_Type<T>(), 
                    MPI_SUM, 
                    A->getMpiGrid()->get_col_comm());

	for(auto i = 0; i < subSize; i++)
        {
            diag[i] = T(1.0) / (M[i*(subSize+1)]);
        }

        for(auto i = 0; i < subSize; i++)
        {
            M[i*(subSize+1)] = T(0.0);
        }

	blaspp::t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, subSize, subSize, subSize, &NegOne,
                M, subSize, W, subSize, &Zero, A->l_data(), subSize); //A = (Diag(M) - M) * A
	
	chase::linalg::internal::cpu_mpi::flipLowerHalfMatrixSign(W1, offset, subSize);
	
	chase::linalg::blaspp::t_gemm(CblasColMajor, 
                                     CblasConjTrans, 
                                     CblasNoTrans, 
                                     subSize, 
                                     subSize, 
                                     W2.l_rows(),
                                     &One, 
                                     W2.l_data() + offset * W2.l_ld(),
                                     W2.l_ld(), 
                                     W1.l_data() + offset * W1.l_ld(), 
                                     W1.l_ld(), 
                                     &Zero, 
                                     M,  
                                     subSize);
        
	// Perform the MPI_Allreduce to sum up the results
        MPI_Allreduce(MPI_IN_PLACE, 
                    M, 
                    subSize * subSize, 
                    chase::mpi::getMPI_Type<T>(), 
                    MPI_SUM, 
                    A->getMpiGrid()->get_row_comm());

        chase::linalg::blaspp::t_axpy(subSize * subSize, 
                                      &One, 
                                      M, 
                                      1, 
                                      A->l_data(), 
                                      1);
	
	//Scale the rows because Ql' * Qr = diag =/= I
        for(auto i = 0; i < subSize; i++)
        {
                blaspp::t_scal(subSize, &diag[i], A->l_data() + i, subSize);
        }

        //Compute the eigenpairs of the non-hermitian rayleigh quotient
        lapackpp::t_geev(LAPACK_COL_MAJOR, 'V', subSize, A->l_data(), subSize, ritzv+offset, ritzvi.data(), W, subSize);
    	
        //Sort indices based on ritz values
        std::vector<Base<T>> sorted_ritzv(ritzv + offset, ritzv + offset + subSize);
        std::vector<std::size_t> indices(subSize);
        std::iota(indices.begin(), indices.end(), 0); // Fill with 0, 1, ..., n-1
        std::sort(indices.begin(), indices.end(), 
                        [&sorted_ritzv](std::size_t i1, std::size_t i2) { return sorted_ritzv[i1] < sorted_ritzv[i2]; });

        // Create temporary storage for sorted eigenvalues and eigenvectors
        T *sorted_W = A->l_data();
        // Reorder eigenvalues and eigenvectors
        for (std::size_t i = 0; i < subSize; ++i) {
                ritzv[i+offset] = sorted_ritzv[indices[i]];
        }

        for (std::size_t i = 0; i < subSize; ++i) {
            std::copy_n(W + indices[i] * subSize, subSize, sorted_W + i * subSize);
        }
	
        // Copy back to original arrays
        std::copy(sorted_W, sorted_W + subSize * subSize, W);
	
	chase::linalg::internal::cpu_mpi::flipLowerHalfMatrixSign(V2, offset, subSize);

        // GEMM for applying eigenvectors back to V1 from V2 * A
        chase::linalg::blaspp::t_gemm(CblasColMajor,
                                     CblasNoTrans,
                                     CblasNoTrans,
                                     V2.l_rows(),
                                     subSize,
                                     subSize,
                                     &One,
                                     V2.l_data() + offset * V2.l_ld(),
                                     V2.l_ld(),
                                     W,
                                     subSize,
                                     &Zero,
                                     V1.l_data() + offset * V1.l_ld(),
                                     V1.l_ld());
    
    }
    
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
     void cpu_mpi::quasi_hermitian_rayleighRitz_v2(MatrixType& H,
                       InputMultiVectorType& V1,
                       InputMultiVectorType& V2,
                       typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type& W1,
                       typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type& W2,
                       chase::Base<typename MatrixType::value_type>* ritzv,
                       std::size_t offset,
                       std::size_t subSize,
                       chase::distMatrix::RedundantMatrix<typename MatrixType::value_type, chase::platform::CPU>* A)                
    {
        using T = typename MatrixType::value_type;

        if (ritzv == nullptr) {
            throw std::invalid_argument("ritzv cannot be a nullptr.");
        }

        std::unique_ptr<chase::distMatrix::RedundantMatrix<T, chase::platform::CPU>> A_ptr;
        
	if (A == nullptr) {
            // Allocate A if not provided
            A_ptr = std::make_unique<chase::distMatrix::RedundantMatrix<T, chase::platform::CPU>>(subSize, 2 * subSize, V1.getMpiGrid_shared_ptr());
            A = A_ptr.get();
        }

        T One    = T(1.0);
        T Zero   = T(0.0);

	T * M = A->l_data() + subSize * subSize; 

        // Perform the distributed matrix-matrix multiplication
        chase::linalg::internal::cpu_mpi::MatrixMultiplyMultiVectorsAndRedistributeAsync(
                        H, 
                        V1, 
                        W1, 
                        V2, 
                        W2,
                        offset,
                        subSize);

	//Flip the sign of the lower part of T to emulate the multiplication SAQ	
	chase::linalg::internal::cpu_mpi::flipLowerHalfMatrixSign(W1, offset, subSize);

        chase::linalg::blaspp::t_gemm(CblasColMajor, 
                                     CblasConjTrans, 
                                     CblasNoTrans, 
                                     subSize, 
                                     subSize, 
                                     W2.l_rows(),
                                     &One, 
                                     W2.l_data() + offset * W2.l_ld(),
                                     W2.l_ld(), 
                                     W1.l_data() + offset * W1.l_ld(), 
                                     W1.l_ld(), 
                                     &Zero, 
                                     A->l_data(),  
                                     subSize);

        // Perform the MPI_Allreduce to sum up the results
        MPI_Allreduce(MPI_IN_PLACE, 
                    A->l_data(), 
                    subSize * subSize, 
                    chase::mpi::getMPI_Type<T>(), 
                    MPI_SUM, 
                    A->getMpiGrid()->get_row_comm());
	
	//Flip the sign of the lower part of V1 for S Q 
	chase::linalg::internal::cpu_mpi::flipLowerHalfMatrixSign(V1, offset, subSize);

        //Compute the matrix M  = Q' S Q
	chase::linalg::blaspp::t_gemm(CblasColMajor, 
                                     CblasConjTrans, 
                                     CblasNoTrans, 
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
                                     subSize);

        // Perform the MPI_Allreduce to sum up the results
        MPI_Allreduce(MPI_IN_PLACE, 
                    M, 
                    subSize * subSize, 
                    chase::mpi::getMPI_Type<T>(), 
                    MPI_SUM, 
                    A->getMpiGrid()->get_col_comm());

	//Factorize the HPD mat Q' SA Q
        lapackpp::t_potrf('L',subSize,A->l_data(),subSize);

        //Create the inverse of the Hermitian Rayleigh Quotient by two successive trsm 
        blaspp::t_trsm('L','L','N','N',subSize,subSize,&One,A->l_data(),subSize,M,subSize);

        blaspp::t_trsm('R','L','C','N',subSize,subSize,&One,A->l_data(),subSize,M,subSize);

        //Compute the invtered ritz pairs of the Hermitian Rayleigh Quotient
        lapackpp::t_heevd(LAPACK_COL_MAJOR, 'V', 'L', subSize, M, subSize, ritzv + offset);

        blaspp::t_trsm('L','L','C','N',subSize,subSize,&One,A->l_data(),subSize,M,subSize);

	//Invert the ritz values and normalize the vectors
        std::vector<T> norms(subSize);

        for(auto idx = 0; idx < subSize; idx++)
        {
                ritzv[idx+offset] = 1.0 / ritzv[idx+offset];
        }
        for(auto idx = 0; idx < subSize; idx++)
        {
                norms[idx] = T(1.0/blaspp::t_nrm2(subSize, M + idx * subSize, 1));
        }
        for(auto idx = 0; idx < subSize; idx++)
        {
                blaspp::t_scal(subSize, &norms[idx], M + idx * subSize, 1);
	}
        // GEMM for applying eigenvectors back to V1 from V2 * A
        chase::linalg::blaspp::t_gemm(CblasColMajor,
                                     CblasNoTrans,
                                     CblasNoTrans,
                                     V2.l_rows(),
                                     subSize,
                                     subSize,
                                     &One,
                                     V2.l_data() + offset * V2.l_ld(),
                                     V2.l_ld(),
                                     M,
                                     subSize,
                                     &Zero,
                                     V1.l_data() + offset * V1.l_ld(),
                                     V1.l_ld());
    
    }
	

}
}
}

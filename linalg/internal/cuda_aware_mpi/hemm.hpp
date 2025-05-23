// This file is a part of ChASE.
// Copyright (c) 2015-2024, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include <mpi.h>
#include "external/blaspp/blaspp.hpp"
#include "external/lapackpp/lapackpp.hpp"
#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"
#include "external/cublaspp/cublaspp.hpp"
#include "linalg/internal/cuda_aware_mpi/cuda_mpi_kernels.hpp"
#include "../typeTraits.hpp"

namespace chase
{
namespace linalg
{
namespace internal
{
    template <typename T, typename MatrixType, typename InputMultiVectorType>
    void cuda_mpi::MatrixMultiplyMultiVectors(cublasHandle_t cublas_handle, T* alpha,
                                        MatrixType& blockMatrix,
                                        InputMultiVectorType& input_multiVector,
                                        T* beta,
                                        typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type& result_multiVector,
                                        std::size_t offset,
                                        std::size_t subSize) 
    {
        //std::cout << "cuda mpi" << std::endl;
        // Ensure the platform type is chase::platform::GPU
        static_assert(std::is_same<typename MatrixType::platform_type, chase::platform::GPU>::value,
                    "Matrix type must be chase::platform::GPU");

        static_assert(std::is_same<typename InputMultiVectorType::platform_type, chase::platform::GPU>::value,
                    "Multivector type must be chase::platform::GPU");

        if (input_multiVector.l_rows() != (ExtractCommType<InputMultiVectorType>::value == chase::distMultiVector::CommunicatorType::row 
                                            ? blockMatrix.l_cols() 
                                            : blockMatrix.l_rows())) {
            throw std::runtime_error("Dimension mismatch: Input multiVector rows must match blockMatrix rows or columns.");
        }


        if (result_multiVector.l_rows() != (ExtractCommType<InputMultiVectorType>::value == chase::distMultiVector::CommunicatorType::row 
                                            ? blockMatrix.l_rows() 
                                            : blockMatrix.l_cols())) {
            throw std::runtime_error("Dimension mismatch: Result multiVector rows must match blockMatrix rows or columns.");
        }

        // Check MPI grid compatibility
        if (input_multiVector.getMpiGrid() != result_multiVector.getMpiGrid() || input_multiVector.getMpiGrid() != blockMatrix.getMpiGrid()) {
            throw std::runtime_error("MPI Grid mismatch between input and result multiVectors.");
        }

        // Check offset and subSize
        std::size_t inputCols = input_multiVector.l_cols();
        if (offset < 0 || offset >= inputCols) {
            throw std::invalid_argument("Offset must be between 0 and the number of input multiVector columns - 1.");
        }
        
        if (subSize <= 0 || subSize > inputCols) {
            throw std::invalid_argument("SubSize must be greater than 0 and less than or equal to the number of input multiVector columns.");
        }
        if (offset + subSize > inputCols) {
            throw std::invalid_argument("Offset plus SubSize must be less than or equal to the number of input multiVector columns.");
        }
                
        int *coords = input_multiVector.getMpiGrid()->get_coords();
        T beta_tmp;
                
        if constexpr (ExtractCommType<InputMultiVectorType>::value == chase::distMultiVector::CommunicatorType::column)
        {
            if (coords[0] != 0)
            {
                beta_tmp = T(0.0); // If not the first row, set beta_tmp to 0
            }
            else
            {
                beta_tmp = *beta; // If the first row, use the provided beta value
            }
	    
	    if constexpr (std::is_same<MatrixType,chase::distMatrix::QuasiHermitianBlockBlockMatrix<T, chase::platform::GPU>>::value ||
                          std::is_same<MatrixType,chase::distMatrix::QuasiHermitianBlockCyclicMatrix<T, chase::platform::GPU>>::value ){

                chase::linalg::internal::cuda_mpi::flipLowerHalfMatrixSign(input_multiVector, offset, subSize);
            }

            if(beta_tmp != T(0.0)){

                if constexpr (std::is_same<MatrixType,chase::distMatrix::QuasiHermitianBlockBlockMatrix<T, chase::platform::GPU>>::value ||
                          std::is_same<MatrixType,chase::distMatrix::QuasiHermitianBlockCyclicMatrix<T, chase::platform::GPU>>::value ){

                        chase::linalg::internal::cuda_mpi::flipLowerHalfMatrixSign(result_multiVector, offset, subSize);
                }
            }

            // Perform the matrix multiplication using BLAS
            CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(cublas_handle,
                                                                    CUBLAS_OP_C,
                                                                    CUBLAS_OP_N,
                                                                    blockMatrix.l_cols(),
                                                                    subSize,
                                                                    blockMatrix.l_rows(), 
                                                                    alpha,
                                                                    blockMatrix.l_data(),
                                                                    blockMatrix.l_ld(),
                                                                    input_multiVector.l_data() + offset * input_multiVector.l_ld(),
                                                                    input_multiVector.l_ld(),
                                                                    &beta_tmp,
                                                                    result_multiVector.l_data() + offset * result_multiVector.l_ld(),
                                                                    result_multiVector.l_ld()));            
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());
            MPI_Allreduce(MPI_IN_PLACE, 
                          result_multiVector.l_data() + offset * result_multiVector.l_ld(), 
                          result_multiVector.l_ld() * subSize,
                          chase::mpi::getMPI_Type<T>(),
                          MPI_SUM,
                          input_multiVector.getMpiGrid()->get_col_comm());
	    
	    if constexpr (std::is_same<MatrixType,chase::distMatrix::QuasiHermitianBlockBlockMatrix<T, chase::platform::GPU>>::value ||
                          std::is_same<MatrixType,chase::distMatrix::QuasiHermitianBlockCyclicMatrix<T, chase::platform::GPU>>::value ){

                chase::linalg::internal::cuda_mpi::flipLowerHalfMatrixSign(input_multiVector, offset, subSize);
            }

            if constexpr (std::is_same<MatrixType,chase::distMatrix::QuasiHermitianBlockBlockMatrix<T, chase::platform::GPU>>::value ||
                          std::is_same<MatrixType,chase::distMatrix::QuasiHermitianBlockCyclicMatrix<T, chase::platform::GPU>>::value ){

                chase::linalg::internal::cuda_mpi::flipLowerHalfMatrixSign(result_multiVector, offset, subSize);
           }
                 
        }
        else // InputCommType is CommunicatorType::column
        {
            if (coords[1] != 0)
            {
                beta_tmp = T(0.0); // If not the first column, set beta_tmp to 0
            }
            else
            {
                beta_tmp = *beta; // If the first column, use the provided beta value
            }

            // Perform the matrix multiplication using BLAS
            CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(cublas_handle,
                                                                    CUBLAS_OP_N,
                                                                    CUBLAS_OP_N,
                                                                    blockMatrix.l_rows(),
                                                                    subSize,
                                                                    blockMatrix.l_cols(), 
                                                                    alpha,
                                                                    blockMatrix.l_data(),
                                                                    blockMatrix.l_ld(),
                                                                    input_multiVector.l_data() + offset * input_multiVector.l_ld(), 
                                                                    input_multiVector.l_ld(),
                                                                    &beta_tmp,
                                                                    result_multiVector.l_data() + offset * result_multiVector.l_ld(),
                                                                    result_multiVector.l_ld()));               

            // Perform reduction across the row communicator
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());
            MPI_Allreduce(MPI_IN_PLACE, 
                          result_multiVector.l_data() + offset * result_multiVector.l_ld(),
                          result_multiVector.l_ld() * subSize,
                          chase::mpi::getMPI_Type<T>(),
                          MPI_SUM,
                          input_multiVector.getMpiGrid()->get_row_comm());                        
        }
       
    }

    template <typename T, typename MatrixType, typename InputMultiVectorType>
    void cuda_mpi::MatrixMultiplyMultiVectors(cublasHandle_t cublas_handle, T* alpha,
                                        MatrixType& blockMatrix,
                                        InputMultiVectorType& input_multiVector,
                                        T* beta,
                                        typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type& result_multiVector) 
    {
        cuda_mpi::MatrixMultiplyMultiVectors(cublas_handle,
                                       alpha,
                                       blockMatrix, 
                                       input_multiVector, 
                                       beta,
                                       result_multiVector,
                                       0,
                                       input_multiVector.l_cols());
    }                                   


    //this operation do: W1<-1.0 * H * V1, while redistribute V2 to W2
    template <typename MatrixType, typename InputMultiVectorType>    
    void cuda_mpi::MatrixMultiplyMultiVectorsAndRedistribute(cublasHandle_t cublas_handle,
                                        MatrixType& blockMatrix, 
                                        InputMultiVectorType& input_multiVector, 
                                        typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type& result_multiVector,
                                        InputMultiVectorType& src_multiVector,   
                                        typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type& target_multiVector,                                                                               
                                        std::size_t offset,
                                        std::size_t subSize)                                                                                

    {
        using T = typename MatrixType::value_type;

        // Ensure the platform type is chase::platform::GPU
        static_assert(std::is_same<typename MatrixType::platform_type, chase::platform::GPU>::value,
                    "Matrix type must be chase::platform::GPU");

        static_assert(std::is_same<typename InputMultiVectorType::platform_type, chase::platform::GPU>::value,
                    "Multivector type must be chase::platform::GPU");

        if (input_multiVector.l_rows() != (ExtractCommType<InputMultiVectorType>::value == chase::distMultiVector::CommunicatorType::row 
                                            ? blockMatrix.l_cols() 
                                            : blockMatrix.l_rows())) {
            throw std::runtime_error("Dimension mismatch: Result multiVector rows must match blockMatrix rows or columns.");
        }


        if (result_multiVector.l_rows() != (ExtractCommType<InputMultiVectorType>::value == chase::distMultiVector::CommunicatorType::row 
                                            ? blockMatrix.l_rows() 
                                            : blockMatrix.l_cols())) {
            throw std::runtime_error("Dimension mismatch: Result multiVector rows must match blockMatrix rows or columns.");
        }

        if (src_multiVector.l_rows() != (ExtractCommType<InputMultiVectorType>::value == chase::distMultiVector::CommunicatorType::row 
                                            ? blockMatrix.l_cols() 
                                            : blockMatrix.l_rows())) {
            throw std::runtime_error("Dimension mismatch: Result multiVector rows must match blockMatrix rows or columns.");
        }


        if (target_multiVector.l_rows() != (ExtractCommType<InputMultiVectorType>::value == chase::distMultiVector::CommunicatorType::row 
                                            ? blockMatrix.l_rows() 
                                            : blockMatrix.l_cols())) {
            throw std::runtime_error("Dimension mismatch: Result multiVector rows must match blockMatrix rows or columns.");
        }

        // Check MPI grid compatibility
        if (input_multiVector.getMpiGrid() != result_multiVector.getMpiGrid() || input_multiVector.getMpiGrid() != blockMatrix.getMpiGrid()) {
            throw std::runtime_error("MPI Grid mismatch between input and result multiVectors.");
        }

        // Check MPI grid compatibility
        if (src_multiVector.getMpiGrid() != target_multiVector.getMpiGrid()) {
            throw std::runtime_error("MPI Grid mismatch between input and result multiVectors.");
        }

        // Check offset and subSize
        std::size_t inputCols = input_multiVector.l_cols();
        if (offset < 0 || offset >= inputCols) {
            throw std::invalid_argument("Offset must be between 0 and the number of input multiVector columns - 1.");
        }
        
        if (subSize <= 0 || subSize > inputCols) {
            throw std::invalid_argument("SubSize must be greater than 0 and less than or equal to the number of input multiVector columns.");
        }
        if (offset + subSize > inputCols) {
            throw std::invalid_argument("Offset plus SubSize must be less than or equal to the number of input multiVector columns.");
        }

        int *coords = input_multiVector.getMpiGrid()->get_coords();

        T One = T(1.0);
        T Zero = T(0.0);
        
        if constexpr (ExtractCommType<InputMultiVectorType>::value == chase::distMultiVector::CommunicatorType::column)
        {
	    if constexpr (std::is_same<MatrixType,chase::distMatrix::QuasiHermitianBlockBlockMatrix<T, chase::platform::GPU>>::value ||
                          std::is_same<MatrixType,chase::distMatrix::QuasiHermitianBlockCyclicMatrix<T, chase::platform::GPU>>::value ){

                chase::linalg::internal::cuda_mpi::flipLowerHalfMatrixSign(input_multiVector, offset, subSize);
            }

            // Perform the matrix multiplication using BLAS
            CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(cublas_handle,
                                                                    CUBLAS_OP_C,
                                                                    CUBLAS_OP_N,
                                                                    blockMatrix.l_cols(),
                                                                    subSize,
                                                                    blockMatrix.l_rows(), 
                                                                    &One,
                                                                    blockMatrix.l_data(),
                                                                    blockMatrix.l_ld(),
                                                                    input_multiVector.l_data() + offset * input_multiVector.l_ld(),
                                                                    input_multiVector.l_ld(),
                                                                    &Zero,
                                                                    result_multiVector.l_data() + offset * result_multiVector.l_ld(),
                                                                    result_multiVector.l_ld()));              

            // Perform reduction across the column communicator
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());
            MPI_Allreduce(MPI_IN_PLACE, 
                          result_multiVector.l_data() + offset * result_multiVector.l_ld(),
                          result_multiVector.l_ld() * subSize, 
                          chase::mpi::getMPI_Type<T>(),
                          MPI_SUM,
                          input_multiVector.getMpiGrid()->get_col_comm());                                           
	   
	    if constexpr (std::is_same<MatrixType,chase::distMatrix::QuasiHermitianBlockBlockMatrix<T, chase::platform::GPU>>::value ||
                          std::is_same<MatrixType,chase::distMatrix::QuasiHermitianBlockCyclicMatrix<T, chase::platform::GPU>>::value ){

                chase::linalg::internal::cuda_mpi::flipLowerHalfMatrixSign(input_multiVector, offset, subSize);
            }

            if constexpr (std::is_same<MatrixType,chase::distMatrix::QuasiHermitianBlockBlockMatrix<T, chase::platform::GPU>>::value ||
                          std::is_same<MatrixType,chase::distMatrix::QuasiHermitianBlockCyclicMatrix<T, chase::platform::GPU>>::value ){

                chase::linalg::internal::cuda_mpi::flipLowerHalfMatrixSign(result_multiVector, offset, subSize);
           }
        }
        else // InputCommType is CommunicatorType::column
        {
            // Perform the matrix multiplication using BLAS

            CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(cublas_handle,
                                                                    CUBLAS_OP_N,
                                                                    CUBLAS_OP_N,
                                                                    blockMatrix.l_rows(),
                                                                    subSize,
                                                                    blockMatrix.l_cols(), 
                                                                    &One,
                                                                    blockMatrix.l_data(),
                                                                    blockMatrix.l_ld(),
                                                                    input_multiVector.l_data() + offset * input_multiVector.l_ld(), 
                                                                    input_multiVector.l_ld(),
                                                                    &Zero,
                                                                    result_multiVector.l_data() + offset * result_multiVector.l_ld(),
                                                                    result_multiVector.l_ld()));             
            // Perform reduction across the row communicator
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());
            MPI_Allreduce(MPI_IN_PLACE,
                          result_multiVector.l_data() + offset * result_multiVector.l_ld(),
                          result_multiVector.l_ld() * subSize, 
                          chase::mpi::getMPI_Type<T>(),
                          MPI_SUM, 
                          input_multiVector.getMpiGrid()->get_row_comm());                       
        }

        src_multiVector.redistributeImpl(&target_multiVector, offset, subSize); //computation of gemm, and redistribute can be overlapped, wiil do later

    }

    template <typename MatrixType, typename InputMultiVectorType>    
    void cuda_mpi::MatrixMultiplyMultiVectorsAndRedistribute(cublasHandle_t cublas_handle,
                                        MatrixType& blockMatrix, 
                                        InputMultiVectorType& input_multiVector, 
                                        typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type& result_multiVector,
                                        InputMultiVectorType& src_multiVector,   
                                        typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type& target_multiVector)                                                                          
    {

        cuda_mpi::MatrixMultiplyMultiVectorsAndRedistribute(cublas_handle,
                                                           blockMatrix, 
                                                           input_multiVector, 
                                                           result_multiVector,
                                                           src_multiVector,
                                                           target_multiVector,
                                                           0,
                                                           input_multiVector.l_cols());
    }

}
}
}

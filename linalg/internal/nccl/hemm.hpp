#pragma once

#include <mpi.h>
#include "linalg/blaspp/blaspp.hpp"
#include "linalg/lapackpp/lapackpp.hpp"
#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"
#include "linalg/cublaspp/cublaspp.hpp"
#include "../typeTraits.hpp"

namespace chase
{
namespace linalg
{
namespace internal
{
namespace nccl
{
    template <typename T, typename MatrixType, typename InputMultiVectorType>
    void MatrixMultiplyMultiVectors(cublasHandle_t cublas_handle, T* alpha,
                                        MatrixType& blockMatrix,
                                        InputMultiVectorType& input_multiVector,
                                        T* beta,
                                        typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type& result_multiVector,
                                        std::size_t offset,
                                        std::size_t subSize) 
    {
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
//from here, for overlappings
/*        cudaStream_t stream_compute_1, stream_compute_2;
        cudaStream_t stream_comm_1, stream_comm_2; 
        cublasHandle_t cublasH_task1, cublasH_task2; 
        CHECK_CUBLAS_ERROR(cublasCreate(&cublasH_task1));
        CHECK_CUBLAS_ERROR(cublasCreate(&cublasH_task2));

        CHECK_CUDA_ERROR(cudaStreamCreate(&stream_compute_1));
        CHECK_CUDA_ERROR(cudaStreamCreate(&stream_compute_2));
        CHECK_CUDA_ERROR(cudaStreamCreate(&stream_comm_1));
        CHECK_CUDA_ERROR(cudaStreamCreate(&stream_comm_2));

        CHECK_CUBLAS_ERROR(cublasSetStream(cublasH_task1, stream_compute_1));
        CHECK_CUBLAS_ERROR(cublasSetStream(cublasH_task2, stream_compute_2));


        std::size_t subSize_firstPart = subSize / 2;
        std::size_t subSize_secondPart = subSize_firstPart + (subSize % 2);
        std::size_t offset_firstPart = offset;
        std::size_t offset_secondPart = subSize_firstPart + offset;

        cudaEvent_t computeDoneEvent1, computeDoneEvent2;
        CHECK_CUDA_ERROR(cudaEventCreate(&computeDoneEvent1));
        CHECK_CUDA_ERROR(cudaEventCreate(&computeDoneEvent2));

        if constexpr (InputCommType == chase::distMultiVector::CommunicatorType::column)
        {
            if (coords[0] != 0)
            {
                beta_tmp = T(0.0); // If not the first row, set beta_tmp to 0
            }
            else
            {
                beta_tmp = *beta; // If the first row, use the provided beta value
            }

            // Perform the matrix multiplication using BLAS
            
            CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(cublasH_task1,
                                                                    CUBLAS_OP_C,
                                                                    CUBLAS_OP_N,
                                                                    blockMatrix.l_cols(),
                                                                    subSize_firstPart,
                                                                    blockMatrix.l_rows(), 
                                                                    alpha,
                                                                    blockMatrix.l_data(),
                                                                    blockMatrix.l_ld(),
                                                                    input_multiVector.l_data() + offset_firstPart * input_multiVector.l_ld(),
                                                                    input_multiVector.l_ld(),
                                                                    &beta_tmp,
                                                                    result_multiVector.l_data() + offset_firstPart * result_multiVector.l_ld(),
                                                                    result_multiVector.l_ld()));            

            CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(cublasH_task2,
                                                                    CUBLAS_OP_C,
                                                                    CUBLAS_OP_N,
                                                                    blockMatrix.l_cols(),
                                                                    subSize_secondPart,
                                                                    blockMatrix.l_rows(), 
                                                                    alpha,
                                                                    blockMatrix.l_data(),
                                                                    blockMatrix.l_ld(),
                                                                    input_multiVector.l_data() + offset_secondPart * input_multiVector.l_ld(),
                                                                    input_multiVector.l_ld(),
                                                                    &beta_tmp,
                                                                    result_multiVector.l_data() + offset_secondPart * result_multiVector.l_ld(),
                                                                    result_multiVector.l_ld()));            

            CHECK_CUDA_ERROR(cudaEventRecord(computeDoneEvent1, stream_compute_1));
            CHECK_CUDA_ERROR(cudaEventRecord(computeDoneEvent2, stream_compute_2));

            CHECK_CUDA_ERROR(cudaStreamWaitEvent(stream_comm_1, computeDoneEvent1, 0));
            // Perform reduction across the column communicator
            CHECK_NCCL_ERROR(chase::Impl::nccl::ncclAllReduceWrapper<T>(result_multiVector.l_data() + offset_firstPart * result_multiVector.l_ld(),  
                                                                        result_multiVector.l_data() + offset_firstPart * result_multiVector.l_ld(),  
                                                                        result_multiVector.l_ld() * subSize_firstPart, 
                                                                        ncclSum, 
                                                                        input_multiVector.getMpiGrid()->get_nccl_col_comm(), &stream_comm_1));   

            // Perform reduction across the column communicator
            CHECK_CUDA_ERROR(cudaStreamWaitEvent(stream_comm_2, computeDoneEvent2, 0));
            CHECK_NCCL_ERROR(chase::Impl::nccl::ncclAllReduceWrapper<T>(result_multiVector.l_data() + offset_secondPart * result_multiVector.l_ld(),  
                                                                        result_multiVector.l_data() + offset_secondPart * result_multiVector.l_ld(),  
                                                                        result_multiVector.l_ld() * subSize_secondPart, 
                                                                        ncclSum, 
                                                                        input_multiVector.getMpiGrid()->get_nccl_col_comm(), &stream_comm_2));   


            CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_comm_1));
            CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_comm_2));
                                                                                        
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
            CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(cublasH_task1,
                                                                    CUBLAS_OP_N,
                                                                    CUBLAS_OP_N,
                                                                    blockMatrix.l_rows(),
                                                                    subSize_firstPart,
                                                                    blockMatrix.l_cols(), 
                                                                    alpha,
                                                                    blockMatrix.l_data(),
                                                                    blockMatrix.l_ld(),
                                                                    input_multiVector.l_data() + offset_firstPart * input_multiVector.l_ld(), 
                                                                    input_multiVector.l_ld(),
                                                                    &beta_tmp,
                                                                    result_multiVector.l_data() + offset_firstPart * result_multiVector.l_ld(),
                                                                    result_multiVector.l_ld()));               


            CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(cublasH_task2,
                                                                    CUBLAS_OP_N,
                                                                    CUBLAS_OP_N,
                                                                    blockMatrix.l_rows(),
                                                                    subSize_secondPart,
                                                                    blockMatrix.l_cols(), 
                                                                    alpha,
                                                                    blockMatrix.l_data(),
                                                                    blockMatrix.l_ld(),
                                                                    input_multiVector.l_data() + offset_secondPart * input_multiVector.l_ld(), 
                                                                    input_multiVector.l_ld(),
                                                                    &beta_tmp,
                                                                    result_multiVector.l_data() + offset_secondPart * result_multiVector.l_ld(),
                                                                    result_multiVector.l_ld()));               

            CHECK_CUDA_ERROR(cudaEventRecord(computeDoneEvent1, stream_compute_1));
            CHECK_CUDA_ERROR(cudaEventRecord(computeDoneEvent2, stream_compute_2));

            CHECK_CUDA_ERROR(cudaStreamWaitEvent(stream_comm_1, computeDoneEvent1, 0));

            // Perform reduction across the row communicator
            CHECK_NCCL_ERROR(chase::Impl::nccl::ncclAllReduceWrapper<T>(result_multiVector.l_data() + offset_firstPart * result_multiVector.l_ld(),  
                                                                        result_multiVector.l_data() + offset_firstPart * result_multiVector.l_ld(),  
                                                                        result_multiVector.l_ld() * subSize_firstPart, 
                                                                        ncclSum, 
                                                                        input_multiVector.getMpiGrid()->get_nccl_row_comm(), &stream_comm_1));
            
            CHECK_CUDA_ERROR(cudaStreamWaitEvent(stream_comm_2, computeDoneEvent2, 0));            
            // Perform reduction across the row communicator
            CHECK_NCCL_ERROR(chase::Impl::nccl::ncclAllReduceWrapper<T>(result_multiVector.l_data() + offset_secondPart * result_multiVector.l_ld(),  
                                                                        result_multiVector.l_data() + offset_secondPart * result_multiVector.l_ld(),  
                                                                        result_multiVector.l_ld() * subSize_secondPart, 
                                                                        ncclSum, 
                                                                        input_multiVector.getMpiGrid()->get_nccl_row_comm(), &stream_comm_2));

            CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_comm_1));
            CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_comm_2));
        }        
*/
                
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

            // Perform reduction across the column communicator
            CHECK_NCCL_ERROR(chase::Impl::nccl::ncclAllReduceWrapper<T>(result_multiVector.l_data() + offset * result_multiVector.l_ld(),  
                                                                        result_multiVector.l_data() + offset * result_multiVector.l_ld(),  
                                                                        result_multiVector.l_ld() * subSize, 
                                                                        ncclSum, 
                                                                        input_multiVector.getMpiGrid()->get_nccl_col_comm()));                   
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
            CHECK_NCCL_ERROR(chase::Impl::nccl::ncclAllReduceWrapper<T>(result_multiVector.l_data() + offset * result_multiVector.l_ld(),  
                                                                        result_multiVector.l_data() + offset * result_multiVector.l_ld(),  
                                                                        result_multiVector.l_ld() * subSize, 
                                                                        ncclSum, 
                                                                        input_multiVector.getMpiGrid()->get_nccl_row_comm()));                           
        }
       
    }

    template <typename T, typename MatrixType, typename InputMultiVectorType>
    void MatrixMultiplyMultiVectors(cublasHandle_t cublas_handle, T* alpha,
                                        MatrixType& blockMatrix,
                                        InputMultiVectorType& input_multiVector,
                                        T* beta,
                                        typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type& result_multiVector) 
    {
        MatrixMultiplyMultiVectors(cublas_handle,
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
    void MatrixMultiplyMultiVectorsAndRedistributeAsync(cublasHandle_t cublas_handle,
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
            CHECK_NCCL_ERROR(chase::Impl::nccl::ncclAllReduceWrapper<T>(result_multiVector.l_data() + offset * result_multiVector.l_ld(),  
                                                                        result_multiVector.l_data() + offset * result_multiVector.l_ld(),  
                                                                        result_multiVector.l_ld() * subSize, 
                                                                        ncclSum, 
                                                                        input_multiVector.getMpiGrid()->get_nccl_col_comm()));                                               
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
            CHECK_NCCL_ERROR(chase::Impl::nccl::ncclAllReduceWrapper<T>(result_multiVector.l_data() + offset * result_multiVector.l_ld(),  
                                                                        result_multiVector.l_data() + offset * result_multiVector.l_ld(),  
                                                                        result_multiVector.l_ld() * subSize, 
                                                                        ncclSum, 
                                                                        input_multiVector.getMpiGrid()->get_nccl_row_comm()));                           
        }

        src_multiVector.redistributeImplAsync(&target_multiVector, offset, subSize); //computation of gemm, and redistribute can be overlapped, wiil do later

    }

    template <typename MatrixType, typename InputMultiVectorType>    
    void MatrixMultiplyMultiVectorsAndRedistributeAsync(cublasHandle_t cublas_handle,
                                        MatrixType& blockMatrix, 
                                        InputMultiVectorType& input_multiVector, 
                                        typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type& result_multiVector,
                                        InputMultiVectorType& src_multiVector,   
                                        typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type& target_multiVector)                                                                          
    {

        MatrixMultiplyMultiVectorsAndRedistributeAsync(cublas_handle,
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
}
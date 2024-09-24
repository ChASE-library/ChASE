#pragma once

#include <mpi.h>
#include "linalg/blaspp/blaspp.hpp"
#include "linalg/lapackpp/lapackpp.hpp"
#include "linalg/matrix/distMatrix.hpp"
#include "linalg/matrix/distMultiVector.hpp"

namespace chase
{
namespace linalg
{
namespace internal
{
namespace mpi
{
    template <typename T, chase::distMultiVector::CommunicatorType InputCommType>
    void BlockBlockMultiplyMultiVectors(T *alpha,
                                        chase::distMatrix::BlockBlockMatrix<T>& blockMatrix, 
                                        chase::distMultiVector::DistMultiVector1D<T, InputCommType>& input_multiVector, 
                                        T *beta,
                                        chase::distMultiVector::DistMultiVector1D<T, 
                                                                                  chase::distMultiVector::OutputCommType<InputCommType>::value>& result_multiVector,
                                        std::size_t offset,
                                        std::size_t subSize) 
    {
        if (input_multiVector.l_rows() != (InputCommType == chase::distMultiVector::CommunicatorType::row 
                                            ? blockMatrix.l_cols() 
                                            : blockMatrix.l_rows())) {
            throw std::runtime_error("Dimension mismatch: Input multiVector rows must match blockMatrix rows or columns.");
        }


        if (result_multiVector.l_rows() != (InputCommType == chase::distMultiVector::CommunicatorType::row 
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
            chase::linalg::blaspp::t_gemm(CblasColMajor, 
                                          CblasConjTrans, 
                                          CblasNoTrans, 
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
                                          result_multiVector.l_ld());

            // Perform reduction across the column communicator
            MPI_Allreduce(MPI_IN_PLACE, 
                         result_multiVector.l_data() + offset * result_multiVector.l_ld(), 
                         result_multiVector.l_ld() * subSize,
                         chase::mpi::getMPI_Type<T>(), 
                         MPI_SUM, 
                         input_multiVector.getMpiGrid()->get_col_comm());                      
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
            chase::linalg::blaspp::t_gemm(CblasColMajor, 
                                          CblasNoTrans, 
                                          CblasNoTrans, 
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
                                          result_multiVector.l_ld());

            // Perform reduction across the row communicator
            MPI_Allreduce(MPI_IN_PLACE, 
                          result_multiVector.l_data() + offset * result_multiVector.l_ld(), 
                          result_multiVector.l_ld() * subSize,
                          chase::mpi::getMPI_Type<T>(), 
                          MPI_SUM, 
                          input_multiVector.getMpiGrid()->get_row_comm());
        }
       
    }

    template <typename T, chase::distMultiVector::CommunicatorType InputCommType>
    void BlockBlockMultiplyMultiVectors(T *alpha,
                                        chase::distMatrix::BlockBlockMatrix<T>& blockMatrix, 
                                        chase::distMultiVector::DistMultiVector1D<T, InputCommType>& input_multiVector, 
                                        T *beta,
                                        chase::distMultiVector::DistMultiVector1D<T, 
                                                                                  chase::distMultiVector::OutputCommType<InputCommType>::value>& result_multiVector) 
    {
        BlockBlockMultiplyMultiVectors(alpha,
                                       blockMatrix, 
                                       input_multiVector, 
                                       beta,
                                       result_multiVector,
                                       0,
                                       input_multiVector.l_cols());
    }

    //this operation do: W1<-1.0 * H * V1, while redistribute V2 to W2
    template <typename T, chase::distMultiVector::CommunicatorType InputCommType>
    void BlockBlockMultiplyMultiVectorsAndRedistributeAsync(
                                        chase::distMatrix::BlockBlockMatrix<T>& blockMatrix, 
                                        chase::distMultiVector::DistMultiVector1D<T, InputCommType>& input_multiVector, 
                                        chase::distMultiVector::DistMultiVector1D<T, 
                                                                                  chase::distMultiVector::OutputCommType<InputCommType>::value>& result_multiVector,
                                        chase::distMultiVector::DistMultiVector1D<T, InputCommType>& src_multiVector,   
                                        chase::distMultiVector::DistMultiVector1D<T, 
                                                                                  chase::distMultiVector::OutputCommType<InputCommType>::value>& target_multiVector,                                                                               
                                        std::size_t offset,
                                        std::size_t subSize)                                                                                   

    {
        if (input_multiVector.l_rows() != (InputCommType == chase::distMultiVector::CommunicatorType::row 
                                            ? blockMatrix.l_cols() 
                                            : blockMatrix.l_rows())) {
            throw std::runtime_error("Dimension mismatch: Result multiVector rows must match blockMatrix rows or columns.");
        }


        if (result_multiVector.l_rows() != (InputCommType == chase::distMultiVector::CommunicatorType::row 
                                            ? blockMatrix.l_rows() 
                                            : blockMatrix.l_cols())) {
            throw std::runtime_error("Dimension mismatch: Result multiVector rows must match blockMatrix rows or columns.");
        }

        if (src_multiVector.l_rows() != (InputCommType == chase::distMultiVector::CommunicatorType::row 
                                            ? blockMatrix.l_cols() 
                                            : blockMatrix.l_rows())) {
            throw std::runtime_error("Dimension mismatch: Result multiVector rows must match blockMatrix rows or columns.");
        }


        if (target_multiVector.l_rows() != (InputCommType == chase::distMultiVector::CommunicatorType::row 
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
        
        if constexpr (InputCommType == chase::distMultiVector::CommunicatorType::column)
        {
            // Perform the matrix multiplication using BLAS
            chase::linalg::blaspp::t_gemm(CblasColMajor, 
                                          CblasConjTrans, 
                                          CblasNoTrans, 
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
                                          result_multiVector.l_ld());

            // Perform reduction across the column communicator
            MPI_Allreduce(MPI_IN_PLACE, 
                         result_multiVector.l_data() + offset * result_multiVector.l_ld(), 
                         result_multiVector.l_ld() * subSize,
                         chase::mpi::getMPI_Type<T>(), 
                         MPI_SUM, 
                         input_multiVector.getMpiGrid()->get_col_comm());                      
        }
        else // InputCommType is CommunicatorType::column
        {
            // Perform the matrix multiplication using BLAS
            chase::linalg::blaspp::t_gemm(CblasColMajor, 
                                          CblasNoTrans, 
                                          CblasNoTrans, 
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
                                          result_multiVector.l_ld());

            // Perform reduction across the row communicator
            MPI_Allreduce(MPI_IN_PLACE, 
                          result_multiVector.l_data() + offset * result_multiVector.l_ld(), 
                          result_multiVector.l_ld() * subSize,
                          chase::mpi::getMPI_Type<T>(), 
                          MPI_SUM, 
                          input_multiVector.getMpiGrid()->get_row_comm());
        }

        src_multiVector.redistributeImpl(&target_multiVector, offset, subSize); //computation of gemm, and redistribute can be overlapped, wiil do later

    }

    template <typename T, chase::distMultiVector::CommunicatorType InputCommType>
    void BlockBlockMultiplyMultiVectorsAndRedistributeAsync(
                                        chase::distMatrix::BlockBlockMatrix<T>& blockMatrix, 
                                        chase::distMultiVector::DistMultiVector1D<T, InputCommType>& input_multiVector, 
                                        chase::distMultiVector::DistMultiVector1D<T, 
                                                                                  chase::distMultiVector::OutputCommType<InputCommType>::value>& result_multiVector,
                                        chase::distMultiVector::DistMultiVector1D<T, InputCommType>& src_multiVector,   
                                        chase::distMultiVector::DistMultiVector1D<T, 
                                                                                  chase::distMultiVector::OutputCommType<InputCommType>::value>& target_multiVector)
    {
        BlockBlockMultiplyMultiVectorsAndRedistributeAsync(blockMatrix, 
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
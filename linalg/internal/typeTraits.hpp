#pragma once

#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"

// Helper struct to deduce InputCommType from the multi-vector type
template <typename MultiVectorType>
struct ExtractCommType;

template <typename T, chase::distMultiVector::CommunicatorType CommType, typename Platform>
struct ExtractCommType<chase::distMultiVector::DistMultiVector1D<T, CommType, Platform>> {
    static constexpr chase::distMultiVector::CommunicatorType value = CommType;
};

template <typename T, chase::distMultiVector::CommunicatorType CommType, typename Platform>
struct ExtractCommType<chase::distMultiVector::DistMultiVectorBlockCyclic1D<T, CommType, Platform>> {
    static constexpr chase::distMultiVector::CommunicatorType value = CommType;
};


// Helper struct to flip the communicator type
template <chase::distMultiVector::CommunicatorType CommType>
struct FlipCommType;

template <>
struct FlipCommType<chase::distMultiVector::CommunicatorType::row> {
    static constexpr chase::distMultiVector::CommunicatorType value = chase::distMultiVector::CommunicatorType::column;
};

template <>
struct FlipCommType<chase::distMultiVector::CommunicatorType::column> {
    static constexpr chase::distMultiVector::CommunicatorType value = chase::distMultiVector::CommunicatorType::row;
};

// Helper trait to deduce the correct result multi-vector type
template <typename MatrixType, typename InputMultiVectorType>
struct ResultMultiVectorType;

// Specialization for BlockBlockMatrix and DistMultiVector1D
template <typename T, chase::distMultiVector::CommunicatorType CommType, typename Platform>
struct ResultMultiVectorType<chase::distMatrix::BlockBlockMatrix<T, Platform>,
                            chase::distMultiVector::DistMultiVector1D<T, CommType, Platform>> {
    // Flip the communicator type and deduce result multi-vector type
    using type = chase::distMultiVector::DistMultiVector1D<T, FlipCommType<CommType>::value, Platform>;
};

// Specialization for BlockCyclicMatrix and DistMultiVectorBlockCyclic1D
template <typename T, chase::distMultiVector::CommunicatorType CommType, typename Platform>
struct ResultMultiVectorType<chase::distMatrix::BlockCyclicMatrix<T, Platform>,
                            chase::distMultiVector::DistMultiVectorBlockCyclic1D<T, CommType, Platform>> {
    // Flip the communicator type and deduce result multi-vector type
    using type = chase::distMultiVector::DistMultiVectorBlockCyclic1D<T, FlipCommType<CommType>::value, Platform>;
};

template <typename MatrixType>
struct ColumnMultiVectorType;

// Specialization for BlockBlockMatrix
template <typename T, typename Platform>    
struct ColumnMultiVectorType<chase::distMatrix::BlockBlockMatrix<T, Platform>>
{
    using type = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::column, Platform>;
};

// Specialization for BlockCyclicMatrix
template <typename T, typename Platform>    
struct ColumnMultiVectorType<chase::distMatrix::BlockCyclicMatrix<T, Platform>>
{
    using type = chase::distMultiVector::DistMultiVectorBlockCyclic1D<T, chase::distMultiVector::CommunicatorType::column, Platform>;
};

template <typename MatrixType>
struct RowMultiVectorType;

// Specialization for BlockBlockMatrix
template <typename T, typename Platform>    
struct RowMultiVectorType<chase::distMatrix::BlockBlockMatrix<T, Platform>>
{
    using type = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::row, Platform>;
};

// Specialization for BlockCyclicMatrix
template <typename T, typename Platform>    
struct RowMultiVectorType<chase::distMatrix::BlockCyclicMatrix<T, Platform>>
{
    using type = chase::distMultiVector::DistMultiVectorBlockCyclic1D<T, chase::distMultiVector::CommunicatorType::row, Platform>;
};


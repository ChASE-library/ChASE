// This file is a part of ChASE.
// Copyright (c) 2015-2024, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"
/**
 * @defgroup MultiVectorHelperTraits Multi-Vector Helper Traits
 *
 * This group contains helper traits and structs for deducing the correct
 * communicator type and result multi-vector types based on matrix and
 * multi-vector types in the distributed matrix and multi-vector system.
 */

/**
 * @ingroup MultiVectorHelperTraits
 * @brief Helper struct to deduce the communicator type from the multi-vector
 * type.
 *
 * This struct is used to extract the communicator type (`CommType`) from a
 * given multi-vector type, allowing the system to handle various communicator
 * types in a flexible manner.
 */
template <typename MultiVectorType>
struct ExtractCommType;

/**
 * @ingroup MultiVectorHelperTraits
 * @brief Specialization for `DistMultiVector1D`.
 *
 * Specialization of `ExtractCommType` for the `DistMultiVector1D` type, which
 * extracts the communicator type from the provided multi-vector type.
 */
template <typename T, chase::distMultiVector::CommunicatorType CommType,
          typename Platform>
struct ExtractCommType<
    chase::distMultiVector::DistMultiVector1D<T, CommType, Platform>>
{
    static constexpr chase::distMultiVector::CommunicatorType value = CommType;
};
/**
 * @ingroup MultiVectorHelperTraits
 * @brief Specialization for `DistMultiVectorBlockCyclic1D`.
 *
 * Specialization of `ExtractCommType` for the `DistMultiVectorBlockCyclic1D`
 * type, which extracts the communicator type from the provided multi-vector
 * type.
 */
template <typename T, chase::distMultiVector::CommunicatorType CommType,
          typename Platform>
struct ExtractCommType<
    chase::distMultiVector::DistMultiVectorBlockCyclic1D<T, CommType, Platform>>
{
    static constexpr chase::distMultiVector::CommunicatorType value = CommType;
};

/**
 * @ingroup MultiVectorHelperTraits
 * @brief Helper struct to flip the communicator type.
 *
 * This struct provides a way to "flip" the communicator type (from row to
 * column and vice versa) for multi-vectors in distributed systems.
 */
template <chase::distMultiVector::CommunicatorType CommType>
struct FlipCommType;
/**
 * @ingroup MultiVectorHelperTraits
 * @brief Specialization to flip from row to column communicator type.
 */
template <>
struct FlipCommType<chase::distMultiVector::CommunicatorType::row>
{
    static constexpr chase::distMultiVector::CommunicatorType value =
        chase::distMultiVector::CommunicatorType::column;
};
/**
 * @ingroup MultiVectorHelperTraits
 * @brief Specialization to flip from column to row communicator type.
 */
template <>
struct FlipCommType<chase::distMultiVector::CommunicatorType::column>
{
    static constexpr chase::distMultiVector::CommunicatorType value =
        chase::distMultiVector::CommunicatorType::row;
};

/**
 * @ingroup MultiVectorHelperTraits
 * @brief Helper trait to deduce the result multi-vector type based on input
 * matrix and multi-vector types.
 *
 * This trait is used to deduce the correct result multi-vector type after
 * performing operations involving matrix and multi-vector types. It handles
 * communicator type flipping where necessary.
 */
template <typename MatrixType, typename InputMultiVectorType>
struct ResultMultiVectorType;

/**
 * @ingroup MultiVectorHelperTraits
 * @brief Specialization for `BlockBlockMatrix` and `DistMultiVector1D`.
 *
 * This specialization defines the result multi-vector type for operations
 * involving a `BlockBlockMatrix` and a `DistMultiVector1D` multi-vector type.
 * It flips the communicator type as part of the deduction.
 */
template <typename T, chase::distMultiVector::CommunicatorType CommType,
          typename Platform>
struct ResultMultiVectorType<
    chase::distMatrix::BlockBlockMatrix<T, Platform>,
    chase::distMultiVector::DistMultiVector1D<T, CommType, Platform>>
{
    // Flip the communicator type and deduce result multi-vector type
    using type = chase::distMultiVector::DistMultiVector1D<
        T, FlipCommType<CommType>::value, Platform>;
};

/**
 * @ingroup MultiVectorHelperTraits
 * @brief Specialization for `PseudoHermitianBlockBlockMatrix` and
 * `DistMultiVector1D`.
 *
 * This specialization defines the result multi-vector type for operations
 * involving a `PseudoHermitianBlockBlockMatrix` and a `DistMultiVector1D`
 * multi-vector type. It flips the communicator type as part of the deduction.
 */
template <typename T, chase::distMultiVector::CommunicatorType CommType,
          typename Platform>
struct ResultMultiVectorType<
    chase::distMatrix::PseudoHermitianBlockBlockMatrix<T, Platform>,
    chase::distMultiVector::DistMultiVector1D<T, CommType, Platform>>
{
    // Flip the communicator type and deduce result multi-vector type
    using type = chase::distMultiVector::DistMultiVector1D<
        T, FlipCommType<CommType>::value, Platform>;
};

/**
 * @ingroup MultiVectorHelperTraits
 * @brief Specialization for `BlockCyclicMatrix` and
 * `DistMultiVectorBlockCyclic1D`.
 *
 * This specialization defines the result multi-vector type for operations
 * involving a `BlockCyclicMatrix` and a `DistMultiVectorBlockCyclic1D`
 * multi-vector type. It flips the communicator type as part of the deduction.
 */
template <typename T, chase::distMultiVector::CommunicatorType CommType,
          typename Platform>
struct ResultMultiVectorType<
    chase::distMatrix::BlockCyclicMatrix<T, Platform>,
    chase::distMultiVector::DistMultiVectorBlockCyclic1D<T, CommType, Platform>>
{
    // Flip the communicator type and deduce result multi-vector type
    using type = chase::distMultiVector::DistMultiVectorBlockCyclic1D<
        T, FlipCommType<CommType>::value, Platform>;
};

/**
 * @ingroup MultiVectorHelperTraits
 * @brief Specialization for `PseudoHermitianBlockCyclicMatrix` and
 * `DistMultiVectorBlockCyclic1D`.
 *
 * This specialization defines the result multi-vector type for operations
 * involving a `PseudoHermitianBlockCyclicMatrix` and a `DistMultiVector1D`
 * multi-vector type. It flips the communicator type as part of the deduction.
 */
template <typename T, chase::distMultiVector::CommunicatorType CommType,
          typename Platform>
struct ResultMultiVectorType<
    chase::distMatrix::PseudoHermitianBlockCyclicMatrix<T, Platform>,
    chase::distMultiVector::DistMultiVectorBlockCyclic1D<T, CommType, Platform>>
{
    // Flip the communicator type and deduce result multi-vector type
    using type = chase::distMultiVector::DistMultiVectorBlockCyclic1D<
        T, FlipCommType<CommType>::value, Platform>;
};

/**
 * @ingroup MultiVectorHelperTraits
 * @brief Deduction of column multi-vector type based on matrix type.
 *
 * This struct defines the correct column multi-vector type for a given matrix
 * type. It is used to define the multi-vector communicator for the column-wise
 * operations.
 */
template <typename MatrixType>
struct ColumnMultiVectorType;

/**
 * @ingroup MultiVectorHelperTraits
 * @brief Specialization for `BlockBlockMatrix`.
 *
 * This specialization defines the column multi-vector type for the
 * `BlockBlockMatrix` type, which corresponds to a `DistMultiVector1D` with a
 * column communicator.
 */
template <typename T, typename Platform>
struct ColumnMultiVectorType<chase::distMatrix::BlockBlockMatrix<T, Platform>>
{
    using type = chase::distMultiVector::DistMultiVector1D<
        T, chase::distMultiVector::CommunicatorType::column, Platform>;
};

/**
 * @ingroup MultiVectorHelperTraits
 * @brief Specialization for `PseudoHermitianBlockBlockMatrix`.
 *
 * This specialization defines the column multi-vector type for the
 * `PseudoHermitianBlockBlockMatrix` type, which corresponds to a
 * `DistMultiVector1D` with a column communicator.
 */
template <typename T, typename Platform>
struct ColumnMultiVectorType<
    chase::distMatrix::PseudoHermitianBlockBlockMatrix<T, Platform>>
{
    using type = chase::distMultiVector::DistMultiVector1D<
        T, chase::distMultiVector::CommunicatorType::column, Platform>;
};

/**
 * @ingroup MultiVectorHelperTraits
 * @brief Specialization for `BlockCyclicMatrix`.
 *
 * This specialization defines the column multi-vector type for the
 * `BlockCyclicMatrix` type, which corresponds to a
 * `DistMultiVectorBlockCyclic1D` with a column communicator.
 */
template <typename T, typename Platform>
struct ColumnMultiVectorType<chase::distMatrix::BlockCyclicMatrix<T, Platform>>
{
    using type = chase::distMultiVector::DistMultiVectorBlockCyclic1D<
        T, chase::distMultiVector::CommunicatorType::column, Platform>;
};

/**
 * @ingroup MultiVectorHelperTraits
 * @brief Specialization for `PseudoHermitianBlockCyclicMatrix`.
 *
 * This specialization defines the column multi-vector type for the
 * `PseudoHermitianBlockCyclicMatrix` type, which corresponds to a
 * `DistMultiVector1D` with a column communicator.
 */
template <typename T, typename Platform>
struct ColumnMultiVectorType<
    chase::distMatrix::PseudoHermitianBlockCyclicMatrix<T, Platform>>
{
    using type = chase::distMultiVector::DistMultiVectorBlockCyclic1D<
        T, chase::distMultiVector::CommunicatorType::column, Platform>;
};

/**
 * @ingroup MultiVectorHelperTraits
 * @brief Deduction of row multi-vector type based on matrix type.
 *
 * This struct defines the correct row multi-vector type for a given matrix
 * type. It is used to define the multi-vector communicator for the row-wise
 * operations.
 */
template <typename MatrixType>
struct RowMultiVectorType;

/**
 * @ingroup MultiVectorHelperTraits
 * @brief Specialization for `BlockBlockMatrix`.
 *
 * This specialization defines the row multi-vector type for the
 * `BlockBlockMatrix` type, which corresponds to a `DistMultiVector1D` with a
 * row communicator.
 */
template <typename T, typename Platform>
struct RowMultiVectorType<chase::distMatrix::BlockBlockMatrix<T, Platform>>
{
    using type = chase::distMultiVector::DistMultiVector1D<
        T, chase::distMultiVector::CommunicatorType::row, Platform>;
};

/**
 * @ingroup MultiVectorHelperTraits
 * @brief Specialization for `PseudoHermitianBlockBlockMatrix`.
 *
 * This specialization defines the row multi-vector type for the
 * `PseudoHermitianBlockBlockMatrix` type, which corresponds to a
 * `DistMultiVector1D` with a row communicator.
 */
template <typename T, typename Platform>
struct RowMultiVectorType<
    chase::distMatrix::PseudoHermitianBlockBlockMatrix<T, Platform>>
{
    using type = chase::distMultiVector::DistMultiVector1D<
        T, chase::distMultiVector::CommunicatorType::row, Platform>;
};

/**
 * @ingroup MultiVectorHelperTraits
 * @brief Specialization for `BlockCyclicMatrix`.
 *
 * This specialization defines the row multi-vector type for the
 * `BlockCyclicMatrix` type, which corresponds to a
 * `DistMultiVectorBlockCyclic1D` with a row communicator.
 */
template <typename T, typename Platform>
struct RowMultiVectorType<chase::distMatrix::BlockCyclicMatrix<T, Platform>>
{
    using type = chase::distMultiVector::DistMultiVectorBlockCyclic1D<
        T, chase::distMultiVector::CommunicatorType::row, Platform>;
};

/**
 * @ingroup MultiVectorHelperTraits
 * @brief Specialization for `PseudoHermitianBlockCyclicMatrix`.
 *
 * This specialization defines the row multi-vector type for the
 * `PseudoHermitianBlockCyclicMatrix` type, which corresponds to a
 * `DistMultiVector1D` with a row communicator.
 */
template <typename T, typename Platform>
struct RowMultiVectorType<
    chase::distMatrix::PseudoHermitianBlockCyclicMatrix<T, Platform>>
{
    using type = chase::distMultiVector::DistMultiVectorBlockCyclic1D<
        T, chase::distMultiVector::CommunicatorType::row, Platform>;
};

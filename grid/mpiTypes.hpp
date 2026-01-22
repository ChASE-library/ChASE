// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include <mpi.h>

/**
 * @page chase_mpi_namespace chase::mpi Namespace
 * @brief Namespace for MPI utilities and type mappings.
 *
 *
 * The `chase::mpi` namespace contains functions and templates used for
 * dealing with MPI types and ensuring compatibility between C++ and MPI
 * data types.
 *
 * This includes specialized template functions to return the correct MPI
 * type for specific C++ data types like `float`, `double`, and complex
 * types `std::complex<float>` and `std::complex<double>`.
 */

/**
 * @defgroup mpi_type_mappings MPI Type Mappings
 *
 * @brief Functions for mapping C++ types to MPI types.
 *
 * This group includes functions that provide the corresponding MPI data
 * types for various C++ data types.
 */
namespace chase
{
namespace mpi
{
/**
 * @ingroup mpi_type_mappings
 * @brief Template function to get the MPI data type corresponding to a C++
 * type.
 *
 * This function is specialized for different C++ types to return the correct
 * MPI data type, allowing compatibility with MPI operations.
 *
 * @tparam T The C++ data type for which the MPI type is required.
 *
 * @return The corresponding MPI data type.
 */
template <typename T>
MPI_Datatype getMPI_Type();

/**
 * @ingroup mpi_type_mappings
 * @brief Specialization for `float` to return the corresponding MPI data type.
 *
 * @return `MPI_FLOAT` corresponding to the `float` type.
 */
template <>
MPI_Datatype getMPI_Type<float>()
{
    return MPI_FLOAT;
}

/**
 * @ingroup mpi_type_mappings
 * @brief Specialization for `double` to return the corresponding MPI data type.
 *
 * @return `MPI_DOUBLE` corresponding to the `double` type.
 */
template <>
MPI_Datatype getMPI_Type<double>()
{
    return MPI_DOUBLE;
}

/**
 * @ingroup mpi_type_mappings
 * @brief Specialization for `std::complex<float>` to return the corresponding
 * MPI data type.
 *
 * @return `MPI_COMPLEX` corresponding to the `std::complex<float>` type.
 */
template <>
MPI_Datatype getMPI_Type<std::complex<float>>()
{
    return MPI_COMPLEX;
}
/**
 * @ingroup mpi_type_mappings
 * @brief Specialization for `std::complex<double>` to return the corresponding
 * MPI data type.
 *
 * @return `MPI_DOUBLE_COMPLEX` corresponding to the `std::complex<double>`
 * type.
 */
template <>
MPI_Datatype getMPI_Type<std::complex<double>>()
{
    return MPI_DOUBLE_COMPLEX;
}

} // namespace mpi
} // namespace chase

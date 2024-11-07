/* -*- Mode: C++; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
// This file is a part of ChASE.
// Copyright (c) 2015-2023, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#ifndef CHASE_ALGORITHM_TYPES_HPP
#define CHASE_ALGORITHM_TYPES_HPP

#include <complex>
#include <fstream>
#include <functional>
#include <sys/stat.h>

namespace chase
{
/**
 * @defgroup PrecisionTraits Precision Traits
 * Traits for converting between single and double precision types.
 * @{
 */

/**
 * @brief Trait for converting a type to its single-precision equivalent.
 * 
 * For standard types, the type remains unchanged. Specializations
 * are provided for `double` and `std::complex<double>`.
 *
 * @tparam T The type to be converted to single precision.
 */

template<typename T>
struct ToSinglePrecisionTrait {
    using Type = T;  // By default, the type remains unchanged
};

/**
 * @brief Specialization for converting `double` to `float`.
 */
template<>
struct ToSinglePrecisionTrait<double> {
    using Type = float;  // Single precision equivalent of double
};

// Specialization for std::complex<double>
/**
 * @brief Specialization for converting `std::complex<double>` to `std::complex<float>`.
 */
template<>
struct ToSinglePrecisionTrait<std::complex<double>> {
    using Type = std::complex<float>;  // Single precision equivalent of std::complex<double>
};

// Trait for converting a type to its double precision equivalent
/**
 * @brief Trait for converting a type to its double-precision equivalent.
 * 
 * For standard types, the type remains unchanged. Specializations
 * are provided for `float` and `std::complex<float>`.
 *
 * @tparam T The type to be converted to double precision.
 */
template<typename T>
struct ToDoublePrecisionTrait {
    using Type = T;  // By default, the type remains unchanged
};

// Specialization for float
/**
 * @brief Specialization for converting `float` to `double`.
 */
template<>
struct ToDoublePrecisionTrait<float> {
    using Type = double;  // Double precision equivalent of float
};

// Specialization for std::complex<float>
/**
 * @brief Specialization for converting `std::complex<float>` to `std::complex<double>`.
 */
template<>
struct ToDoublePrecisionTrait<std::complex<float>> {
    using Type = std::complex<double>;  // Double precision equivalent of std::complex<float>
};
/** @} */ // end of PrecisionTraits group

/**
 * @brief Converts a value to its single-precision equivalent, if applicable.
 * 
 * @tparam T The type of the value to be converted.
 * @param val The value to be converted to single precision.
 * @return The single-precision equivalent of `val`.
 */
template<typename T>
typename ToSinglePrecisionTrait<T>::Type convertToSinglePrecision(const T& val) {
    return static_cast<typename ToSinglePrecisionTrait<T>::Type>(val);
}

/**
 * @brief Converts a value to its double-precision equivalent, if applicable.
 * 
 * @tparam T The type of the value to be converted.
 * @param val The value to be converted to double precision.
 * @return The double-precision equivalent of `val`.
 */
template<typename T>
typename ToDoublePrecisionTrait<T>::Type convertToDoublePrecision(const T& val) {
    return static_cast<typename ToDoublePrecisionTrait<T>::Type>(val);
}

/**
 * @defgroup BaseConversion Base Type Conversion
 * Helper templates to extract the base type from complex types.
 * @{
 */

/**
 * @brief Primary template for extracting the base type.
 * 
 * @tparam Q The type to extract the base type from.
 */
template <class Q>
struct Base_Class
{
    typedef Q type; ///< Defines the base type as `Q`
};

/**
 * @brief Specialization for `std::complex`, extracting the underlying type `Q`.
 * 
 * @tparam Q The base type of the complex type.
 */
template <class Q>
struct Base_Class<std::complex<Q>>
{
    typedef Q type; ///< The underlying type of `std::complex<Q>`
};

/**
 * @brief Alias for `Base_Class` that simplifies access to the underlying type.
 * 
 * @tparam Q The type to extract the base type from.
 */
template <typename Q>
using Base = typename Base_Class<Q>::type;
/** @} */ // end of BaseConversion group

/**
 * @ingroup PrecisionTraits
 * @brief Primary template for converting a type to its lower precision equivalent.
 * 
 * Provides single-precision equivalents for specific types.
 * @tparam T The type to convert.
 */
template<typename T>
struct PrecisionTrait {
    using Type = T;  ///< By default, the type remains unchanged
};

/**
 * @brief Specialization for converting `double` to `float`.
 */
template<>
struct PrecisionTrait<double> {
    using Type = float;  ///< Single precision equivalent of `double`
};

/**
 * @brief Specialization for converting `std::complex<double>` to `std::complex<float>`.
 */
template<>
struct PrecisionTrait<std::complex<double>> {
    using Type = std::complex<float>;  ///< Single precision equivalent of `std::complex<double>`
};

} // namespace chase

/**
 * @brief Checks if a given file path exists.
 * 
 * @param s The file path as a string.
 * @return `true` if the path exists, `false` otherwise.
 */
bool isPathExist(const std::string& s)
{
    struct stat buffer;
    return (stat(s.c_str(), &buffer) == 0);
}

/**
 * @defgroup RandomGenerators Random Generators
 * Utilities for generating random values of various types.
 * @{
 */

/**
 * @brief Generates a random value of type `T` using a provided generator function.
 * 
 * @tparam T The type of value to generate.
 * @param f A generator function that produces a random `double`.
 * @return A random value of type `T`.
 */
template <typename T>
T getRandomT(std::function<double(void)> f);

/**
 * @brief Specialization for generating a random `double`.
 * 
 * @param f A generator function that produces a random `double`.
 * @return A random `double` value.
 */
template <>
double getRandomT(std::function<double(void)> f)
{
    return double(f());
}

/**
 * @brief Specialization for generating a random `float`.
 * 
 * @param f A generator function that produces a random `double`.
 * @return A random `float` value.
 */
template <>
float getRandomT(std::function<double(void)> f)
{
    return float(f());
}

/**
 * @brief Specialization for generating a random `std::complex<double>`.
 * 
 * @param f A generator function that produces a random `double`.
 * @return A random `std::complex<double>` value.
 */
template <>
std::complex<double> getRandomT(std::function<double(void)> f)
{
    return std::complex<double>(f(), f());
}

/**
 * @brief Specialization for generating a random `std::complex<float>`.
 * 
 * @param f A generator function that produces a random `double`.
 * @return A random `std::complex<float>` value.
 */
template <>
std::complex<float> getRandomT(std::function<double(void)> f)
{
    return std::complex<float>(f(), f());
}

/** @} */ // end of RandomGenerators group

/**
 * @brief Computes the conjugate of a scalar value.
 * 
 * This function computes the conjugate of a scalar value, supporting both real and complex types.
 * 
 * @tparam T The type of the scalar value.
 * @param scalar The scalar value to compute the conjugate for.
 * @return The conjugate of the scalar.
 * 
 * @note For real numbers, the conjugate is the same as the original value.
 */
template<typename T>
T conjugate(const T& scalar) {
    static_assert(std::is_arithmetic<T>::value || std::is_same<T, std::complex<float>>::value || std::is_same<T, std::complex<double>>::value,
                  "Type must be float, double, std::complex<float> or std::complex<double>");    
    if constexpr (std::is_arithmetic<T>::value) {
        return scalar; // For real scalars, conjugate is the same as the
// original value
    } else {
        return std::conj(scalar); // For complex scalars, use std::conj
    }
}

#endif

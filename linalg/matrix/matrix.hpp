// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "algorithm/types.hpp"
#include "external/cpu/lapackpp/lapackpp.hpp"
#include <chrono>      // For std::chrono
#include <complex>    // std::complex (if you're using complex types)
#include <cstddef>    // std::size_t
#include <functional> // std::function
#include <memory>     // std::unique_ptr, std::allocator
#include <omp.h>
#include <stdexcept>   // std::runtime_error
#include <type_traits> // std::enable_if, std::is_same

/**
 * @page chase_namespace chase Namespace
 * @brief Top-level namespace for all components related to chase.
 *
 * This namespace contains the core classes and functions for chase
 */

/**
 * @page matrix_namespace chase::matrix Namespace
 * @brief Namespace containing matrix classes, allocators, and enumerations.
 *
 * The `chase::matrix` namespace includes CPU and GPU matrix classes,
 * allocator types, and enumeration for managing memory locations.
 */

/**
 * @defgroup MatrixClasses Matrix Classes
 * @brief Matrix classes for CPU and GPU with memory management and
 * mixed-precision support.
 */

/**
 * @defgroup Allocators Memory Allocators
 * @brief Allocators for CPU and GPU memory management.
 */

/**
 * @defgroup BufferTypes Buffer Types and Utilities
 * @brief Enumeration and utilities for handling CPU and GPU buffers.
 */

/**
 * @page platform_namespace chase::platform Namespace
 * @brief Namespace containing platform-specific structures and identifiers.
 *
 * The `chase::platform` namespace includes identifiers for different platforms,
 * such as CPU and GPU.
 */

/**
 * @defgroup PlatformTypes Platform Types
 * @brief Represents different platforms (CPU, GPU) used in matrix operations.
 */

namespace chase
{
namespace platform
{
/**
 * @ingroup PlatformTypes
 * @brief Represents the CPU platform.
 *
 * This structure is used to specify that an operation or matrix class
 * is targeting the CPU platform.
 */
struct CPU
{
};
/**
 * @ingroup PlatformTypes
 * @brief Represents the GPU platform.
 *
 * This structure is used to specify that an operation or matrix class
 * is targeting the GPU platform.
 */
struct GPU
{
};
} // namespace platform
} // namespace chase

#if defined(HAS_CUDA) || defined(HAS_HIP)
#include "matrix_gpu_runtime.hpp"
#endif

namespace chase
{
namespace matrix
{
struct Hermitian
{
};
struct PseudoHermitian
{
};
} // namespace matrix
} // namespace chase

/**
 * @page platform_namespace chase::platform Namespace
 * @brief Namespace containing platform-specific structures and identifiers.
 *
 * The `chase::platform` namespace includes identifiers for different platforms,
 * such as CPU and GPU.
 */

/** @page chase_matrix
 *  @brief The Matrix Namespace
 *
 * This namespace contains matrix classes that work with different platforms.
 * Each matrix type is designed to operate on either CPU or GPU platforms.
 */
namespace chase
{
namespace matrix
{

/**
 * @ingroup BufferTypes
 * @brief Specifies the type of external buffer used for matrix data.
 */
enum class BufferType
{
    CPU, ///< Indicates that the external buffer is in CPU memory
    GPU  ///< Indicates that the external buffer is in GPU memory
};

/**
 * @ingroup Allocators
 * @brief Default allocator type trait for CPU platform.
 *
 * Uses std::allocator on CPU.
 * @tparam Platform The platform type (CPU).
 */
template <typename Platform>
struct DefaultAllocator
{
    template <typename T>
    /// Uses std::allocator as default for CPU platform.
    using type = std::allocator<T>;
};

#if defined(HAS_CUDA) || defined(HAS_HIP)
template <>
struct DefaultAllocator<chase::platform::GPU>
{
    template <typename T>
    using type = gpu_runtime::GpuAllocator<T>;
};
#endif

template <typename T, typename Platform = chase::platform::CPU,
          template <typename> class Allocator =
              DefaultAllocator<Platform>::template type>
class Matrix;

/**
 * @ingroup MatrixClasses
 * @brief Abstract base class template for matrix objects.
 *
 * Provides the interface for derived CPU and GPU matrix classes,
 * supporting operations like data access, dimensions, and binary I/O.
 *
 * @tparam T Data type of matrix elements.
 * @tparam Derived Derived matrix class.
 * @tparam Platform Platform type (CPU or GPU).
 * @tparam Allocator Memory allocator type.
 */
template <
    typename T,
    template <typename, typename, template <typename> class> class Derived,
    typename Platform, template <typename> class Allocator>
class AbstractMatrix
{
protected:
    using SinglePrecisionType = typename chase::ToSinglePrecisionTrait<T>::Type;
    using SinglePrecisionDerived =
        Derived<SinglePrecisionType, Platform, Allocator>;
    std::unique_ptr<SinglePrecisionDerived> single_precision_matrix_;
    bool is_single_precision_enabled_ = false;

    // Double precision matrix
    using DoublePrecisionType = typename chase::ToDoublePrecisionTrait<T>::Type;
    using DoublePrecisionDerived =
        Derived<DoublePrecisionType, Platform, Allocator>;
    std::unique_ptr<DoublePrecisionDerived> double_precision_matrix_;
    bool is_double_precision_enabled_ = false;

    std::chrono::high_resolution_clock::time_point start, end;
    std::chrono::duration<double> elapsed;

public:
    using value_type = T;
    using platform_type = Platform;
    virtual ~AbstractMatrix() = default;
    /// Returns a pointer to the matrix data.
    virtual T* data() = 0;
    /// Returns a pointer to the CPU data.
    virtual T* cpu_data() = 0;
    /// Returns the number of rows in the matrix.
    virtual std::size_t rows() const = 0;
    /// Returns the number of columns in the matrix.
    virtual std::size_t cols() const = 0;
    /// Returns the leading dimension of the matrix.
    virtual std::size_t ld() const = 0;
    /// Returns the leading dimension of the CPU data.
    virtual std::size_t cpu_ld() const = 0;
    /**
     * Saves the matrix data to a binary file.
     * @param filename Path to the output file.
     * @throws std::runtime_error if the file cannot be opened or data is
     * uninitialized.
     */
    void saveToBinaryFile(const std::string& filename)
    {
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open())
        {
            throw std::runtime_error(
                "[Matrix]: Failed to open file for writing.");
        }

        if (this->cpu_data() == nullptr)
        {
            throw std::runtime_error(
                "[Matrix]: Original CPU data is not initialized.");
        }

        std::size_t dataSize = this->rows() * this->cols() * sizeof(T);
        if (this->cpu_ld() > this->rows())
        {
            std::vector<T> buffer(this->rows() * this->cols());
            chase::linalg::lapackpp::t_lacpy('A', this->rows(), this->cols(),
                                             this->cpu_data(), this->cpu_ld(),
                                             buffer.data(), this->rows());
            file.write(reinterpret_cast<const char*>(buffer.data()), dataSize);
        }
        else
        {
            file.write(reinterpret_cast<const char*>(this->cpu_data()),
                       dataSize);
        }
    }

    /**
     * Reads matrix data from a binary file.
     * @param filename Path to the input file.
     * @throws std::runtime_error if the file cannot be opened, or its size is
     * insufficient.
     */
    void readFromBinaryFile(const std::string& filename)
    {
        std::ifstream file(filename, std::ios::binary | std::ios::in);
        if (!file.is_open())
        {
            throw std::runtime_error(
                "[Matrix]: Failed to open file for reading.");
        }

        if (this->cpu_data() == nullptr)
        {
            throw std::runtime_error(
                "[Matrix]: Original CPU data is not initialized.");
        }

        // Check file size
        file.seekg(0, std::ios::end);
        std::streamsize fileSize = file.tellg();
        file.seekg(0, std::ios::beg);

        std::streamsize requiredSize = this->rows() * this->cols() * sizeof(T);
        if (fileSize < requiredSize)
        {
            throw std::runtime_error("[Matrix]: File size is smaller than the "
                                     "required matrix size.");
        }

        if (this->cpu_ld() > this->rows())
        {
            std::vector<T> buffer(this->rows() * this->cols());
            file.read(reinterpret_cast<char*>(buffer.data()), requiredSize);
            chase::linalg::lapackpp::t_lacpy('A', this->rows(), this->cols(),
                                             buffer.data(), this->rows(),
                                             this->cpu_data(), this->cpu_ld());
        }
        else
        {
            file.read(reinterpret_cast<char*>(this->cpu_data()), requiredSize);
        }
    }

    /**
     * Enables single-precision representation for supported data types.
     * @tparam U Enables this function for double or complex double only.
     * @throws std::runtime_error if the single-precision matrix is
     * uninitialized.
     */
    template <typename U = T,
              typename std::enable_if<
                  std::is_same<U, double>::value ||
                      std::is_same<U, std::complex<double>>::value,
                  int>::type = 0>
    void enableSinglePrecision()
    {
        if (!single_precision_matrix_)
        {
            single_precision_matrix_ = std::make_unique<SinglePrecisionDerived>(
                this->rows(), this->cols());
        }

        if constexpr (std::is_same<Platform, chase::platform::CPU>::value)
        {
#pragma omp parallel for
            for (std::size_t j = 0; j < this->cols(); ++j)
            {
                for (std::size_t i = 0; i < this->rows(); ++i)
                {
                    single_precision_matrix_
                        ->data()[j * single_precision_matrix_->ld() + i] =
                        static_cast<SinglePrecisionType>(
                            this->data()[j * this->ld() + i]);
                }
            }
        }
#ifdef CHASE_GPU_MATRIX
        else
        {
            chase::matrix::gpu_runtime::convert_dp_to_sp(
                this->data(), single_precision_matrix_->data(),
                this->rows() * this->cols());
        }
#endif
        is_single_precision_enabled_ = true;
    }

    /**
     * Disables single-precision representation.
     * @tparam U Enables this function for double or complex double only.
     * @param copyback If true, copies data back to original precision.
     * @throws std::runtime_error if single precision is not enabled or
     * uninitialized.
     */
    template <typename U = T,
              typename std::enable_if<
                  std::is_same<U, double>::value ||
                      std::is_same<U, std::complex<double>>::value,
                  int>::type = 0>
    void disableSinglePrecision(bool copyback = false)
    {
        if (!is_single_precision_enabled_)
        {
            throw std::runtime_error("Single precision is not enabled.");
        }

        if (single_precision_matrix_ == nullptr)
        {
            throw std::runtime_error("Original SP data is not initialized.");
        }

        if (copyback)
        {
            if constexpr (std::is_same<Platform, chase::platform::CPU>::value)
            {
                // Convert single-precision data back to original precision
#pragma omp parallel for
                for (std::size_t j = 0; j < this->cols(); ++j)
                {
                    for (std::size_t i = 0; i < this->rows(); ++i)
                    {
                        this->data()[j * this->ld() + i] = static_cast<T>(
                            single_precision_matrix_
                                ->data()[j * single_precision_matrix_->ld() +
                                         i]);
                    }
                }
            }
#ifdef CHASE_GPU_MATRIX
            else
            {
                chase::matrix::gpu_runtime::convert_sp_to_dp(
                    single_precision_matrix_->data(), this->data(),
                    this->rows() * this->cols());
            }
#endif
        }

        single_precision_matrix_.reset();
        is_single_precision_enabled_ = false;
    }

    // Check if single precision is enabled
    template <typename U = T,
              typename std::enable_if<
                  std::is_same<U, double>::value ||
                      std::is_same<U, std::complex<double>>::value,
                  int>::type = 0>
    bool isSinglePrecisionEnabled() const
    {
        return is_single_precision_enabled_;
    }

    // Get the single precision matrix itself
    template <typename U = T,
              typename std::enable_if<
                  std::is_same<U, double>::value ||
                      std::is_same<U, std::complex<double>>::value,
                  int>::type = 0>
    SinglePrecisionDerived* matrix_sp()
    {
        if (is_single_precision_enabled_)
        {
            return single_precision_matrix_.get();
        }
        else
        {
            throw std::runtime_error("Single precision is not enabled.");
        }
    }

    // If T is already single precision, these methods should not be available
    template <typename U = T,
              typename std::enable_if<
                  !std::is_same<U, double>::value &&
                      !std::is_same<U, std::complex<double>>::value,
                  int>::type = 0>
    void enableSinglePrecision()
    {
        throw std::runtime_error("[DistMatrix]: Single precision operations "
                                 "not supported for this type.");
    }

    template <typename U = T,
              typename std::enable_if<
                  !std::is_same<U, double>::value &&
                      !std::is_same<U, std::complex<double>>::value,
                  int>::type = 0>
    void disableSinglePrecision()
    {
        throw std::runtime_error("[DistMatrix]: Single precision operations "
                                 "not supported for this type.");
    }

    // Enable double precision for float types (and complex<float>)
    template <
        typename U = T,
        typename std::enable_if<std::is_same<U, float>::value ||
                                    std::is_same<U, std::complex<float>>::value,
                                int>::type = 0>
    void enableDoublePrecision()
    {
        if (!double_precision_matrix_)
        {
            start = std::chrono::high_resolution_clock::now();
            double_precision_matrix_ = std::make_unique<DoublePrecisionDerived>(
                this->rows(), this->cols());
        }

        if constexpr (std::is_same<Platform, chase::platform::CPU>::value)
        {
#pragma omp parallel for
            for (std::size_t j = 0; j < this->cols(); ++j)
            {
                for (std::size_t i = 0; i < this->rows(); ++i)
                {
                    double_precision_matrix_
                        ->data()[j * double_precision_matrix_->ld() + i] =
                        static_cast<DoublePrecisionType>(
                            this->data()[j * this->ld() + i]);
                }
            }
        }
#ifdef CHASE_GPU_MATRIX
        else
        {
            chase::matrix::gpu_runtime::convert_sp_to_dp(
                this->data(), double_precision_matrix_->data(),
                this->rows() * this->cols());
        }
#endif
        is_double_precision_enabled_ = true;
        end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(
            end - start);
    }

    // Disable double precision for float types
    template <
        typename U = T,
        typename std::enable_if<std::is_same<U, float>::value ||
                                    std::is_same<U, std::complex<float>>::value,
                                int>::type = 0>
    void disableDoublePrecision(bool copyback = false)
    {
        if (!is_double_precision_enabled_)
        {
            throw std::runtime_error("Double precision is not enabled.");
        }

        if (double_precision_matrix_ == nullptr)
        {
            throw std::runtime_error("Original DP data is not initialized.");
        }

        if (copyback)
        {
            if constexpr (std::is_same<Platform, chase::platform::CPU>::value)
            {
#pragma omp parallel for
                for (std::size_t j = 0; j < this->cols(); ++j)
                {
                    for (std::size_t i = 0; i < this->rows(); ++i)
                    {
                        this->data()[j * this->ld() + i] = static_cast<T>(
                            double_precision_matrix_
                                ->data()[j * double_precision_matrix_->ld() +
                                         i]);
                    }
                }
            }
#ifdef CHASE_GPU_MATRIX
            else
            {
                chase::matrix::gpu_runtime::convert_dp_to_sp(
                    double_precision_matrix_->data(), this->data(),
                    this->rows() * this->cols());
            }
#endif
        }

        double_precision_matrix_.reset();
        is_double_precision_enabled_ = false;
    }

    // Check if double precision is enabled
    template <
        typename U = T,
        typename std::enable_if<std::is_same<U, float>::value ||
                                    std::is_same<U, std::complex<float>>::value,
                                int>::type = 0>
    bool isDoublePrecisionEnabled() const
    {
        return is_double_precision_enabled_;
    }

    // Get the double precision matrix itself
    template <
        typename U = T,
        typename std::enable_if<std::is_same<U, float>::value ||
                                    std::is_same<U, std::complex<float>>::value,
                                int>::type = 0>
    DoublePrecisionDerived* matrix_dp()
    {
        if (is_double_precision_enabled_)
        {
            return double_precision_matrix_.get();
        }
        else
        {
            throw std::runtime_error("Double precision is not enabled.");
        }
    }
    
    template <
        typename U = T,
        typename std::enable_if<std::is_same<U, float>::value ||
                                    std::is_same<U, std::complex<float>>::value,
                                int>::type = 0>
    DoublePrecisionDerived* getDoublePrecisionMatrix()
    {
        return matrix_dp();
    }

#ifdef CHASE_GPU_MATRIX
    template <
        typename U = T,
        typename std::enable_if<std::is_same<U, float>::value ||
                                    std::is_same<U, std::complex<float>>::value,
                                int>::type = 0>
    void enableDoublePrecisionAsync(
        gpu_runtime::chase_gpu_stream_ptr_t stream_ = nullptr)
    {
        if (!double_precision_matrix_)
        {
            start = std::chrono::high_resolution_clock::now();
            double_precision_matrix_ = std::make_unique<DoublePrecisionDerived>(
                this->rows(), this->cols());
        }

        if constexpr (std::is_same<Platform, chase::platform::GPU>::value)
        {
            chase::matrix::gpu_runtime::convert_sp_to_dp(
                this->data(), double_precision_matrix_->data(),
                this->rows() * this->cols(), stream_);
        }
        else
        {
            throw std::runtime_error(
                "GPU async double-precision conversion requires GPU support.");
        }
        is_double_precision_enabled_ = true;
        end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(
            end - start);
    }

    template <
        typename U = T,
        typename std::enable_if<std::is_same<U, float>::value ||
                                    std::is_same<U, std::complex<float>>::value,
                                int>::type = 0>
    void disableDoublePrecisionAsync(
        bool copyback = false,
        gpu_runtime::chase_gpu_stream_ptr_t stream_ = nullptr)
    {
        if (!is_double_precision_enabled_)
        {
            throw std::runtime_error("Double precision is not enabled.");
        }

        if (double_precision_matrix_ == nullptr)
        {
            throw std::runtime_error("Original DP data is not initialized.");
        }

        if (copyback)
        {
            if constexpr (std::is_same<Platform, chase::platform::GPU>::value)
            {
                chase::matrix::gpu_runtime::convert_dp_to_sp(
                    double_precision_matrix_->data(), this->data(),
                    this->rows() * this->cols(), stream_);
            }
            else
            {
                throw std::runtime_error(
                    "GPU async double-precision conversion requires GPU support.");
            }
        }

        double_precision_matrix_.reset();
        is_double_precision_enabled_ = false;
    }

    template <
        typename U = T,
        typename std::enable_if<std::is_same<U, float>::value ||
                                    std::is_same<U, std::complex<float>>::value,
                                int>::type = 0>
    void copyToSubBlockAsync(std::size_t offset, std::size_t size,
                             gpu_runtime::chase_gpu_stream_ptr_t stream_ =
                                 nullptr)
    {
        if (!double_precision_matrix_)
        {
            throw std::runtime_error("Double precision matrix is not enabled.");
        }

        const std::size_t max_size = this->ld() * this->cols();
        if (offset + size > max_size)
        {
            throw std::runtime_error(
                "copyToSubBlockAsync range exceeds matrix storage.");
        }

        if constexpr (std::is_same<Platform, chase::platform::GPU>::value)
        {
            chase::matrix::gpu_runtime::convert_sp_to_dp(
                this->data() + offset, double_precision_matrix_->data() + offset,
                size, stream_);
        }
        else
        {
            throw std::runtime_error(
                "GPU async sub-block copy requires GPU support.");
        }
    }

    template <
        typename U = T,
        typename std::enable_if<std::is_same<U, float>::value ||
                                    std::is_same<U, std::complex<float>>::value,
                                int>::type = 0>
    void copyToSubBlock(std::size_t offset, std::size_t size)
    {
        copyToSubBlockAsync(offset, size, nullptr);
    }

    template <
        typename U = T,
        typename std::enable_if<std::is_same<U, float>::value ||
                                    std::is_same<U, std::complex<float>>::value,
                                int>::type = 0>
    void copyBackSubBlockAsync(std::size_t offset, std::size_t size,
                               gpu_runtime::chase_gpu_stream_ptr_t stream_ =
                                   nullptr)
    {
        if (!double_precision_matrix_)
        {
            throw std::runtime_error("Double precision matrix is not enabled.");
        }

        const std::size_t max_size = this->ld() * this->cols();
        if (offset + size > max_size)
        {
            throw std::runtime_error(
                "copyBackSubBlockAsync range exceeds matrix storage.");
        }

        if constexpr (std::is_same<Platform, chase::platform::GPU>::value)
        {
            chase::matrix::gpu_runtime::convert_dp_to_sp(
                double_precision_matrix_->data() + offset, this->data() + offset,
                size, stream_);
        }
        else
        {
            throw std::runtime_error(
                "GPU async sub-block copy requires GPU support.");
        }
    }

    template <
        typename U = T,
        typename std::enable_if<std::is_same<U, float>::value ||
                                    std::is_same<U, std::complex<float>>::value,
                                int>::type = 0>
    void copyBackSubBlock(std::size_t offset, std::size_t size)
    {
        copyBackSubBlockAsync(offset, size, nullptr);
    }
#endif
    // If T is already double precision, these methods should not be available
    template <typename U = T,
              typename std::enable_if<
                  !std::is_same<U, float>::value &&
                      !std::is_same<U, std::complex<float>>::value,
                  int>::type = 0>
    void enableDoublePrecision()
    {
        throw std::runtime_error("[Matrix]: Double precision operations not "
                                 "supported for this type.");
    }

    template <typename U = T,
              typename std::enable_if<
                  !std::is_same<U, float>::value &&
                      !std::is_same<U, std::complex<float>>::value,
                  int>::type = 0>
    void disableDoublePrecision()
    {
        throw std::runtime_error("[Matrix]: Double precision operations not "
                                 "supported for this type.");
    }

    // Copy from double to single precision (if T is double)
    template <typename U = T,
              typename std::enable_if<
                  std::is_same<U, double>::value ||
                      std::is_same<U, std::complex<double>>::value,
                  int>::type = 0>
    void copyTo()
    {
        if (!single_precision_matrix_)
        {
            throw std::runtime_error("Single precision matrix is not enabled.");
        }

        if constexpr (std::is_same<Platform, chase::platform::CPU>::value)
        {
#pragma omp parallel for
            for (std::size_t j = 0; j < this->cols(); ++j)
            {
                for (std::size_t i = 0; i < this->rows(); ++i)
                {
                    single_precision_matrix_
                        ->data()[j * single_precision_matrix_->ld() + i] =
                        static_cast<SinglePrecisionType>(
                            this->data()[j * this->ld() + i]);
                }
            }
        }
#ifdef CHASE_GPU_MATRIX
        else
        {
            chase::matrix::gpu_runtime::convert_dp_to_sp(
                this->data(), single_precision_matrix_->data(),
                this->rows() * this->cols());
        }
#endif
    }

    // Copy from single to double precision (if T is double)
    template <typename U = T,
              typename std::enable_if<
                  std::is_same<U, double>::value ||
                      std::is_same<U, std::complex<double>>::value,
                  int>::type = 0>
    void copyBack()
    {
        if (!single_precision_matrix_)
        {
            throw std::runtime_error("Single precision matrix is not enabled.");
        }

        if constexpr (std::is_same<Platform, chase::platform::CPU>::value)
        {
#pragma omp parallel for
            for (std::size_t j = 0; j < this->cols(); ++j)
            {
                for (std::size_t i = 0; i < this->rows(); ++i)
                {
                    this->data()[j * this->ld() + i] = static_cast<T>(
                        single_precision_matrix_
                            ->data()[j * single_precision_matrix_->ld() + i]);
                }
            }
        }
#ifdef CHASE_GPU_MATRIX
        else
        {
            chase::matrix::gpu_runtime::convert_sp_to_dp(
                single_precision_matrix_->data(), this->data(),
                this->rows() * this->cols());
        }
#endif
    }

    // Copy from single to double precision (if T is float)
    template <
        typename U = T,
        typename std::enable_if<std::is_same<U, float>::value ||
                                    std::is_same<U, std::complex<float>>::value,
                                int>::type = 0>
    void copyTo()
    {
        if (!double_precision_matrix_)
        {
            throw std::runtime_error("Double precision matrix is not enabled.");
        }

        if constexpr (std::is_same<Platform, chase::platform::CPU>::value)
        {
#pragma omp parallel for
            for (std::size_t j = 0; j < this->cols(); ++j)
            {
                for (std::size_t i = 0; i < this->rows(); ++i)
                {
                    double_precision_matrix_
                        ->data()[j * double_precision_matrix_->ld() + i] =
                        static_cast<DoublePrecisionType>(
                            this->data()[j * this->ld() + i]);
                }
            }
        }
#ifdef CHASE_GPU_MATRIX
        else
        {
            chase::matrix::gpu_runtime::convert_sp_to_dp(
                this->data(), double_precision_matrix_->data(),
                this->rows() * this->cols());
        }
#endif
    }

    // Copy from double to single precision (if T is float)
    template <
        typename U = T,
        typename std::enable_if<std::is_same<U, float>::value ||
                                    std::is_same<U, std::complex<float>>::value,
                                int>::type = 0>
    void copyBack()
    {
        if (!double_precision_matrix_)
        {
            throw std::runtime_error("Double precision matrix is not enabled.");
        }

        if constexpr (std::is_same<Platform, chase::platform::CPU>::value)
        {
#pragma omp parallel for
            for (std::size_t j = 0; j < this->cols(); ++j)
            {
                for (std::size_t i = 0; i < this->rows(); ++i)
                {
                    this->data()[j * this->ld() + i] = static_cast<T>(
                        double_precision_matrix_
                            ->data()[j * double_precision_matrix_->ld() + i]);
                }
            }
        }
#ifdef CHASE_GPU_MATRIX
        else
        {
            chase::matrix::gpu_runtime::convert_dp_to_sp(
                double_precision_matrix_->data(), this->data(),
                this->rows() * this->cols());
        }
#endif
    }
};

/**
 * @ingroup MatrixClasses
 * @brief A matrix class for CPU-based matrix operations.
 *
 * This class represents a matrix stored in CPU memory, with support for memory
 * allocation and deallocation, as well as external data handling. It also
 * supports move semantics and memory management via custom allocators.
 *
 * @tparam T Data type of matrix elements.
 * @tparam Allocator Memory allocator type (default is std::allocator).
 */
template <typename T, template <typename> class Allocator>
class Matrix<T, chase::platform::CPU, Allocator>
    : public AbstractMatrix<T, Matrix, chase::platform::CPU, Allocator>
{
private:
    std::unique_ptr<T[], std::function<void(T*)>>
        data_; ///< Smart pointer to manage matrix data with custom deleter.
    T* external_data_; ///< Pointer to external data, if provided by the user.
    std::size_t rows_; ///< Number of rows in the matrix.
    std::size_t cols_; ///< Number of columns in the matrix.
    std::size_t ld_; ///< Leading dimension of the matrix (number of elements in
                     ///< a column).
    bool owns_mem_;  ///< Flag to indicate whether the matrix owns its memory.
    Allocator<T> allocator_; ///< Memory allocator for matrix elements.

public:
    using hermitian_type =
        chase::matrix::Hermitian; ///< Alias for the Hermitian matrix type.

    /**
     * @brief Default constructor.
     *
     * Initializes an empty matrix with no allocated memory.
     */
    Matrix()
        : data_(nullptr, [](T*) {}), external_data_(nullptr), rows_(0),
          cols_(0), ld_(0), owns_mem_(false)
    {
    }

    /**
     * @brief Constructor to allocate a matrix with specified dimensions.
     *
     * Allocates memory using the provided allocator and sets the matrix size to
     * the specified number of rows and columns. The matrix is initialized with
     * zeros.
     *
     * @param rows Number of rows in the matrix.
     * @param cols Number of columns in the matrix.
     */
    Matrix(std::size_t rows, std::size_t cols)
        : rows_(rows), cols_(cols), ld_(rows), owns_mem_(true)
    {
        // Allocate memory using the allocator
        T* raw_data = allocator_.allocate(rows_ * cols_);
        std::fill_n(raw_data, rows_ * cols_, T(0.0));

        data_ = std::unique_ptr<T[], std::function<void(T*)>>(
            raw_data,
            [this](T* ptr) { allocator_.deallocate(ptr, rows_ * cols_); });

        external_data_ = nullptr;
    }

    /**
     * @brief Constructor wrapping user-provided data.
     *
     * Initializes the matrix with user-supplied data, but does not take
     * ownership of the memory.
     *
     * @param rows Number of rows in the matrix.
     * @param cols Number of columns in the matrix.
     * @param ld Leading dimension of the matrix.
     * @param data Pointer to the user-provided matrix data.
     */
    Matrix(std::size_t rows, std::size_t cols, std::size_t ld, T* data)
        : data_(nullptr, [](T*) {}), external_data_(data), rows_(rows),
          cols_(cols), ld_(ld), owns_mem_(false)
    {
    }

    /**
     * @brief Move constructor.
     *
     * Transfers ownership of memory and matrix data from another matrix object.
     * The moved-from matrix will not own memory after the move.
     *
     * @param other Another matrix object to move from.
     */
    Matrix(Matrix&& other) noexcept
        : data_(std::move(other.data_)), rows_(other.rows_), cols_(other.cols_),
          ld_(other.ld_), allocator_(std::move(other.allocator_)),
          owns_mem_(other.owns_mem_)
    {
        this->single_precision_matrix_ =
            std::move(other.single_precision_matrix_);
        this->is_single_precision_enabled_ = other.is_single_precision_enabled_;
        this->double_precision_matrix_ =
            std::move(other.double_precision_matrix_);
        this->is_double_precision_enabled_ = other.is_double_precision_enabled_;
        other.owns_mem_ =
            false; // Make sure the moved-from object no longer owns the memory
    }

    /**
     * @brief Move assignment operator.
     *
     * Transfers ownership of memory and matrix data from another matrix object,
     * freeing any existing memory.
     *
     * @param other Another matrix object to move from.
     * @return Reference to this matrix object.
     */
    Matrix& operator=(Matrix&& other) noexcept
    {
        if (this != &other)
        {
            if (owns_mem_)
            {
                allocator_.deallocate(data_.get(),
                                      rows_ * cols_); // Free existing memory
            }

            data_ = std::move(other.data_); // Transfer ownership of memory
            external_data_ = other.external_data_;
            rows_ = other.rows_;
            cols_ = other.cols_;
            ld_ = other.ld_;
            allocator_ = std::move(other.allocator_);
            owns_mem_ = other.owns_mem_;
            this->single_precision_matrix_ =
                std::move(other.single_precision_matrix_);
            this->is_single_precision_enabled_ =
                other.is_single_precision_enabled_;
            this->double_precision_matrix_ =
                std::move(other.double_precision_matrix_);
            this->is_double_precision_enabled_ =
                other.is_double_precision_enabled_;
            other.single_precision_matrix_ =
                nullptr; // Transfer ownership of low precision data
            other.double_precision_matrix_ =
                nullptr; // Transfer ownership of high precision data
            other.owns_mem_ =
                false; // The moved-from object should no longer own the memory
        }
        return *this;
    }

    /**
     * @brief Swap function to exchange the data of two matrices.
     *
     * Exchanges all matrix properties (data, size, memory ownership) with
     * another matrix.
     *
     * @param other Another matrix object to swap with.
     */
    void swap(Matrix& other)
    {
        std::swap(data_, other.data_);
        std::swap(external_data_, other.external_data_);
        std::swap(rows_, other.rows_);
        std::swap(cols_, other.cols_);
        std::swap(ld_, other.ld_);
        std::swap(owns_mem_, other.owns_mem_);
        std::swap(allocator_, other.allocator_);
        std::swap(this->is_single_precision_enabled_,
                  other.is_single_precision_enabled_);
        std::swap(this->single_precision_matrix_,
                  other.single_precision_matrix_);
        std::swap(this->is_double_precision_enabled_,
                  other.is_double_precision_enabled_);
        std::swap(this->double_precision_matrix_,
                  other.double_precision_matrix_);
    }

    // Swap only the underlying data pointers/ownership for data().
    void swapDataPointer(Matrix& other)
    {
        std::swap(data_, other.data_);
        std::swap(external_data_, other.external_data_);
        std::swap(owns_mem_, other.owns_mem_);
    }

    /**
     * @brief Get a pointer to the matrix data.
     *
     * Returns either the internally allocated data or the externally provided
     * data.
     *
     * @return Pointer to the matrix data.
     */
    T* data() override { return owns_mem_ ? data_.get() : external_data_; }

    /**
     * @brief Get a pointer to the CPU data (same as data in this case).
     *
     * @return Pointer to the CPU matrix data.
     */
    T* cpu_data() override { return owns_mem_ ? data_.get() : external_data_; }

    /**
     * @brief Get the number of rows in the matrix.
     *
     * @return The number of rows.
     */
    std::size_t rows() const override { return rows_; }
    /**
     * @brief Get the number of columns in the matrix.
     *
     * @return The number of columns.
     */
    std::size_t cols() const override { return cols_; }
    /**
     * @brief Get the leading dimension of the matrix.
     *
     * @return The leading dimension.
     */
    std::size_t ld() const override { return ld_; }
    /**
     * @brief Get the leading dimension of the CPU data.
     *
     * @return The leading dimension of the CPU data.
     */
    std::size_t cpu_ld() const override { return ld_; }
};

template <typename T, typename Platform = chase::platform::CPU,
          template <typename> class Allocator =
              DefaultAllocator<Platform>::template type>
class PseudoHermitianMatrix;

template <typename T, template <typename> class Allocator>
class PseudoHermitianMatrix<T, chase::platform::CPU, Allocator>
    : public Matrix<T, chase::platform::CPU, Allocator>
{
public:
    using hermitian_type = chase::matrix::PseudoHermitian;

    PseudoHermitianMatrix() : Matrix<T, chase::platform::CPU, Allocator>() {}

    PseudoHermitianMatrix(std::size_t rows, std::size_t cols)
        : Matrix<T, chase::platform::CPU, Allocator>(rows, cols)
    {
    }

    PseudoHermitianMatrix(std::size_t rows, std::size_t cols, std::size_t ld,
                          T* data)
        : Matrix<T, chase::platform::CPU, Allocator>(rows, cols, ld, data)
    {
    }
};

#if defined(CHASE_GPU_MATRIX)
#include "matrix_gpu.hpp"
#endif

} // namespace matrix
} // namespace chase

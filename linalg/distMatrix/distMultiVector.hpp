// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include <algorithm> // For std::max_element
#include <complex>   // For std::complex types
#include <iostream>  // For std::cout
#include <memory>    // For std::unique_ptr, std::shared_ptr
#include <mpi.h>
#include <omp.h>     // For OpenMP parallelization
#include <stdexcept> // For throwing runtime errors

#include "algorithm/types.hpp"
#include "external/lapackpp/lapackpp.hpp"
#include "grid/mpiGrid2D.hpp"
#include "grid/mpiTypes.hpp"
#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/matrix/matrix.hpp"

#ifdef HAS_CUDA
#include "Impl/chase_gpu/cuda_utils.hpp"
#include "external/cublaspp/cublaspp.hpp"
#include "linalg/internal/cuda/lacpy.hpp"
#include "linalg/internal/cuda/precision_conversion.cuh"
#include <cuda_runtime.h>
#endif

#include "Impl/chase_gpu/nvtx.hpp"

/**
 * @page chase_distMultiVector_namespace chase::distMultiVector Namespace
 * The `chase::distMultiVector` namespace contains classes, structures, and
 * utilities for distributed block-vectors, including support for distributed
 * multi-vector computations and precision conversion traits.
 */

namespace chase
{
/**
 * @defgroup distMultiVector Distributed Multi-Vector class
 * @brief Classes and utilities for distributed multi-vector operations.
 * @{
 */
namespace distMultiVector
{
/**
 * @brief Enumeration for distribution types.
 *
 * Specifies the type of distribution for multi-vector data.
 */
enum class DistributionType
{
    Block,      /**< Block distribution */
    BlockCyclic /**< Block cyclic distribution */
};

/**
 * @brief Enumeration for communicator types.
 *
 * Specifies the type of communicator to be used for distributed operations.
 */
enum class CommunicatorType
{
    row,    /**< Row communicator */
    column, /**< Column communicator */
    all     /**< All communicators */
};

/*
struct row_comm {};
struct col_comm {};
*/

/**
 * @brief Distributed multi-vector with 1D distribution.
 *
 * This class template represents a distributed multi-vector with 1D data
 * distribution, templated by type, communicator type, and platform.
 */
template <typename T, CommunicatorType comm_type, typename Platform>
class DistMultiVector1D;

/**
 * @brief Distributed multi-vector with block-cyclic 1D distribution.
 *
 * This class template represents a distributed multi-vector with 1D
 * block-cyclic data distribution, templated by type, communicator type, and
 * platform.
 */
template <typename T, CommunicatorType comm_type, typename Platform>
class DistMultiVectorBlockCyclic1D;

/**
 * @brief Abstract base class for distributed multi-vectors.
 *
 * The AbstractDistMultiVector class defines an interface for distributed
 * multi-vectors, providing methods for querying distribution, communicator
 * type, and other vector properties.
 *
 * @tparam T           Data type (e.g., double, std::complex<double>)
 * @tparam comm_type   Communicator type (row, column, or all)
 * @tparam Derived     Derived class template implementing specific multi-vector
 * functionality
 * @tparam Platform    Platform type (default: chase::platform::CPU)
 */
template <typename T, CommunicatorType comm_type,
          template <typename, CommunicatorType, typename> class Derived,
          typename Platform = chase::platform::CPU>
class AbstractDistMultiVector
{
protected:
    // Single precision matrix
    using SinglePrecisionType =
        typename chase::ToSinglePrecisionTrait<T>::Type; /**< Single precision
                                                            type */
    using SinglePrecisionDerived =
        Derived<SinglePrecisionType, comm_type,
                Platform>; /**< Derived class in single precision */
    std::unique_ptr<SinglePrecisionDerived>
        single_precision_multivec_; /**< Pointer to single precision
                                       multi-vector */
    bool is_single_precision_enabled_ =
        false; /**< Flag indicating if single precision is enabled */
    std::chrono::high_resolution_clock::time_point start,
        end; /**< Timing points for performance measurement */
    std::chrono::duration<double>
        elapsed; /**< Duration for single precision operations */

    // Double precision matrix
    using DoublePrecisionType =
        typename chase::ToDoublePrecisionTrait<T>::Type; /**< Double precision
                                                           type */
    using DoublePrecisionDerived =
        Derived<DoublePrecisionType, comm_type,
                Platform>; /**< Derived class in double precision */
    std::unique_ptr<DoublePrecisionDerived>
        double_precision_multivec_; /**< Pointer to double precision
                                      multi-vector */
    bool is_double_precision_enabled_ =
        false; /**< Flag indicating if double precision is enabled */

public:
    virtual ~AbstractDistMultiVector() = default;
    /**
     * @brief Get the distribution type of the multi-vector.
     * @return DistributionType Enum value representing the distribution type.
     */
    virtual DistributionType getMultiVectorDistributionType() const = 0;
    /**
     * @brief Get the communicator type of the multi-vector.
     * @return CommunicatorType Enum value representing the communicator type.
     */
    virtual CommunicatorType getMultiVectorCommunicatorType() const = 0;
    /**
     * @brief Get the MPI grid object.
     * @return Pointer to MpiGrid2DBase representing the MPI grid.
     */
    virtual chase::grid::MpiGrid2DBase* getMpiGrid() const = 0;
    /**
     * @brief Get the MPI grid object as a shared pointer.
     * @return Shared pointer to MpiGrid2DBase representing the MPI grid.
     */
    virtual std::shared_ptr<chase::grid::MpiGrid2DBase>
    getMpiGrid_shared_ptr() const = 0;
    /**
     * @brief Get the global row count.
     * @return The number of rows in the multi-vector.
     */
    virtual std::size_t g_rows() const = 0;
    /**
     * @brief Get the global column count.
     * @return The number of columns in the multi-vector.
     */
    virtual std::size_t g_cols() const = 0;
    /**
     * @brief Get the global offset.
     * @return The global offset of the local data.
     */
    virtual std::size_t g_off() const = 0;
    /**
     * @brief Get the local row count.
     * @return The number of local rows in the multi-vector.
     */
    virtual std::size_t l_rows() const = 0;
    /**
     * @brief Get the local column count.
     * @return The number of local columns in the multi-vector.
     */
    virtual std::size_t l_cols() const = 0;
    /**
     * @brief Get the leading dimension for local storage.
     * @return The leading dimension size.
     */
    virtual std::size_t l_ld() const = 0;
    /**
     * @brief Get the pointer to local data.
     * @return Pointer to local data in the multi-vector.
     */
    virtual T* l_data() = 0;
    /**
     * @brief Get the block size for block-cyclic distribution.
     * @return Block size for distribution.
     */
    virtual std::size_t mb() const = 0;
    /**
     * @brief Returns the number of row blocks.
     *
     * @return The number of row blocks.
     */
    virtual std::size_t mblocks() const = 0;
    /**
     * @brief Get the local index for the lower half.
     * @return The local index for the lower half.
     */
    virtual std::size_t l_half() const = 0;

    // virtual typename chase::platform::MatrixTypePlatform<T, Platform>::type&
    // loc_matrix() = 0;
    /**
     * @brief Get the local matrix object.
     * @return Reference to Matrix object representing local matrix data.
     */
    virtual chase::matrix::Matrix<T, Platform>& loc_matrix() = 0;
    /**
     * @brief Get the global rank in the MPI communicator.
     * @return The rank of the process within the global MPI communicator.
     */
    int grank()
    {
        int grank = 0;
        MPI_Comm_rank(this->getMpiGrid()->get_comm(), &grank);
        return grank;
    }
    /**
     * @brief Allocate memory for local data on the CPU.
     */
    void allocate_cpu_data()
    {
        auto& loc_matrix = this->loc_matrix();
        loc_matrix.allocate_cpu_data();
    }
    // Enable single precision for double types (and complex<double>)
    /**
     * @brief Enable single precision for double types.
     */
    template <typename U = T,
              typename std::enable_if<
                  std::is_same<U, double>::value ||
                      std::is_same<U, std::complex<double>>::value,
                  int>::type = 0>
    void enableSinglePrecision()
    {
        if (!single_precision_multivec_)
        {
            start = std::chrono::high_resolution_clock::now();
            if constexpr (std::is_same<Derived<T, comm_type, Platform>,
                                       chase::distMultiVector::
                                           DistMultiVectorBlockCyclic1D<
                                               T, comm_type, Platform>>::value)
            {
                single_precision_multivec_ =
                    std::make_unique<SinglePrecisionDerived>(
                        this->g_rows(), this->g_cols(), this->mb(),
                        this->getMpiGrid_shared_ptr());
            }
            else
            {
                single_precision_multivec_ =
                    std::make_unique<SinglePrecisionDerived>(
                        this->g_rows(), this->g_cols(),
                        this->getMpiGrid_shared_ptr());
            }
            //
            if constexpr (std::is_same<Platform, chase::platform::CPU>::value)
            {
#pragma omp parallel for collapse(2) schedule(static, 16)
                for (std::size_t j = 0; j < this->l_cols(); ++j)
                {
                    for (std::size_t i = 0; i < this->l_rows(); ++i)
                    {
                        single_precision_multivec_->l_data()
                            [j * single_precision_multivec_.get()->l_ld() + i] =
                            chase::convertToSinglePrecision(
                                this->l_data()[j * this->l_ld() + i]);
                    }
                }
            }
            else
#ifdef HAS_CUDA
            {

                chase::linalg::internal::cuda::convert_DP_TO_SP_GPU(
                    this->l_data(), single_precision_multivec_->l_data(),
                    this->l_cols() * this->l_rows());
            }
#else
            {
                throw std::runtime_error(
                    "GPU is not supported in AbstractDistMultiVector");
            }
#endif
            is_single_precision_enabled_ = true;
            end = std::chrono::high_resolution_clock::now();
            elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(
                end - start);

            if (this->grank() == 0)
                std::cout << "Single precision matrix enabled in "
                             "AbstractDistMultiVector. \n";
        }
        else
        {
            throw std::runtime_error("Single precision already enabled.");
        }
    }

    // Disable single precision for double types
    /**
     * @brief Disable single precision and optionally copy data back.
     * @param copyback Whether to copy data back to double precision.
     */
    template <typename U = T,
              typename std::enable_if<
                  std::is_same<U, double>::value ||
                      std::is_same<U, std::complex<double>>::value,
                  int>::type = 0>
    void disableSinglePrecision(bool copyback = false)
    {
        start = std::chrono::high_resolution_clock::now();
        if (copyback)
        {
            if (single_precision_multivec_)
            {
                if constexpr (std::is_same<Platform,
                                           chase::platform::CPU>::value)
                {

#pragma omp parallel for collapse(2) schedule(static, 16)
                    for (std::size_t j = 0; j < this->l_cols(); ++j)
                    {
                        for (std::size_t i = 0; i < this->l_rows(); ++i)
                        {
                            this->l_data()[j * this->l_ld() + i] =
                                chase::convertToDoublePrecision<T>(
                                    single_precision_multivec_->l_data()
                                        [j * single_precision_multivec_.get()
                                                 ->l_ld() +
                                         i]);
                        }
                    }
                }
                else
#ifdef HAS_CUDA
                {

                    chase::linalg::internal::cuda::convert_SP_TO_DP_GPU(
                        single_precision_multivec_->l_data(), this->l_data(),
                        this->l_cols() * this->l_rows());
                }
#else
                {
                    throw std::runtime_error(
                        "GPU is not supported in AbstractDistMultiVector");
                }
#endif
            }
            else
            {
                throw std::runtime_error("Single precision is not enabled.");
            }
        }

        end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(
            end - start);

        if (this->grank() == 0)
            std::cout << "Single precision matrix disabled in "
                         "AbstractDistMultiVector. \n";

        single_precision_multivec_.reset(); // Free the single precision memory
        is_single_precision_enabled_ = false;
    }

    // Check if single precision is enabled
    /**
     * @brief Check if single precision is enabled.
     * @return True if single precision is enabled, false otherwise.
     */
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
    /**
     * @brief Get the single precision multi-vector.
     * @return Pointer to the single precision multi-vector.
     */
    template <typename U = T,
              typename std::enable_if<
                  std::is_same<U, double>::value ||
                      std::is_same<U, std::complex<double>>::value,
                  int>::type = 0>
    SinglePrecisionDerived* getSinglePrecisionMatrix()
    {
        if (is_single_precision_enabled_)
        {
            return single_precision_multivec_.get();
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
        throw std::runtime_error("[DistMultiVector]: Single precision "
                                 "operations not supported for this type.");
    }

    template <typename U = T,
              typename std::enable_if<
                  !std::is_same<U, double>::value &&
                      !std::is_same<U, std::complex<double>>::value,
                  int>::type = 0>
    void disableSinglePrecision()
    {
        throw std::runtime_error("[DistMultiVector]: Single precision "
                                 "operations not supported for this type.");
    }

    // Enable double precision for float types
    template <
        typename U = T,
        typename std::enable_if<std::is_same<U, float>::value ||
                                    std::is_same<U, std::complex<float>>::value,
                                int>::type = 0>
    void enableDoublePrecision()
    {
        if (!double_precision_multivec_)
        {
            start = std::chrono::high_resolution_clock::now();
            if constexpr (std::is_same<Derived<T, comm_type, Platform>,
                                       chase::distMultiVector::
                                           DistMultiVectorBlockCyclic1D<
                                               T, comm_type, Platform>>::value)
            {
                double_precision_multivec_ =
                    std::make_unique<DoublePrecisionDerived>(
                        this->g_rows(), this->g_cols(), this->mb(),
                        this->getMpiGrid_shared_ptr());
            }
            else
            {
                double_precision_multivec_ =
                    std::make_unique<DoublePrecisionDerived>(
                        this->g_rows(), this->g_cols(),
                        this->getMpiGrid_shared_ptr());
            }

            if constexpr (std::is_same<Platform, chase::platform::CPU>::value)
            {
#pragma omp parallel for collapse(2) schedule(static, 16)
                for (std::size_t j = 0; j < this->l_cols(); ++j)
                {
                    for (std::size_t i = 0; i < this->l_rows(); ++i)
                    {
                        double_precision_multivec_->l_data()
                            [j * double_precision_multivec_.get()->l_ld() + i] =
                            chase::convertToDoublePrecision(
                                this->l_data()[j * this->l_ld() + i]);
                    }
                }
            }
            else
#ifdef HAS_CUDA
            {
                chase::linalg::internal::cuda::convert_SP_TO_DP_GPU(
                    this->l_data(), double_precision_multivec_->l_data(),
                    this->l_cols() * this->l_rows());
            }
#else
            {
                throw std::runtime_error(
                    "GPU is not supported in AbstractDistMultiVector");
            }
#endif
            is_double_precision_enabled_ = true;
            end = std::chrono::high_resolution_clock::now();
            elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(
                end - start);

            if (this->grank() == 0)
                std::cout << "Double precision matrix enabled in "
                             "AbstractDistMultiVector. \n";
        }
        else
        {
            throw std::runtime_error("Double precision already enabled.");
        }
    }

    // Disable double precision for float types
    template <
        typename U = T,
        typename std::enable_if<std::is_same<U, float>::value ||
                                    std::is_same<U, std::complex<float>>::value,
                                int>::type = 0>
    void disableDoublePrecision(bool copyback = false)
    {
        start = std::chrono::high_resolution_clock::now();
        if (copyback)
        {
            if (double_precision_multivec_)
            {
                if constexpr (std::is_same<Platform,
                                           chase::platform::CPU>::value)
                {
#pragma omp parallel for collapse(2) schedule(static, 16)
                    for (std::size_t j = 0; j < this->l_cols(); ++j)
                    {
                        for (std::size_t i = 0; i < this->l_rows(); ++i)
                        {
                            this->l_data()[j * this->l_ld() + i] =
                                chase::convertToSinglePrecision<T>(
                                    double_precision_multivec_->l_data()
                                        [j * double_precision_multivec_.get()
                                                 ->l_ld() +
                                         i]);
                        }
                    }
                }
                else
#ifdef HAS_CUDA
                {
                    chase::linalg::internal::cuda::convert_DP_TO_SP_GPU(
                        double_precision_multivec_->l_data(), this->l_data(),
                        this->l_cols() * this->l_rows());
                }
#else
                {
                    throw std::runtime_error(
                        "GPU is not supported in AbstractDistMultiVector");
                }
#endif
            }
            else
            {
                throw std::runtime_error("Double precision is not enabled.");
            }
        }

        double_precision_multivec_.reset(); // Free the double precision memory
        is_double_precision_enabled_ = false;
        end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(
            end - start);

        if (this->grank() == 0)
            std::cout << "Double precision matrix disabled in "
                         "AbstractDistMultiVector. \n";
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
    DoublePrecisionDerived* getDoublePrecisionMatrix()
    {
        if (is_double_precision_enabled_)
        {
            return double_precision_multivec_.get();
        }
        else
        {
            throw std::runtime_error("Double precision is not enabled.");
        }
    }

    // If T is already double precision, these methods should not be available
    template <typename U = T,
              typename std::enable_if<
                  !std::is_same<U, float>::value &&
                      !std::is_same<U, std::complex<float>>::value,
                  int>::type = 0>
    void enableDoublePrecision()
    {
        throw std::runtime_error("[DistMultiVector]: Double precision "
                                 "operations not supported for this type.");
    }

    template <typename U = T,
              typename std::enable_if<
                  !std::is_same<U, float>::value &&
                      !std::is_same<U, std::complex<float>>::value,
                  int>::type = 0>
    void disableDoublePrecision()
    {
        throw std::runtime_error("[DistMultiVector]: Double precision "
                                 "operations not supported for this type.");
    }

    // Copy from single to double precision (if T is double)
    template <typename U = T,
              typename std::enable_if<
                  std::is_same<U, double>::value ||
                      std::is_same<U, std::complex<double>>::value,
                  int>::type = 0>
    void copyback()
    {
        start = std::chrono::high_resolution_clock::now();
        if (!single_precision_multivec_)
        {
            throw std::runtime_error("Single precision matrix is not enabled.");
        }

        if constexpr (std::is_same<Platform, chase::platform::CPU>::value)
        {
#pragma omp parallel for collapse(2) schedule(static, 16)
            for (std::size_t j = 0; j < this->l_cols(); ++j)
            {
                for (std::size_t i = 0; i < this->l_rows(); ++i)
                {
                    this->l_data()[j * this->l_ld() + i] =
                        chase::convertToDoublePrecision<T>(
                            single_precision_multivec_->l_data()
                                [j * single_precision_multivec_.get()->l_ld() +
                                 i]);
                }
            }
        }
        else
#ifdef HAS_CUDA
        {
            chase::linalg::internal::cuda::convert_SP_TO_DP_GPU(
                single_precision_multivec_->l_data(), this->l_data(),
                this->l_cols() * this->l_rows());
        }
#else
        {
            throw std::runtime_error(
                "GPU is not supported in AbstractDistMultiVector");
        }
#endif
        end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(
            end - start);

        if (this->grank() == 0)
            std::cout
                << "Single precision matrix copied back to double precision in "
                   "AbstractDistMultiVector. \n";
    }

    // Copy from double to single precision (if T is float)
    template <
        typename U = T,
        typename std::enable_if<std::is_same<U, float>::value ||
                                    std::is_same<U, std::complex<float>>::value,
                                int>::type = 0>
    void copyback()
    {
        start = std::chrono::high_resolution_clock::now();
        if (!double_precision_multivec_)
        {
            throw std::runtime_error("Double precision matrix is not enabled.");
        }

        if constexpr (std::is_same<Platform, chase::platform::CPU>::value)
        {
#pragma omp parallel for collapse(2) schedule(static, 16)
            for (std::size_t j = 0; j < this->l_cols(); ++j)
            {
                for (std::size_t i = 0; i < this->l_rows(); ++i)
                {
                    this->l_data()[j * this->l_ld() + i] = static_cast<T>(
                        double_precision_multivec_->l_data()
                            [j * double_precision_multivec_.get()->l_ld() + i]);
                }
            }
        }
        else
#ifdef HAS_CUDA
        {
            chase::linalg::internal::cuda::convert_DP_TO_SP_GPU(
                double_precision_multivec_->l_data(), this->l_data(),
                this->l_cols() * this->l_rows());
        }
#else
        {
            throw std::runtime_error(
                "GPU is not supported in AbstractDistMultiVector");
        }
#endif
        end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(
            end - start);

        if (this->grank() == 0)
            std::cout
                << "Double precision matrix copied back to single precision in "
                   "AbstractDistMultiVector. \n";
    }

    // Copy from double to single precision (if T is double)
    template <typename U = T,
              typename std::enable_if<
                  std::is_same<U, double>::value ||
                      std::is_same<U, std::complex<double>>::value,
                  int>::type = 0>
    void copyTo()
    {
        start = std::chrono::high_resolution_clock::now();
        if (!single_precision_multivec_)
        {
            throw std::runtime_error("Single precision matrix is not enabled.");
        }

        if constexpr (std::is_same<Platform, chase::platform::CPU>::value)
        {
#pragma omp parallel for collapse(2) schedule(static, 16)
            for (std::size_t j = 0; j < this->l_cols(); ++j)
            {
                for (std::size_t i = 0; i < this->l_rows(); ++i)
                {
                    single_precision_multivec_->l_data()
                        [j * single_precision_multivec_.get()->l_ld() + i] =
                        chase::convertToSinglePrecision<T>(
                            this->l_data()[j * this->l_ld() + i]);
                }
            }
        }
        else
#ifdef HAS_CUDA
        {
            chase::linalg::internal::cuda::convert_DP_TO_SP_GPU(
                this->l_data(), single_precision_multivec_->l_data(),
                this->l_cols() * this->l_rows());
        }
#else
        {
            throw std::runtime_error(
                "GPU is not supported in AbstractDistMultiVector");
        }
#endif
        end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(
            end - start);

        if (this->grank() == 0)
            std::cout
                << "Single precision matrix copied to double precision in "
                   "AbstractDistMultiVector. \n";
    }

    // Copy from single to double precision (if T is float)
    template <
        typename U = T,
        typename std::enable_if<std::is_same<U, float>::value ||
                                    std::is_same<U, std::complex<float>>::value,
                                int>::type = 0>
    void copyTo()
    {
        start = std::chrono::high_resolution_clock::now();
        if (!double_precision_multivec_)
        {
            throw std::runtime_error("Double precision matrix is not enabled.");
        }

        if constexpr (std::is_same<Platform, chase::platform::CPU>::value)
        {
#pragma omp parallel for collapse(2) schedule(static, 16)
            for (std::size_t j = 0; j < this->l_cols(); ++j)
            {
                for (std::size_t i = 0; i < this->l_rows(); ++i)
                {
                    double_precision_multivec_->l_data()
                        [j * double_precision_multivec_.get()->l_ld() + i] =
                        chase::convertToDoublePrecision<T>(
                            this->l_data()[j * this->l_ld() + i]);
                }
            }
        }
        else
#ifdef HAS_CUDA
        {
            chase::linalg::internal::cuda::convert_SP_TO_DP_GPU(
                this->l_data(), double_precision_multivec_->l_data(),
                this->l_cols() * this->l_rows());
        }
#else
        {
            throw std::runtime_error(
                "GPU is not supported in AbstractDistMultiVector");
        }
#endif
        end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(
            end - start);

        if (this->grank() == 0)
            std::cout
                << "Single precision matrix copied to double precision in "
                   "AbstractDistMultiVector. \n";
    }

    // If T is neither float nor double, these methods should not be available
    template <typename U = T,
              typename std::enable_if<
                  !std::is_same<U, float>::value &&
                      !std::is_same<U, std::complex<float>>::value &&
                      !std::is_same<U, double>::value &&
                      !std::is_same<U, std::complex<double>>::value,
                  int>::type = 0>
    void copyTo()
    {
        throw std::runtime_error("[DistMultiVector]: CopyTo operations not "
                                 "supported for this type.");
    }
};

/**
 * @brief Distributed MultiVector class for 1D block distribution within either
 * row or column communicator of a 2D MPI grid.
 *
 * This class represents a distributed matrix structure where data can be split
 * across either the row or column communicators in a 2D MPI grid setup. The
 * implementation is platform-specific, allowing for both CPU and GPU usage.
 *
 * @tparam T Element type.
 * @tparam comm_type Specifies the communicator type (row or column) for 1D
 * distribution within the 2D MPI grid.
 * @tparam Platform Specifies the computational platform, with a default to
 * chase::platform::CPU.
 */
template <typename T, CommunicatorType comm_type,
          typename Platform = chase::platform::CPU>
class DistMultiVector1D
    : public AbstractDistMultiVector<T, comm_type, DistMultiVector1D,
                                     Platform> // distribute either within row
                                               // or column communicator of 2D
                                               // MPI grid
{
public:
    using platform_type =
        Platform;         /**< Alias for the computational platform type. */
    using value_type = T; /**< Alias for the element type. */
    static constexpr chase::distMultiVector::CommunicatorType
        communicator_type = comm_type;

    /**
     * @brief Default destructor.
     */
    ~DistMultiVector1D() override {};
    /**
     * @brief Default constructor.
     */
    DistMultiVector1D();
    /**
     * @brief Constructs a distributed multivector with specified global
     * dimensions and MPI grid.
     *
     * Initializes the 1D distribution according to the specified communicator
     * type, handling dimensions and block sizes.
     *
     * @param M Global number of rows.
     * @param N Global number of columns.
     * @param mpi_grid Shared pointer to the 2D MPI grid for distribution.
     */
    DistMultiVector1D(std::size_t M, std::size_t N,
                      std::shared_ptr<chase::grid::MpiGrid2DBase> mpi_grid)
        : M_(M), N_(N), mpi_grid_(mpi_grid)
    {
        int* dims_ = mpi_grid_.get()->get_dims();
        int* coord_ = mpi_grid_.get()->get_coords();
        std::size_t len;
        n_ = N_;
        int dim, coord;

        if constexpr (comm_type ==
                      chase::distMultiVector::CommunicatorType::
                          row) // distributed within row communicator
        {
            dim = dims_[1];
            coord = coord_[1];
        }
        else if constexpr (comm_type ==
                           chase::distMultiVector::CommunicatorType::column)
        {
            dim = dims_[0];
            coord = coord_[0];
        }
        else
        {
            std::runtime_error("no CommunicatorType supported");
        }

        if (M_ % dim == 0)
        {
            len = M_ / dim;
        }
        else
        {
            len = std::min(M_, M_ / dim + 1);
        }

        if (coord < dim - 1)
        {
            m_ = len;
        }
        else
        {
            m_ = M_ - (dim - 1) * len;
        }

        off_ = coord * len;

        ld_ = m_;

        // local_matrix_ =  typename chase::platform::MatrixTypePlatform<T,
        // Platform>::type(m_, n_);
        local_matrix_ = chase::matrix::Matrix<T, Platform>(m_, n_);
        mb_ = m_;

        if (coord == dim - 1 && dim != 1)
        {
            mb_ = (M_ - m_) / (dim - 1);
        }

        if (off_ + m_ > M_ / 2)
        {
            l_half_ = std::size_t(std::max(0, int(M_ / 2) - int(off_)));
        }
        else
        {
            l_half_ = m_;
        }
    }
    /**
     * @brief Constructs a distributed multivector with specified local
     * dimensions, leading dimension, and data pointer.
     *
     * @param m Local number of rows.
     * @param n Local number of columns.
     * @param ld Leading dimension of the data pointer.
     * @param data Pointer to the data array.
     * @param mpi_grid Shared pointer to the 2D MPI grid.
     */
    DistMultiVector1D(std::size_t m, std::size_t n, std::size_t ld, T* data,
                      std::shared_ptr<chase::grid::MpiGrid2DBase> mpi_grid)
        : m_(m), n_(n), ld_(ld), mpi_grid_(mpi_grid)
    {
        N_ = n_;
        uint64_t lv = static_cast<uint64_t>(m_);
        uint64_t res = 0;
        std::size_t len;

        MPI_Comm comm;

        if constexpr (comm_type ==
                      chase::distMultiVector::CommunicatorType::
                          row) // distributed within row communicator
        {
            comm = mpi_grid_.get()->get_row_comm();
        }
        else if constexpr (comm_type ==
                           chase::distMultiVector::CommunicatorType::column)
        {
            comm = mpi_grid_.get()->get_col_comm();
        }
        else
        {
            std::runtime_error("no CommunicatorType supported");
        }

        MPI_Allreduce(&lv, &res, 1, MPI_UINT64_T, MPI_SUM, comm);
        M_ = static_cast<std::size_t>(res);

        // local_matrix_ = typename chase::platform::MatrixTypePlatform<T,
        // Platform>::type(m_, n_, ld_, data);
        local_matrix_ = chase::matrix::Matrix<T, Platform>(m_, n_, ld_, data);

        if constexpr (std::is_same<Platform, chase::platform::GPU>::value)
        {
            ld_ = local_matrix_.ld();
        }

        int* dims_ = mpi_grid_.get()->get_dims();
        int* coord_ = mpi_grid_.get()->get_coords();

        int dim, coord;

        if constexpr (comm_type ==
                      chase::distMultiVector::CommunicatorType::
                          row) // distributed within row communicator
        {
            dim = dims_[1];
            coord = coord_[1];
        }
        else if constexpr (comm_type ==
                           chase::distMultiVector::CommunicatorType::column)
        {
            dim = dims_[0];
            coord = coord_[0];
        }
        else
        {
            std::runtime_error("no CommunicatorType supported");
        }

        mb_ = m_;

        if (coord == dim - 1 && dim != 1)
        {
            mb_ = (M_ - m_) / (dim - 1);
        }

        if (M_ % dim == 0)
        {
            len = M_ / dim;
        }
        else
        {
            len = std::min(M_, M_ / dim + 1);
        }

        if (coord < dim - 1)
        {
            m_ = len;
        }
        else
        {
            m_ = M_ - (dim - 1) * len;
        }

        off_ = coord * len;

        if (off_ + m_ > M_ / 2)
        {
            l_half_ = std::size_t(std::max(0, int(M_ / 2) - int(off_)));
        }
        else
        {
            l_half_ = m_;
        }
    }
    /**
     * @brief Retrieves the distribution type of the multi-vector.
     *
     * @return Returns the distribution type as Block.
     */
    DistributionType getMultiVectorDistributionType() const override
    {
        return DistributionType::Block;
    }
    /**
     * @brief Retrieves the communicator type for distribution.
     *
     * @return Returns the specified communicator type (row or column).
     */
    CommunicatorType getMultiVectorCommunicatorType() const override
    {
        return comm_type;
    }

    /**
     * @brief Provides a raw pointer to the MPI grid used for distribution.
     *
     * @return Raw pointer to the MPI grid object.
     */
    chase::grid::MpiGrid2DBase* getMpiGrid() const override
    {
        return mpi_grid_.get();
    }
    /**
     * @brief Provides a shared pointer to the MPI grid used for distribution.
     *
     * @return Shared pointer to the MPI grid object.
     */
    std::shared_ptr<chase::grid::MpiGrid2DBase>
    getMpiGrid_shared_ptr() const override
    {
        return mpi_grid_;
    }
    /**
     * @brief Clones the distributed multivector to a specified type with the
     * same global dimensions and MPI grid.
     *
     * @tparam CloneType The type of the cloned object.
     * @return Cloned object of specified type.
     */
    template <typename CloneType>
    CloneType clone()
    {
        static_assert(std::is_same_v<T, typename CloneType::value_type>,
                      "Cloned type must have the same value_type");
        /// using NewCommType = typename CloneType::communicator_type;
        return CloneType(M_, N_, mpi_grid_);
    }
    /**
     * @brief Clones the distributed multivector to a specified type with new
     * global dimensions.
     *
     * @tparam CloneType The type of the cloned object.
     * @param g_M New global row dimension.
     * @param g_N New global column dimension.
     * @return Cloned object of specified type with new dimensions.
     */
    template <typename CloneType>
    CloneType clone(std::size_t g_M, std::size_t g_N)
    {
        static_assert(std::is_same_v<T, typename CloneType::value_type>,
                      "Cloned type must have the same value_type");
        /// using NewCommType = typename CloneType::communicator_type;
        return CloneType(g_M, g_N, mpi_grid_);
    }
    /**
     * @brief Creates a unique pointer to a clone of the distributed multivector
     * with the same dimensions.
     *
     * @tparam CloneType The type of the cloned object.
     * @return Unique pointer to the cloned object.
     */
    template <typename CloneType>
    std::unique_ptr<CloneType> clone2()
    {
        static_assert(std::is_same_v<T, typename CloneType::value_type>,
                      "Cloned type must have the same value_type");
        /// using NewCommType = typename CloneType::communicator_type;
        return std::make_unique<CloneType>(M_, N_, mpi_grid_);
    }
    /**
     * @brief Creates a unique pointer to a clone of the distributed multivector
     * with new global dimensions.
     *
     * @tparam CloneType The type of the cloned object.
     * @param g_M New global row dimension.
     * @param g_N New global column dimension.
     * @return Unique pointer to the cloned object with new dimensions.
     */
    template <typename CloneType>
    std::unique_ptr<CloneType> clone2(std::size_t g_M, std::size_t g_N)
    {
        static_assert(std::is_same_v<T, typename CloneType::value_type>,
                      "Cloned type must have the same value_type");
        /// using NewCommType = typename CloneType::communicator_type;
        return std::make_unique<CloneType>(g_M, g_N, mpi_grid_);
    }

    /**
     * @brief Swaps data and metadata with another distributed multivector with
     * compatible types.
     *
     * @tparam OtherCommType Communicator type of the other multivector.
     * @tparam OtherPlatform Platform type of the other multivector.
     * @param other The other multivector to swap with.
     *
     * @throws std::runtime_error if communicator types or platforms do not
     * match.
     */
    template <CommunicatorType OtherCommType, typename OtherPlatform>
    void swap(DistMultiVector1D<T, OtherCommType, OtherPlatform>& other)
    {

        SCOPED_NVTX_RANGE();

        // Check if the communicator types are the same
        if constexpr (comm_type != OtherCommType)
        {
            throw std::runtime_error(
                "Cannot swap: Communicator types do not match.");
        }

        if constexpr (!std::is_same<Platform, OtherPlatform>::value)
        {
            throw std::runtime_error(
                "Cannot swap: Platform types do not match.");
        }
        // Ensure both objects have the same MPI grid
        if (mpi_grid_.get() != other.mpi_grid_.get())
        {
            throw std::runtime_error("Cannot swap: MPI grids do not match.");
        }

        std::swap(M_, other.M_);
        std::swap(N_, other.N_);
        std::swap(m_, other.m_);
        std::swap(n_, other.n_);
        std::swap(ld_, other.ld_);
        std::swap(mb_, other.mb_);
        local_matrix_.swap(other.local_matrix_);
        std::swap(this->is_single_precision_enabled_,
                  other.is_single_precision_enabled_);
        std::swap(this->single_precision_multivec_,
                  other.single_precision_multivec_);
        std::swap(this->is_double_precision_enabled_,
                  other.is_double_precision_enabled_);
        std::swap(this->double_precision_multivec_,
                  other.double_precision_multivec_);
    }
    /**
     * @brief Swaps columns i and j in the distributed multivector.
     *
     * @param i Index of the first column.
     * @param j Index of the second column.
     */
    void swap_ij(std::size_t i, std::size_t j)
    {
        if constexpr (std::is_same<Platform, chase::platform::CPU>::value)
        {
            std::vector<T> tmp(m_);
            chase::linalg::lapackpp::t_lacpy(
                'A', m_, 1, this->l_data() + i * ld_, 1, tmp.data(), 1);
            chase::linalg::lapackpp::t_lacpy('A', m_, 1,
                                             this->l_data() + j * ld_, 1,
                                             this->l_data() + i * ld_, 1);
            chase::linalg::lapackpp::t_lacpy('A', m_, 1, tmp.data(), 1,
                                             this->l_data() + j * ld_, 1);
        }
#ifdef HAS_CUDA
        else
        {
            T* tmp;
            CHECK_CUDA_ERROR(cudaMalloc(&tmp, m_ * sizeof(T)));
            chase::linalg::internal::cuda::t_lacpy(
                'A', m_, 1, this->l_data() + i * ld_, 1, tmp, 1);
            chase::linalg::internal::cuda::t_lacpy('A', m_, 1,
                                                   this->l_data() + j * ld_, 1,
                                                   this->l_data() + i * ld_, 1);
            chase::linalg::internal::cuda::t_lacpy('A', m_, 1, tmp, 1,
                                                   this->l_data() + j * ld_, 1);
            cudaFree(tmp);
        }
#endif
    }

#ifdef HAS_CUDA
    /**
     * @brief Transfers data from device to host.
     *
     * This operation is only supported on GPU platforms.
     *
     * @throws std::runtime_error if executed on a CPU platform.
     */
    void D2H()
    {
        if constexpr (std::is_same<Platform, chase::platform::GPU>::value)
        {
            local_matrix_.D2H();
        }
        else
        {
            throw std::runtime_error("[DistMultiVector]: CPU type of matrix do "
                                     "not support D2H operation");
        }
    }
    /**
     * @brief Transfers data from host to device.
     *
     * This operation is only supported on GPU platforms.
     *
     * @throws std::runtime_error if executed on a CPU platform.
     */
    void H2D()
    {
        if constexpr (std::is_same<Platform, chase::platform::GPU>::value)
        {
            local_matrix_.H2D();
        }
        else
        {
            throw std::runtime_error("[DistMultiVector]: CPU type of matrix do "
                                     "not support H2D operation");
        }
    }
#endif
    /**
     * @brief Provides a pointer to the CPU data.
     *
     * @return Pointer to the CPU data.
     */
    T* cpu_data() { return local_matrix_.cpu_data(); }

    /**
     * @brief Retrieves the leading dimension for the CPU data.
     *
     * @return Leading dimension of the CPU data.
     */
    std::size_t cpu_ld() { return local_matrix_.cpu_ld(); }

    /**
     * @brief Redistributes data to another distributed multivector with a
     * specified communicator type.
     *
     * @tparam target_comm_type Communicator type of the target multivector.
     * @tparam OtherPlatform Platform type of the target multivector.
     * @param targetMultiVector Target multivector for redistribution.
     * @param offset Column offset for redistribution.
     * @param subsetSize Number of columns to redistribute.
     *
     * @throws std::runtime_error if platforms or communicator types do not
     * match.
     * @throws std::invalid_argument if subset range is invalid.
     */
    template <CommunicatorType target_comm_type, typename OtherPlatform>
    void redistributeImpl(DistMultiVector1D<T, target_comm_type, OtherPlatform>*
                              targetMultiVector,
                          std::size_t offset, std::size_t subsetSize)
    {
        // Validate the subset range
        if (offset + subsetSize > this->g_cols() ||
            subsetSize > targetMultiVector->g_cols())
        {
            throw std::invalid_argument("Invalid subset range");
        }

        if constexpr (!std::is_same<Platform, OtherPlatform>::value)
        {
            throw std::runtime_error(
                "Cannot redistribute: Platform types do not match.");
        }

        // Check if the target matrix's communicator type matches the allowed
        // types
        if constexpr (comm_type == CommunicatorType::row &&
                      target_comm_type == CommunicatorType::column)
        {
            // Implement redistribution from row to column
            redistributeRowToColumn<OtherPlatform>(targetMultiVector, offset,
                                                   subsetSize);
        }
        else if constexpr (comm_type == CommunicatorType::column &&
                           target_comm_type == CommunicatorType::row)
        {
            // Implement redistribution from column to row
            redistributeColumnToRow<OtherPlatform>(targetMultiVector, offset,
                                                   subsetSize);
        }
        else
        {
            throw std::runtime_error(
                "Invalid redistribution between matrix types");
        }
    }
    /**
     * @brief Redistributes the full data range to another distributed
     * multivector.
     *
     * @tparam target_comm_type Communicator type of the target multivector.
     * @tparam OtherPlatform Platform type of the target multivector.
     * @param targetMultiVector Target multivector for redistribution.
     */
    template <CommunicatorType target_comm_type, typename OtherPlatform>
    void redistributeImpl(DistMultiVector1D<T, target_comm_type, OtherPlatform>*
                              targetMultiVector)
    {
        this->redistributeImpl(targetMultiVector, 0, this->n_);
    }

#ifdef HAS_NCCL
    /**
     * @brief Asynchronous redistribution of data to another distributed
     * multivector on GPU with specified communicator type.
     *
     * @tparam target_comm_type Communicator type of the target multivector.
     * @param targetMultiVector Target multivector for redistribution.
     * @param offset Column offset for redistribution.
     * @param subsetSize Number of columns to redistribute.
     * @param stream_ CUDA stream for asynchronous operation (optional).
     */
    template <CommunicatorType target_comm_type>
    void redistributeImplAsync(
        DistMultiVector1D<T, target_comm_type, chase::platform::GPU>*
            targetMultiVector,
        std::size_t offset, std::size_t subsetSize,
        cudaStream_t* stream_ = nullptr)
    {

        cudaStream_t usedStream = (stream_ == nullptr) ? 0 : *stream_;

        // Validate the subset range
        if (offset + subsetSize > this->g_cols() ||
            subsetSize > targetMultiVector->g_cols())
        {
            throw std::invalid_argument("Invalid subset range");
        }

        if constexpr (!std::is_same<Platform, chase::platform::GPU>::value)
        {
            throw std::runtime_error(
                "NCCL based redistribution support only GPU.");
        }

        // Check if the target matrix's communicator type matches the allowed
        // types
        if constexpr (comm_type == CommunicatorType::row &&
                      target_comm_type == CommunicatorType::column)
        {
            // Implement redistribution from row to column
            redistributeRowToColumnAsync(targetMultiVector, offset, subsetSize,
                                         usedStream);
        }
        else if constexpr (comm_type == CommunicatorType::column &&
                           target_comm_type == CommunicatorType::row)
        {
            // Implement redistribution from column to row
            redistributeColumnToRowAsync(targetMultiVector, offset, subsetSize,
                                         usedStream);
        }
        else
        {
            throw std::runtime_error(
                "Invalid redistribution between matrix types");
        }
    }
    /**
     * @brief Redistributes the full data on GPU with range to another
     * distributed multivector.
     *
     * @tparam target_comm_type Communicator type of the target multivector.
     * @tparam OtherPlatform Platform type of the target multivector.
     * @param targetMultiVector Target multivector for redistribution.
     */
    template <CommunicatorType target_comm_type>
    void redistributeImplAsync(
        DistMultiVector1D<T, target_comm_type, chase::platform::GPU>*
            targetMultiVector,
        cudaStream_t* stream_ = nullptr)
    {
        this->redistributeImplAsync(targetMultiVector, 0, this->n_, stream_);
    }

#endif
    /**
     * @brief Get the number of rows in the global matrix.
     *
     * @return The global row count.
     */
    std::size_t g_rows() const override { return M_; }
    /**
     * @brief Get the number of columns in the global matrix.
     *
     * @return The global column count.
     */
    std::size_t g_cols() const override { return N_; }
    /**
     * @brief Get the global offset.
     *
     * @return The global offset of the local data.
     */
    std::size_t g_off() const override { return off_; }
    /**
     * @brief Get the number of rows in the local matrix (i.e., rows stored on
     * this process).
     *
     * @return The local row count.
     */
    std::size_t l_rows() const override { return m_; }
    /**
     * @brief Get the number of columns in the local matrix (i.e., columns
     * stored on this process).
     *
     * @return The local column count.
     */
    std::size_t l_cols() const override { return n_; }
    /**
     * @brief Get the leading dimension of the local matrix storage.
     *
     * @return The leading dimension for the local matrix.
     */
    std::size_t l_ld() const override { return ld_; }
    /**
     * @brief Get the block size used in the distributed matrix layout.
     *
     * @return The matrix block size.
     */
    std::size_t mb() const override { return mb_; }
    /**
     * @brief Returns the number of row blocks.
     *
     * @return The number of row blocks.
     */
    std::size_t mblocks() const override { return mblocks_; }
    /**
     * @brief Get the block size used in the distributed matrix layout.
     *
     * @return The matrix block size.
     */
    std::size_t l_half() const override { return l_half_; }
    /**
     * @brief Access the raw pointer to the local matrix data.
     *
     * @return Pointer to the local matrix data.
     */
    T* l_data() override { return local_matrix_.data(); }

    // typename chase::platform::MatrixTypePlatform<T, Platform>::type&
    // loc_matrix() override { return local_matrix_;}
    /**
     * @brief Access the local matrix object.
     *
     * @return Reference to the local matrix of type chase::matrix::Matrix.
     */
    chase::matrix::Matrix<T, Platform>& loc_matrix() override
    {
        return local_matrix_;
    }
#ifdef HAS_SCALAPACK
    /**
     * @brief Get the ScaLAPACK descriptor for the matrix.
     *
     * @return Pointer to the ScaLAPACK descriptor array.
     */
    std::size_t* get_scalapack_desc() { return desc_; }
    /**
     * @brief Initialize the ScaLAPACK descriptor for the matrix.
     *
     * Initializes the ScaLAPACK descriptor array with parameters based on the
     * MPI grid and matrix layout.
     *
     * If the platform is GPU, CPU data is allocated if it is not already
     * allocated.
     *
     * @return Pointer to the initialized ScaLAPACK descriptor.
     */
    std::size_t* scalapack_descriptor_init()
    {
        if constexpr (std::is_same<Platform, chase::platform::GPU>::value)
        {
            if (local_matrix_.cpu_data() == nullptr)
            {
                local_matrix_.allocate_cpu_data();
            }
        }

        std::size_t ldd = this->cpu_ld();

        if constexpr (comm_type == CommunicatorType::column)
        {
            std::size_t mb = m_;
            int* coords = mpi_grid_.get()->get_coords();
            int* dims = mpi_grid_.get()->get_dims();

            if (coords[0] == dims[0] - 1 && dims[0] != 1)
            {
                mb = (M_ - m_) / (dims[0] - 1);
            }

            std::size_t default_blocksize = 64;
            std::size_t nb = std::min(n_, default_blocksize);
            int zero = 0;
            int one = 1;
            int info;
            int colcomm1D_ctxt = mpi_grid_.get()->get_blacs_colcomm_ctxt();
            chase::linalg::scalapackpp::t_descinit(
                desc_, &M_, &N_, &mb, &nb, &zero, &zero, &colcomm1D_ctxt, &ldd,
                &info);
        }
        else
        {
            // row based will be implemented later
        }

        return desc_;
    }
#endif

private:
    /**
     * @brief The total number of rows in the global matrix.
     */
    std::size_t M_;
    /**
     * @brief The total number of columns in the global matrix.
     */
    std::size_t N_;
    /**
     * @brief The global offset of the local data.
     */
    std::size_t off_;
    /**
     * @brief The number of rows in the local matrix, stored on the current
     * process.
     */
    std::size_t m_;
    /**
     * @brief The number of columns in the local matrix, stored on the current
     * process.
     */
    std::size_t n_;
    /**
     * @brief The leading dimension for the local matrix, used for efficient
     * memory access.
     */
    std::size_t ld_;
    /**
     * @brief The block size used in the distributed matrix layout.
     */
    std::size_t mb_;
    /**
     * @brief The number of blocks along the row dimension.
     *
     * This is used to determine how the matrix is divided into smaller blocks
     * for distributed computation.
     */
    std::size_t mblocks_;
    /**
     * @brief The local index of the lower half
     */
    std::size_t l_half_;

    // typename chase::platform::MatrixTypePlatform<T, Platform>::type
    // local_matrix_;
    /**
     * @brief The matrix object that holds the local matrix data, stored based
     * on the specified platform.
     */
    chase::matrix::Matrix<T, Platform> local_matrix_;
    /**
     * @brief Shared pointer to the MPI grid object that manages the 2D process
     * grid configuration.
     */
    std::shared_ptr<chase::grid::MpiGrid2DBase> mpi_grid_;
#ifdef HAS_SCALAPACK
    /**
     * @brief ScaLAPACK descriptor array for the matrix, used to manage
     * distributed matrix layout and storage.
     *
     * Contains 9 elements, where each element holds specific metadata required
     * for ScaLAPACK operations.
     */
    std::size_t desc_[9];
#endif
    // data for redistribution
    std::vector<std::size_t> orig_dests;
    std::vector<std::size_t> orig_srcs;
    std::vector<std::size_t> orig_lens;
    std::vector<std::size_t> target_disps;
    std::vector<std::size_t> orig_disps;

    template <typename OtherPlatform,
              chase::distMultiVector::CommunicatorType OtherCommType>
    void init_redistribution(
        DistMultiVector1D<T, OtherCommType, OtherPlatform>* targetMultiVector)
    {
        orig_dests = std::vector<std::size_t>();
        orig_srcs = std::vector<std::size_t>();
        orig_lens = std::vector<std::size_t>();
        target_disps = std::vector<std::size_t>();
        orig_disps = std::vector<std::size_t>();

        std::size_t orig_dest = 0;
        std::size_t orig_src = 0;
        orig_dests.push_back(orig_dest);
        orig_srcs.push_back(orig_src);
        std::size_t len = 1;
        std::size_t orig_disp = 0;
        std::size_t target_disp = 0;
        orig_disps.push_back(orig_disp);
        target_disps.push_back(target_disp);

        std::size_t mb = this->mb();
        std::size_t nb = targetMultiVector->mb();
        int* coords = mpi_grid_.get()->get_coords();
        int* dims = mpi_grid_.get()->get_dims();
        int dim0, dim1;
        if constexpr (comm_type ==
                      chase::distMultiVector::CommunicatorType::column)
        {
            dim0 = dims[0];
            dim1 = dims[1];
        }
        else if constexpr (comm_type ==
                           chase::distMultiVector::CommunicatorType::row)
        {
            dim0 = dims[1];
            dim1 = dims[0];
        }

        for (auto i = 1; i < M_; i++)
        {
            auto src_tmp = (i / mb) % dim0;
            auto dest_tmp = (i / nb) % dim1;
            if (dest_tmp == orig_dest && src_tmp == orig_src)
            {
                len += 1;
            }
            else
            {
                orig_lens.push_back(len);
                orig_dest = (i / nb) % dim1;
                target_disp = i % nb + ((i / nb) / dim1) * nb;
                orig_disp = i % mb + ((i / mb) / dim0) * mb;
                orig_src = (i / mb) % dim0;
                orig_srcs.push_back(orig_src);
                orig_dests.push_back(orig_dest);
                target_disps.push_back(target_disp);
                orig_disps.push_back(orig_disp);
                len = 1;
            }
        }
        orig_lens.push_back(len);
    }

    template <typename OtherPlatform>
    void
    redistributeRowToColumn(DistMultiVector1D<T, CommunicatorType::column,
                                              OtherPlatform>* targetMultiVector,
                            std::size_t offset, std::size_t subsetSize)
    {
        // Ensure the dimensions are compatible
        if (this->M_ != targetMultiVector->g_rows() ||
            this->N_ != targetMultiVector->g_cols())
        {
            throw std::runtime_error(
                "Dimension mismatch during redistribution");
        }

        if (this->mpi_grid_.get() != targetMultiVector->getMpiGrid())
        {
            throw std::runtime_error("MPI Grid mismatch during redistribution");
        }

        int* dims = this->mpi_grid_->get_dims();
        int* coords = this->mpi_grid_->get_coords();
        if (dims[0] == dims[1]) // squared grid
        {
            for (auto i = 0; i < dims[1]; i++)
            {
                if (coords[0] == i)
                {
                    if (coords[1] == i)
                    {
                        MPI_Bcast(this->l_data() + offset * this->ld_,
                                  this->m_ * subsetSize,
                                  chase::mpi::getMPI_Type<T>(), i,
                                  this->mpi_grid_->get_row_comm());
                    }
                    else
                    {
                        MPI_Bcast(targetMultiVector->l_data() +
                                      offset * targetMultiVector->l_ld(),
                                  targetMultiVector->l_rows() * subsetSize,
                                  chase::mpi::getMPI_Type<T>(), i,
                                  this->mpi_grid_->get_row_comm());
                    }
                }
            }

            for (auto i = 0; i < dims[1]; i++)
            {
                if (coords[0] == coords[1])
                {
                    if constexpr (std::is_same<Platform,
                                               chase::platform::CPU>::value)
                    {
                        chase::linalg::lapackpp::t_lacpy(
                            'A', this->m_, subsetSize,
                            this->l_data() + offset * this->ld_, this->ld_,
                            targetMultiVector->l_data() +
                                offset * targetMultiVector->l_ld(),
                            targetMultiVector->l_ld());
                    }
#ifdef HAS_CUDA
                    else
                    {
                        chase::linalg::internal::cuda::t_lacpy(
                            'A', this->m_, subsetSize,
                            this->l_data() + offset * this->ld_, this->ld_,
                            targetMultiVector->l_data() +
                                offset * targetMultiVector->l_ld(),
                            targetMultiVector->l_ld());
                    }
#endif
                }
#ifdef HAS_CUDA
                cudaDeviceSynchronize();
#endif
            }
        }
        else
        {
            init_redistribution<
                OtherPlatform,
                chase::distMultiVector::CommunicatorType::column>(
                targetMultiVector);

            for (auto i = 0; i < orig_lens.size(); i++)
            {
                if (coords[0] == orig_dests[i])
                {
                    if constexpr (std::is_same<Platform,
                                               chase::platform::CPU>::value)
                    {
                        auto max_c_len =
                            *max_element(orig_lens.begin(), orig_lens.end());
                        std::unique_ptr<chase::matrix::Matrix<T>> buff =
                            std::make_unique<chase::matrix::Matrix<T>>(
                                max_c_len, subsetSize);
                        chase::linalg::lapackpp::t_lacpy(
                            'A', orig_lens[i], subsetSize,
                            this->l_data() + offset * this->ld_ + orig_disps[i],
                            this->ld_, buff->data(), orig_lens[i]);

                        MPI_Bcast(buff->data(), orig_lens[i] * subsetSize,
                                  chase::mpi::getMPI_Type<T>(), orig_srcs[i],
                                  this->mpi_grid_->get_row_comm());
                        chase::linalg::lapackpp::t_lacpy(
                            'A', orig_lens[i], subsetSize, buff->data(),
                            orig_lens[i],
                            targetMultiVector->l_data() + target_disps[i] +
                                offset * targetMultiVector->l_ld(),
                            targetMultiVector->l_ld());
                    }
#ifdef HAS_CUDA
                    else
                    {
                        auto max_c_len =
                            *max_element(orig_lens.begin(), orig_lens.end());
                        std::unique_ptr<
                            chase::matrix::Matrix<T, chase::platform::GPU>>
                            buff = std::make_unique<
                                chase::matrix::Matrix<T, chase::platform::GPU>>(
                                max_c_len, subsetSize);
                        chase::linalg::internal::cuda::t_lacpy(
                            'A', orig_lens[i], subsetSize,
                            this->l_data() + offset * this->ld_ + orig_disps[i],
                            this->ld_, buff->data(), orig_lens[i]);

                        MPI_Bcast(buff->data(), orig_lens[i] * subsetSize,
                                  chase::mpi::getMPI_Type<T>(), orig_srcs[i],
                                  this->mpi_grid_->get_row_comm());
                        chase::linalg::internal::cuda::t_lacpy(
                            'A', orig_lens[i], subsetSize, buff->data(),
                            orig_lens[i],
                            targetMultiVector->l_data() + target_disps[i] +
                                offset * targetMultiVector->l_ld(),
                            targetMultiVector->l_ld());
                    }
#endif
                }
            }
        }
    }

    template <typename OtherPlatform>
    void
    redistributeColumnToRow(DistMultiVector1D<T, CommunicatorType::row,
                                              OtherPlatform>* targetMultiVector,
                            std::size_t offset, std::size_t subsetSize)
    {
        // Ensure the dimensions are compatible
        if (this->M_ != targetMultiVector->g_rows() ||
            this->N_ != targetMultiVector->g_cols())
        {
            throw std::runtime_error(
                "Dimension mismatch during redistribution");
        }

        if (this->mpi_grid_.get() != targetMultiVector->getMpiGrid())
        {
            throw std::runtime_error("MPI Grid mismatch during redistribution");
        }

        int* dims = this->mpi_grid_->get_dims();
        int* coords = this->mpi_grid_->get_coords();
        if (dims[0] == dims[1]) // squared grid
        {
            for (auto i = 0; i < dims[0]; i++)
            {
                if (coords[1] == i)
                {
                    if (coords[0] == i)
                    {
                        MPI_Bcast(this->l_data() + offset * this->ld_,
                                  this->m_ * subsetSize,
                                  chase::mpi::getMPI_Type<T>(), i,
                                  this->mpi_grid_->get_col_comm());
                    }
                    else
                    {
                        MPI_Bcast(targetMultiVector->l_data() +
                                      offset * targetMultiVector->l_ld(),
                                  targetMultiVector->l_rows() * subsetSize,
                                  chase::mpi::getMPI_Type<T>(), i,
                                  this->mpi_grid_->get_col_comm());
                    }
                }
            }

            for (auto i = 0; i < dims[0]; i++)
            {
                if (coords[0] == coords[1])
                {
                    if constexpr (std::is_same<Platform,
                                               chase::platform::CPU>::value)
                    {
                        chase::linalg::lapackpp::t_lacpy(
                            'A', this->m_, subsetSize,
                            this->l_data() + offset * this->ld_, this->ld_,
                            targetMultiVector->l_data() +
                                offset * targetMultiVector->l_ld(),
                            targetMultiVector->l_ld());
                    }
#ifdef HAS_CUDA
                    else
                    {
                        chase::linalg::internal::cuda::t_lacpy(
                            'A', this->m_, subsetSize,
                            this->l_data() + offset * this->ld_, this->ld_,
                            targetMultiVector->l_data() +
                                offset * targetMultiVector->l_ld(),
                            targetMultiVector->l_ld());
                    }
#endif
                }
            }
        }
        else
        {
            init_redistribution<OtherPlatform,
                                chase::distMultiVector::CommunicatorType::row>(
                targetMultiVector);

            for (auto i = 0; i < orig_lens.size(); i++)
            {
                if (coords[1] == orig_dests[i])
                {
                    if constexpr (std::is_same<Platform,
                                               chase::platform::CPU>::value)
                    {
                        auto max_c_len =
                            *max_element(orig_lens.begin(), orig_lens.end());
                        std::unique_ptr<chase::matrix::Matrix<T>> buff =
                            std::make_unique<chase::matrix::Matrix<T>>(
                                max_c_len, subsetSize);
                        chase::linalg::lapackpp::t_lacpy(
                            'A', orig_lens[i], subsetSize,
                            this->l_data() + offset * this->ld_ + orig_disps[i],
                            this->ld_, buff->data(), orig_lens[i]);

                        MPI_Bcast(buff->data(), orig_lens[i] * subsetSize,
                                  chase::mpi::getMPI_Type<T>(), orig_srcs[i],
                                  this->mpi_grid_->get_col_comm());
                        chase::linalg::lapackpp::t_lacpy(
                            'A', orig_lens[i], subsetSize, buff->data(),
                            orig_lens[i],
                            targetMultiVector->l_data() + target_disps[i] +
                                offset * targetMultiVector->l_ld(),
                            targetMultiVector->l_ld());
                    }
#ifdef HAS_CUDA
                    else
                    {
                        auto max_c_len =
                            *max_element(orig_lens.begin(), orig_lens.end());
                        std::unique_ptr<
                            chase::matrix::Matrix<T, chase::platform::GPU>>
                            buff = std::make_unique<
                                chase::matrix::Matrix<T, chase::platform::GPU>>(
                                max_c_len, subsetSize);
                        chase::linalg::internal::cuda::t_lacpy(
                            'A', orig_lens[i], subsetSize,
                            this->l_data() + offset * this->ld_ + orig_disps[i],
                            this->ld_, buff->data(), orig_lens[i]);

                        MPI_Bcast(buff->data(), orig_lens[i] * subsetSize,
                                  chase::mpi::getMPI_Type<T>(), orig_srcs[i],
                                  this->mpi_grid_->get_col_comm());
                        chase::linalg::internal::cuda::t_lacpy(
                            'A', orig_lens[i], subsetSize, buff->data(),
                            orig_lens[i],
                            targetMultiVector->l_data() + target_disps[i] +
                                offset * targetMultiVector->l_ld(),
                            targetMultiVector->l_ld());
                    }
#endif
                }
            }
        }
    }
#ifdef HAS_CUDA
#ifdef HAS_NCCL
    void redistributeRowToColumnAsync(
        DistMultiVector1D<T, CommunicatorType::column, chase::platform::GPU>*
            targetMultiVector,
        std::size_t offset, std::size_t subsetSize, cudaStream_t stream)
    {
        // Ensure the dimensions are compatible
        if (this->M_ != targetMultiVector->g_rows() ||
            this->N_ != targetMultiVector->g_cols())
        {
            throw std::runtime_error(
                "Dimension mismatch during redistribution");
        }

        if (this->mpi_grid_.get() != targetMultiVector->getMpiGrid())
        {
            throw std::runtime_error("MPI Grid mismatch during redistribution");
        }

        int* dims = this->mpi_grid_->get_dims();
        int* coords = this->mpi_grid_->get_coords();
        if (dims[0] == dims[1]) // squared grid
        {
            for (auto i = 0; i < dims[1]; i++)
            {
                if (coords[0] == i)
                {
                    if (coords[1] == i)
                    {
                        CHECK_NCCL_ERROR(chase::nccl::ncclBcastWrapper(
                            this->l_data() + offset * this->ld_,
                            this->m_ * subsetSize, i,
                            this->mpi_grid_->get_nccl_row_comm(), &stream));
                    }
                    else
                    {
                        CHECK_NCCL_ERROR(chase::nccl::ncclBcastWrapper(
                            targetMultiVector->l_data() +
                                offset * targetMultiVector->l_ld(),
                            targetMultiVector->l_rows() * subsetSize, i,
                            this->mpi_grid_->get_nccl_row_comm(), &stream));
                    }
                }
            }

            for (auto i = 0; i < dims[1]; i++)
            {
                if (coords[0] == coords[1])
                {

                    chase::linalg::internal::cuda::t_lacpy(
                        'A', this->m_, subsetSize,
                        this->l_data() + offset * this->ld_, this->ld_,
                        targetMultiVector->l_data() +
                            offset * targetMultiVector->l_ld(),
                        targetMultiVector->l_ld());
                }
            }
        }
        else
        {
            init_redistribution<
                chase::platform::GPU,
                chase::distMultiVector::CommunicatorType::column>(
                targetMultiVector);

            for (auto i = 0; i < orig_lens.size(); i++)
            {
                if (coords[0] == orig_dests[i])
                {
                    auto max_c_len =
                        *max_element(orig_lens.begin(), orig_lens.end());
                    std::unique_ptr<
                        chase::matrix::Matrix<T, chase::platform::GPU>>
                        buff = std::make_unique<
                            chase::matrix::Matrix<T, chase::platform::GPU>>(
                            max_c_len, subsetSize);
                    chase::linalg::internal::cuda::t_lacpy(
                        'A', orig_lens[i], subsetSize,
                        this->l_data() + offset * this->ld_ + orig_disps[i],
                        this->ld_, buff->data(), orig_lens[i]);

                    CHECK_NCCL_ERROR(chase::nccl::ncclBcastWrapper(
                        buff->data(), orig_lens[i] * subsetSize, orig_srcs[i],
                        this->mpi_grid_->get_nccl_row_comm(), &stream));

                    chase::linalg::internal::cuda::t_lacpy(
                        'A', orig_lens[i], subsetSize, buff->data(),
                        orig_lens[i],
                        targetMultiVector->l_data() + target_disps[i] +
                            offset * targetMultiVector->l_ld(),
                        targetMultiVector->l_ld());
                }
            }
        }
    }

    void redistributeColumnToRowAsync(
        DistMultiVector1D<T, CommunicatorType::row, chase::platform::GPU>*
            targetMultiVector,
        std::size_t offset, std::size_t subsetSize, cudaStream_t stream)
    {
        // Ensure the dimensions are compatible
        if (this->M_ != targetMultiVector->g_rows() ||
            this->N_ != targetMultiVector->g_cols())
        {
            throw std::runtime_error(
                "Dimension mismatch during redistribution");
        }

        if (this->mpi_grid_.get() != targetMultiVector->getMpiGrid())
        {
            throw std::runtime_error("MPI Grid mismatch during redistribution");
        }

        int* dims = this->mpi_grid_->get_dims();
        int* coords = this->mpi_grid_->get_coords();
        if (dims[0] == dims[1]) // squared grid
        {
            for (auto i = 0; i < dims[0]; i++)
            {
                if (coords[1] == i)
                {
                    if (coords[0] == i)
                    {
                        CHECK_NCCL_ERROR(chase::nccl::ncclBcastWrapper(
                            this->l_data() + offset * this->ld_,
                            this->m_ * subsetSize, i,
                            this->mpi_grid_->get_nccl_col_comm(), &stream));
                    }
                    else
                    {
                        CHECK_NCCL_ERROR(chase::nccl::ncclBcastWrapper(
                            targetMultiVector->l_data() +
                                offset * targetMultiVector->l_ld(),
                            targetMultiVector->l_rows() * subsetSize, i,
                            this->mpi_grid_->get_nccl_col_comm(), &stream));
                    }
                }
            }

            for (auto i = 0; i < dims[0]; i++)
            {
                if (coords[0] == coords[1])
                {
                    chase::linalg::internal::cuda::t_lacpy(
                        'A', this->m_, subsetSize,
                        this->l_data() + offset * this->ld_, this->ld_,
                        targetMultiVector->l_data() +
                            offset * targetMultiVector->l_ld(),
                        targetMultiVector->l_ld());
                }
            }
        }
        else
        {
            init_redistribution<chase::platform::GPU,
                                chase::distMultiVector::CommunicatorType::row>(
                targetMultiVector);

            for (auto i = 0; i < orig_lens.size(); i++)
            {
                if (coords[1] == orig_dests[i])
                {
                    auto max_c_len =
                        *max_element(orig_lens.begin(), orig_lens.end());
                    std::unique_ptr<
                        chase::matrix::Matrix<T, chase::platform::GPU>>
                        buff = std::make_unique<
                            chase::matrix::Matrix<T, chase::platform::GPU>>(
                            max_c_len, subsetSize);
                    chase::linalg::internal::cuda::t_lacpy(
                        'A', orig_lens[i], subsetSize,
                        this->l_data() + offset * this->ld_ + orig_disps[i],
                        this->ld_, buff->data(), orig_lens[i]);

                    CHECK_NCCL_ERROR(chase::nccl::ncclBcastWrapper(
                        buff->data(), orig_lens[i] * subsetSize, orig_srcs[i],
                        this->mpi_grid_->get_nccl_col_comm(), &stream));

                    chase::linalg::internal::cuda::t_lacpy(
                        'A', orig_lens[i], subsetSize, buff->data(),
                        orig_lens[i],
                        targetMultiVector->l_data() + target_disps[i] +
                            offset * targetMultiVector->l_ld(),
                        targetMultiVector->l_ld());
                }
            }
        }
    }

#endif
#endif
};

/**
 * @brief A distributed block-cyclic 1D matrix, used to distribute a
 * multi-vector either within the row or column communicator of a 2D MPI grid.
 *
 * This class is designed to manage a block-cyclic 1D distribution of a matrix
 * in a parallelized setting, supporting both CPU and GPU platforms. The matrix
 * data is distributed in a way that each process stores a subset of the data.
 * It can handle redistribution between different communicator types
 * (row/column).
 *
 * @tparam T The data type of the matrix elements.
 * @tparam comm_type The type of communicator (either row or column).
 * @tparam Platform The platform type, defaulting to CPU.
 */
template <typename T, CommunicatorType comm_type,
          typename Platform = chase::platform::CPU>
class DistMultiVectorBlockCyclic1D
    : public AbstractDistMultiVector<T, comm_type, DistMultiVectorBlockCyclic1D,
                                     Platform> // distribute either within row
                                               // or column communicator of 2D
                                               // MPI grid
{
public:
    using platform_type = Platform; ///< Alias for platform type.
    using value_type = T;           ///< Alias for element type.
    static constexpr chase::distMultiVector::CommunicatorType
        communicator_type = comm_type;

    /**
     * @brief Destructor for the DistMultiVectorBlockCyclic1D class.
     */
    ~DistMultiVectorBlockCyclic1D() override {};
    /**
     * @brief Default constructor.
     */
    DistMultiVectorBlockCyclic1D();
    /**
     * @brief Constructs a block-cyclic 1D distributed multi-vector.
     *
     * @param M Total number of rows in the matrix.
     * @param N Total number of columns in the matrix.
     * @param mb The block size for distribution.
     * @param mpi_grid A shared pointer to the 2D MPI grid used for
     * distribution.
     */
    DistMultiVectorBlockCyclic1D(
        std::size_t M, std::size_t N, std::size_t mb,
        std::shared_ptr<chase::grid::MpiGrid2DBase> mpi_grid)
        : M_(M), N_(N), mpi_grid_(mpi_grid), mb_(mb)
    {
        int* dims_ = mpi_grid_.get()->get_dims();
        int* coord_ = mpi_grid_.get()->get_coords();
        int dim, coord;
        if constexpr (comm_type == CommunicatorType::column)
        {
            dim = dims_[0];
            coord = coord_[0];
        }
        else
        {
            dim = dims_[1];
            coord = coord_[1];
        }

        std::tie(m_, mblocks_) = numroc(M_, mb_, coord, dim);
        n_ = N_;
        ld_ = m_;
        // local_matrix_ = typename chase::platform::MatrixTypePlatform<T,
        // Platform>::type(m_, n_);
        local_matrix_ = chase::matrix::Matrix<T, Platform>(m_, n_);

        init_contiguous_buffer_info();
    }

    /**
     * @brief Constructs a block-cyclic 1D distributed multi-vector with
     * pre-allocated data.
     *
     * @param M Total number of rows in the matrix.
     * @param m The local number of rows in this process.
     * @param n The number of columns.
     * @param mb The block size for distribution.
     * @param ld The leading dimension of the local matrix.
     * @param data Pointer to the pre-allocated data for the matrix.
     * @param mpi_grid A shared pointer to the 2D MPI grid used for
     * distribution.
     */
    DistMultiVectorBlockCyclic1D(
        std::size_t M, std::size_t m, std::size_t n, std::size_t mb,
        std::size_t ld, T* data,
        std::shared_ptr<chase::grid::MpiGrid2DBase> mpi_grid)
        : M_(M), N_(n), n_(n), mb_(mb), mpi_grid_(mpi_grid), ld_(ld)
    {
        int* dims_ = mpi_grid_.get()->get_dims();
        int* coord_ = mpi_grid_.get()->get_coords();

        int dim, coord;
        if constexpr (comm_type == CommunicatorType::column)
        {
            dim = dims_[0];
            coord = coord_[0];
        }
        else
        {
            dim = dims_[1];
            coord = coord_[1];
        }

        std::tie(m_, mblocks_) = numroc(M_, mb_, coord, dim);

        if (m_ != m)
        {
            throw std::runtime_error(
                "the local row number of input matrix is not correctly "
                "matching the given block-cyclic distribution");
        }

        if (ld_ < m_)
        {
            throw std::runtime_error(
                "the leading dimension of local matrix is not correctly "
                "matching the given block-cyclic distribution");
        }

        // local_matrix_ = typename chase::platform::MatrixTypePlatform<T,
        // Platform>::type(m_, n_, ld_, data);
        local_matrix_ = chase::matrix::Matrix<T, Platform>(m_, n_, ld_, data);
        if constexpr (std::is_same<Platform, chase::platform::GPU>::value)
        {
            ld_ = local_matrix_.ld();
        }

        init_contiguous_buffer_info();
    }

    /**
     * @brief Gets the distribution type of the multi-vector.
     *
     * @return The distribution type, which is always BlockCyclic.
     */
    DistributionType getMultiVectorDistributionType() const override
    {
        return DistributionType::BlockCyclic;
    }

    /**
     * @brief Gets the communicator type used for distribution.
     *
     * @return The communicator type, either row or column.
     */
    CommunicatorType getMultiVectorCommunicatorType() const override
    {
        return comm_type;
    }

    /**
     * @brief Retrieves the 2D MPI grid.
     *
     * @return A pointer to the MPI grid.
     */
    chase::grid::MpiGrid2DBase* getMpiGrid() const override
    {
        return mpi_grid_.get();
    }

    /**
     * @brief Retrieves the 2D MPI grid as a shared pointer.
     *
     * @return A shared pointer to the MPI grid.
     */
    std::shared_ptr<chase::grid::MpiGrid2DBase>
    getMpiGrid_shared_ptr() const override
    {
        return mpi_grid_;
    }

    /**
     * @brief Clones the current object into another instance of a different
     * type.
     *
     * @tparam CloneType The type of the clone.
     * @return A cloned instance of the current object.
     */
    template <typename CloneType>
    CloneType clone()
    {
        static_assert(std::is_same_v<T, typename CloneType::value_type>,
                      "Cloned type must have the same value_type");
        /// using NewCommType = typename CloneType::communicator_type;
        return CloneType(M_, N_, mb_, mpi_grid_);
    }

    /**
     * @brief Clones the current object with new global dimensions.
     *
     * @tparam CloneType The type of the clone.
     * @param g_M New global row size.
     * @param g_N New global column size.
     * @return A cloned instance with the updated global dimensions.
     */
    template <typename CloneType>
    CloneType clone(std::size_t g_M, std::size_t g_N)
    {
        static_assert(std::is_same_v<T, typename CloneType::value_type>,
                      "Cloned type must have the same value_type");
        /// using NewCommType = typename CloneType::communicator_type;
        return CloneType(g_M, g_N, mb_, mpi_grid_);
    }

    /**
     * @brief Clones the current object into a new instance of a different type,
     * returned as a unique pointer.
     *
     * @tparam CloneType The type of the clone.
     * @return A unique pointer to a cloned instance of the current object.
     */
    template <typename CloneType>
    std::unique_ptr<CloneType> clone2()
    {
        static_assert(std::is_same_v<T, typename CloneType::value_type>,
                      "Cloned type must have the same value_type");
        /// using NewCommType = typename CloneType::communicator_type;
        return std::make_unique<CloneType>(M_, N_, mb_, mpi_grid_);
    }
    /**
     * @brief Clones the current object with new global dimensions, returned as
     * a unique pointer.
     *
     * @tparam CloneType The type of the clone.
     * @param g_M New global row size.
     * @param g_N New global column size.
     * @return A unique pointer to a cloned instance with the updated global
     * dimensions.
     */
    template <typename CloneType>
    std::unique_ptr<CloneType> clone2(std::size_t g_M, std::size_t g_N)
    {
        static_assert(std::is_same_v<T, typename CloneType::value_type>,
                      "Cloned type must have the same value_type");
        /// using NewCommType = typename CloneType::communicator_type;
        return std::make_unique<CloneType>(g_M, g_N, mb_, mpi_grid_);
    }
    /**
     * @brief Swaps the contents of this matrix with another matrix of a
     * compatible type.
     *
     * @tparam OtherCommType The communicator type of the other matrix.
     * @tparam OtherPlatform The platform type of the other matrix.
     * @param other The other matrix to swap with.
     */
    template <CommunicatorType OtherCommType, typename OtherPlatform>
    void
    swap(DistMultiVectorBlockCyclic1D<T, OtherCommType, OtherPlatform>& other)
    {
        // Check if the communicator types are the same
        if constexpr (comm_type != OtherCommType)
        {
            throw std::runtime_error(
                "Cannot swap: Communicator types do not match.");
        }

        if constexpr (!std::is_same<Platform, OtherPlatform>::value)
        {
            throw std::runtime_error(
                "Cannot swap: Platform types do not match.");
        }
        // Ensure both objects have the same MPI grid
        if (mpi_grid_.get() != other.mpi_grid_.get())
        {
            throw std::runtime_error("Cannot swap: MPI grids do not match.");
        }

        std::swap(M_, other.M_);
        std::swap(N_, other.N_);
        std::swap(m_, other.m_);
        std::swap(mb_, other.mb_);
        std::swap(n_, other.n_);
        std::swap(ld_, other.ld_);
        std::swap(mblocks_, other.mblocks_);
        local_matrix_.swap(other.local_matrix_);
        std::swap(this->is_single_precision_enabled_,
                  other.is_single_precision_enabled_);
        std::swap(this->single_precision_multivec_,
                  other.single_precision_multivec_);
        std::swap(this->is_double_precision_enabled_,
                  other.is_double_precision_enabled_);
        std::swap(this->double_precision_multivec_,
                  other.double_precision_multivec_);
    }

    /**
     * @brief Swaps columns i and j in the local matrix.
     *
     * @param i The first column index to swap.
     * @param j The second column index to swap.
     */
    void swap_ij(std::size_t i, std::size_t j)
    {
        if constexpr (std::is_same<Platform, chase::platform::CPU>::value)
        {
            std::vector<T> tmp(m_);
            chase::linalg::lapackpp::t_lacpy(
                'A', m_, 1, this->l_data() + i * ld_, 1, tmp.data(), 1);
            chase::linalg::lapackpp::t_lacpy('A', m_, 1,
                                             this->l_data() + j * ld_, 1,
                                             this->l_data() + i * ld_, 1);
            chase::linalg::lapackpp::t_lacpy('A', m_, 1, tmp.data(), 1,
                                             this->l_data() + j * ld_, 1);
        }
#ifdef HAS_CUDA
        else
        {
            T* tmp;
            CHECK_CUDA_ERROR(cudaMalloc(&tmp, m_ * sizeof(T)));
            chase::linalg::internal::cuda::t_lacpy(
                'A', m_, 1, this->l_data() + i * ld_, 1, tmp, 1);
            chase::linalg::internal::cuda::t_lacpy('A', m_, 1,
                                                   this->l_data() + j * ld_, 1,
                                                   this->l_data() + i * ld_, 1);
            chase::linalg::internal::cuda::t_lacpy('A', m_, 1, tmp, 1,
                                                   this->l_data() + j * ld_, 1);
            cudaFree(tmp);
        }
#endif
    }

#ifdef HAS_CUDA
    /**
     * @brief Transfers data from the device (GPU) to the host (CPU).
     */
    void D2H()
    {
        if constexpr (std::is_same<Platform, chase::platform::GPU>::value)
        {
            local_matrix_.D2H();
        }
        else
        {
            throw std::runtime_error("[DistMultiVector]: CPU type of matrix do "
                                     "not support D2H operation");
        }
    }
    /**
     * @brief Transfers data from the host (CPU) to the device (GPU).
     */
    void H2D()
    {
        if constexpr (std::is_same<Platform, chase::platform::GPU>::value)
        {
            local_matrix_.H2D();
        }
        else
        {
            throw std::runtime_error("[DistMultiVector]: CPU type of matrix do "
                                     "not support H2D operation");
        }
    }
#endif
    /**
     * @brief Retrieves a pointer to the data on the CPU.
     *
     * @return A pointer to the local matrix data on the CPU.
     */
    T* cpu_data() { return local_matrix_.cpu_data(); }
    /**
     * @brief Retrieves the leading dimension of the data on the CPU.
     *
     * @return The leading dimension of the local matrix on the CPU.
     */
    std::size_t cpu_ld() { return local_matrix_.cpu_ld(); }

    /**
     * @brief Redistributes the multi-vector data between two matrices with
     * different communicator types.
     *
     * @tparam target_comm_type The target communicator type (row or column).
     * @tparam OtherPlatform The target platform type.
     * @param targetMultiVector The target multi-vector to redistribute data to.
     * @param offset The starting offset for redistribution.
     * @param subsetSize The number of elements to redistribute.
     */
    template <CommunicatorType target_comm_type, typename OtherPlatform>
    void redistributeImpl(
        DistMultiVectorBlockCyclic1D<T, target_comm_type, OtherPlatform>*
            targetMultiVector,
        std::size_t offset, std::size_t subsetSize)
    {
        // Validate the subset range
        if (offset + subsetSize > this->g_cols() ||
            subsetSize > targetMultiVector->g_cols())
        {
            throw std::invalid_argument("Invalid subset range");
        }

        if constexpr (!std::is_same<Platform, OtherPlatform>::value)
        {
            throw std::runtime_error(
                "Cannot redistribute: Platform types do not match.");
        }

        // Check if the target matrix's communicator type matches the allowed
        // types
        if constexpr (comm_type == CommunicatorType::row &&
                      target_comm_type == CommunicatorType::column)
        {
            // Implement redistribution from row to column
            redistributeRowToColumn<OtherPlatform>(targetMultiVector, offset,
                                                   subsetSize);
        }
        else if constexpr (comm_type == CommunicatorType::column &&
                           target_comm_type == CommunicatorType::row)
        {
            // Implement redistribution from column to row
            redistributeColumnToRow<OtherPlatform>(targetMultiVector, offset,
                                                   subsetSize);
        }
        else
        {
            throw std::runtime_error(
                "Invalid redistribution between matrix types");
        }
    }

    /**
     * @brief Redistributes the multi-vector data between two matrices with
     * different communicator types.
     *
     * @tparam target_comm_type The target communicator type (row or column).
     * @tparam OtherPlatform The target platform type.
     * @param targetMultiVector The target multi-vector to redistribute data to.
     */
    template <CommunicatorType target_comm_type, typename OtherPlatform>
    void redistributeImpl(
        DistMultiVectorBlockCyclic1D<T, target_comm_type, OtherPlatform>*
            targetMultiVector)
    {
        this->redistributeImpl(targetMultiVector, 0, this->n_);
    }

#ifdef HAS_NCCL
    /**
     * @brief Asynchronously redistributes the multi-vector data between two
     * matrices with different communicator types using NCCL.
     *
     * @tparam target_comm_type The target communicator type (row or column).
     * @param targetMultiVector The target multi-vector to redistribute data to.
     * @param offset The starting offset for redistribution.
     * @param subsetSize The number of elements to redistribute.
     * @param stream_ Optional CUDA stream for asynchronous operation.
     */
    template <CommunicatorType target_comm_type>
    void redistributeImplAsync(
        DistMultiVectorBlockCyclic1D<T, target_comm_type, chase::platform::GPU>*
            targetMultiVector,
        std::size_t offset, std::size_t subsetSize,
        cudaStream_t* stream_ = nullptr)
    {

        cudaStream_t usedStream = (stream_ == nullptr) ? 0 : *stream_;

        // Validate the subset range
        if (offset + subsetSize > this->g_cols() ||
            subsetSize > targetMultiVector->g_cols())
        {
            throw std::invalid_argument("Invalid subset range");
        }

        if constexpr (!std::is_same<Platform, chase::platform::GPU>::value)
        {
            throw std::runtime_error(
                "NCCL based redistribution support only GPU.");
        }

        // Check if the target matrix's communicator type matches the allowed
        // types
        if constexpr (comm_type == CommunicatorType::row &&
                      target_comm_type == CommunicatorType::column)
        {
            // Implement redistribution from row to column
            redistributeRowToColumnAsync(targetMultiVector, offset, subsetSize,
                                         usedStream);
        }
        else if constexpr (comm_type == CommunicatorType::column &&
                           target_comm_type == CommunicatorType::row)
        {
            // Implement redistribution from column to row
            redistributeColumnToRowAsync(targetMultiVector, offset, subsetSize,
                                         usedStream);
        }
        else
        {
            throw std::runtime_error(
                "Invalid redistribution between matrix types");
        }
    }
    /**
     * @brief Asynchronously redistributes the multi-vector data between two
     * matrices with different communicator types using NCCL.
     *
     * @tparam target_comm_type The target communicator type (row or column).
     * @param targetMultiVector The target multi-vector to redistribute data to.
     * @param stream_ Optional CUDA stream for asynchronous operation.
     */
    template <CommunicatorType target_comm_type>
    void redistributeImplAsync(
        DistMultiVectorBlockCyclic1D<T, target_comm_type, chase::platform::GPU>*
            targetMultiVector,
        cudaStream_t* stream_ = nullptr)
    {
        this->redistributeImplAsync(targetMultiVector, 0, this->n_, stream_);
    }
#endif

    /**
     * @brief Get the global number of rows in the matrix.
     *
     * @return The global number of rows in the matrix across all processes.
     */
    std::size_t g_rows() const override { return M_; }

    /**
     * @brief Get the global number of columns in the matrix.
     *
     * @return The global number of columns in the matrix across all processes.
     */
    std::size_t g_cols() const override { return N_; }

    /**
     * @brief Get the global offset.
     *
     * @return The global offset of the local data.
     */
    std::size_t g_off() const override { return off_; }

    /**
     * @brief Get the local number of rows in this process.
     *
     * @return The local number of rows assigned to the current process.
     */
    std::size_t l_rows() const override { return m_; }

    /**
     * @brief Get the local number of columns in this process.
     *
     * @return The local number of columns assigned to the current process.
     */
    std::size_t l_cols() const override { return n_; }

    /**
     * @brief Get the leading dimension of the local matrix.
     *
     * @return The leading dimension (stride) of the local matrix.
     */
    std::size_t l_ld() const override { return ld_; }

    /**
     * @brief Get the block size for block-cyclic distribution.
     *
     * @return The block size used when distributing the matrix across
     * processes.
     */
    std::size_t mb() const override { return mb_; }

    /**
     * @brief Returns the number of row blocks.
     *
     * @return The number of row blocks.
     */
    std::size_t mblocks() const override { return mblocks_; }

    /**
     * @brief Get the block size used in the distributed matrix layout.
     *
     * @return The matrix block size.
     */
    std::size_t l_half() const override { return l_half_; }

    /**
     * @brief Get the pointer to the local matrix data.
     *
     * @return A pointer to the local matrix data stored in the current process.
     */
    T* l_data() override { return local_matrix_.data(); }

    // typename chase::platform::MatrixTypePlatform<T, Platform>::type&
    // loc_matrix() override { return local_matrix_;}
    /**
     * @brief Access the local matrix object.
     *
     * @return Reference to the local matrix of type chase::matrix::Matrix.
     */
    chase::matrix::Matrix<T, Platform>& loc_matrix() override
    {
        return local_matrix_;
    }

    std::vector<std::size_t> m_contiguous_global_offs()
    {
        return m_contiguous_global_offs_;
    }
    std::vector<std::size_t> m_contiguous_local_offs()
    {
        return m_contiguous_local_offs_;
    }
    std::vector<std::size_t> m_contiguous_lens() { return m_contiguous_lens_; }

#ifdef HAS_SCALAPACK
    std::size_t* get_scalapack_desc() { return desc_; }

    std::size_t* scalapack_descriptor_init()
    {
        if constexpr (std::is_same<Platform, chase::platform::GPU>::value)
        {
            if (local_matrix_.cpu_data() == nullptr)
            {
                local_matrix_.allocate_cpu_data();
            }
        }

        std::size_t ldd = this->cpu_ld();

        if constexpr (comm_type == CommunicatorType::column)
        {
            int* coords = mpi_grid_.get()->get_coords();
            int* dims = mpi_grid_.get()->get_dims();

            std::size_t default_blocksize = 64;
            std::size_t nb = std::min(n_, default_blocksize);
            int zero = 0;
            int one = 1;
            int info;
            int colcomm1D_ctxt = mpi_grid_.get()->get_blacs_colcomm_ctxt();
            chase::linalg::scalapackpp::t_descinit(
                desc_, &M_, &N_, &mb_, &nb, &zero, &zero, &colcomm1D_ctxt, &ldd,
                &info);
        }
        else
        {
            // row based will be implemented later
        }

        return desc_;
    }
#endif

private:
    /**
     * @brief The global number of rows in the matrix.
     *
     * This represents the total number of rows in the matrix across all
     * processes.
     */
    std::size_t M_;

    /**
     * @brief The global number of columns in the matrix.
     *
     * This represents the total number of columns in the matrix across all
     * processes.
     */
    std::size_t N_;

    /**
     * @brief The global offset for the local data.
     */

    std::size_t off_;

    /**
     * @brief The local number of rows in this process.
     *
     * This is the number of rows that are assigned to the current process in a
     * distributed setting.
     */
    std::size_t m_;

    /**
     * @brief The local number of columns in this process.
     *
     * This is the number of columns that are assigned to the current process in
     * a distributed setting.
     */
    std::size_t n_;

    /**
     * @brief The leading dimension of the local matrix.
     *
     * This is the stride (or padding) between consecutive rows of the local
     * matrix in memory.
     */
    std::size_t ld_;

    /**
     * @brief The block size for block-cyclic distribution.
     *
     * This specifies the block size used when distributing the matrix across
     * different processes.
     */
    std::size_t mb_;

    /**
     * @brief The number of blocks along the row dimension.
     *
     * This is used to determine how the matrix is divided into smaller blocks
     * for distributed computation.
     */
    std::size_t mblocks_;

    /**
     * @brief The local matrix stored in the current process.
     *
     * This matrix holds the local portion of the matrix assigned to the current
     * process. The type of this matrix is determined by the platform (CPU or
     * GPU).
     */
    chase::matrix::Matrix<T, Platform> local_matrix_;

    /**
     * @brief A shared pointer to the 2D MPI grid.
     *
     * This grid is used for distributing the matrix across processes in a 2D
     * decomposition.
     */
    std::shared_ptr<chase::grid::MpiGrid2DBase> mpi_grid_;
#ifdef HAS_SCALAPACK
    /**
     * @brief ScaLAPACK descriptor array for the matrix, used to manage
     * distributed matrix layout and storage.
     *
     * Contains 9 elements, where each element holds specific metadata required
     * for ScaLAPACK operations.
     */
    std::size_t desc_[9];
#endif

    std::size_t l_half_; ///< Local index for the lower half
    std::vector<std::size_t> m_contiguous_global_offs_;
    std::vector<std::size_t> m_contiguous_local_offs_;
    std::vector<std::size_t> m_contiguous_lens_;

    void init_contiguous_buffer_info()
    {
        int axis;

        if constexpr (comm_type ==
                      chase::distMultiVector::CommunicatorType::column)
        {
            axis = 0;
        }
        else if constexpr (comm_type ==
                           chase::distMultiVector::CommunicatorType::row)
        {
            axis = 1;
        }

        int coord = mpi_grid_.get()->get_coords()[axis];
        int dim = mpi_grid_.get()->get_dims()[axis];

        std::size_t nr;
        int sendr = 0;

        for (std::size_t r = 0; r < M_; r += mb_, sendr = (sendr + 1) % dim)
        {
            nr = mb_;
            if (M_ - r < mb_)
            {
                nr = M_ - r;
            }

            if (coord == sendr)
            {
                m_contiguous_global_offs_.push_back(r);
                m_contiguous_lens_.push_back(nr);
            }
        }

        m_contiguous_local_offs_.resize(mblocks_);

        m_contiguous_local_offs_[0] = 0;

        for (std::size_t i = 1; i < mblocks_; i++)
        {
            m_contiguous_local_offs_[i] =
                m_contiguous_local_offs_[i - 1] + m_contiguous_lens_[i - 1];
        }

        l_half_ = 0;
        std::size_t idx = 0;

        while (idx < mblocks_ &&
               m_contiguous_global_offs_[idx] + m_contiguous_lens_[idx] <=
                   (M_ / 2))
        {
            l_half_ += m_contiguous_lens_[idx];
            idx += 1;
        }

        if (idx < mblocks_)
        {
            if (int(M_ / 2) - int(m_contiguous_global_offs_[idx]) > 0)
            {
                l_half_ += std::size_t(int(M_ / 2) -
                                       int(m_contiguous_global_offs_[idx]));
            }
        }
    }
    // data for redistribution
    std::vector<std::size_t> orig_dests;
    std::vector<std::size_t> orig_srcs;
    std::vector<std::size_t> orig_lens;
    std::vector<std::size_t> target_disps;
    std::vector<std::size_t> orig_disps;

    template <typename OtherPlatform,
              chase::distMultiVector::CommunicatorType OtherCommType>
    void init_redistribution(
        DistMultiVectorBlockCyclic1D<T, OtherCommType, OtherPlatform>*
            targetMultiVector)
    {
        orig_dests = std::vector<std::size_t>();
        orig_srcs = std::vector<std::size_t>();
        orig_lens = std::vector<std::size_t>();
        target_disps = std::vector<std::size_t>();
        orig_disps = std::vector<std::size_t>();

        std::size_t orig_dest = 0;
        std::size_t orig_src = 0;
        orig_dests.push_back(orig_dest);
        orig_srcs.push_back(orig_src);
        std::size_t len = 1;
        std::size_t orig_disp = 0;
        std::size_t target_disp = 0;
        orig_disps.push_back(orig_disp);
        target_disps.push_back(target_disp);

        std::size_t mb = this->mb();
        std::size_t nb = targetMultiVector->mb();
        int* coords = mpi_grid_.get()->get_coords();
        int* dims = mpi_grid_.get()->get_dims();
        int dim0, dim1;
        if constexpr (comm_type ==
                      chase::distMultiVector::CommunicatorType::column)
        {
            dim0 = dims[0];
            dim1 = dims[1];
        }
        else if constexpr (comm_type ==
                           chase::distMultiVector::CommunicatorType::row)
        {
            dim0 = dims[1];
            dim1 = dims[0];
        }

        for (auto i = 1; i < M_; i++)
        {
            auto src_tmp = (i / mb) % dim0;
            auto dest_tmp = (i / nb) % dim1;
            if (dest_tmp == orig_dest && src_tmp == orig_src)
            {
                len += 1;
            }
            else
            {
                orig_lens.push_back(len);
                orig_dest = (i / nb) % dim1;
                target_disp = i % nb + ((i / nb) / dim1) * nb;
                orig_disp = i % mb + ((i / mb) / dim0) * mb;
                orig_src = (i / mb) % dim0;
                orig_srcs.push_back(orig_src);
                orig_dests.push_back(orig_dest);
                target_disps.push_back(target_disp);
                orig_disps.push_back(orig_disp);
                len = 1;
            }
        }
        orig_lens.push_back(len);
    }

    template <typename OtherPlatform>
    void redistributeRowToColumn(
        DistMultiVectorBlockCyclic1D<T, CommunicatorType::column,
                                     OtherPlatform>* targetMultiVector,
        std::size_t offset, std::size_t subsetSize)
    {
        // Ensure the dimensions are compatible
        if (this->M_ != targetMultiVector->g_rows() ||
            this->N_ != targetMultiVector->g_cols())
        {
            throw std::runtime_error(
                "Dimension mismatch during redistribution");
        }

        if (this->mpi_grid_.get() != targetMultiVector->getMpiGrid())
        {
            throw std::runtime_error("MPI Grid mismatch during redistribution");
        }

        if (this->mb_ != targetMultiVector->mb())
        {
            throw std::runtime_error("Blocksize of the original and target "
                                     "matrices mismatch during redistribution");
        }

        int* dims = this->mpi_grid_->get_dims();
        int* coords = this->mpi_grid_->get_coords();
        if (dims[0] == dims[1]) // squared grid
        {
            for (auto i = 0; i < dims[1]; i++)
            {
                if (coords[0] == i)
                {
                    if (coords[1] == i)
                    {
                        MPI_Bcast(this->l_data() + offset * this->ld_,
                                  this->m_ * subsetSize,
                                  chase::mpi::getMPI_Type<T>(), i,
                                  this->mpi_grid_->get_row_comm());
                    }
                    else
                    {
                        MPI_Bcast(targetMultiVector->l_data() +
                                      offset * targetMultiVector->l_ld(),
                                  targetMultiVector->l_rows() * subsetSize,
                                  chase::mpi::getMPI_Type<T>(), i,
                                  this->mpi_grid_->get_row_comm());
                    }
                }
            }

            for (auto i = 0; i < dims[1]; i++)
            {
                if (coords[0] == coords[1])
                {
                    if constexpr (std::is_same<Platform,
                                               chase::platform::CPU>::value)
                    {
                        chase::linalg::lapackpp::t_lacpy(
                            'A', this->m_, subsetSize,
                            this->l_data() + offset * this->ld_, this->ld_,
                            targetMultiVector->l_data() +
                                offset * targetMultiVector->l_ld(),
                            targetMultiVector->l_ld());
                    }
#ifdef HAS_CUDA
                    else
                    {
                        chase::linalg::internal::cuda::t_lacpy(
                            'A', this->m_, subsetSize,
                            this->l_data() + offset * this->ld_, this->ld_,
                            targetMultiVector->l_data() +
                                offset * targetMultiVector->l_ld(),
                            targetMultiVector->l_ld());
                    }
#endif
                }
            }
        }
        else
        {
            init_redistribution<
                OtherPlatform,
                chase::distMultiVector::CommunicatorType::column>(
                targetMultiVector);

            for (auto i = 0; i < orig_lens.size(); i++)
            {
                if (coords[0] == orig_dests[i])
                {
                    if constexpr (std::is_same<Platform,
                                               chase::platform::CPU>::value)
                    {
                        auto max_c_len =
                            *max_element(orig_lens.begin(), orig_lens.end());
                        std::unique_ptr<chase::matrix::Matrix<T>> buff =
                            std::make_unique<chase::matrix::Matrix<T>>(
                                max_c_len, subsetSize);
                        chase::linalg::lapackpp::t_lacpy(
                            'A', orig_lens[i], subsetSize,
                            this->l_data() + offset * this->ld_ + orig_disps[i],
                            this->ld_, buff->data(), orig_lens[i]);

                        MPI_Bcast(buff->data(), orig_lens[i] * subsetSize,
                                  chase::mpi::getMPI_Type<T>(), orig_srcs[i],
                                  this->mpi_grid_->get_row_comm());
                        chase::linalg::lapackpp::t_lacpy(
                            'A', orig_lens[i], subsetSize, buff->data(),
                            orig_lens[i],
                            targetMultiVector->l_data() + target_disps[i] +
                                offset * targetMultiVector->l_ld(),
                            targetMultiVector->l_ld());
                    }
#ifdef HAS_CUDA
                    else
                    {
                        auto max_c_len =
                            *max_element(orig_lens.begin(), orig_lens.end());
                        std::unique_ptr<
                            chase::matrix::Matrix<T, chase::platform::GPU>>
                            buff = std::make_unique<
                                chase::matrix::Matrix<T, chase::platform::GPU>>(
                                max_c_len, subsetSize);
                        chase::linalg::internal::cuda::t_lacpy(
                            'A', orig_lens[i], subsetSize,
                            this->l_data() + offset * this->ld_ + orig_disps[i],
                            this->ld_, buff->data(), orig_lens[i]);

                        MPI_Bcast(buff->data(), orig_lens[i] * subsetSize,
                                  chase::mpi::getMPI_Type<T>(), orig_srcs[i],
                                  this->mpi_grid_->get_row_comm());
                        chase::linalg::internal::cuda::t_lacpy(
                            'A', orig_lens[i], subsetSize, buff->data(),
                            orig_lens[i],
                            targetMultiVector->l_data() + target_disps[i] +
                                offset * targetMultiVector->l_ld(),
                            targetMultiVector->l_ld());
                    }
#endif
                }
            }
        }
    }

    template <typename OtherPlatform>
    void redistributeColumnToRow(
        DistMultiVectorBlockCyclic1D<T, CommunicatorType::row, OtherPlatform>*
            targetMultiVector,
        std::size_t offset, std::size_t subsetSize)
    {
        // Ensure the dimensions are compatible
        if (this->M_ != targetMultiVector->g_rows() ||
            this->N_ != targetMultiVector->g_cols())
        {
            throw std::runtime_error(
                "Dimension mismatch during redistribution");
        }

        if (this->mpi_grid_.get() != targetMultiVector->getMpiGrid())
        {
            throw std::runtime_error("MPI Grid mismatch during redistribution");
        }

        if (this->mb_ != targetMultiVector->mb())
        {
            throw std::runtime_error("Blocksize of the original and target "
                                     "matrices mismatch during redistribution");
        }

        int* dims = this->mpi_grid_->get_dims();
        int* coords = this->mpi_grid_->get_coords();
        if (dims[0] == dims[1]) // squared grid
        {
            for (auto i = 0; i < dims[0]; i++)
            {
                if (coords[1] == i)
                {
                    if (coords[0] == i)
                    {
                        MPI_Bcast(this->l_data() + offset * this->ld_,
                                  this->m_ * subsetSize,
                                  chase::mpi::getMPI_Type<T>(), i,
                                  this->mpi_grid_->get_col_comm());
                    }
                    else
                    {
                        MPI_Bcast(targetMultiVector->l_data() +
                                      offset * targetMultiVector->l_ld(),
                                  targetMultiVector->l_rows() * subsetSize,
                                  chase::mpi::getMPI_Type<T>(), i,
                                  this->mpi_grid_->get_col_comm());
                    }
                }
            }

            for (auto i = 0; i < dims[0]; i++)
            {
                if (coords[0] == coords[1])
                {
                    if constexpr (std::is_same<Platform,
                                               chase::platform::CPU>::value)
                    {
                        chase::linalg::lapackpp::t_lacpy(
                            'A', this->m_, subsetSize,
                            this->l_data() + offset * this->ld_, this->ld_,
                            targetMultiVector->l_data() +
                                offset * targetMultiVector->l_ld(),
                            targetMultiVector->l_ld());
                    }
#ifdef HAS_CUDA
                    else
                    {
                        chase::linalg::internal::cuda::t_lacpy(
                            'A', this->m_, subsetSize,
                            this->l_data() + offset * this->ld_, this->ld_,
                            targetMultiVector->l_data() +
                                offset * targetMultiVector->l_ld(),
                            targetMultiVector->l_ld());
                    }
#endif
                }
            }
        }
        else
        {
            init_redistribution<OtherPlatform,
                                chase::distMultiVector::CommunicatorType::row>(
                targetMultiVector);

            for (auto i = 0; i < orig_lens.size(); i++)
            {
                if (coords[1] == orig_dests[i])
                {
                    if constexpr (std::is_same<Platform,
                                               chase::platform::CPU>::value)
                    {
                        auto max_c_len =
                            *max_element(orig_lens.begin(), orig_lens.end());
                        std::unique_ptr<chase::matrix::Matrix<T>> buff =
                            std::make_unique<chase::matrix::Matrix<T>>(
                                max_c_len, subsetSize);
                        chase::linalg::lapackpp::t_lacpy(
                            'A', orig_lens[i], subsetSize,
                            this->l_data() + offset * this->ld_ + orig_disps[i],
                            this->ld_, buff->data(), orig_lens[i]);

                        MPI_Bcast(buff->data(), orig_lens[i] * subsetSize,
                                  chase::mpi::getMPI_Type<T>(), orig_srcs[i],
                                  this->mpi_grid_->get_col_comm());
                        chase::linalg::lapackpp::t_lacpy(
                            'A', orig_lens[i], subsetSize, buff->data(),
                            orig_lens[i],
                            targetMultiVector->l_data() + target_disps[i] +
                                offset * targetMultiVector->l_ld(),
                            targetMultiVector->l_ld());
                    }
#ifdef HAS_CUDA
                    else
                    {
                        auto max_c_len =
                            *max_element(orig_lens.begin(), orig_lens.end());
                        std::unique_ptr<
                            chase::matrix::Matrix<T, chase::platform::GPU>>
                            buff = std::make_unique<
                                chase::matrix::Matrix<T, chase::platform::GPU>>(
                                max_c_len, subsetSize);
                        chase::linalg::internal::cuda::t_lacpy(
                            'A', orig_lens[i], subsetSize,
                            this->l_data() + offset * this->ld_ + orig_disps[i],
                            this->ld_, buff->data(), orig_lens[i]);

                        MPI_Bcast(buff->data(), orig_lens[i] * subsetSize,
                                  chase::mpi::getMPI_Type<T>(), orig_srcs[i],
                                  this->mpi_grid_->get_col_comm());
                        chase::linalg::internal::cuda::t_lacpy(
                            'A', orig_lens[i], subsetSize, buff->data(),
                            orig_lens[i],
                            targetMultiVector->l_data() + target_disps[i] +
                                offset * targetMultiVector->l_ld(),
                            targetMultiVector->l_ld());
                    }
#endif
                }
            }
        }
    }

#ifdef HAS_CUDA
#ifdef HAS_NCCL
    void redistributeRowToColumnAsync(
        DistMultiVectorBlockCyclic1D<T, CommunicatorType::column,
                                     chase::platform::GPU>* targetMultiVector,
        std::size_t offset, std::size_t subsetSize, cudaStream_t stream)
    {
        // Ensure the dimensions are compatible
        if (this->M_ != targetMultiVector->g_rows() ||
            this->N_ != targetMultiVector->g_cols())
        {
            throw std::runtime_error(
                "Dimension mismatch during redistribution");
        }

        if (this->mpi_grid_.get() != targetMultiVector->getMpiGrid())
        {
            throw std::runtime_error("MPI Grid mismatch during redistribution");
        }

        if (this->mb_ != targetMultiVector->mb())
        {
            throw std::runtime_error("Blocksize of the original and target "
                                     "matrices mismatch during redistribution");
        }

        int* dims = this->mpi_grid_->get_dims();
        int* coords = this->mpi_grid_->get_coords();
        if (dims[0] == dims[1]) // squared grid
        {
            for (auto i = 0; i < dims[1]; i++)
            {
                if (coords[0] == i)
                {
                    if (coords[1] == i)
                    {
                        CHECK_NCCL_ERROR(chase::nccl::ncclBcastWrapper(
                            this->l_data() + offset * this->ld_,
                            this->m_ * subsetSize, i,
                            this->mpi_grid_->get_nccl_row_comm(), &stream));
                    }
                    else
                    {
                        CHECK_NCCL_ERROR(chase::nccl::ncclBcastWrapper(
                            targetMultiVector->l_data() +
                                offset * targetMultiVector->l_ld(),
                            targetMultiVector->l_rows() * subsetSize, i,
                            this->mpi_grid_->get_nccl_row_comm(), &stream));
                    }
                }
            }

            for (auto i = 0; i < dims[1]; i++)
            {
                if (coords[0] == coords[1])
                {

                    chase::linalg::internal::cuda::t_lacpy(
                        'A', this->m_, subsetSize,
                        this->l_data() + offset * this->ld_, this->ld_,
                        targetMultiVector->l_data() +
                            offset * targetMultiVector->l_ld(),
                        targetMultiVector->l_ld());
                }
            }
        }
        else
        {
            init_redistribution<
                chase::platform::GPU,
                chase::distMultiVector::CommunicatorType::column>(
                targetMultiVector);

            for (auto i = 0; i < orig_lens.size(); i++)
            {
                if (coords[0] == orig_dests[i])
                {
                    auto max_c_len =
                        *max_element(orig_lens.begin(), orig_lens.end());
                    std::unique_ptr<
                        chase::matrix::Matrix<T, chase::platform::GPU>>
                        buff = std::make_unique<
                            chase::matrix::Matrix<T, chase::platform::GPU>>(
                            max_c_len, subsetSize);
                    chase::linalg::internal::cuda::t_lacpy(
                        'A', orig_lens[i], subsetSize,
                        this->l_data() + offset * this->ld_ + orig_disps[i],
                        this->ld_, buff->data(), orig_lens[i]);

                    CHECK_NCCL_ERROR(chase::nccl::ncclBcastWrapper(
                        buff->data(), orig_lens[i] * subsetSize, orig_srcs[i],
                        this->mpi_grid_->get_nccl_row_comm(), &stream));

                    chase::linalg::internal::cuda::t_lacpy(
                        'A', orig_lens[i], subsetSize, buff->data(),
                        orig_lens[i],
                        targetMultiVector->l_data() + target_disps[i] +
                            offset * targetMultiVector->l_ld(),
                        targetMultiVector->l_ld());
                }
            }
        }
    }

    void redistributeColumnToRowAsync(
        DistMultiVectorBlockCyclic1D<T, CommunicatorType::row,
                                     chase::platform::GPU>* targetMultiVector,
        std::size_t offset, std::size_t subsetSize, cudaStream_t stream)
    {
        // Ensure the dimensions are compatible
        if (this->M_ != targetMultiVector->g_rows() ||
            this->N_ != targetMultiVector->g_cols())
        {
            throw std::runtime_error(
                "Dimension mismatch during redistribution");
        }

        if (this->mpi_grid_.get() != targetMultiVector->getMpiGrid())
        {
            throw std::runtime_error("MPI Grid mismatch during redistribution");
        }

        if (this->mb_ != targetMultiVector->mb())
        {
            throw std::runtime_error("Blocksize of the original and target "
                                     "matrices mismatch during redistribution");
        }

        int* dims = this->mpi_grid_->get_dims();
        int* coords = this->mpi_grid_->get_coords();
        if (dims[0] == dims[1]) // squared grid
        {
            for (auto i = 0; i < dims[0]; i++)
            {
                if (coords[1] == i)
                {
                    if (coords[0] == i)
                    {
                        CHECK_NCCL_ERROR(chase::nccl::ncclBcastWrapper(
                            this->l_data() + offset * this->ld_,
                            this->m_ * subsetSize, i,
                            this->mpi_grid_->get_nccl_col_comm(), &stream));
                    }
                    else
                    {
                        CHECK_NCCL_ERROR(chase::nccl::ncclBcastWrapper(
                            targetMultiVector->l_data() +
                                offset * targetMultiVector->l_ld(),
                            targetMultiVector->l_rows() * subsetSize, i,
                            this->mpi_grid_->get_nccl_col_comm(), &stream));
                    }
                }
            }

            for (auto i = 0; i < dims[0]; i++)
            {
                if (coords[0] == coords[1])
                {
                    chase::linalg::internal::cuda::t_lacpy(
                        'A', this->m_, subsetSize,
                        this->l_data() + offset * this->ld_, this->ld_,
                        targetMultiVector->l_data() +
                            offset * targetMultiVector->l_ld(),
                        targetMultiVector->l_ld());
                }
            }
        }
        else
        {
            init_redistribution<chase::platform::GPU,
                                chase::distMultiVector::CommunicatorType::row>(
                targetMultiVector);

            for (auto i = 0; i < orig_lens.size(); i++)
            {
                if (coords[1] == orig_dests[i])
                {
                    auto max_c_len =
                        *max_element(orig_lens.begin(), orig_lens.end());
                    std::unique_ptr<
                        chase::matrix::Matrix<T, chase::platform::GPU>>
                        buff = std::make_unique<
                            chase::matrix::Matrix<T, chase::platform::GPU>>(
                            max_c_len, subsetSize);
                    chase::linalg::internal::cuda::t_lacpy(
                        'A', orig_lens[i], subsetSize,
                        this->l_data() + offset * this->ld_ + orig_disps[i],
                        this->ld_, buff->data(), orig_lens[i]);

                    CHECK_NCCL_ERROR(chase::nccl::ncclBcastWrapper(
                        buff->data(), orig_lens[i] * subsetSize, orig_srcs[i],
                        this->mpi_grid_->get_nccl_col_comm(), &stream));

                    chase::linalg::internal::cuda::t_lacpy(
                        'A', orig_lens[i], subsetSize, buff->data(),
                        orig_lens[i],
                        targetMultiVector->l_data() + target_disps[i] +
                            offset * targetMultiVector->l_ld(),
                        targetMultiVector->l_ld());
                }
            }
        }
    }

#endif
#endif
};
/** @} */ // End of distMultiVector group

} // namespace distMultiVector
} // namespace chase

// This file is a part of ChASE.
// Copyright (c) 2015-2024, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include <memory>       // std::unique_ptr, std::allocator
#include <functional>   // std::function
#include <stdexcept>    // std::runtime_error
#include <type_traits>  // std::enable_if, std::is_same
#include <cstddef>      // std::size_t
#include <complex>      // std::complex (if you're using complex types)
#include "algorithm/types.hpp"
#include "external/lapackpp/lapackpp.hpp"
#include <omp.h>
#ifdef HAS_CUDA
#include <cuda_runtime.h>
#include "Impl/chase_gpu/cuda_utils.hpp"
#include "external/cublaspp/cublaspp.hpp"
#include "linalg/internal/cuda/precision_conversion.cuh"
#endif

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
 * @brief Matrix classes for CPU and GPU with memory management and mixed-precision support.
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
 * The `chase::platform` namespace includes identifiers for different platforms, such as CPU and GPU.
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
    struct CPU {};  
    /**
    * @ingroup PlatformTypes
    * @brief Represents the GPU platform.
    * 
    * This structure is used to specify that an operation or matrix class
    * is targeting the GPU platform.
    */
    struct GPU {};  
}
}

namespace chase {
namespace matrix {
    struct Hermitian {};
    struct PseudoHermitian {};
}
}

/**
 * @page platform_namespace chase::platform Namespace
 * @brief Namespace containing platform-specific structures and identifiers.
 *
 * The `chase::platform` namespace includes identifiers for different platforms, such as CPU and GPU.
 */

/** @page chase_matrix
 *  @brief The Matrix Namespace
 * 
 * This namespace contains matrix classes that work with different platforms.
 * Each matrix type is designed to operate on either CPU or GPU platforms.
 */
namespace chase {
namespace matrix {

/**
 * @ingroup BufferTypes
 * @brief Specifies the type of external buffer used for matrix data.
 */
enum class BufferType { 
    CPU, ///< Indicates that the external buffer is in CPU memory
    GPU  ///< Indicates that the external buffer is in GPU memory
};

#ifdef HAS_CUDA
/**
 * @ingroup Allocators
 * @brief Allocator for GPU memory allocation using CUDA.
 * @tparam T Data type of matrix elements.
 */
template <typename T>
class GpuAllocator {
public:
    using value_type = T;

    /// Default constructor.
    GpuAllocator() = default;

    /// Constructor for allocator type conversions.
    template <class U>
    constexpr GpuAllocator(const GpuAllocator<U>&) noexcept {}

    /**
     * Allocates memory on the GPU.
     * @param n Number of elements to allocate.
     * @return Pointer to the allocated memory.
     * @throws std::bad_alloc if allocation fails.
     */
    T* allocate(std::size_t n) {
        T* ptr = nullptr;
        if (cudaMalloc(&ptr, n * sizeof(T)) != cudaSuccess) {
            throw std::bad_alloc();
        }
        return ptr;
    }

    /**
     * Deallocates GPU memory.
     * @param ptr Pointer to memory to deallocate.
     * @param n Size of the allocation (unused).
     */
    void deallocate(T* ptr, std::size_t) noexcept {
        cudaFree(ptr);
    }
};
#endif

/**
 * @ingroup Allocators
 * @brief Default allocator type trait for CPU platform.
 *
 * Uses std::allocator on CPU.
 * @tparam Platform The platform type (CPU).
 */
template <typename Platform>
struct DefaultAllocator {
    template <typename T>
    /// Uses std::allocator as default for CPU platform.
    using type = std::allocator<T>; 
};

#ifdef HAS_CUDA
/// Specialization of DefaultAllocator for GPU platform.
template <>
struct DefaultAllocator<chase::platform::GPU> {
    template <typename T>
    using type = GpuAllocator<T>;  ///< Uses GpuAllocator for GPU platform.
};
#endif

template <typename T, typename Platform = chase::platform::CPU, template <typename> class Allocator = DefaultAllocator<Platform>::template type>
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
template <typename T, template <typename, typename, template <typename> class> class Derived, typename Platform, template <typename> class Allocator>
class AbstractMatrix
{
protected:
    using SinglePrecisionType = typename chase::ToSinglePrecisionTrait<T>::Type;
    using SinglePrecisionDerived = Derived<SinglePrecisionType, Platform, Allocator>;
    std::unique_ptr<SinglePrecisionDerived> single_precision_matrix_;
    bool is_single_precision_enabled_ = false; 

    // Double precision matrix
    using DoublePrecisionType = typename chase::ToDoublePrecisionTrait<T>::Type;
    using DoublePrecisionDerived = Derived<DoublePrecisionType, Platform, Allocator>;
    std::unique_ptr<DoublePrecisionDerived> double_precision_matrix_;
    bool is_double_precision_enabled_ = false;

    std::chrono::high_resolution_clock::time_point start, end;
    std::chrono::duration<double> elapsed;

public:
    using value_type = T;
    using platform_type = Platform;
    virtual ~AbstractMatrix() = default;
    /// Returns a pointer to the matrix data.
    virtual T *        data() = 0;
    /// Returns a pointer to the CPU data.
    virtual T *        cpu_data() = 0;
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
     * @throws std::runtime_error if the file cannot be opened or data is uninitialized.
     */
    void saveToBinaryFile(const std::string& filename) {
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("[Matrix]: Failed to open file for writing.");
        }
        
        if (this->cpu_data() == nullptr) {
            throw std::runtime_error("[Matrix]: Original CPU data is not initialized.");
        }

        std::size_t dataSize = this->rows() * this->cols() * sizeof(T);
        if (this->cpu_ld() > this->rows()) {
            std::vector<T> buffer( this->rows() * this->cols());
            chase::linalg::lapackpp::t_lacpy('A', this->rows(), this->cols(), this->cpu_data(), this->cpu_ld(), buffer.data(), this->rows());
            file.write(reinterpret_cast<const char*>(buffer.data()), dataSize);
        } else {
            file.write(reinterpret_cast<const char*>(this->cpu_data()), dataSize);
        }
    }

    /**
     * Reads matrix data from a binary file.
     * @param filename Path to the input file.
     * @throws std::runtime_error if the file cannot be opened, or its size is insufficient.
     */
    void readFromBinaryFile(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary | std::ios::in);
        if (!file.is_open()) {
            throw std::runtime_error("[Matrix]: Failed to open file for reading.");
        }
        
        if (this->cpu_data() == nullptr) {
            throw std::runtime_error("[Matrix]: Original CPU data is not initialized.");
        }

        // Check file size
        file.seekg(0, std::ios::end);
        std::streamsize fileSize = file.tellg();
        file.seekg(0, std::ios::beg);

        std::streamsize requiredSize = this->rows() * this->cols() * sizeof(T);
        if (fileSize < requiredSize) {
            throw std::runtime_error("[Matrix]: File size is smaller than the required matrix size.");
        }

        if (this->cpu_ld() > this->rows()) {
            std::vector<T> buffer(this->rows() * this->cols());
            file.read(reinterpret_cast<char*>(buffer.data()), requiredSize);
            chase::linalg::lapackpp::t_lacpy('A', this->rows(), this->cols(), buffer.data(), this->rows(), this->cpu_data(), this->cpu_ld());
        } else {
            file.read(reinterpret_cast<char*>(this->cpu_data()), requiredSize);
        }
    }

    /**
     * Enables single-precision representation for supported data types.
     * @tparam U Enables this function for double or complex double only.
     * @throws std::runtime_error if the single-precision matrix is uninitialized.
     */    
    template<typename U = T, typename std::enable_if<
        std::is_same<U, double>::value || std::is_same<U, std::complex<double>>::value, int>::type = 0>
    void enableSinglePrecision() 
    {
        if (!single_precision_matrix_) 
        {
            single_precision_matrix_ = std::make_unique<SinglePrecisionDerived>(this->rows(), this->cols());
        }    

        if constexpr (std::is_same<Platform, chase::platform::CPU>::value)
        {
            #pragma omp parallel for        
            for (std::size_t j = 0; j < this->cols(); ++j) {
                for (std::size_t i = 0; i < this->rows(); ++i) {
                    single_precision_matrix_->data()[j * single_precision_matrix_->ld() + i] = static_cast<SinglePrecisionType>(this->data()[j * this->ld() + i]);
                }
            }
        }
#ifdef HAS_CUDA
        else
        {
            chase::linalg::internal::cuda::convert_DP_TO_SP_GPU(this->data(), single_precision_matrix_->data(), this->rows() * this->cols());
        }
#endif
        is_single_precision_enabled_ = true;
    }

    /**
     * Disables single-precision representation.
     * @tparam U Enables this function for double or complex double only.
     * @param copyback If true, copies data back to original precision.
     * @throws std::runtime_error if single precision is not enabled or uninitialized.
     */
    template<typename U = T, typename std::enable_if<
        std::is_same<U, double>::value || std::is_same<U, std::complex<double>>::value, int>::type = 0>
    void disableSinglePrecision(bool copyback = false) {
        if (!is_single_precision_enabled_) {
            throw std::runtime_error("Single precision is not enabled.");
        }

        if (single_precision_matrix_ == nullptr) {
            throw std::runtime_error("Original SP data is not initialized.");
        }

        if(copyback){
            if constexpr (std::is_same<Platform, chase::platform::CPU>::value){
            // Convert single-precision data back to original precision
                #pragma omp parallel for        
                for (std::size_t j = 0; j < this->cols(); ++j) {
                    for (std::size_t i = 0; i < this->rows(); ++i) {
                        this->data()[j * this->ld() + i] = static_cast<T>(single_precision_matrix_->data()[j * single_precision_matrix_->ld() + i]);
                    }
                }  
                
            }
#ifdef HAS_CUDA
            else
            {
                chase::linalg::internal::cuda::convert_SP_TO_DP_GPU(single_precision_matrix_->data(), this->data(), this->rows() * this->cols());
            }
#endif
        }
   
        single_precision_matrix_.reset();
        is_single_precision_enabled_ = false;     
    }

    // Check if single precision is enabled
    template <typename U = T, typename std::enable_if<std::is_same<U, double>::value || std::is_same<U, std::complex<double>>::value, int>::type = 0>
    bool isSinglePrecisionEnabled() const {
        return is_single_precision_enabled_;
    }

    // Get the single precision matrix itself
    template <typename U = T, typename std::enable_if<std::is_same<U, double>::value || std::is_same<U, std::complex<double>>::value, int>::type = 0>
    SinglePrecisionDerived* matrix_sp() {
        if (is_single_precision_enabled_) {
            return single_precision_matrix_.get();
        } else {
            throw std::runtime_error("Single precision is not enabled.");
        }
    }

    // If T is already single precision, these methods should not be available
    template <typename U = T, typename std::enable_if<!std::is_same<U, double>::value && !std::is_same<U, std::complex<double>>::value, int>::type = 0>
    void enableSinglePrecision() {
        throw std::runtime_error("[DistMatrix]: Single precision operations not supported for this type.");
    }

    template <typename U = T, typename std::enable_if<!std::is_same<U, double>::value && !std::is_same<U, std::complex<double>>::value, int>::type = 0>
    void disableSinglePrecision() {
        throw std::runtime_error("[DistMatrix]: Single precision operations not supported for this type.");
    }

    // Enable double precision for float types (and complex<float>)
    template<typename U = T, typename std::enable_if<
        std::is_same<U, float>::value || std::is_same<U, std::complex<float>>::value, int>::type = 0>
    void enableDoublePrecision() 
    {
        if (!double_precision_matrix_) 
        {
            start = std::chrono::high_resolution_clock::now();
            double_precision_matrix_ = std::make_unique<DoublePrecisionDerived>(this->rows(), this->cols());
        }    

        if constexpr (std::is_same<Platform, chase::platform::CPU>::value)
        {
            #pragma omp parallel for        
            for (std::size_t j = 0; j < this->cols(); ++j) {
                for (std::size_t i = 0; i < this->rows(); ++i) {
                    double_precision_matrix_->data()[j * double_precision_matrix_->ld() + i] = static_cast<DoublePrecisionType>(this->data()[j * this->ld() + i]);
                }
            }
        }
#ifdef HAS_CUDA
        else
        {
            chase::linalg::internal::cuda::convert_SP_TO_DP_GPU(this->data(), double_precision_matrix_->data(), this->rows() * this->cols());
        }
#endif
        is_double_precision_enabled_ = true;
        end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    }

    // Disable double precision for float types
    template<typename U = T, typename std::enable_if<
        std::is_same<U, float>::value || std::is_same<U, std::complex<float>>::value, int>::type = 0>
    void disableDoublePrecision(bool copyback = false) {
        if (!is_double_precision_enabled_) {
            throw std::runtime_error("Double precision is not enabled.");
        }

        if (double_precision_matrix_ == nullptr) {
            throw std::runtime_error("Original DP data is not initialized.");
        }

        if(copyback){
            if constexpr (std::is_same<Platform, chase::platform::CPU>::value){
                #pragma omp parallel for        
                for (std::size_t j = 0; j < this->cols(); ++j) {
                    for (std::size_t i = 0; i < this->rows(); ++i) {
                        this->data()[j * this->ld() + i] = static_cast<T>(double_precision_matrix_->data()[j * double_precision_matrix_->ld() + i]);
                    }
                }  
            }
#ifdef HAS_CUDA
            else
            {
                chase::linalg::internal::cuda::convert_DP_TO_SP_GPU(double_precision_matrix_->data(), this->data(), this->rows() * this->cols());
            }
#endif
        }
   
        double_precision_matrix_.reset();
        is_double_precision_enabled_ = false;     
    }

    // Check if double precision is enabled
    template <typename U = T, typename std::enable_if<std::is_same<U, float>::value || std::is_same<U, std::complex<float>>::value, int>::type = 0>
    bool isDoublePrecisionEnabled() const {
        return is_double_precision_enabled_;
    }

    // Get the double precision matrix itself
    template <typename U = T, typename std::enable_if<std::is_same<U, float>::value || std::is_same<U, std::complex<float>>::value, int>::type = 0>
    DoublePrecisionDerived* matrix_dp() {
        if (is_double_precision_enabled_) {
            return double_precision_matrix_.get();
        } else {
            throw std::runtime_error("Double precision is not enabled.");
        }
    }

    // If T is already double precision, these methods should not be available
    template <typename U = T, typename std::enable_if<!std::is_same<U, float>::value && !std::is_same<U, std::complex<float>>::value, int>::type = 0>
    void enableDoublePrecision() {
        throw std::runtime_error("[Matrix]: Double precision operations not supported for this type.");
    }

    template <typename U = T, typename std::enable_if<!std::is_same<U, float>::value && !std::is_same<U, std::complex<float>>::value, int>::type = 0>
    void disableDoublePrecision() {
        throw std::runtime_error("[Matrix]: Double precision operations not supported for this type.");
    }

    // Copy from double to single precision (if T is double)
    template<typename U = T, typename std::enable_if<
        std::is_same<U, double>::value || std::is_same<U, std::complex<double>>::value, int>::type = 0>
    void copyTo() {
        if (!single_precision_matrix_) {
            throw std::runtime_error("Single precision matrix is not enabled.");
        }

        if constexpr (std::is_same<Platform, chase::platform::CPU>::value) {
            #pragma omp parallel for        
            for (std::size_t j = 0; j < this->cols(); ++j) {
                for (std::size_t i = 0; i < this->rows(); ++i) {
                    single_precision_matrix_->data()[j * single_precision_matrix_->ld() + i] = 
                        static_cast<SinglePrecisionType>(this->data()[j * this->ld() + i]);
                }
            }
        }
#ifdef HAS_CUDA
        else {
            chase::linalg::internal::cuda::convert_DP_TO_SP_GPU(
                this->data(), single_precision_matrix_->data(), this->rows() * this->cols());
        }
#endif
    }

    // Copy from single to double precision (if T is double)
    template<typename U = T, typename std::enable_if<
        std::is_same<U, double>::value || std::is_same<U, std::complex<double>>::value, int>::type = 0>
    void copyBack() {
        if (!single_precision_matrix_) {
            throw std::runtime_error("Single precision matrix is not enabled.");
        }

        if constexpr (std::is_same<Platform, chase::platform::CPU>::value) {
            #pragma omp parallel for        
            for (std::size_t j = 0; j < this->cols(); ++j) {
                for (std::size_t i = 0; i < this->rows(); ++i) {
                    this->data()[j * this->ld() + i] = 
                        static_cast<T>(single_precision_matrix_->data()[j * single_precision_matrix_->ld() + i]);
                }
            }
        }
#ifdef HAS_CUDA
        else {
            chase::linalg::internal::cuda::convert_SP_TO_DP_GPU(
                single_precision_matrix_->data(), this->data(), this->rows() * this->cols());
        }
#endif
    }

    // Copy from single to double precision (if T is float)
    template<typename U = T, typename std::enable_if<
        std::is_same<U, float>::value || std::is_same<U, std::complex<float>>::value, int>::type = 0>
    void copyTo() {
        if (!double_precision_matrix_) {
            throw std::runtime_error("Double precision matrix is not enabled.");
        }

        if constexpr (std::is_same<Platform, chase::platform::CPU>::value) {
            #pragma omp parallel for        
            for (std::size_t j = 0; j < this->cols(); ++j) {
                for (std::size_t i = 0; i < this->rows(); ++i) {
                    double_precision_matrix_->data()[j * double_precision_matrix_->ld() + i] = 
                        static_cast<DoublePrecisionType>(this->data()[j * this->ld() + i]);
                }
            }
        }
#ifdef HAS_CUDA
        else {
            chase::linalg::internal::cuda::convert_SP_TO_DP_GPU(
                this->data(), double_precision_matrix_->data(), this->rows() * this->cols());
        }
#endif
    }

    // Copy from double to single precision (if T is float)
    template<typename U = T, typename std::enable_if<
        std::is_same<U, float>::value || std::is_same<U, std::complex<float>>::value, int>::type = 0>
    void copyBack() {
        if (!double_precision_matrix_) {
            throw std::runtime_error("Double precision matrix is not enabled.");
        }

        if constexpr (std::is_same<Platform, chase::platform::CPU>::value) {
            #pragma omp parallel for        
            for (std::size_t j = 0; j < this->cols(); ++j) {
                for (std::size_t i = 0; i < this->rows(); ++i) {
                    this->data()[j * this->ld() + i] = 
                        static_cast<T>(double_precision_matrix_->data()[j * double_precision_matrix_->ld() + i]);
                }
            }
        }
#ifdef HAS_CUDA
        else {
            chase::linalg::internal::cuda::convert_DP_TO_SP_GPU(
                double_precision_matrix_->data(), this->data(), this->rows() * this->cols());
        }
#endif
    }
};

/**
 * @ingroup MatrixClasses
 * @brief A matrix class for CPU-based matrix operations.
 * 
 * This class represents a matrix stored in CPU memory, with support for memory allocation and deallocation, 
 * as well as external data handling. It also supports move semantics and memory management via custom allocators.
 *
 * @tparam T Data type of matrix elements.
 * @tparam Allocator Memory allocator type (default is std::allocator).
 */
template <typename T, template <typename> class Allocator>
class Matrix<T, chase::platform::CPU, Allocator> : public AbstractMatrix<T, Matrix, chase::platform::CPU, Allocator> 
{
private:
    std::unique_ptr<T[], std::function<void(T*)>> data_; ///< Smart pointer to manage matrix data with custom deleter.
    T* external_data_; ///< Pointer to external data, if provided by the user.
    std::size_t rows_; ///< Number of rows in the matrix.
    std::size_t cols_; ///< Number of columns in the matrix.
    std::size_t ld_; ///< Leading dimension of the matrix (number of elements in a column).
    bool owns_mem_; ///< Flag to indicate whether the matrix owns its memory.
    Allocator<T> allocator_; ///< Memory allocator for matrix elements.

public:
    using hermitian_type = chase::matrix::Hermitian; ///< Alias for the Hermitian matrix type.

    /**
     * @brief Default constructor.
     * 
     * Initializes an empty matrix with no allocated memory.
     */
    Matrix()
        : data_(nullptr, [](T*){}), external_data_(nullptr), rows_(0), cols_(0), ld_(0), 
          owns_mem_(false)        
          {}

    /**
     * @brief Constructor to allocate a matrix with specified dimensions.
     * 
     * Allocates memory using the provided allocator and sets the matrix size to the specified number of rows and columns.
     * The matrix is initialized with zeros.
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
            [this](T* ptr) { allocator_.deallocate(ptr, rows_ * cols_); }
        );

        external_data_ = nullptr; 
    }

    /**
     * @brief Constructor wrapping user-provided data.
     * 
     * Initializes the matrix with user-supplied data, but does not take ownership of the memory.
     * 
     * @param rows Number of rows in the matrix.
     * @param cols Number of columns in the matrix.
     * @param ld Leading dimension of the matrix.
     * @param data Pointer to the user-provided matrix data.
     */
    Matrix(std::size_t rows, std::size_t cols, std::size_t ld, T *data)
        : data_(nullptr, [](T*){}), external_data_(data), rows_(rows), cols_(cols), ld_(ld), owns_mem_(false)       
        {}

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
          ld_(other.ld_), allocator_(std::move(other.allocator_)), owns_mem_(other.owns_mem_)          
    {
        this->single_precision_matrix_ = std::move(other.single_precision_matrix_);
        this->is_single_precision_enabled_ = other.is_single_precision_enabled_;
        this->double_precision_matrix_ = std::move(other.double_precision_matrix_);
        this->is_double_precision_enabled_ = other.is_double_precision_enabled_;
        other.owns_mem_ = false;  // Make sure the moved-from object no longer owns the memory
    }

    /**
     * @brief Move assignment operator.
     * 
     * Transfers ownership of memory and matrix data from another matrix object, freeing any existing memory.
     * 
     * @param other Another matrix object to move from.
     * @return Reference to this matrix object.
     */
    Matrix& operator=(Matrix&& other) noexcept {
        if (this != &other) {
            if (owns_mem_) {
                allocator_.deallocate(data_.get(), rows_ * cols_);  // Free existing memory
            }

            data_ = std::move(other.data_);  // Transfer ownership of memory
            external_data_ = other.external_data_;
            rows_ = other.rows_;
            cols_ = other.cols_;
            ld_ = other.ld_;
            allocator_ = std::move(other.allocator_);
            owns_mem_ = other.owns_mem_;
            this->single_precision_matrix_ = std::move(other.single_precision_matrix_);
            this->is_single_precision_enabled_ = other.is_single_precision_enabled_;
            this->double_precision_matrix_ = std::move(other.double_precision_matrix_);
            this->is_double_precision_enabled_ = other.is_double_precision_enabled_;
            other.single_precision_matrix_ = nullptr;  // Transfer ownership of low precision data
            other.double_precision_matrix_ = nullptr;  // Transfer ownership of high precision data
            other.owns_mem_ = false;  // The moved-from object should no longer own the memory
        }
        return *this;
    }

    /**
     * @brief Swap function to exchange the data of two matrices.
     * 
     * Exchanges all matrix properties (data, size, memory ownership) with another matrix.
     * 
     * @param other Another matrix object to swap with.
     */
    void swap(Matrix& other) {
        std::swap(data_, other.data_);
        std::swap(external_data_, other.external_data_);
        std::swap(rows_, other.rows_);
        std::swap(cols_, other.cols_);
        std::swap(ld_, other.ld_);
        std::swap(owns_mem_, other.owns_mem_);
        std::swap(allocator_, other.allocator_);
        std::swap(this->is_single_precision_enabled_, other.is_single_precision_enabled_);
        std::swap(this->single_precision_matrix_, other.single_precision_matrix_);
        std::swap(this->is_double_precision_enabled_, other.is_double_precision_enabled_);
        std::swap(this->double_precision_matrix_, other.double_precision_matrix_);
    }

    /**
     * @brief Get a pointer to the matrix data.
     * 
     * Returns either the internally allocated data or the externally provided data.
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

#ifdef HAS_CUDA
/**
 * @ingroup MatrixClasses
 * @brief A matrix class for handling matrix data on both CPU and GPU.
 * 
 * This class manages a matrix that can reside both on the GPU and CPU. It uses CUDA to allocate and copy data between the CPU and GPU. It handles memory management, including external memory and internal allocation.
 * 
 * @tparam T Type of elements in the matrix.
 * @tparam Allocator Type of the allocator for managing memory.
 */
template <typename T, template <typename> class Allocator>
class Matrix<T, chase::platform::GPU, Allocator> : public AbstractMatrix<T, Matrix, chase::platform::GPU, Allocator> {
private:
    std::unique_ptr<T[], std::function<void(T*)>> gpu_data_; ///< Pointer to the GPU data.
    std::unique_ptr<T[], std::function<void(T*)>> cpu_data_; ///< Pointer to the CPU data.
    T *external_gpu_data_ = nullptr; ///< Pointer to external GPU data, if provided.
    T *external_cpu_data_ = nullptr; ///< Pointer to external CPU data, if provided.
    std::size_t rows_; ///< Number of rows in the matrix.
    std::size_t cols_; ///< Number of columns in the matrix.
    std::size_t gpu_ld_; ///< Leading dimension for the GPU data.
    std::size_t cpu_ld_; ///< Leading dimension for the CPU data.
    bool owns_cpu_mem_; ///< Flag to indicate if the CPU memory is owned by the object.
    bool owns_gpu_mem_; ///< Flag to indicate if the GPU memory is owned by the object.
    Allocator<T> allocator_; ///< Allocator used to manage memory.

public:
    using hermitian_type = chase::matrix::Hermitian; ///< Alias for the Hermitian matrix type.

    /**
     * @brief Default constructor for a Matrix.
     * Initializes a matrix with no data and sets default values for dimensions and memory ownership flags.
     */
    Matrix()
        : gpu_data_(nullptr, [](T*){}), cpu_data_(nullptr, [](T*){}), rows_(0), cols_(0), gpu_ld_(0),
          cpu_ld_(0), owns_cpu_mem_(false), owns_gpu_mem_(false)
        {}

    /**
     * @brief Constructor for a Matrix with specified dimensions.
     * Allocates memory on the GPU for the matrix data.
     * 
     * @param rows The number of rows in the matrix.
     * @param cols The number of columns in the matrix.
     */
    Matrix(std::size_t rows, std::size_t cols)
        : gpu_data_(nullptr, [](T*){}), cpu_data_(nullptr, [](T*){}), rows_(rows), cols_(cols), gpu_ld_(rows), cpu_ld_(0), owns_cpu_mem_(false), owns_gpu_mem_(true)
    {     
        T* raw_data = allocator_.allocate(rows_ * cols_);
        CHECK_CUDA_ERROR(cudaMemset(raw_data, 0, rows * cols * sizeof(T)));

        gpu_data_ = std::unique_ptr<T[], std::function<void(T*)>>(
            raw_data,
            [this](T* ptr) { allocator_.deallocate(ptr, rows_ * cols_); }
        );
    }

    /**
     * @brief Constructor that initializes the matrix with external data.
     * This constructor supports both CPU and GPU buffers and handles memory allocation accordingly.
     * 
     * @param rows The number of rows in the matrix.
     * @param cols The number of columns in the matrix.
     * @param ld The leading dimension of the matrix.
     * @param external_data Pointer to external data (CPU or GPU).
     * @param buffer_type The type of external buffer (CPU or GPU).
     */
    Matrix(std::size_t rows, std::size_t cols, std::size_t ld, T *external_data, BufferType buffer_type = BufferType::CPU)
        : gpu_data_(nullptr, [](T*){}), cpu_data_(nullptr, [](T*){}),
          external_gpu_data_(buffer_type == BufferType::GPU ? external_data : nullptr),
          external_cpu_data_(buffer_type == BufferType::CPU ? external_data : nullptr),
          rows_(rows), cols_(cols),
          gpu_ld_(buffer_type == BufferType::GPU ? ld : rows), 
          cpu_ld_(buffer_type == BufferType::CPU ? ld : 0),
          owns_cpu_mem_(false),
          owns_gpu_mem_(buffer_type == BufferType::GPU ? false : true)
    {
        if(buffer_type == BufferType::CPU)
        {
            //if provided external cpu, allocaate gpu internally
            T* raw_data = allocator_.allocate(rows_ * cols_);

            gpu_data_ = std::unique_ptr<T[], std::function<void(T*)>>(
                raw_data,
                [this](T* ptr) { allocator_.deallocate(ptr, rows_ * cols_); }
            );

            CHECK_CUBLAS_ERROR(cublasSetMatrix(rows_, cols_, sizeof(T), external_cpu_data_, cpu_ld_,
                            gpu_data_.get(), gpu_ld_)); 
        }
        //if provided external gpu, cpu is not allocated, until necessary
    }

    /**
     * @brief Move constructor to transfer ownership of resources.
     * 
     * This constructor moves the resources (GPU and CPU data) from another Matrix object.
     * 
     * @param other The Matrix object from which resources are moved.
     */
    Matrix(Matrix&& other) noexcept
        : gpu_data_(std::move(other.gpu_data_)), 
          cpu_data_(std::move(other.cpu_data_)), 
          external_gpu_data_(other.external_gpu_data_), 
          external_cpu_data_(other.external_cpu_data_),
          rows_(other.rows_),
          cols_(other.cols_), 
          gpu_ld_(other.gpu_ld_), 
          cpu_ld_(other.cpu_ld_),
          owns_cpu_mem_(other.owns_cpu_mem_),
          owns_gpu_mem_(other.owns_gpu_mem_)
    {
        this->single_precision_matrix_ = std::move(other.single_precision_matrix_);
        this->is_single_precision_enabled_ = other.is_single_precision_enabled_;
        this->double_precision_matrix_ = std::move(other.double_precision_matrix_);
        this->is_double_precision_enabled_ = other.is_double_precision_enabled_;
        other.gpu_data_ = nullptr; // The moved-from object no longer owns the GPU data
        other.cpu_data_ = nullptr; // The moved-from object no longer owns the CPU data
        other.owns_cpu_mem_ = false; // No longer owns the memory
        other.owns_gpu_mem_ = false; // No longer owns the memory
        other.external_cpu_data_ = nullptr;
        other.external_gpu_data_ = nullptr;
    }

    /**
     * @brief Move assignment operator to transfer ownership of resources.
     * 
     * This operator moves the resources (GPU and CPU data) from another Matrix object to the current one.
     * 
     * @param other The Matrix object from which resources are moved.
     * @return Reference to the current object.
     */
    Matrix& operator=(Matrix&& other) noexcept {
        if (this != &other) {
            // Transfer ownership
            gpu_data_ = std::move(other.gpu_data_);
            cpu_data_ = std::move(other.cpu_data_);
            external_gpu_data_ = other.external_gpu_data_;            
            external_cpu_data_ = other.external_cpu_data_;
            rows_ = other.rows_;
            cols_ = other.cols_;
            gpu_ld_ = other.gpu_ld_;
            cpu_ld_ = other.cpu_ld_;
            owns_cpu_mem_ = other.owns_cpu_mem_;
            owns_gpu_mem_ = other.owns_gpu_mem_;
            this->single_precision_matrix_ = std::move(other.single_precision_matrix_);
            this->is_single_precision_enabled_ = other.is_single_precision_enabled_;
            this->double_precision_matrix_ = std::move(other.double_precision_matrix_);
            this->is_double_precision_enabled_ = other.is_double_precision_enabled_;
            other.single_precision_matrix_ = nullptr;  // Transfer ownership of low precision data
            other.double_precision_matrix_ = nullptr;  // Transfer ownership of high precision data
            other.owns_cpu_mem_ = false; // No longer owns the memory
            other.owns_gpu_mem_ = false; // No longer owns the memory
            other.external_cpu_data_ = nullptr;
            other.external_gpu_data_ = nullptr;
        }
        return *this;
    }

    /**
     * @brief Swap function to exchange data between two matrices.
     * 
     * This function swaps the GPU, CPU, and other matrix data between two Matrix objects.
     * 
     * @param other The other Matrix object to swap data with.
     */
    void swap(Matrix& other) noexcept {
        std::swap(gpu_data_, other.gpu_data_);
        std::swap(cpu_data_, other.cpu_data_);
        std::swap(external_gpu_data_, other.external_gpu_data_);
        std::swap(external_cpu_data_, other.external_cpu_data_);
        std::swap(rows_, other.rows_);
        std::swap(cols_, other.cols_);
        std::swap(gpu_ld_, other.gpu_ld_);
        std::swap(cpu_ld_, other.cpu_ld_);
        std::swap(owns_cpu_mem_, other.owns_cpu_mem_);
        std::swap(owns_gpu_mem_, other.owns_gpu_mem_);
        std::swap(this->is_single_precision_enabled_, other.is_single_precision_enabled_);
        std::swap(this->single_precision_matrix_, other.single_precision_matrix_);
        std::swap(this->is_double_precision_enabled_, other.is_double_precision_enabled_);
        std::swap(this->double_precision_matrix_, other.double_precision_matrix_);
    }

    /**
     * @brief Destructor for the Matrix class.
     * Releases any resources held by the Matrix object.
     */
    ~Matrix() {       
    }

    /**
     * @brief Allocates memory for CPU data.
     * 
     * This function allocates pinned memory for the CPU data if the matrix owns the CPU memory.
     */
    void allocate_cpu_data()
    {
        if(!owns_cpu_mem_ || cpu_data_ == nullptr)
        {
            T* raw_data;
            CHECK_CUDA_ERROR(cudaMallocHost(&raw_data, rows_ * cols_ * sizeof(T))); // Allocate pinned CPU buffer
            //raw_data = (T *)malloc(rows_ * cols_ * sizeof(T));
            memset(raw_data, 0, rows_ * cols_ * sizeof(T));            
            //cudaHostUnregister(raw_data);

            cpu_data_ = std::unique_ptr<T[], std::function<void(T*)>>(
                raw_data,
                [this](T* ptr) 
                { 
                    cudaFreeHost(ptr); 
                    //free(ptr);
                }
            );

            owns_cpu_mem_ = true;
            cpu_ld_ = rows_;
        }
    }
    /**
     * @brief Returns a pointer to the CPU data buffer.
     * 
     * This function returns the CPU data pointer. If the matrix does not own the CPU memory but has external CPU data, it returns the external CPU data. If the matrix owns the CPU memory, it returns the internal allocated CPU data. 
     * If neither condition is met, it throws a runtime error.
     * 
     * @return Pointer to the CPU data.
     * @throws std::runtime_error If the CPU buffer is not allocated or provided.
     */
    T* cpu_data () override
    {       
        if(!owns_cpu_mem_ && external_cpu_data_ != nullptr)
        {
            return external_cpu_data_;
        }else if(owns_cpu_mem_ && cpu_data_ != nullptr)
        {
            return cpu_data_.get();
        }else
        {
            throw std::runtime_error("CPU buffer is not allocated or provided");
        }
    }

    /**
     * @brief Returns a pointer to the matrix data (either GPU or external GPU data).
     * 
     * This function returns the matrix data pointer. If the matrix owns the GPU memory, it returns the internal GPU data pointer. If not, it returns the external GPU data pointer.
     * 
     * @return Pointer to the matrix data (either GPU or external).
     */
    T *data() override 
    {
        return owns_gpu_mem_ ? gpu_data_.get() : external_gpu_data_; 
    }

    /**
     * @brief Copies the matrix data from CPU to GPU.
     * 
     * This function copies the data from the CPU buffer (if allocated or external) to the GPU buffer. It uses cuBLAS functions to perform the matrix transfer.
     */
    void H2D()
    {
        T *src_data = this->cpu_data();
        T *target_data = this->data();
        CHECK_CUBLAS_ERROR(cublasSetMatrix(rows_, cols_, sizeof(T), src_data, cpu_ld_,
                        target_data, gpu_ld_));        
    }

    /**
     * @brief Copies the matrix data from GPU to CPU.
     * 
     * This function copies the data from the GPU buffer to the CPU buffer. It uses cuBLAS functions to perform the matrix transfer. If the CPU buffer is not already allocated, it is allocated before copying the data.
     */
    void D2H()
    {
        if(cpu_data_ == nullptr && external_cpu_data_ == nullptr)
        {
            this->allocate_cpu_data();
        }
        T *dest_data = this->cpu_data();
        CHECK_CUBLAS_ERROR(cublasGetMatrix(rows_, cols_, sizeof(T), this->data(), gpu_ld_,
                        dest_data, this->cpu_ld()));        
    }
    /**
     * @brief Returns the number of rows in the matrix.
     * 
     * This function returns the number of rows in the matrix.
     * 
     * @return The number of rows in the matrix.
     */
    std::size_t rows() const { return rows_; }
    /**
     * @brief Returns the number of columns in the matrix.
     * 
     * This function returns the number of columns in the matrix.
     * 
     * @return The number of columns in the matrix.
     */    
    std::size_t cols() const override { return cols_; }
    /**
     * @brief Returns the leading dimension of the matrix (GPU).
     * 
     * This function returns the leading dimension (stride) of the matrix as it is stored in GPU memory. 
     * The leading dimension is used to calculate the memory layout for matrix operations.
     * 
     * @return The leading dimension of the matrix in GPU memory.
     */    
    std::size_t ld() const override { return gpu_ld_; }
    /**
     * @brief Returns the leading dimension of the matrix (CPU).
     * 
     * This function returns the leading dimension (stride) of the matrix as it is stored in CPU memory.
     * 
     * @return The leading dimension of the matrix in CPU memory.
     */    
    std::size_t cpu_ld() const override { return cpu_ld_; }

};
#endif

template <typename T, typename Platform = chase::platform::CPU, template <typename> class Allocator = DefaultAllocator<Platform>::template type>
class PseudoHermitianMatrix;

template <typename T, template <typename> class Allocator>
class PseudoHermitianMatrix<T, chase::platform::CPU, Allocator> : public Matrix<T, chase::platform::CPU, Allocator> {
public:
    using hermitian_type = chase::matrix::PseudoHermitian; ///< Alias for the Pseudo-Hermitian matrix type.
    // Default constructor
    PseudoHermitianMatrix() : Matrix<T, chase::platform::CPU, Allocator>() {}
    
    // Constructor with dimensions
    PseudoHermitianMatrix(std::size_t rows, std::size_t cols) 
        : Matrix<T, chase::platform::CPU, Allocator>(rows, cols) {}
    
    // Constructor with external data
    PseudoHermitianMatrix(std::size_t rows, std::size_t cols, std::size_t ld, T* data)
        : Matrix<T, chase::platform::CPU, Allocator>(rows, cols, ld, data) {}
};

#ifdef HAS_CUDA
template <typename T, template <typename> class Allocator>
class PseudoHermitianMatrix<T, chase::platform::GPU, Allocator> : public Matrix<T, chase::platform::GPU, Allocator> {
public:
    using hermitian_type = chase::matrix::PseudoHermitian; ///< Alias for the Pseudo-Hermitian matrix type.
    // Default constructor
    PseudoHermitianMatrix() : Matrix<T, chase::platform::GPU, Allocator>() {}
    
    // Constructor with dimensions
    PseudoHermitianMatrix(std::size_t rows, std::size_t cols) 
        : Matrix<T, chase::platform::GPU, Allocator>(rows, cols) {}
    
    // Constructor with external data
    PseudoHermitianMatrix(std::size_t rows, std::size_t cols, std::size_t ld, T* data, 
                         chase::matrix::BufferType buffer_type = chase::matrix::BufferType::CPU)
        : Matrix<T, chase::platform::GPU, Allocator>(rows, cols, ld, data, buffer_type) {}

};
#endif

}  // namespace matrix
}  // namespace chase



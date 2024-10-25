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
#include "Impl/cuda/cuda_utils.hpp"
#include "external/cublaspp/cublaspp.hpp"
#include "linalg/internal/cuda/precision_conversion.cuh"
#endif

namespace chase
{
namespace platform
{
    struct CPU {};  // Represents CPU platform
    struct GPU {};  // Represents GPU platform
}
}

namespace chase {
namespace matrix {

enum class BufferType { 
    CPU, // Indicates that the external buffer is in CPU memory
    GPU  // Indicates that the external buffer is in GPU memory
};

#ifdef HAS_CUDA
template <typename T>
class GpuAllocator {
public:
    using value_type = T;

    GpuAllocator() = default;

    template <class U>
    constexpr GpuAllocator(const GpuAllocator<U>&) noexcept {}

    T* allocate(std::size_t n) {
        T* ptr = nullptr;
        if (cudaMalloc(&ptr, n * sizeof(T)) != cudaSuccess) {
            throw std::bad_alloc();
        }
        return ptr;
    }

    void deallocate(T* ptr, std::size_t) noexcept {
        cudaFree(ptr);
    }
};
#endif

// DefaultAllocator type trait
template <typename Platform>
struct DefaultAllocator {
    template <typename T>
    using type = std::allocator<T>;  // Default to std::allocator for unknown platforms
};

#ifdef HAS_CUDA
// Specialize DefaultAllocator for GPU
template <>
struct DefaultAllocator<chase::platform::GPU> {
    template <typename T>
    using type = GpuAllocator<T>;  // Use GpuAllocator for GPU platform
};
#endif

template <typename T, typename Platform = chase::platform::CPU, template <typename> class Allocator = DefaultAllocator<Platform>::template type>
class Matrix;

template <typename T, template <typename, typename, template <typename> class> class Derived, typename Platform, template <typename> class Allocator>
class AbstractMatrix
{
    using value_type = T;
    using platform_type = Platform;
protected:
#ifdef ENABLE_MIXED_PRECISION       
    using SinglePrecisionType = typename chase::ToSinglePrecisionTrait<T>::Type;
    using SinglePrecisionDerived = Derived<SinglePrecisionType, Platform, Allocator>;
    std::unique_ptr<SinglePrecisionDerived> single_precision_matrix_;
    bool is_single_precision_enabled_ = false; 
#endif

public:
    virtual ~AbstractMatrix() = default;
    virtual T *        data() = 0;
    virtual T *        cpu_data() = 0;
    virtual std::size_t rows() const = 0;
    virtual std::size_t cols() const = 0;
    virtual std::size_t ld() const = 0;
    virtual std::size_t cpu_ld() const = 0;

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

    // Read matrix data from a binary file
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

#ifdef ENABLE_MIXED_PRECISION       
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

    // Disable low precision: only enabled if T is double or std::complex<double>
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

#endif
};

template <typename T, template <typename> class Allocator>
class Matrix<T, chase::platform::CPU, Allocator> : public AbstractMatrix<T, Matrix, chase::platform::CPU, Allocator> 
{
private:
    std::unique_ptr<T[], std::function<void(T*)>> data_;
    T* external_data_;
    std::size_t rows_;
    std::size_t cols_;
    std::size_t ld_;
    bool owns_mem_;
    Allocator<T> allocator_;

public:
    // Default constructor
    Matrix()
        : data_(nullptr, [](T*){}), external_data_(nullptr), rows_(0), cols_(0), ld_(0), 
          owns_mem_(false)        
          {}

    // Constructor with default allocator
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

    // Constructor wrapping up user-provided pointers
    Matrix(std::size_t rows, std::size_t cols, std::size_t ld, T *data)
        : data_(nullptr, [](T*){}), rows_(rows), cols_(cols), ld_(ld), external_data_(data), owns_mem_(false)       
        {}


    // Move constructor
    Matrix(Matrix&& other) noexcept
        : data_(std::move(other.data_)), rows_(other.rows_), cols_(other.cols_), 
          ld_(other.ld_), allocator_(std::move(other.allocator_)), owns_mem_(other.owns_mem_)          
    {
#ifdef ENABLE_MIXED_PRECISION       
        this->single_precision_matrix_ = std::move(other.single_precision_matrix_);
        this->is_single_precision_enabled_ = other.is_single_precision_enabled_;
#endif
        other.owns_mem_ = false;  // Make sure the moved-from object no longer owns the memory
    }

    // Move assignment operator
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
#ifdef ENABLE_MIXED_PRECISION
            this->single_precision_matrix_ = std::move(other.single_precision_matrix_);
            this->is_single_precision_enabled_ = other.is_single_precision_enabled_;
            other.single_precision_matrix_ = nullptr;  // Transfer ownership of low precision data
#endif            
            other.owns_mem_ = false;  // The moved-from object should no longer own the memory

        }
        return *this;
    }

    // Swap function to exchange data between two matrices
    void swap(Matrix& other) {
        std::swap(data_, other.data_);
        std::swap(external_data_, other.external_data_);
        std::swap(rows_, other.rows_);
        std::swap(cols_, other.cols_);
        std::swap(ld_, other.ld_);
        std::swap(owns_mem_, other.owns_mem_);
        std::swap(allocator_, other.allocator_);
#ifdef ENABLE_MIXED_PRECISION
        std::swap(this->is_single_precision_enabled_, other.is_single_precision_enabled_);
        std::swap(this->single_precision_matrix_, other.single_precision_matrix_);
#endif        
    }

    T* data() override { return owns_mem_ ? data_.get() : external_data_; }
    T* cpu_data() override { return owns_mem_ ? data_.get() : external_data_; }
    std::size_t rows() const override { return rows_; }
    std::size_t cols() const override { return cols_; }
    std::size_t ld() const override { return ld_; }
    std::size_t cpu_ld() const override { return ld_; }

};

#ifdef HAS_CUDA
template <typename T, template <typename> class Allocator>
class Matrix<T, chase::platform::GPU, Allocator> : public AbstractMatrix<T, Matrix, chase::platform::GPU, Allocator> {
private:
    std::unique_ptr<T[], std::function<void(T*)>> gpu_data_;
    std::unique_ptr<T[], std::function<void(T*)>> cpu_data_;
    T *external_gpu_data_ = nullptr;
    T *external_cpu_data_ = nullptr;
    std::size_t rows_;
    std::size_t cols_;
    std::size_t gpu_ld_;
    std::size_t cpu_ld_;
    bool owns_cpu_mem_;
    bool owns_gpu_mem_;
    Allocator<T> allocator_;

public:
    // Default constructor
    Matrix()
        : cpu_data_(nullptr, [](T*){}), gpu_data_(nullptr, [](T*){}), rows_(0), cols_(0), gpu_ld_(0),
          cpu_ld_(0), owns_cpu_mem_(false), owns_gpu_mem_(false)
        {}

    Matrix(std::size_t rows, std::size_t cols)
        : cpu_data_(nullptr, [](T*){}), rows_(rows), cols_(cols), cpu_ld_(0), gpu_ld_(rows), owns_cpu_mem_(false), owns_gpu_mem_(true)
    {     
        T* raw_data = allocator_.allocate(rows_ * cols_);
        CHECK_CUDA_ERROR(cudaMemset(raw_data, 0, rows * cols * sizeof(T)));

        gpu_data_ = std::unique_ptr<T[], std::function<void(T*)>>(
            raw_data,
            [this](T* ptr) { allocator_.deallocate(ptr, rows_ * cols_); }
        );
    }

    Matrix(std::size_t rows, std::size_t cols, std::size_t ld, T *external_data, BufferType buffer_type = BufferType::CPU)
        : rows_(rows), cols_(cols),
          gpu_ld_(buffer_type == BufferType::GPU ? ld : rows), 
          cpu_ld_(buffer_type == BufferType::CPU ? ld : 0),
          cpu_data_(nullptr, [](T*){}), gpu_data_(nullptr, [](T*){}),
          external_gpu_data_(buffer_type == BufferType::GPU ? external_data : nullptr),
          external_cpu_data_(buffer_type == BufferType::CPU ? external_data : nullptr),        
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

    // Move constructor
    Matrix(Matrix&& other) noexcept
        : gpu_data_(std::move(other.gpu_data_)), 
          cpu_data_(std::move(other.cpu_data_)), 
          rows_(other.rows_),
          cols_(other.cols_), 
          gpu_ld_(other.gpu_ld_), 
          cpu_ld_(other.cpu_ld_),
          owns_cpu_mem_(other.owns_cpu_mem_),
          owns_gpu_mem_(other.owns_gpu_mem_),
          external_cpu_data_(other.external_cpu_data_),
          external_gpu_data_(other.external_gpu_data_)   
    {
#ifdef ENABLE_MIXED_PRECISION
          this->single_precision_matrix_ = std::move(other.single_precision_matrix_);
          this->is_single_precision_enabled_(other.is_single_precision_enabled_) ;
#endif          
        other.gpu_data_ = nullptr; // The moved-from object no longer owns the GPU data
        other.cpu_data_ = nullptr; // The moved-from object no longer owns the CPU data
        other.owns_cpu_mem_ = false; // No longer owns the memory
        other.owns_gpu_mem_ = false; // No longer owns the memory
        other.external_cpu_data_ = nullptr;
        other.external_gpu_data_ = nullptr;
    }

    // Move assignment operator
    Matrix& operator=(Matrix&& other) noexcept {
        if (this != &other) {
            // Transfer ownership
            gpu_data_ = std::move(other.gpu_data_);
            cpu_data_ = std::move(other.cpu_data_);
            rows_ = other.rows_;
            cols_ = other.cols_;
            gpu_ld_ = other.gpu_ld_;
            cpu_ld_ = other.cpu_ld_;
            external_cpu_data_ = other.external_cpu_data_;
            external_gpu_data_ = other.external_gpu_data_;            
            owns_cpu_mem_ = other.owns_cpu_mem_;
            owns_gpu_mem_ = other.owns_gpu_mem_;
#ifdef ENABLE_MIXED_PRECISION
            this->single_precision_matrix_ = std::move(other.single_precision_matrix_);
            this->is_single_precision_enabled_ = other.is_single_precision_enabled_;
            other.single_precision_matrix_ = nullptr;  // Transfer ownership of low precision data
#endif  

            // Nullify the other
            other.gpu_data_ = nullptr; // The moved-from object no longer owns the GPU data
            other.cpu_data_ = nullptr; // The moved-from object no longer owns the CPU data
            other.owns_cpu_mem_ = false; // No longer owns the memory
            other.owns_gpu_mem_ = false; // No longer owns the memory
            other.external_cpu_data_ = nullptr;
            other.external_gpu_data_ = nullptr;
        }
        return *this;
    }

    // Swap function to exchange data between two matrices
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
#ifdef ENABLE_MIXED_PRECISION
        std::swap(this->is_single_precision_enabled_, other.is_single_precision_enabled_);
        std::swap(this->single_precision_matrix_, other.single_precision_matrix_);
#endif 
    }

    ~Matrix() {       
    }

    void allocate_cpu_data()
    {
        if(!owns_cpu_mem_ || cpu_data_ == nullptr)
        {
            T* raw_data;
            CHECK_CUDA_ERROR(cudaMallocHost(&raw_data, rows_ * cols_ * sizeof(T))); // Allocate pinned CPU buffer
            memset(raw_data, 0, rows_ * cols_ * sizeof(T));            

            cpu_data_ = std::unique_ptr<T[], std::function<void(T*)>>(
                raw_data,
                [this](T* ptr) { cudaFreeHost(ptr);; }
            );

            owns_cpu_mem_ = true;
            cpu_ld_ = rows_;
        }
    }

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

    T *data() override 
    {
        return owns_gpu_mem_ ? gpu_data_.get() : external_gpu_data_; 
    }

    void H2D()
    {
        T *src_data = this->cpu_data();
        T *target_data = this->data();
        CHECK_CUBLAS_ERROR(cublasSetMatrix(rows_, cols_, sizeof(T), src_data, cpu_ld_,
                        target_data, gpu_ld_));        
    }

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

    std::size_t rows() const { return rows_; }
    std::size_t cols() const override { return cols_; }
    std::size_t ld() const override { return gpu_ld_; }
    std::size_t cpu_ld() const override { return cpu_ld_; }

};
#endif

}  // namespace matrix
}  // namespace chase



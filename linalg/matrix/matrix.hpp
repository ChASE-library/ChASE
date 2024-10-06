#pragma once

#include <memory>       // std::unique_ptr, std::allocator
#include <functional>   // std::function
#include <stdexcept>    // std::runtime_error
#include <type_traits>  // std::enable_if, std::is_same
#include <cstddef>      // std::size_t
#include <complex>      // std::complex (if you're using complex types)
#include "algorithm/types.hpp"
#include "linalg/lapackpp/lapackpp.hpp"
#include <omp.h>
#ifdef HAS_CUDA
#include <cuda_runtime.h>
#include "Impl/cuda/cuda_utils.hpp"
#include "linalg/cublaspp/cublaspp.hpp"
#include "linalg/internal/cuda/precision_conversion.cuh"
#endif

namespace chase {
namespace matrix {

template <typename T, typename Allocator = std::allocator<T>>
class MatrixCPU {
private:
    std::unique_ptr<T[], std::function<void(T*)>> data_;
    T* external_data_;
    std::size_t rows_;
    std::size_t cols_;
    std::size_t ld_;
    bool owns_mem_;
    Allocator allocator_;
#ifdef ENABLE_MIXED_PRECISION
    using singlePrecisionT = typename chase::ToSinglePrecisionTrait<T>::Type;
    //singlePrecisionT* single_precision_data_;  // Low precision data (raw pointer)
    std::unique_ptr<MatrixCPU<singlePrecisionT>> single_precision_data_;
    bool single_precision_enabled_;  // Flag to track if low precision is enabled
#endif
public:
    // Default constructor
    MatrixCPU()
        : data_(nullptr, [](T*){}), external_data_(nullptr), rows_(0), cols_(0), ld_(0), 
          owns_mem_(false)
#ifdef ENABLE_MIXED_PRECISION
          , single_precision_data_(nullptr), single_precision_enabled_(false) 
#endif          
          {}

    // Constructor with default allocator
    MatrixCPU(std::size_t rows, std::size_t cols)
        : rows_(rows), cols_(cols), ld_(rows), owns_mem_(true)
#ifdef ENABLE_MIXED_PRECISION
        , single_precision_data_(nullptr), single_precision_enabled_(false) 
#endif        
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
    MatrixCPU(std::size_t rows, std::size_t cols, std::size_t ld, T *data)
        : data_(nullptr, [](T*){}), rows_(rows), cols_(cols), ld_(ld), external_data_(data), owns_mem_(false)
#ifdef ENABLE_MIXED_PRECISION
        , single_precision_data_(nullptr), single_precision_enabled_(false) 
#endif        
        {}

    // Destructor
    ~MatrixCPU() {
    }

    // Move constructor
    MatrixCPU(MatrixCPU&& other) noexcept
        : data_(std::move(other.data_)), rows_(other.rows_), cols_(other.cols_), 
          ld_(other.ld_), allocator_(std::move(other.allocator_)), owns_mem_(other.owns_mem_)
#ifdef ENABLE_MIXED_PRECISION
          ,single_precision_data_(std::move(other.single_precision_data_)),
          single_precision_enabled_(other.single_precision_enabled_) 
#endif          
    {
        other.owns_mem_ = false;  // Make sure the moved-from object no longer owns the memory
    }

    // Move assignment operator
    MatrixCPU& operator=(MatrixCPU&& other) noexcept {
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
            single_precision_data_ = std::move(other.single_precision_data_);
            single_precision_enabled_ = other.single_precision_enabled_;
            other.single_precision_data_ = nullptr;  // Transfer ownership of low precision data
#endif            
            other.owns_mem_ = false;  // The moved-from object should no longer own the memory

        }
        return *this;
    }

    // Swap function to exchange data between two matrices
    void swap(MatrixCPU<T, Allocator>& other) {
        std::swap(data_, other.data_);
        std::swap(external_data_, other.external_data_);
        std::swap(rows_, other.rows_);
        std::swap(cols_, other.cols_);
        std::swap(ld_, other.ld_);
        std::swap(owns_mem_, other.owns_mem_);
        std::swap(allocator_, other.allocator_);
#ifdef ENABLE_MIXED_PRECISION
        std::swap(single_precision_enabled_, other.single_precision_enabled_);
        std::swap(single_precision_data_, other.single_precision_data_);
#endif        
    }

    void saveToBinaryFile(const std::string& filename) {
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("[MatrixCPU]: Failed to open file for writing.");
        }
        
        if (this->data() == nullptr) {
            throw std::runtime_error("[MatrixCPU]: Original data is not initialized.");
        }

        std::size_t dataSize = rows_ * cols_ * sizeof(T);
        if (ld_ > rows_) {
            std::vector<T> buffer(rows_ * cols_);
            chase::linalg::lapackpp::t_lacpy('A', rows_, cols_, this->data(), ld_, buffer.data(), rows_);
            file.write(reinterpret_cast<const char*>(buffer.data()), dataSize);
        } else {
            file.write(reinterpret_cast<const char*>(this->data()), dataSize);
        }
    }

    // Read matrix data from a binary file
    void readFromBinaryFile(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary | std::ios::in);
        if (!file.is_open()) {
            throw std::runtime_error("[MatrixCPU]: Failed to open file for reading.");
        }
        
        if (this->data() == nullptr) {
            throw std::runtime_error("[MatrixCPU]: Original data is not initialized.");
        }

        // Check file size
        file.seekg(0, std::ios::end);
        std::streamsize fileSize = file.tellg();
        file.seekg(0, std::ios::beg);

        std::streamsize requiredSize = rows_ * cols_ * sizeof(T);
        if (fileSize < requiredSize) {
            throw std::runtime_error("[MatrixCPU]: File size is smaller than the required matrix size.");
        }

        if (ld_ > rows_) {
            std::vector<T> buffer(rows_ * cols_);
            file.read(reinterpret_cast<char*>(buffer.data()), requiredSize);
            chase::linalg::lapackpp::t_lacpy('A', rows_, cols_, buffer.data(), rows_, this->data(), ld_);
        } else {
            file.read(reinterpret_cast<char*>(this->data()), requiredSize);
        }
    }
#ifdef ENABLE_MIXED_PRECISION
    // Enable low precision: only enabled if T is double or std::complex<double>
    template<typename U = T, typename std::enable_if<
        std::is_same<U, double>::value || std::is_same<U, std::complex<double>>::value, int>::type = 0>
    void enableSinglePrecision() {
        if (single_precision_enabled_) {
            throw std::runtime_error("Single precision already enabled.");
        }

        if (this->data() == nullptr) {
            throw std::runtime_error("Original data is not initialized.");
        }

        //single_precision_data_ = new singlePrecisionT[ld_ * cols_];
        single_precision_data_ = std::make_unique<MatrixCPU<singlePrecisionT>>(rows_, cols_);
        // Convert original data to single precision
        #pragma omp parallel for        
        for (std::size_t j = 0; j < cols_; ++j) {
            for (std::size_t i = 0; i < rows_; ++i) {
                single_precision_data_->data()[j * rows_ + i] = static_cast<singlePrecisionT>(this->data()[j * ld_ + i]);
            }
        }
        single_precision_enabled_ = true;
    }

    // Disable low precision: only enabled if T is double or std::complex<double>
    template<typename U = T, typename std::enable_if<
        std::is_same<U, double>::value || std::is_same<U, std::complex<double>>::value, int>::type = 0>
    void disableSinglePrecision(bool copyback = false) {
        if (!single_precision_enabled_) {
            throw std::runtime_error("Single precision is not enabled.");
        }

        if (single_precision_data_ == nullptr) {
            throw std::runtime_error("Original SP data is not initialized.");
        }

        if(copyback){
        // Convert single-precision data back to original precision
            #pragma omp parallel for        
            for (std::size_t j = 0; j < cols_; ++j) {
                for (std::size_t i = 0; i < rows_; ++i) {
                    this->data()[j * ld_ + i] = static_cast<T>(single_precision_data_->data()[j * ld_ + i]);
                }
            }     
        }
   
        single_precision_data_.reset();
        single_precision_enabled_ = false;     
    }
#endif
    // Accessor methods
    T* data() { return owns_mem_ ? data_.get() : external_data_; }
#ifdef ENABLE_MIXED_PRECISION
    MatrixCPU<singlePrecisionT>* matrix_sp() { return single_precision_data_.get(); }
#endif    
    std::size_t rows() const { return rows_; }
    std::size_t cols() const { return cols_; }
    std::size_t ld() const { return ld_; }
#ifdef ENABLE_MIXED_PRECISION
    bool isSinglePrecisionEnabled() const { return single_precision_enabled_; }

    // Disabled methods when T is not double or complex<double>
    template<typename U = T, typename std::enable_if<
        !(std::is_same<U, double>::value || std::is_same<U, std::complex<double>>::value), int>::type = 0>
    void enableSinglePrecision() {
        throw std::runtime_error("Single precision operations are only supported for double or complex<double>.");
    }

    template<typename U = T, typename std::enable_if<
        !(std::is_same<U, double>::value || std::is_same<U, std::complex<double>>::value), int>::type = 0>
    void disableSinglePrecision(bool copyback = false) {
        throw std::runtime_error("Single precision operations are only supported for double or complex<double>.");
    }
#endif    
};


#ifdef HAS_CUDA
template <typename T>
class MatrixGPU {
private:
    T *gpu_data_ = nullptr;
    T *cpu_data_ = nullptr;
    std::size_t rows_;
    std::size_t cols_;
    std::size_t gpu_ld_;
    std::size_t cpu_ld_;
    bool owns_cpu_mem_;
#ifdef ENABLE_MIXED_PRECISION
    using singlePrecisionT = typename chase::ToSinglePrecisionTrait<T>::Type;
    //singlePrecisionT* single_precision_data_;  // Low precision data (raw pointer)
    std::unique_ptr<MatrixGPU<singlePrecisionT>> single_precision_data_ = nullptr;
    bool single_precision_enabled_ = false;  // Flag to track if low precision is enabled
#endif

public:
    // Default constructor
    MatrixGPU()
        : cpu_data_(nullptr), gpu_data_(nullptr), rows_(0), cols_(0), gpu_ld_(0),
          cpu_ld_(0), owns_cpu_mem_(false)
        {}

    MatrixGPU(std::size_t rows, std::size_t cols)
        : rows_(rows), cols_(cols), cpu_ld_(0), gpu_ld_(rows), owns_cpu_mem_(false)
    {        
        CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void **>(&gpu_data_), rows * cols * sizeof(T)));      // Allocate GPU buffer
        CHECK_CUDA_ERROR(cudaMemset(gpu_data_, 0, rows * cols * sizeof(T)));
    }

    MatrixGPU(std::size_t rows, std::size_t cols, std::size_t ld, T *cpu_data)
        : rows_(rows), cols_(cols), cpu_ld_(ld), gpu_ld_(rows), cpu_data_(cpu_data), owns_cpu_mem_(true)
    {
        CHECK_CUDA_ERROR(cudaMalloc(&gpu_data_, rows * cols * sizeof(T)));      // Allocate GPU buffer
        CHECK_CUBLAS_ERROR(cublasSetMatrix(rows_, cols_, sizeof(T), cpu_data_, cpu_ld_,
                        gpu_data_, gpu_ld_)); 
    }

    // Move constructor
    MatrixGPU(MatrixGPU&& other) noexcept
        : gpu_data_(other.gpu_data_), cpu_data_(other.cpu_data_), rows_(other.rows_),
          cols_(other.cols_), gpu_ld_(other.gpu_ld_), cpu_ld_(other.cpu_ld_),
          owns_cpu_mem_(other.owns_cpu_mem_)
#ifdef ENABLE_MIXED_PRECISION
          ,single_precision_data_(std::move(other.single_precision_data_)),
          single_precision_enabled_(other.single_precision_enabled_) 
#endif     
    {
        other.gpu_data_ = nullptr; // The moved-from object no longer owns the GPU data
        other.cpu_data_ = nullptr; // The moved-from object no longer owns the CPU data
        other.owns_cpu_mem_ = false; // No longer owns the memory
    }

    // Move assignment operator
    MatrixGPU& operator=(MatrixGPU&& other) noexcept {
        if (this != &other) {
            // Free existing GPU memory if it owns it
            cudaFree(gpu_data_);
            if (owns_cpu_mem_ && cpu_data_ != nullptr) {
                cudaFreeHost(cpu_data_);
            }

            // Transfer ownership
            gpu_data_ = other.gpu_data_;
            cpu_data_ = other.cpu_data_;
            rows_ = other.rows_;
            cols_ = other.cols_;
            gpu_ld_ = other.gpu_ld_;
            cpu_ld_ = other.cpu_ld_;
            owns_cpu_mem_ = other.owns_cpu_mem_;
#ifdef ENABLE_MIXED_PRECISION
            single_precision_data_ = std::move(other.single_precision_data_);
            single_precision_enabled_ = other.single_precision_enabled_;
            other.single_precision_data_ = nullptr;  // Transfer ownership of low precision data
#endif  

            // Nullify the other
            other.gpu_data_ = nullptr;
            other.cpu_data_ = nullptr;
            other.owns_cpu_mem_ = false;
        }
        return *this;
    }

    // Swap function to exchange data between two matrices
    void swap(MatrixGPU& other) noexcept {
        std::swap(gpu_data_, other.gpu_data_);
        std::swap(cpu_data_, other.cpu_data_);
        std::swap(rows_, other.rows_);
        std::swap(cols_, other.cols_);
        std::swap(gpu_ld_, other.gpu_ld_);
        std::swap(cpu_ld_, other.cpu_ld_);
        std::swap(owns_cpu_mem_, other.owns_cpu_mem_);
#ifdef ENABLE_MIXED_PRECISION
        std::swap(single_precision_enabled_, other.single_precision_enabled_);
        std::swap(single_precision_data_, other.single_precision_data_);
#endif 
    }

    ~MatrixGPU() {
        cudaFree(gpu_data_);
        if(owns_cpu_mem_ && cpu_data_!= nullptr)
        {
            cudaFreeHost(cpu_data_);
        }        
    }

    void allocate_cpu_data()
    {
        if(!owns_cpu_mem_ || cpu_data_ == nullptr)
        {
            CHECK_CUDA_ERROR(cudaMallocHost(&cpu_data_, rows_ * cols_ * sizeof(T))); // Allocate pinned CPU buffer
            memset(cpu_data_, 0, rows_ * cols_ * sizeof(T));            
            owns_cpu_mem_ = true;
            cpu_ld_ = rows_;
        }else
        {
            //std::cerr << "[MatrixGPU]: Warning> CPU data is already allocated" << std::endl;
        }
    }

    T* cpu_data() 
    {       
        if(!owns_cpu_mem_ || cpu_data_ == nullptr)
        {
            //std::cerr << "[MatrixGPU]: Warning> CPU data is not allocated yet, nullptr is returned" << std::endl;
        }  
        return cpu_data_; 
    }

    void H2D()
    {
        T *src_data = this->cpu_data();
        CHECK_CUBLAS_ERROR(cublasSetMatrix(rows_, cols_, sizeof(T), src_data, cpu_ld_,
                        this->gpu_data(), gpu_ld_));        
    }

    void D2H()
    {
        if(cpu_data_ == nullptr)
        {
            this->allocate_cpu_data();
        }
        T *dest_data = this->cpu_data();
        CHECK_CUBLAS_ERROR(cublasGetMatrix(rows_, cols_, sizeof(T), this->gpu_data(), gpu_ld_,
                        dest_data, this->cpu_ld()));        
    }

    void saveToBinaryFile(const std::string& filename) {
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("[MatrixGPU]: Failed to open file for writing.");
        }

        std::size_t dataSize = rows_ * cols_ * sizeof(T);
        if (cpu_ld_ > rows_) {
            std::vector<T> buffer(rows_ * cols_);
            chase::linalg::lapackpp::t_lacpy('A', rows_, cols_, this->cpu_data(), cpu_ld_, buffer.data(), rows_);
            file.write(reinterpret_cast<const char*>(buffer.data()), dataSize);
        } else {
            file.write(reinterpret_cast<const char*>(this->cpu_data()), dataSize);
        }
    }

    // Read matrix data from a binary file
    void readFromBinaryFile(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary | std::ios::in);
        if (!file.is_open()) {
            throw std::runtime_error("[MatrixGPU]: Failed to open file for reading.");
        }
        
        // Check file size
        file.seekg(0, std::ios::end);
        std::streamsize fileSize = file.tellg();
        file.seekg(0, std::ios::beg);

        std::streamsize requiredSize = rows_ * cols_ * sizeof(T);
        if (fileSize < requiredSize) {
            throw std::runtime_error("[MatrixCPU]: File size is smaller than the required matrix size.");
        }

        if (cpu_ld_ > rows_) {
            std::vector<T> buffer(rows_ * cols_);
            file.read(reinterpret_cast<char*>(buffer.data()), requiredSize);
            chase::linalg::lapackpp::t_lacpy('A', rows_, cols_, buffer.data(), rows_, this->cpu_data(), cpu_ld_);
        } else {
            file.read(reinterpret_cast<char*>(this->cpu_data()), requiredSize);
        }
    }

#ifdef ENABLE_MIXED_PRECISION
    // Enable low precision: only enabled if T is double or std::complex<double>
    template<typename U = T, typename std::enable_if<
        std::is_same<U, double>::value || std::is_same<U, std::complex<double>>::value, int>::type = 0>
    void enableSinglePrecision() {
        if (single_precision_enabled_) {
            throw std::runtime_error("Single precision already enabled.");
        }

        single_precision_data_ = std::make_unique<MatrixGPU<singlePrecisionT>>(rows_, cols_);
        chase::linalg::internal::cuda::convert_DP_TO_SP_GPU(this->gpu_data(), single_precision_data_->gpu_data(), rows_ * cols_);
        //std::cout << this->gpu_data() << " vs " << single_precision_data_->gpu_data() << std::endl;
        single_precision_enabled_ = true;
    }

    // Disable low precision: only enabled if T is double or std::complex<double>
    template<typename U = T, typename std::enable_if<
        std::is_same<U, double>::value || std::is_same<U, std::complex<double>>::value, int>::type = 0>
    void disableSinglePrecision(bool copyback = false) {
        if (!single_precision_enabled_) {
            throw std::runtime_error("Single precision is not enabled.");
        }

        if (single_precision_data_ == nullptr) {
            throw std::runtime_error("Original SP data is not initialized.");
        }

        if(copyback){
            chase::linalg::internal::cuda::convert_SP_TO_DP_GPU(single_precision_data_->gpu_data(), this->gpu_data(), rows_ * cols_);
        }
   
        single_precision_data_.reset();
        single_precision_enabled_ = false;     
    }
#endif

    T* gpu_data() { return gpu_data_; }
    std::size_t rows() const { return rows_; }
    std::size_t cols() const { return cols_; }
    std::size_t gpu_ld() const { return gpu_ld_; }
    std::size_t cpu_ld() const { return cpu_ld_; }
#ifdef ENABLE_MIXED_PRECISION
    MatrixGPU<singlePrecisionT>* matrix_sp() { return single_precision_data_.get(); }
    bool isSinglePrecisionEnabled() const { return single_precision_enabled_; }    

    // Disabled methods when T is not double or complex<double>
    template<typename U = T, typename std::enable_if<
        !(std::is_same<U, double>::value || std::is_same<U, std::complex<double>>::value), int>::type = 0>
    void enableSinglePrecision() {
        throw std::runtime_error("Single precision operations are only supported for double or complex<double>.");
    }

    template<typename U = T, typename std::enable_if<
        !(std::is_same<U, double>::value || std::is_same<U, std::complex<double>>::value), int>::type = 0>
    void disableSinglePrecision(bool copyback = false) {
        throw std::runtime_error("Single precision operations are only supported for double or complex<double>.");
    }
#endif    


};

#endif
}  // namespace matrix
}  // namespace chase



namespace chase
{
namespace platform
{
struct CPU {};  // Represents CPU platform
struct GPU {};  // Represents GPU platform

// Type trait to select the appropriate matrix class
template<typename T, typename Platform>
struct MatrixTypePlatform;

// Specialization for CPU
template<typename T>
struct MatrixTypePlatform<T, chase::platform::CPU> {
    using type = chase::matrix::MatrixCPU<T>;
};

#ifdef HAS_CUDA
// Specialization for GPU
template<typename T>
struct MatrixTypePlatform<T, chase::platform::GPU> {
    using type = chase::matrix::MatrixGPU<T>;
};
#endif
}    
}


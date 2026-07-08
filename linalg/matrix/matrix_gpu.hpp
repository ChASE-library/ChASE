// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#ifdef CHASE_GPU_MATRIX

/**
 * @ingroup MatrixClasses
 * @brief A matrix class for handling matrix data on both CPU and GPU.
 */
template <typename T, template <typename> class Allocator>
class Matrix<T, chase::platform::GPU, Allocator>
    : public AbstractMatrix<T, Matrix, chase::platform::GPU, Allocator>
{
private:
    std::unique_ptr<T[], std::function<void(T*)>> gpu_data_;
    std::unique_ptr<T[], std::function<void(T*)>> cpu_data_;
    T* external_gpu_data_ = nullptr;
    T* external_cpu_data_ = nullptr;
    std::size_t rows_ = 0;
    std::size_t cols_ = 0;
    std::size_t gpu_ld_ = 0;
    std::size_t cpu_ld_ = 0;
    bool owns_cpu_mem_ = false;
    bool owns_gpu_mem_ = false;
    Allocator<T> allocator_;

public:
    using hermitian_type = chase::matrix::Hermitian;

    Matrix()
        : gpu_data_(nullptr, [](T*) {}), cpu_data_(nullptr, [](T*) {})
    {
    }

    Matrix(std::size_t rows, std::size_t cols)
        : gpu_data_(nullptr, [](T*) {}), cpu_data_(nullptr, [](T*) {}),
          rows_(rows), cols_(cols), gpu_ld_(rows), owns_gpu_mem_(true)
    {
        T* raw_data = allocator_.allocate(rows_ * cols_);
        gpu_runtime::device_memset(raw_data, 0, rows * cols * sizeof(T));
        gpu_data_ = std::unique_ptr<T[], std::function<void(T*)>>(
            raw_data,
            [this](T* ptr) { allocator_.deallocate(ptr, rows_ * cols_); });
    }

    Matrix(std::size_t rows, std::size_t cols, std::size_t ld, T* external_data,
           BufferType buffer_type = BufferType::CPU)
        : gpu_data_(nullptr, [](T*) {}), cpu_data_(nullptr, [](T*) {}),
          external_gpu_data_(buffer_type == BufferType::GPU ? external_data
                                                            : nullptr),
          external_cpu_data_(buffer_type == BufferType::CPU ? external_data
                                                            : nullptr),
          rows_(rows), cols_(cols),
          gpu_ld_(buffer_type == BufferType::GPU ? ld : rows),
          cpu_ld_(buffer_type == BufferType::CPU ? ld : 0),
          owns_gpu_mem_(buffer_type != BufferType::GPU)
    {
        if (buffer_type == BufferType::CPU)
        {
            T* raw_data = allocator_.allocate(rows_ * cols_);
            gpu_data_ = std::unique_ptr<T[], std::function<void(T*)>>(
                raw_data,
                [this](T* ptr) { allocator_.deallocate(ptr, rows_ * cols_); });
            gpu_runtime::copy_matrix_h2d(rows_, cols_, sizeof(T),
                                         external_cpu_data_, cpu_ld_,
                                         gpu_data_.get(), gpu_ld_);
        }
    }

    Matrix(Matrix&& other) noexcept
        : gpu_data_(std::move(other.gpu_data_)),
          cpu_data_(std::move(other.cpu_data_)),
          external_gpu_data_(other.external_gpu_data_),
          external_cpu_data_(other.external_cpu_data_), rows_(other.rows_),
          cols_(other.cols_), gpu_ld_(other.gpu_ld_), cpu_ld_(other.cpu_ld_),
          owns_cpu_mem_(other.owns_cpu_mem_), owns_gpu_mem_(other.owns_gpu_mem_)
    {
        this->single_precision_matrix_ =
            std::move(other.single_precision_matrix_);
        this->is_single_precision_enabled_ = other.is_single_precision_enabled_;
        this->double_precision_matrix_ =
            std::move(other.double_precision_matrix_);
        this->is_double_precision_enabled_ = other.is_double_precision_enabled_;
        other.gpu_data_ = nullptr;
        other.cpu_data_ = nullptr;
        other.owns_cpu_mem_ = false;
        other.owns_gpu_mem_ = false;
        other.external_cpu_data_ = nullptr;
        other.external_gpu_data_ = nullptr;
    }

    Matrix& operator=(Matrix&& other) noexcept
    {
        if (this != &other)
        {
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
            this->single_precision_matrix_ =
                std::move(other.single_precision_matrix_);
            this->is_single_precision_enabled_ =
                other.is_single_precision_enabled_;
            this->double_precision_matrix_ =
                std::move(other.double_precision_matrix_);
            this->is_double_precision_enabled_ =
                other.is_double_precision_enabled_;
            other.single_precision_matrix_ = nullptr;
            other.double_precision_matrix_ = nullptr;
            other.owns_cpu_mem_ = false;
            other.owns_gpu_mem_ = false;
            other.external_cpu_data_ = nullptr;
            other.external_gpu_data_ = nullptr;
        }
        return *this;
    }

    void swap(Matrix& other) noexcept
    {
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
        std::swap(this->is_single_precision_enabled_,
                  other.is_single_precision_enabled_);
        std::swap(this->single_precision_matrix_, other.single_precision_matrix_);
        std::swap(this->is_double_precision_enabled_,
                  other.is_double_precision_enabled_);
        std::swap(this->double_precision_matrix_, other.double_precision_matrix_);
    }

    void swapDataPointer(Matrix& other) noexcept
    {
        std::swap(gpu_data_, other.gpu_data_);
        std::swap(external_gpu_data_, other.external_gpu_data_);
        std::swap(owns_gpu_mem_, other.owns_gpu_mem_);
    }

    ~Matrix() = default;

    void allocate_cpu_data()
    {
        if (!owns_cpu_mem_ || cpu_data_ == nullptr)
        {
            void* raw_data = nullptr;
            gpu_runtime::host_pinned_malloc(&raw_data, rows_ * cols_ * sizeof(T));
            std::memset(raw_data, 0, rows_ * cols_ * sizeof(T));
            cpu_data_ = std::unique_ptr<T[], std::function<void(T*)>>(
                static_cast<T*>(raw_data), [](T* ptr) {
                    gpu_runtime::host_pinned_free(ptr);
                });
            owns_cpu_mem_ = true;
            cpu_ld_ = rows_;
        }
    }

    T* cpu_data() override
    {
        if (!owns_cpu_mem_ && external_cpu_data_ != nullptr)
        {
            return external_cpu_data_;
        }
        if (owns_cpu_mem_ && cpu_data_ != nullptr)
        {
            return cpu_data_.get();
        }
        throw std::runtime_error("CPU buffer is not allocated or provided");
    }

    T* data() override
    {
        return owns_gpu_mem_ ? gpu_data_.get() : external_gpu_data_;
    }

    void H2D()
    {
        gpu_runtime::copy_matrix_h2d(rows_, cols_, sizeof(T), this->cpu_data(),
                                     cpu_ld_, this->data(), gpu_ld_);
    }

    void D2H()
    {
        if (cpu_data_ == nullptr && external_cpu_data_ == nullptr)
        {
            this->allocate_cpu_data();
        }
        gpu_runtime::copy_matrix_d2h(rows_, cols_, sizeof(T), this->data(),
                                     gpu_ld_, this->cpu_data(), cpu_ld_);
    }

    std::size_t rows() const override { return rows_; }
    std::size_t cols() const override { return cols_; }
    std::size_t ld() const override { return gpu_ld_; }
    std::size_t cpu_ld() const override { return cpu_ld_; }
};

template <typename T, template <typename> class Allocator>
class PseudoHermitianMatrix<T, chase::platform::GPU, Allocator>
    : public Matrix<T, chase::platform::GPU, Allocator>
{
public:
    using hermitian_type = chase::matrix::PseudoHermitian;

    PseudoHermitianMatrix() = default;

    PseudoHermitianMatrix(std::size_t rows, std::size_t cols)
        : Matrix<T, chase::platform::GPU, Allocator>(rows, cols)
    {
    }

    PseudoHermitianMatrix(std::size_t rows, std::size_t cols, std::size_t ld,
                          T* data,
                          BufferType buffer_type = BufferType::CPU)
        : Matrix<T, chase::platform::GPU, Allocator>(rows, cols, ld, data,
                                                     buffer_type)
    {
    }
};

#endif // CHASE_GPU_MATRIX

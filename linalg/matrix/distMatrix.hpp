#pragma once

#include <iostream>                // For std::cout
#include <memory>                  // For std::unique_ptr, std::shared_ptr
#include <complex>                 // For std::complex types
#include <stdexcept>               // For throwing runtime errors
#include <omp.h>                   // For OpenMP parallelization
#include <mpi.h>

#include "algorithm/types.hpp"
#include "linalg/matrix/matrix.hpp"
#include "Impl/mpi/mpiGrid2D.hpp"

namespace chase {
namespace distMatrix {

enum class MatrixType {
    BlockBlock,
    BlockCyclic,
    Redundant
};

template<typename T>
class RedundantMatrix;

template<typename T>
class BlockBlockMatrix;

template<typename T>
class BlockCyclicMatrix;

// Type trait to map matrix classes to their respective MatrixType
template <typename T>
struct MatrixTypeTrait;

// Specialize for RedundantMatrix
template <typename T>
struct MatrixTypeTrait<RedundantMatrix<T>> {
    static constexpr MatrixType value = MatrixType::Redundant;
};

// Specialize for BlockBlockMatrix
template <typename T>
struct MatrixTypeTrait<BlockBlockMatrix<T>> {
    static constexpr MatrixType value = MatrixType::BlockBlock;
};

template <MatrixType matrix_type, typename T>
struct MatrixConstructorTrait;

// Specialization for RedundantMatrix
template <typename T>
struct MatrixConstructorTrait<MatrixType::Redundant, T> {
    using type = RedundantMatrix<T>;
};

// Specialization for BlockBlockMatrix
template <typename T>
struct MatrixConstructorTrait<MatrixType::BlockBlock, T> {
    using type = BlockBlockMatrix<T>;
};


template <typename T, template <typename> class Derived>
class AbstractDistMatrix {
protected:
    // Single precision matrix
#ifdef ENABLE_MIXED_PRECISION       
    using SinglePrecisionType = typename chase::ToSinglePrecisionTrait<T>::Type;
    using SinglePrecisionDerived = Derived<SinglePrecisionType>;
    std::unique_ptr<SinglePrecisionDerived> single_precision_matrix_;
    bool is_single_precision_enabled_ = false; 
#endif

public:
    virtual ~AbstractDistMatrix() = default;
    // Get the matrix type via type traits
    MatrixType getMatrixType() const {
        return MatrixTypeTrait<Derived<T>>::value;
    }
    virtual chase::Impl::mpi::MpiGrid2DBase* getMpiGrid() const = 0;   
    virtual std::shared_ptr<chase::Impl::mpi::MpiGrid2DBase> getMpiGrid_shared_ptr() const = 0;
    virtual std::size_t g_rows() const = 0;
    virtual std::size_t g_cols() const = 0;
    virtual std::size_t l_rows() const = 0;
    virtual std::size_t l_cols() const = 0;
    virtual std::size_t l_ld() const = 0;
    virtual T *         l_data() = 0;
#ifdef ENABLE_MIXED_PRECISION       
    // Enable single precision for double types (and complex<double>)
    template <typename U = T, typename std::enable_if<std::is_same<U, double>::value || std::is_same<U, std::complex<double>>::value, int>::type = 0>
    void enableSinglePrecision() {
        if (!single_precision_matrix_) {
            single_precision_matrix_ = std::make_unique<SinglePrecisionDerived>(this->g_rows(), this->g_cols(), this->getMpiGrid_shared_ptr());
            #pragma omp parallel for
            for (std::size_t j = 0; j < this->l_cols(); ++j) {
                for (std::size_t i = 0; i < this->l_rows(); ++i) {
                    single_precision_matrix_->l_data()[j * single_precision_matrix_.get()->l_ld() + i] 
                                        = chase::convertToSinglePrecision(this->l_data()[j * this->l_ld() + i]);
                }
            }
            is_single_precision_enabled_ = true;
            std::cout << "Single precision matrix enabled in AbstractDistMatrix.\n";
        } else {
            throw std::runtime_error("Single precision already enabled.");
        }
    }

    // Disable single precision for double types
    template <typename U = T, typename std::enable_if<std::is_same<U, double>::value || std::is_same<U, std::complex<double>>::value, int>::type = 0>
    void disableSinglePrecision(bool copyback = false) {
        if(copyback)
        {
            if (single_precision_matrix_) {
                #pragma omp parallel for
                for (std::size_t j = 0; j < this->l_cols(); ++j) {
                    for (std::size_t i = 0; i < this->l_rows(); ++i) {
                        this->l_data()[j * this->l_ld() + i] = 
                                chase::convertToDoublePrecision<T>(single_precision_matrix_->l_data()[j * single_precision_matrix_.get()->l_ld() + i]);
                    }
                }
                single_precision_matrix_.reset();  // Free the single precision memory
                is_single_precision_enabled_ = false;
                std::cout << "Single precision matrix disabled in AbstractDistMatrix.\n";
            } else {
                throw std::runtime_error("Single precision is not enabled.");
            }
        }
    }

    // Check if single precision is enabled
    template <typename U = T, typename std::enable_if<std::is_same<U, double>::value || std::is_same<U, std::complex<double>>::value, int>::type = 0>
    bool isSinglePrecisionEnabled() const {
        return is_single_precision_enabled_;
    }

    // Get the single precision matrix itself
    template <typename U = T, typename std::enable_if<std::is_same<U, double>::value || std::is_same<U, std::complex<double>>::value, int>::type = 0>
    SinglePrecisionDerived* getSinglePrecisionMatrix() {
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

template<typename T> 
class RedundantMatrix : public AbstractDistMatrix<T, RedundantMatrix>
{

public:
    ~RedundantMatrix() override {};
    RedundantMatrix();
    RedundantMatrix(std::size_t m, std::size_t n,
                    std::shared_ptr<chase::Impl::mpi::MpiGrid2DBase> mpi_grid)
                    :m_(m), n_(n), ld_(m_), mpi_grid_(mpi_grid)
    {
        M_ = m_;
        N_ = n_;
        local_matrix_ = chase::matrix::MatrixCPU<T>(m_, n_);
    }

    RedundantMatrix(std::size_t m, std::size_t n, std::size_t ld, T *data, 
                    std::shared_ptr<chase::Impl::mpi::MpiGrid2DBase> mpi_grid)
                    :m_(m), n_(n), ld_(ld), mpi_grid_(mpi_grid)
    {
        M_ = m_;
        N_ = n_;        
        local_matrix_ = chase::matrix::MatrixCPU<T>(m_, n_, ld_, data);
    }

    std::size_t g_rows() const override { return M_;}
    std::size_t g_cols() const override { return N_;}
    std::size_t l_rows() const override { return m_;}
    std::size_t l_cols() const override { return n_;}
    std::size_t l_ld() const override { return ld_;}
    T *         l_data() override { return local_matrix_.data();}

    chase::matrix::MatrixCPU<T> loc_matrix() { return local_matrix_;}

    // Accessors for MPI grid
    chase::Impl::mpi::MpiGrid2DBase* getMpiGrid() const override {
        return mpi_grid_.get();
    }

    std::shared_ptr<chase::Impl::mpi::MpiGrid2DBase> getMpiGrid_shared_ptr() const override
    {
        return mpi_grid_;
    }

    template<MatrixType TargetType>
    void redistributeImpl(typename MatrixConstructorTrait<TargetType, T>::type* targetMatrix)//,
                            //std::size_t offset, std::size_t subsetSize)
    {
        if(M_ != targetMatrix->g_rows() || N_ != targetMatrix->g_cols() )
        {
            std::runtime_error("[RedundantMatrix]: redistribution requires original and target matrices have same global size");
        }

        if constexpr (TargetType == MatrixType::BlockBlock)
        {
            redistributeToBlockBlock(targetMatrix);
        }else if constexpr (TargetType == MatrixType::Redundant)
        {
            std::runtime_error("[RedundantMatrix]: no need to redistribute from redundant to redundant");
        }else if constexpr (TargetType == MatrixType::BlockCyclic)
        {
            std::runtime_error("[RedundantMatrix]: no support for redistribution from redundant to block-cyclic yet");
        }
    }

private:
    std::size_t M_;
    std::size_t N_;
    std::size_t m_;
    std::size_t n_;
    std::size_t ld_;

    chase::matrix::MatrixCPU<T> local_matrix_;
    std::shared_ptr<chase::Impl::mpi::MpiGrid2DBase> mpi_grid_;    

    void redistributeToBlockBlock(BlockBlockMatrix<T>* targetMatrix)
    {
        std::size_t *g_offs = targetMatrix->g_offs();
        std::size_t l_cols = targetMatrix->l_cols();
        std::size_t l_rows = targetMatrix->l_rows();
        #pragma omp parallel for
        for (auto y = 0; y < l_cols; y++)
        {
            for (auto x = 0; x < l_rows; x++)
            {
                targetMatrix->l_data()[x + targetMatrix->l_ld() * y] =
                    this->l_data()[(g_offs[1] + y) * this->l_ld() + (g_offs[0] + x)];
            }
        }
        
    }
};


template<typename T> 
class BlockBlockMatrix : public AbstractDistMatrix<T, BlockBlockMatrix>
{
private:
    std::size_t M_;
    std::size_t N_;
    std::size_t m_;
    std::size_t n_;
    std::size_t ld_;
    std::size_t g_offs_[2];

    chase::matrix::MatrixCPU<T> local_matrix_;
    std::shared_ptr<chase::Impl::mpi::MpiGrid2DBase> mpi_grid_;

public:
    ~BlockBlockMatrix() override {};
    BlockBlockMatrix();
    BlockBlockMatrix(std::size_t M, std::size_t N,
                    std::shared_ptr<chase::Impl::mpi::MpiGrid2DBase> mpi_grid)
                    :M_(M), N_(N), mpi_grid_(mpi_grid)
    {
        int *dims_ = mpi_grid_.get()->get_dims();
        int *coord_ = mpi_grid_.get()->get_coords();
        std::size_t len;

        if (M_ % dims_[0] == 0)
        {
            len = M_ / dims_[0];
        }
        else
        {
            len = std::min(M_, M_ / dims_[0] + 1);
        }

        g_offs_[0] = coord_[0] * len;

        if (coord_[0] < dims_[0] - 1)
        {
            m_ = len;
        }
        else
        {
            m_ = M_ - (dims_[0] - 1) * len;
        }

//
        if (N_ % dims_[1] == 0)
        {
            len = N_ / dims_[1];
        }
        else
        {
            len = std::min(N_, N_ / dims_[1] + 1);
        }

        g_offs_[1] = coord_[1] * len;

        if (coord_[1] < dims_[1] - 1)
        {
            n_ = len;
        }
        else
        {
            n_ = N_ - (dims_[1] - 1) * len;
        }

        ld_ = m_;
        local_matrix_ = chase::matrix::MatrixCPU<T>(m_, n_);
    }

    BlockBlockMatrix(std::size_t m, std::size_t n, std::size_t ld, T *data,
                    std::shared_ptr<chase::Impl::mpi::MpiGrid2DBase> mpi_grid)
                    :m_(m), n_(n), ld_(ld), mpi_grid_(mpi_grid)
    {
        uint64_t lv = static_cast<uint64_t>(m_);
        uint64_t res = 0; 
        MPI_Allreduce(&lv, &res, 1, MPI_UINT64_T, MPI_SUM, mpi_grid_.get()->get_col_comm());
        M_ = static_cast<std::size_t>(res);

        lv = static_cast<uint64_t>(n_);
        res = 0;
        MPI_Allreduce(&lv, &res, 1, MPI_UINT64_T, MPI_SUM, mpi_grid_.get()->get_row_comm());
        N_ = static_cast<std::size_t>(res);

        local_matrix_ = chase::matrix::MatrixCPU<T>(m_, n_, ld_, data);

        int *coord_ = mpi_grid_.get()->get_coords();
        int *dims_ = mpi_grid_.get()->get_dims();
        std::size_t len;
        if (M_ % dims_[0] == 0)
        {
            len = M_ / dims_[0];
        }
        else
        {
            len = std::min(M_, M_ / dims_[0] + 1);
        }

        g_offs_[0] = coord_[0] * len;

        if (N_ % dims_[1] == 0)
        {
            len = N_ / dims_[1];
        }
        else
        {
            len = std::min(N_, N_ / dims_[1] + 1);
        }

        g_offs_[1] = coord_[1] * len;
    }

    std::size_t g_rows() const override { return M_;}
    std::size_t g_cols() const override { return N_;}
    std::size_t l_ld() const override { return ld_;}
    std::size_t l_rows() const override { return m_;}
    std::size_t l_cols() const override { return n_;}
    T *         l_data() override { return local_matrix_.data();}
    std::size_t *g_offs() { return g_offs_;}

    chase::matrix::MatrixCPU<T> loc_matrix() { return local_matrix_;}

    // Accessors for MPI grid
    chase::Impl::mpi::MpiGrid2DBase* getMpiGrid() const override {
        return mpi_grid_.get();
    }

    std::shared_ptr<chase::Impl::mpi::MpiGrid2DBase> getMpiGrid_shared_ptr() const override
    {
        return mpi_grid_;
    }
};


template<typename T> 
class BlockCyclicMatrix : public AbstractDistMatrix<T, BlockCyclicMatrix>
{
    //impl later
};


}
}
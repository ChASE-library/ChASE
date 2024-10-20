#pragma once

#include <iostream>                // For std::cout
#include <memory>                  // For std::unique_ptr, std::shared_ptr
#include <complex>                 // For std::complex types
#include <stdexcept>               // For throwing runtime errors
#include <omp.h>                   // For OpenMP parallelization
#include <mpi.h>
#include <chrono>

#include "algorithm/types.hpp"
#include "linalg/matrix/matrix.hpp"
#include "linalg/scalapackpp/scalapackpp.hpp"
#include "Impl/grid/mpiGrid2D.hpp"
#include "Impl/grid/mpiTypes.hpp"

#ifdef HAS_CUDA
#include <cuda_runtime.h>
#include "Impl/cuda/cuda_utils.hpp"
#include "linalg/internal/cuda/lacpy.hpp"
#include "linalg/cublaspp/cublaspp.hpp"
#include "linalg/internal/cuda/precision_conversion.cuh"
#endif

namespace chase {

    std::pair<std::size_t, std::size_t> numroc(std::size_t n, std::size_t nb,
                                            int iproc, int nprocs)
    {

        std::size_t numroc;
        std::size_t extrablks, mydist, nblocks;
        mydist = (nprocs + iproc) % nprocs;
        nblocks = n / nb;
        numroc = (nblocks / nprocs) * nb;
        extrablks = nblocks % nprocs;

        if (mydist < extrablks)
            numroc = numroc + nb;
        else if (mydist == extrablks)
            numroc = numroc + n % nb;

        std::size_t nb_loc = numroc / nb;

        if (numroc % nb != 0)
        {
            nb_loc += 1;
        }
        return std::make_pair(numroc, nb_loc);
    }

namespace distMatrix {

enum class MatrixType {
    BlockBlock,
    BlockCyclic,
    Redundant
};

struct BlockBlock {}; 
struct Redundant {}; 
struct BlockCyclic {}; 

template<typename T, typename Platform>
class RedundantMatrix;

template<typename T, typename Platform>
class BlockBlockMatrix;

template<typename T, typename Platform>
class BlockCyclicMatrix;

template<typename Type, typename T, typename Platform>
struct distMatrixTypeTrait;

template<typename T, typename Platform>
struct distMatrixTypeTrait<Redundant, T, Platform> {
    using type = chase::distMatrix::RedundantMatrix<T, Platform>;
};

template<typename T, typename Platform>
struct distMatrixTypeTrait<BlockBlock, T, Platform> {
    using type = chase::distMatrix::BlockBlockMatrix<T, Platform>;
};

template<typename T, typename Platform>
struct distMatrixTypeTrait<BlockCyclic, T, Platform> {
    using type = chase::distMatrix::BlockCyclicMatrix<T, Platform>;
};

template <typename T, template <typename, typename> class Derived, typename Platform = chase::platform::CPU>
class AbstractDistMatrix {
protected:
    // Single precision matrix
#ifdef ENABLE_MIXED_PRECISION       
    using SinglePrecisionType = typename chase::ToSinglePrecisionTrait<T>::Type;
    using SinglePrecisionDerived = Derived<SinglePrecisionType, Platform>;
    std::unique_ptr<SinglePrecisionDerived> single_precision_matrix_;
    bool is_single_precision_enabled_ = false; 
    std::chrono::high_resolution_clock::time_point start, end;
    std::chrono::duration<double> elapsed;
#endif

public:
    virtual ~AbstractDistMatrix() = default;
    virtual chase::Impl::mpi::MpiGrid2DBase* getMpiGrid() const = 0;   
    virtual std::shared_ptr<chase::Impl::mpi::MpiGrid2DBase> getMpiGrid_shared_ptr() const = 0;
    virtual std::size_t g_rows() const = 0;
    virtual std::size_t g_cols() const = 0;
    virtual std::size_t l_rows() const = 0;
    virtual std::size_t l_cols() const = 0;
    virtual std::size_t l_ld() const = 0;
    virtual T *         l_data() = 0;
    virtual typename chase::platform::MatrixTypePlatform<T, Platform>::type& loc_matrix() = 0;
    virtual std::size_t mb() const = 0;
    virtual std::size_t nb() const = 0;  
    //virtual std::unique_ptr<SinglePrecisionDerived> createSinglePrecisionMatrix() = 0;
    int grank()
    {
        int grank = 0;
        MPI_Comm_rank(this->getMpiGrid()->get_comm(), &grank);
        return grank;
    }

#ifdef HAS_CUDA
    void D2H()
    {
        auto& loc_matrix = this->loc_matrix();
        if constexpr (std::is_same<Platform, chase::platform::GPU>::value)
        {
            loc_matrix.D2H();
        }else
        {
            throw std::runtime_error("[DistMatrix]: CPU type of matrix do not support D2H operation");
        }
    }

    void H2D()
    {
        auto& loc_matrix = this->loc_matrix();
        if constexpr (std::is_same<Platform, chase::platform::GPU>::value)
        {
            loc_matrix.H2D();
        }else
        {
            throw std::runtime_error("[DistMatrix]: CPU type of matrix do not support H2D operation");
        }
    }

    void allocate_cpu_data()
    {
        auto& loc_matrix = this->loc_matrix();
        loc_matrix.allocate_cpu_data();   
    }
#endif

    T *cpu_data()
    {
        auto& loc_matrix = this->loc_matrix();
        if constexpr (std::is_same<Platform, chase::platform::GPU>::value)
        {
            return loc_matrix.cpu_data();
        }else
        {
            return loc_matrix.data();
        }        
    }

    std::size_t cpu_ld()
    {
        auto& loc_matrix = this->loc_matrix();
        if constexpr (std::is_same<Platform, chase::platform::GPU>::value)
        {
            return loc_matrix.cpu_ld();
        }else
        {
            return loc_matrix.ld();
        }           
    }

#ifdef ENABLE_MIXED_PRECISION       
    // Enable single precision for double types (and complex<double>)
    template <typename U = T, typename std::enable_if<std::is_same<U, double>::value || std::is_same<U, std::complex<double>>::value, int>::type = 0>
    void enableSinglePrecision() {
        if (!single_precision_matrix_) {
            start = std::chrono::high_resolution_clock::now();

            if constexpr(std::is_same<Derived<T, Platform>, chase::distMatrix::BlockCyclicMatrix<T, Platform>>::value)
            {
                single_precision_matrix_ = std::make_unique<SinglePrecisionDerived>(this->g_rows(), this->g_cols(), this->mb(), this->nb(), this->getMpiGrid_shared_ptr());
            }else
            {
                single_precision_matrix_ = std::make_unique<SinglePrecisionDerived>(this->g_rows(), this->g_cols(), this->getMpiGrid_shared_ptr());
            }
            
            if constexpr (std::is_same<Platform, chase::platform::CPU>::value) 
            {
                #pragma omp parallel for
                for (std::size_t j = 0; j < this->l_cols(); ++j) {
                    for (std::size_t i = 0; i < this->l_rows(); ++i) {
                        single_precision_matrix_->l_data()[j * single_precision_matrix_.get()->l_ld() + i] 
                                            = chase::convertToSinglePrecision(this->l_data()[j * this->l_ld() + i]);
                    }
                }
            }else
#ifdef HAS_CUDA            
            {

                chase::linalg::internal::cuda::convert_DP_TO_SP_GPU(this->l_data(), single_precision_matrix_->l_data(), this->l_cols() * this->l_rows());
            }
#else
            {
                throw std::runtime_error("GPU is not supported in AbstractDistMultiVector");
            }
#endif                  
            is_single_precision_enabled_ = true;
            end = std::chrono::high_resolution_clock::now();
            elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

            if(this->grank() == 0)
                std::cout << "Single precision matrix enabled in AbstractDistMatrix in " << elapsed.count() << " s\n";
        } else {
            throw std::runtime_error("Single precision already enabled.");
        }
    }


    // Disable single precision for double types
    template <typename U = T, typename std::enable_if<std::is_same<U, double>::value || std::is_same<U, std::complex<double>>::value, int>::type = 0>
    void disableSinglePrecision(bool copyback = false) {
        start = std::chrono::high_resolution_clock::now();
        if(copyback)
        {
            if (single_precision_matrix_) {
                if constexpr (std::is_same<Platform, chase::platform::CPU>::value) 
                {
                    #pragma omp parallel for
                    for (std::size_t j = 0; j < this->l_cols(); ++j) {
                        for (std::size_t i = 0; i < this->l_rows(); ++i) {
                            this->l_data()[j * this->l_ld() + i] = 
                                    chase::convertToDoublePrecision<T>(single_precision_matrix_->l_data()[j * single_precision_matrix_.get()->l_ld() + i]);
                        }
                    }
                }
    #ifdef HAS_CUDA            
                {
                    chase::linalg::internal::cuda::convert_SP_TO_DP_GPU(single_precision_matrix_->l_data(), this->l_data(), this->l_cols() * this->l_rows());
                }
    #else
                {
                    throw std::runtime_error("GPU is not supported in AbstractDistMultiVector");
                }
    #endif            
            } else {
                throw std::runtime_error("Single precision is not enabled.");
            }
        }
        end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

        if(this->grank() == 0)
            std::cout << "Single precision matrix disabled in AbstractDistMatrix in " << elapsed.count() << " s\n";
        single_precision_matrix_.reset();  // Free the single precision memory
        is_single_precision_enabled_ = false;

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

template<typename T, typename Platform = chase::platform::CPU> 
class RedundantMatrix : public AbstractDistMatrix<T, RedundantMatrix, Platform>
{

public:
    using platform_type = Platform;
    using value_type = T;  // Alias for element type

    ~RedundantMatrix() override {};
    RedundantMatrix();
    RedundantMatrix(std::size_t m, std::size_t n,
                    std::shared_ptr<chase::Impl::mpi::MpiGrid2DBase> mpi_grid)
                    :m_(m), n_(n), ld_(m_), mpi_grid_(mpi_grid)
    {
        M_ = m_;
        N_ = n_;
        local_matrix_ = typename chase::platform::MatrixTypePlatform<T, Platform>::type(m_, n_); 
    }

    RedundantMatrix(std::size_t m, std::size_t n, std::size_t ld, T *data, 
                    std::shared_ptr<chase::Impl::mpi::MpiGrid2DBase> mpi_grid)
                    :m_(m), n_(n), ld_(ld), mpi_grid_(mpi_grid)
    {
        M_ = m_;
        N_ = n_;        
        local_matrix_ = typename chase::platform::MatrixTypePlatform<T, Platform>::type(m_, n_, ld_, data);
        
        if constexpr (std::is_same<Platform, chase::platform::GPU>::value)
        {
            ld_ = local_matrix_.gpu_ld();
        }    
    }

    std::size_t g_rows() const override { return M_;}
    std::size_t g_cols() const override { return N_;}
    std::size_t l_rows() const override { return m_;}
    std::size_t l_cols() const override { return n_;}
    std::size_t l_ld() const override { return ld_;}
    std::size_t mb() const override { return -1;}
    std::size_t nb() const override { return -1;} 

    T *         l_data() override { 
        if constexpr (std::is_same<Platform, chase::platform::CPU>::value)
        {
            return local_matrix_.data();
        }
#ifdef HAS_CUDA        
        else
        {
            return local_matrix_.gpu_data();
        }
#endif        
    }
    typename chase::platform::MatrixTypePlatform<T, Platform>::type& loc_matrix() override { return local_matrix_;}


    // Accessors for MPI grid
    chase::Impl::mpi::MpiGrid2DBase* getMpiGrid() const override {
        return mpi_grid_.get();
    }

    std::shared_ptr<chase::Impl::mpi::MpiGrid2DBase> getMpiGrid_shared_ptr() const override
    {
        return mpi_grid_;
    }

    void mapToNewMpiGrid(std::shared_ptr<chase::Impl::mpi::MpiGrid2DBase> new_mpi_grid)
    {
        mpi_grid_ = new_mpi_grid;
    }

    //here the startrow/col indices should be the global indices
    template<template <typename, typename> class targetType>
    void redistributeImpl(targetType<T, Platform>* targetMatrix,
                          std::size_t startRow, std::size_t subRows, std::size_t startCol, std::size_t subCols)
    {
        if(M_ != targetMatrix->g_rows() || N_ != targetMatrix->g_cols() )
        {
            throw std::runtime_error("[RedundantMatrix]: redistribution requires original and target matrices have same global size");
        }

        else if constexpr (std::is_same<typename distMatrixTypeTrait<BlockBlock, T, Platform>::type, targetType<T, Platform>>::value) 
        {
            redistributeToBlockBlock(targetMatrix, startRow, subRows, startCol, subCols);
        }
        else if constexpr (std::is_same<typename distMatrixTypeTrait<BlockCyclic, T, Platform>::type, targetType<T, Platform>>::value) 
        {
            redistributeToBlockCyclic(targetMatrix, startRow, subRows, startCol, subCols);
        }        
        else if constexpr (std::is_same<typename distMatrixTypeTrait<Redundant, T, Platform>::type, targetType<T, Platform>>::value) 
        {
            throw std::runtime_error("[RedundantMatrix]: no need to redistribute from redundant to redundant");
        }else
        {
            throw std::runtime_error("[RedundantMatrix]: no support for redistribution from redundant to othertypes yet");
        }      
    }   

    template<template <typename, typename> class targetType>
    void redistributeImpl(targetType<T, Platform>* targetMatrix)
    {
        this->redistributeImpl(targetMatrix, 0, this->g_rows(), 0, this->g_cols());
    }
private:
    std::size_t M_;
    std::size_t N_;
    std::size_t m_;
    std::size_t n_;
    std::size_t ld_;

    typename chase::platform::MatrixTypePlatform<T, Platform>::type local_matrix_;
    std::shared_ptr<chase::Impl::mpi::MpiGrid2DBase> mpi_grid_;    

    void redistributeToBlockBlock(BlockBlockMatrix<T, Platform>* targetMatrix,
                                   std::size_t startRow, std::size_t subRows, std::size_t startCol, std::size_t subCols)
    {
        //attention for submatrix should be check later, seems not fully correct
        if constexpr (std::is_same<Platform, chase::platform::CPU>::value) {
            std::size_t *g_offs = targetMatrix->g_offs();
            std::size_t l_cols = targetMatrix->l_cols();
            std::size_t l_rows = targetMatrix->l_rows();
            for(auto y = 0; y < l_cols; y++)
            {
                for(auto x = 0; x < l_rows; x++)
                {
                    std::size_t x_g_off = g_offs[0] + x;
                    std::size_t y_g_off = g_offs[1] + y;
                    if(x_g_off >= startRow && x_g_off < startRow + subRows && y_g_off >= startCol && y_g_off < startCol + subCols )
                    {
                        targetMatrix->l_data()[x + targetMatrix->l_ld() * y] = this->l_data()[(g_offs[1] + y) * this->l_ld() + (g_offs[0] + x)];
                    }

                }
            }     
        }
#ifdef HAS_CUDA
        else if constexpr (std::is_same<Platform, chase::platform::GPU>::value)
        {
            std::size_t *g_offs = targetMatrix->g_offs();
            std::size_t l_cols = targetMatrix->l_cols();
            std::size_t l_rows = targetMatrix->l_rows();
            for(auto y = 0; y < l_cols; y++)
            {
                for(auto x = 0; x < l_rows; x++)
                {
                    std::size_t x_g_off = g_offs[0] + x;
                    std::size_t y_g_off = g_offs[1] + y;
                    if(x_g_off >= startRow && x_g_off < startRow + subRows && y_g_off >= startCol && y_g_off < startCol + subCols )
                    {
                        targetMatrix->cpu_data()[x + targetMatrix->cpu_ld() * y] = this->cpu_data()[(g_offs[1] + y) * this->cpu_ld() + (g_offs[0] + x)];
                    }

                }
            }
            //targetMatrix->H2D();              
            //throw std::runtime_error("[RedundantMatrix]: redistribution for GPU data from redundant to BlockBlock is not supported yet.");
        }
#endif        
    }

    void redistributeToBlockCyclic(BlockCyclicMatrix<T, Platform>* targetMatrix,
                                   std::size_t startRow, std::size_t subRows, std::size_t startCol, std::size_t subCols)
    {
        //attention for submatrix should be check later, seems not fully correct
        if constexpr (std::is_same<Platform, chase::platform::CPU>::value) {
            auto m_contiguous_global_offs = targetMatrix->m_contiguous_global_offs();
            auto n_contiguous_global_offs = targetMatrix->n_contiguous_global_offs();
            auto m_contiguous_local_offs = targetMatrix->m_contiguous_local_offs();
            auto n_contiguous_local_offs = targetMatrix->n_contiguous_local_offs();
            auto m_contiguous_lens = targetMatrix->m_contiguous_lens();
            auto n_contiguous_lens = targetMatrix->n_contiguous_lens();
            auto mblocks = targetMatrix->mblocks();
            auto nblocks = targetMatrix->nblocks();
            
            for(std::size_t j = 0; j < nblocks; j++)
            {
                for(std::size_t i = 0; i < mblocks; i++)
                {
                    for(std::size_t q = 0; q < n_contiguous_lens[j]; q++)
                    {
                        for(std::size_t p = 0; p < m_contiguous_lens[i]; p++)
                        {
                            std::size_t x_g_off = p + m_contiguous_local_offs[i];
                            std::size_t y_g_off = q + n_contiguous_local_offs[j];
                            
                            if(x_g_off >= startRow && x_g_off < startRow + subRows && y_g_off >= startCol && y_g_off < startCol + subCols )
                            {
                                targetMatrix->l_data()[(q + n_contiguous_local_offs[j]) * targetMatrix->l_ld() + p + m_contiguous_local_offs[i]]
                                    = this->l_data()[(q + n_contiguous_global_offs[j]) * this->l_ld() + p + m_contiguous_global_offs[i]];
                            }
                        }
                    }
                }
            }   
        }
#ifdef HAS_CUDA
        else if constexpr (std::is_same<Platform, chase::platform::GPU>::value)
        {
            auto m_contiguous_global_offs = targetMatrix->m_contiguous_global_offs();
            auto n_contiguous_global_offs = targetMatrix->n_contiguous_global_offs();
            auto m_contiguous_local_offs = targetMatrix->m_contiguous_local_offs();
            auto n_contiguous_local_offs = targetMatrix->n_contiguous_local_offs();
            auto m_contiguous_lens = targetMatrix->m_contiguous_lens();
            auto n_contiguous_lens = targetMatrix->n_contiguous_lens();
            auto mblocks = targetMatrix->mblocks();
            auto nblocks = targetMatrix->nblocks();
            
            for(std::size_t j = 0; j < nblocks; j++)
            {
                for(std::size_t i = 0; i < mblocks; i++)
                {
                    for(std::size_t q = 0; q < n_contiguous_lens[j]; q++)
                    {
                        for(std::size_t p = 0; p < m_contiguous_lens[i]; p++)
                        {
                            std::size_t x_g_off = p + m_contiguous_local_offs[i];
                            std::size_t y_g_off = q + n_contiguous_local_offs[j];
                            
                            if(x_g_off >= startRow && x_g_off < startRow + subRows && y_g_off >= startCol && y_g_off < startCol + subCols )
                            {
                                targetMatrix->cpu_data()[(q + n_contiguous_local_offs[j]) * targetMatrix->cpu_ld() + p + m_contiguous_local_offs[i]]
                                    = this->cpu_data()[(q + n_contiguous_global_offs[j]) * this->cpu_ld() + p + m_contiguous_global_offs[i]];
                            }
                        }
                    }
                }
            }
            //targetMatrix->H2D();            
            //throw std::runtime_error("[RedundantMatrix]: redistribution for GPU data from redundant to BlockCyclic is not supported yet.");
        }
#endif        
    }

};


template<typename T, typename Platform = chase::platform::CPU> 
class BlockBlockMatrix : public AbstractDistMatrix<T, BlockBlockMatrix, Platform>
{

public:
    using platform_type = Platform;
    using value_type = T;  // Alias for element type

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
        local_matrix_ = typename chase::platform::MatrixTypePlatform<T, Platform>::type(m_, n_);     
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

        local_matrix_ = typename chase::platform::MatrixTypePlatform<T, Platform>::type(m_, n_, ld_, data);

        if constexpr (std::is_same<Platform, chase::platform::GPU>::value)
        {
            ld_ = local_matrix_.gpu_ld();
        }   

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
    std::size_t *g_offs() { return g_offs_;}
    std::size_t mb() const override { return -1;}
    std::size_t nb() const override { return -1;}    
    T *         l_data() override { 
        if constexpr (std::is_same<Platform, chase::platform::CPU>::value)
        {
            return local_matrix_.data();
        }
#ifdef HAS_CUDA        
        else
        {
            return local_matrix_.gpu_data();
        }
#endif        
    }
    typename chase::platform::MatrixTypePlatform<T, Platform>::type& loc_matrix() override { return local_matrix_;}

    // Accessors for MPI grid
    chase::Impl::mpi::MpiGrid2DBase* getMpiGrid() const override {
        return mpi_grid_.get();
    }

    std::shared_ptr<chase::Impl::mpi::MpiGrid2DBase> getMpiGrid_shared_ptr() const override
    {
        return mpi_grid_;
    }

    template<typename CloneVectorType>
    CloneVectorType cloneMultiVector(std::size_t g_M, std::size_t g_N)
    {
        static_assert(
            std::is_same_v<T, typename CloneVectorType::value_type>,
            "Cloned type must have the same value_type"
        );
        ///using NewCommType = typename CloneType::communicator_type;
        return CloneVectorType(g_M, g_N, mpi_grid_);        
    }

    //only save from CPU buffer
    //for saving GPU data, need to copy to CPU by D2H()
    void saveToBinaryFile(const std::string& filename) {
    	MPI_File fileHandle;
        MPI_Status status;
        T *buff;
        if constexpr (std::is_same<Platform, chase::platform::GPU>::value)
        {
            if(local_matrix_.cpu_data() == nullptr)
            {
               throw std::runtime_error("[BlockBlockMatrix]: only can save data from CPU buffer");
            }
            buff = local_matrix_.cpu_data();
        }else
        {
            buff = local_matrix_.data();
        }

        if(MPI_File_open(this->mpi_grid_.get()->get_comm(), filename.data(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fileHandle) != MPI_SUCCESS)
        {
            if(this->grank() == 0)
                std::cout << "Can't open input matrix - " << filename << std::endl;
            MPI_Abort(this->mpi_grid_.get()->get_comm(), EXIT_FAILURE);
        }


        if (this->l_data() == nullptr) {
            throw std::runtime_error("[BlockBlockMatrix]: Original data is not initialized.");
        }

        MPI_Count count_write = m_ * n_;

        MPI_Datatype subarray;
        int global_matrix_size[] = {(int)M_, (int)N_};
        int local_matrix_size[] = {(int)m_,(int)n_};
        int offsets[] = {(int)g_offs_[0], (int)g_offs_[1]};

        MPI_Type_create_subarray(2, global_matrix_size, local_matrix_size, offsets, MPI_ORDER_FORTRAN, chase::mpi::getMPI_Type<T>(), &subarray);
        MPI_Type_commit(&subarray);

        MPI_File_set_view(fileHandle, 0, chase::mpi::getMPI_Type<T>(), subarray, "native", MPI_INFO_NULL);
        MPI_File_write_all(fileHandle, buff, count_write, chase::mpi::getMPI_Type<T>(), &status);

        MPI_Type_free(&subarray);

    	if (MPI_File_close(&fileHandle) != MPI_SUCCESS)
    	{
            MPI_Abort(this->mpi_grid_.get()->get_comm(), EXIT_FAILURE);
        }
    }

    // Read matrix data from a binary file
    void readFromBinaryFile(const std::string& filename) {
        T *buff;
        if constexpr (std::is_same<Platform, chase::platform::GPU>::value)
        {
            if(local_matrix_.cpu_data() == nullptr)
            {
               local_matrix_.allocate_cpu_data();
            }
            buff = local_matrix_.cpu_data();
        }else
        {
            buff = local_matrix_.data();
        }

#ifdef USE_MPI_IO	    
	    MPI_File fileHandle;
        MPI_Status status;
        int access_mode = MPI_MODE_RDONLY;

        if(MPI_File_open(this->mpi_grid_.get()->get_comm(), filename.data(), access_mode, MPI_INFO_NULL, &fileHandle) != MPI_SUCCESS)
        {
            if(this->grank() == 0)
                std::cout << "Can't open input matrix - " << filename << std::endl;
            MPI_Abort(this->mpi_grid_.get()->get_comm(), EXIT_FAILURE);
        }

        if (this->l_data() == nullptr) {
            throw std::runtime_error("[BlockBlockMatrix]: Original data is not initialized.");
        }

        MPI_Count count_read = m_ * n_;

        MPI_Datatype subarray;
        int global_matrix_size[] = {(int)M_, (int)N_};
        int local_matrix_size[] = {(int)m_,(int)n_};
        int offsets[] = {(int)g_offs_[0], (int)g_offs_[1]};

	    MPI_Type_create_subarray(2, global_matrix_size, local_matrix_size, offsets, MPI_ORDER_FORTRAN, chase::mpi::getMPI_Type<T>(), &subarray);
        MPI_Type_commit(&subarray);

        MPI_File_set_view(fileHandle, 0, chase::mpi::getMPI_Type<T>(), subarray, "native", MPI_INFO_NULL);
        MPI_File_read_all(fileHandle, buff, count_read, chase::mpi::getMPI_Type<T>(), &status);

        MPI_Type_free(&subarray);
    
        if (MPI_File_close(&fileHandle) != MPI_SUCCESS)
        {
            MPI_Abort(this->mpi_grid_.get()->get_comm(), EXIT_FAILURE);
        }
#else
	    std::ifstream input(filename.data(), std::ios::binary);	
        if (!input.is_open()) {
            throw std::runtime_error("[BlockBlockMatrix]: Failed to open file for reading.");
        }
        
        if (this->l_data() == nullptr) {
            throw std::runtime_error("[BlockBlockMatrix]: Original data is not initialized.");
        }

        for (std::size_t y = 0; y < n_; y++)
        {
            input.seekg(((g_offs_[0]) + M_ * (g_offs_[1] + y)) * sizeof(T));
            input.read(reinterpret_cast<char*>(buff + this->cpu_ld() * y), m_ * sizeof(T));
        }
        
        input.close();
#endif
    }

#ifdef HAS_SCALAPACK
    std::size_t *get_scalapack_desc(){ return desc_; }
#endif

#ifdef HAS_SCALAPACK
    std::size_t *scalapack_descriptor_init()
    {
        std::size_t mb = m_;
        std::size_t nb = n_;
        int *coords = mpi_grid_.get()->get_coords();
        int *dims = mpi_grid_.get()->get_dims();

        if (coords[1] == dims[1] - 1 && dims[1] != 1)
        {
            nb = (N_ - n_) / (dims[1] - 1);
        }

        if (coords[0] == dims[0] - 1 && dims[0] != 1)
        {
            mb = (M_ - m_) / (dims[0] - 1);
        }
        int zero = 0;
        int one = 1;
        int info;
        int comm2D_ctxt = mpi_grid_.get()->get_blacs_comm2D_ctxt();
        int grank;
        MPI_Comm_rank(MPI_COMM_WORLD, &grank);
        if constexpr (std::is_same<Platform, chase::platform::GPU>::value)
        {
            if(local_matrix_.cpu_data() == nullptr)
            {
                local_matrix_.allocate_cpu_data();
            }
        }
        std::size_t ldd = this->cpu_ld();
        //std::cout << "grank = " << grank << ", " <<  M_ << "x" << N_ << " " << mb << "x" << nb << " " << ld_ << std::endl;
        //std::cout << "comm2D_ctxt: = " << comm2D_ctxt << std::endl;
        chase::linalg::scalapackpp::t_descinit(desc_, 
                                               &M_, 
                                               &N_, 
                                               &mb, 
                                               &nb, 
                                               &zero, 
                                               &zero, 
                                               &comm2D_ctxt, 
                                               &ldd, 
                                               &info);  

        return desc_;
    }
#endif

    template<template <typename, typename> class targetType>
    void redistributeImpl(targetType<T, Platform>* targetMatrix)//,
                            //std::size_t offset, std::size_t subsetSize)
    {
        if(M_ != targetMatrix->g_rows() || N_ != targetMatrix->g_cols() )
        {
            throw std::runtime_error("[BlockBlockMatrix]: redistribution requires original and target matrices have same global size");
        }

        if constexpr (std::is_same<typename distMatrixTypeTrait<Redundant, T, Platform>::type, targetType<T, Platform>>::value) 
        {
            redistributeToRedundant(targetMatrix);
        }
        else if constexpr (std::is_same<typename distMatrixTypeTrait<BlockBlock, T, Platform>::type, targetType<T, Platform>>::value) 
        {
            throw std::runtime_error("[BlockBlockMatrix]: no need to redistribute from BlockBlock to BlockBlock");
        }else
        {
            throw std::runtime_error("[BlockBlockMatrix]:  no support for redistribution from redundant to othertypes yet");
        }
    }

private:
    std::size_t M_;
    std::size_t N_;
    std::size_t m_;
    std::size_t n_;
    std::size_t ld_;
    std::size_t g_offs_[2];

    typename chase::platform::MatrixTypePlatform<T, Platform>::type local_matrix_;
    std::shared_ptr<chase::Impl::mpi::MpiGrid2DBase> mpi_grid_;

    void redistributeToRedundant(RedundantMatrix<T, Platform>* targetMatrix)
    {
        if constexpr (std::is_same<Platform, chase::platform::GPU>::value)
        {
            throw std::runtime_error("[BlockBlockMatrix]: Redistribution for GPU matrix is not supported yet");
        }else
        {
            //redistribute either thing in column communicator
            //packing uppacking data maunally rather than using MPI new types, 
            //for helping later implementation with nccl, and which is not supported new types
            int *dims = mpi_grid_.get()->get_dims();
            int *coords = mpi_grid_.get()->get_coords();
            MPI_Comm row_comm = mpi_grid_.get()->get_row_comm();
            MPI_Comm col_comm = mpi_grid_.get()->get_col_comm();

            std::size_t row_sendrecv_lens_ = m_;
            std::size_t row_block_size = m_; //global offset will be coord[0] * row_block_size_
            std::size_t column_sendrecv_lens_ = n_;
            std::size_t column_block_size = n_; //global offset will be coord[1] * column_block_size_   

            //ensure good size of the case M/N is not divisible by dims[0]/dims[1]
            //the block size is always m_ for the rank from 0 to dims[0]-2
            //for rank dims[0]-1 with potential smaller m_, if M is not divisible by dims[0]
            //row_block_size is computed as follows
            if(coords[0] == dims[0] - 1) //last rank in the column communicator
            {
                row_block_size = (M_ - m_) / (dims[0] - 1);
            }
            //same also for row_block_size in row communicator
            if(coords[1] == dims[1] - 1) //last rank in the row communicator
            {
                column_block_size = (N_ - n_) / (dims[1] - 1);
            }
            std::vector<T> buff(M_ * column_block_size);

            //if bcast one by one from the root as 0 to dims[0]-2, the
            //sendrecv length is always the same row_block_size for all ranks
            //we perform these bcast in a loop
            //std::vector<T> buff(row_block_size * n_);

            for(auto i = 0; i < dims[0] - 1; i++)
            {
                //packing
                if(coords[0] == i)
                {
                    chase::linalg::lapackpp::t_lacpy('A',
                                                    row_block_size,
                                                    n_,
                                                    this->l_data(),
                                                    this->l_ld(),
                                                    buff.data(),
                                                    row_block_size);                
                }

                MPI_Bcast(buff.data(), 
                        row_block_size * n_, 
                        chase::mpi::getMPI_Type<T>(), 
                        i, 
                        col_comm);                
                
                //unpacking
                chase::linalg::lapackpp::t_lacpy('A',
                                                row_block_size,
                                                n_,
                                                buff.data(),
                                                row_block_size,
                                                targetMatrix->l_data() + i * row_block_size + coords[1] * column_block_size * targetMatrix->l_ld(),
                                                targetMatrix->l_ld());            
            }

            //for last rank in the column communicator
            row_sendrecv_lens_ = m_;
            //if not last rank
            if(coords[0] != dims[0] - 1)
            {
                row_sendrecv_lens_ = M_ - (dims[0] - 1) * m_;
            }

            //for last rank, as row_sendrecv_lens_ <= row_block_size,
            //no need to reallocate buff.
            //packing
            if(coords[0] == dims[0] - 1)
            {
                chase::linalg::lapackpp::t_lacpy('A',
                                                row_sendrecv_lens_,
                                                n_,
                                                this->l_data(),
                                                this->l_ld(),
                                                buff.data(),
                                                row_sendrecv_lens_);                
            }

            MPI_Bcast(buff.data(), 
                        row_sendrecv_lens_ * n_, 
                        chase::mpi::getMPI_Type<T>(), 
                        dims[0] - 1, 
                        col_comm);                
            
            //unpacking
            chase::linalg::lapackpp::t_lacpy('A',
                                                row_sendrecv_lens_,
                                                n_,
                                                buff.data(),
                                                row_sendrecv_lens_,
                                                targetMatrix->l_data() + (dims[0] - 1) * row_block_size + coords[1] * column_block_size * targetMatrix->l_ld(),
                                                targetMatrix->l_ld());

            //now the collected data should be bcast within row communicator
            //it should follow the same scheme, but the data size is ~ M_ * n
            //since data is column major, so 1 unpacking operation is enough.      
            //same as for column comm, start with the first dims[1]-1 ranks
            //buff.resize(column_block_size * M_);

            for(auto i = 0; i < dims[1] - 1; i++)
            {
                //packing
                if(coords[1] == i)
                {
                    chase::linalg::lapackpp::t_lacpy('A',
                                                    M_,
                                                    column_block_size,
                                                    targetMatrix->l_data() + i * column_block_size * targetMatrix->l_ld(),
                                                    targetMatrix->l_ld(),
                                                    buff.data(),
                                                    M_);                
                }

                MPI_Bcast(buff.data(), 
                        column_block_size * M_, 
                        chase::mpi::getMPI_Type<T>(), 
                        i, 
                        row_comm);                
                
                //unpacking
                chase::linalg::lapackpp::t_lacpy('A',
                                                M_,
                                                column_block_size,
                                                buff.data(),
                                                M_,
                                                targetMatrix->l_data() + i * column_block_size * targetMatrix->l_ld(),
                                                targetMatrix->l_ld());
                
            }

            //for last rank in the row communicator
            column_sendrecv_lens_ = n_;
            //if not last rank
            if(coords[1] != dims[1] - 1)
            {
                column_sendrecv_lens_ = N_ - (dims[1] - 1) * n_;
            }

            //packing
            if(coords[1] == dims[1] - 1)
            {
                chase::linalg::lapackpp::t_lacpy('A',
                                                M_,
                                                column_sendrecv_lens_,
                                                targetMatrix->l_data() + (dims[1] - 1) * column_block_size * targetMatrix->l_ld(),
                                                targetMatrix->l_ld(),
                                                buff.data(),
                                                M_);                
            }

            MPI_Bcast(buff.data(), 
                        column_sendrecv_lens_ * M_, 
                        chase::mpi::getMPI_Type<T>(), 
                        dims[1] - 1, 
                        row_comm);                
            
            //unpacking
            chase::linalg::lapackpp::t_lacpy('A',
                                                M_,
                                                column_sendrecv_lens_,
                                                buff.data(),
                                                M_,
                                                targetMatrix->l_data() + (dims[1] - 1) * column_block_size * targetMatrix->l_ld(),
                                                targetMatrix->l_ld());
                        
        }                     
    }
#ifdef HAS_SCALAPACK
    std::size_t desc_[9];
#endif
};


template<typename T, typename Platform = chase::platform::CPU> 
class BlockCyclicMatrix : public AbstractDistMatrix<T, BlockCyclicMatrix, Platform>
{

public:
    using platform_type = Platform;
    using value_type = T;  // Alias for element type
    
    ~BlockCyclicMatrix() override {};
    BlockCyclicMatrix();
    BlockCyclicMatrix(std::size_t M, std::size_t N, std::size_t mb, std::size_t nb,
                    std::shared_ptr<chase::Impl::mpi::MpiGrid2DBase> mpi_grid)
                    :M_(M), N_(N), mpi_grid_(mpi_grid), mb_(mb), nb_(nb)
    {
        int *dims_ = mpi_grid_.get()->get_dims();
        int *coord_ = mpi_grid_.get()->get_coords();
        std::tie(m_, mblocks_) = numroc(M_, mb_, coord_[0], dims_[0]);
        std::tie(n_, nblocks_) = numroc(N_, nb_, coord_[1], dims_[1]);   
        ld_ = m_;
        local_matrix_ = typename chase::platform::MatrixTypePlatform<T, Platform>::type(m_, n_);      

        init_contiguous_buffer_info();

    }

    BlockCyclicMatrix(std::size_t M, std::size_t N, 
                      std::size_t m, std::size_t n,
                      std::size_t mb, std::size_t nb,
                      std::size_t ld, T *data,
                      std::shared_ptr<chase::Impl::mpi::MpiGrid2DBase> mpi_grid)
                      : M_(M), N_(N), mpi_grid_(mpi_grid), mb_(mb), nb_(nb), ld_(ld)
    {
        int *dims_ = mpi_grid_.get()->get_dims();
        int *coord_ = mpi_grid_.get()->get_coords();

        std::tie(m_, mblocks_) = numroc(M_, mb_, coord_[0], dims_[0]);
        std::tie(n_, nblocks_) = numroc(N_, nb_, coord_[1], dims_[1]); 
        if(m_ != m)
        {
            throw std::runtime_error("the local row number of input matrix is not correctly matching the given block-cyclic distribution");
        }

        if(n_ != n)
        {
            throw std::runtime_error("the local column number of input matrix is not correctly matching the given block-cyclic distribution");
        }

        if(ld_ < m_)
        {
            throw std::runtime_error("the leading dimension of local matrix is not correctly matching the given block-cyclic distribution");
        }

        local_matrix_ = typename chase::platform::MatrixTypePlatform<T, Platform>::type(m_, n_, ld_, data);        

        if constexpr (std::is_same<Platform, chase::platform::GPU>::value)
        {
            ld_ = local_matrix_.gpu_ld();
        }
        
        init_contiguous_buffer_info();
           
    }

    std::size_t g_rows() const override { return M_; }
    std::size_t g_cols() const override { return N_; }
    std::size_t l_ld() const override { return ld_; }
    std::size_t l_rows() const override { return m_; }
    std::size_t l_cols() const override { return n_;}
    std::size_t mb() const override { return mb_;}
    std::size_t nb() const override { return nb_;}
    std::size_t mblocks() {return mblocks_; }
    std::size_t nblocks() {return nblocks_; }
    std::vector<std::size_t> m_contiguous_global_offs() { return m_contiguous_global_offs_; }
    std::vector<std::size_t> n_contiguous_global_offs() { return n_contiguous_global_offs_; }
    std::vector<std::size_t> m_contiguous_local_offs() { return m_contiguous_local_offs_; }
    std::vector<std::size_t> n_contiguous_local_offs() { return n_contiguous_local_offs_; }
    std::vector<std::size_t> m_contiguous_lens() { return m_contiguous_lens_; }
    std::vector<std::size_t> n_contiguous_lens() {return n_contiguous_lens_; }
    
    //std::size_t *g_offs() {}
    T *         l_data() override { 
        if constexpr (std::is_same<Platform, chase::platform::CPU>::value)
        {
            return local_matrix_.data();
        }
#ifdef HAS_CUDA        
        else
        {
            return local_matrix_.gpu_data();
        }
#endif 

    }
    
    typename chase::platform::MatrixTypePlatform<T, Platform>::type& loc_matrix() override { return local_matrix_;}

    // Accessors for MPI grid
    chase::Impl::mpi::MpiGrid2DBase* getMpiGrid() const override {
        return mpi_grid_.get();
    }

    std::shared_ptr<chase::Impl::mpi::MpiGrid2DBase> getMpiGrid_shared_ptr() const override
    {
        return mpi_grid_;
    }

    template<typename CloneVectorType>
    CloneVectorType cloneMultiVector(std::size_t g_M, std::size_t g_N)
    {
        static_assert(
            std::is_same_v<T, typename CloneVectorType::value_type>,
            "Cloned type must have the same value_type"
        );
        ///using NewCommType = typename CloneType::communicator_type;
        return CloneVectorType(g_M, g_N, mb_, mpi_grid_);        
    }

    //only save from CPU buffer
    //for saving GPU data, need to copy to CPU by D2H()
    void saveToBinaryFile(const std::string& filename) {
    	MPI_File fileHandle;
        MPI_Status status;
        T *buff;
        std::size_t cpu_ld;

        if constexpr (std::is_same<Platform, chase::platform::GPU>::value)
        {
            if(local_matrix_.cpu_data() == nullptr)
            {
               throw std::runtime_error("[BlockBlockMatrix]: only can save data from CPU buffer");
            }
            buff = local_matrix_.cpu_data();
            cpu_ld = local_matrix_.cpu_ld();
        }else
        {
            buff = local_matrix_.data();
            cpu_ld = local_matrix_.ld();
        }

        if(MPI_File_open(this->mpi_grid_.get()->get_comm(), filename.data(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fileHandle) != MPI_SUCCESS)
        {
            if(this->grank() == 0)
                std::cout << "Can't open input matrix - " << filename << std::endl;
            MPI_Abort(this->mpi_grid_.get()->get_comm(), EXIT_FAILURE);
        }

        if (this->l_data() == nullptr) {
            throw std::runtime_error("[BlockBlockMatrix]: Original data is not initialized.");
        }

        std::vector<T> tmp;
        if(cpu_ld > m_)
        {
            tmp.resize(m_ * n_);
            chase::linalg::lapackpp::t_lacpy('A', m_, n_, buff, cpu_ld, tmp.data(), m_);
            buff = tmp.data();
        }

        int *dims_ = mpi_grid_.get()->get_dims();

        int gsizes[2] = {(int)M_, (int)N_};
        int distribs[2] = {MPI_DISTRIBUTE_CYCLIC, MPI_DISTRIBUTE_CYCLIC};
    	int dargs[2] = {(int)mb_,(int)nb_};
	    int psizes[2] = {dims_[0], dims_[1]};
        int order = MPI_ORDER_FORTRAN;

        MPI_Datatype darray;
        MPI_Type_create_darray(this->mpi_grid_.get()->get_nprocs(), this->mpi_grid_.get()->get_myRank(), 2, gsizes, distribs, dargs, psizes, order, chase::mpi::getMPI_Type<T>(), &darray);
        MPI_Type_commit(&darray);

        MPI_Count count_write = m_ * n_;
        MPI_File_set_view(fileHandle, 0, chase::mpi::getMPI_Type<T>(), darray, "native", MPI_INFO_NULL);

        MPI_File_write_all(fileHandle, buff, count_write, chase::mpi::getMPI_Type<T>(), &status);

        MPI_Type_free(&darray);

        if (MPI_File_close(&fileHandle) != MPI_SUCCESS)
        {
            MPI_Abort(this->mpi_grid_.get()->get_comm(), EXIT_FAILURE);
        }
    }

    // Read matrix data from a binary file
    void readFromBinaryFile(const std::string& filename) 
    {
        T *buff;
        std::size_t cpu_ld;
        if constexpr (std::is_same<Platform, chase::platform::GPU>::value)
        {
            if(local_matrix_.cpu_data() == nullptr)
            {
               local_matrix_.allocate_cpu_data();
            }
            buff = local_matrix_.cpu_data();
            cpu_ld = local_matrix_.cpu_ld();
        }else
        {
            buff = local_matrix_.data();
            cpu_ld = local_matrix_.ld();
        }
#ifdef USE_MPI_IO
        int *dims_ = mpi_grid_.get()->get_dims();
        int gsizes[2] = {(int)M_, (int)N_};
        int distribs[2] = {MPI_DISTRIBUTE_CYCLIC, MPI_DISTRIBUTE_CYCLIC};
        int dargs[2] = {(int)mb_,(int)nb_};
	    int psizes[2] = {dims_[0], dims_[1]};
        int order = MPI_ORDER_FORTRAN;

        MPI_Datatype darray;
        MPI_Type_create_darray(this->mpi_grid_.get()->get_nprocs(), this->mpi_grid_.get()->get_myRank(), 2, gsizes, distribs, dargs, psizes, order, chase::mpi::getMPI_Type<T>(), &darray);
        MPI_Type_commit(&darray);

    	MPI_File fileHandle;
        MPI_Status status;
        int access_mode = MPI_MODE_RDONLY;

        if(MPI_File_open(this->mpi_grid_.get()->get_comm(), filename.data(), access_mode, MPI_INFO_NULL, &fileHandle) != MPI_SUCCESS)
        {
            if(this->grank() == 0)
                std::cout << "Can't open input matrix - " << filename << std::endl;
            MPI_Abort(this->mpi_grid_.get()->get_comm(), EXIT_FAILURE);
        }

        if (this->l_data() == nullptr) {
            throw std::runtime_error("[BlockBlockMatrix]: Original data is not initialized.");
        }
        std::vector<T> tmp;
        if(cpu_ld > m_)
        {
            tmp.resize(m_ * n_);
        }

        MPI_Count count_read = m_ * n_;
        MPI_File_set_view(fileHandle, 0, chase::mpi::getMPI_Type<T>(), darray, "native", MPI_INFO_NULL);
        if(cpu_ld > m_)
        {
            MPI_File_read_all(fileHandle, tmp.data(), count_read, chase::mpi::getMPI_Type<T>(), &status);
            chase::linalg::lapackpp::t_lacpy('A', m_, n_, tmp.data(), m_, buff, cpu_ld);
        }else
        {
            MPI_File_read_all(fileHandle, buff, count_read, chase::mpi::getMPI_Type<T>(), &status);
        }
        MPI_Type_free(&darray);

        if (MPI_File_close(&fileHandle) != MPI_SUCCESS)
        {
            MPI_Abort(this->mpi_grid_.get()->get_comm(), EXIT_FAILURE);
        }
#else
	    std::ifstream input(filename.data(), std::ios::binary);	
        if (!input.is_open()) {
            throw std::runtime_error("[BlockCyclicMatrix]: Failed to open file for reading.");
        }
        
        if (this->l_data() == nullptr) {
            throw std::runtime_error("[BlockCyclicMatrix]: Original data is not initialized.");
        }
	
        for (std::size_t j = 0; j < nblocks_; j++)
	    {
            for (std::size_t i = 0; i < mblocks_; i++)
            {
                for (std::size_t q = 0; q < n_contiguous_lens_[j]; q++)
                {
                    input.seekg(((q + n_contiguous_global_offs_[j]) * M_ + m_contiguous_global_offs_[i]) * sizeof(T));
                    input.read(reinterpret_cast<char*>(buff + (q + n_contiguous_local_offs_[j]) * this->cpu_ld() +
                                                       m_contiguous_local_offs_[i]),
                               m_contiguous_lens_[i] * sizeof(T));
                }
            }
        }
        input.close();

#endif

    }

#ifdef HAS_SCALAPACK
    std::size_t *get_scalapack_desc(){ return desc_; }
#endif

#ifdef HAS_SCALAPACK
    std::size_t *scalapack_descriptor_init()
    {
        int comm2D_ctxt = mpi_grid_.get()->get_blacs_comm2D_ctxt();
        int zero = 0;
        int one = 1;
        int info;

        if constexpr (std::is_same<Platform, chase::platform::GPU>::value)
        {
            if(local_matrix_.cpu_data() == nullptr)
            {
                local_matrix_.allocate_cpu_data();
            }
        }

        std::size_t ldd = this->cpu_ld();

        chase::linalg::scalapackpp::t_descinit(desc_, 
                                               &M_, 
                                               &N_, 
                                               &mb_, 
                                               &nb_, 
                                               &zero, 
                                               &zero, 
                                               &comm2D_ctxt, 
                                               &ldd, 
                                               &info); 


        return desc_;
    }
#endif

    template<template <typename, typename> class targetType>
    void redistributeImpl(targetType<T, Platform>* targetMatrix)//,
                            //std::size_t offset, std::size_t subsetSize)
    {
        if(M_ != targetMatrix->g_rows() || N_ != targetMatrix->g_cols() )
        {
            throw std::runtime_error("[BlockCyclicMatrix]: redistribution requires original and target matrices have same global size");
        }

        if constexpr (std::is_same<typename distMatrixTypeTrait<Redundant, T, Platform>::type, targetType<T, Platform>>::value) 
        {
            //redistributeToRedundant(targetMatrix);
        }
        else if constexpr (std::is_same<typename distMatrixTypeTrait<BlockCyclic, T, Platform>::type, targetType<T, Platform>>::value) 
        {
            throw std::runtime_error("[BlockCyclicMatrix]: no need to redistribute from BlockBlock to BlockBlock");
        }else
        {
            throw std::runtime_error("[BlockCyclicMatrix]:  no support for redistribution from redundant to othertypes yet");
        }        
    }

private:
    std::size_t M_;
    std::size_t N_;
    std::size_t m_;
    std::size_t n_;
    std::size_t ld_;
    std::size_t mb_;
    std::size_t nb_;
    std::size_t mblocks_;
    std::size_t nblocks_;    
    std::vector<std::size_t> m_contiguous_global_offs_;
    std::vector<std::size_t> n_contiguous_global_offs_;
    std::vector<std::size_t> m_contiguous_local_offs_;
    std::vector<std::size_t> n_contiguous_local_offs_;
    std::vector<std::size_t> m_contiguous_lens_;
    std::vector<std::size_t> n_contiguous_lens_;

    typename chase::platform::MatrixTypePlatform<T, Platform>::type local_matrix_;
    std::shared_ptr<chase::Impl::mpi::MpiGrid2DBase> mpi_grid_;

    void init_contiguous_buffer_info()
    {
        int *coords =  mpi_grid_.get()->get_coords();
        int *dims = mpi_grid_.get()->get_dims();

        std::size_t nr, nc;
        int sendr = 0;
        int sendc = 0;
        for (std::size_t r = 0; r < M_; r += mb_, sendr = (sendr + 1) % dims[0])
        {
            nr = mb_;
            if(M_ - r < mb_)
            {
                nr = M_ - r;
            }

            if(coords[0] == sendr)
            {
                m_contiguous_global_offs_.push_back(r);
                m_contiguous_lens_.push_back(nr);
            }
        }

        for (std::size_t c = 0; c < N_; c += nb_, sendc = (sendc + 1) % dims[1])
        {
            nc = nb_;
            if(N_ - c < nb_)
            {
                nc = N_ - c;
            }

            if(coords[1] == sendc)
            {
                n_contiguous_global_offs_.push_back(c);
                n_contiguous_lens_.push_back(nc);
            }
        }

        m_contiguous_local_offs_.resize(mblocks_);
        n_contiguous_local_offs_.resize(nblocks_);

        m_contiguous_local_offs_[0] = 0;
        n_contiguous_local_offs_[0] = 0;

        for (std::size_t i = 1; i < mblocks_; i++)
        {
            m_contiguous_local_offs_[i] = m_contiguous_local_offs_[i - 1] 
                                          + m_contiguous_lens_[i - 1];
        }

        for (std::size_t j = 1; j < nblocks_; j++)
        {
            n_contiguous_local_offs_[j] = n_contiguous_local_offs_[j - 1] 
                                          + n_contiguous_lens_[j - 1];
        }
    }

    void redistributeToRedundant(RedundantMatrix<T, Platform>* targetMatrix)
    {}
#ifdef HAS_SCALAPACK
    std::size_t desc_[9];
#endif
};


}
}
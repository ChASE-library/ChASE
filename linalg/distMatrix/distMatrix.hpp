// This file is a part of ChASE.
// Copyright (c) 2015-2024, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

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
#include "external/scalapackpp/scalapackpp.hpp"
#include "grid/mpiGrid2D.hpp"
#include "grid/mpiTypes.hpp"

#ifdef HAS_CUDA
#include <cuda_runtime.h>
#include "Impl/chase_gpu/cuda_utils.hpp"
#include "linalg/internal/cuda/lacpy.hpp"
#include "external/cublaspp/cublaspp.hpp"
#include "linalg/internal/cuda/precision_conversion.cuh"
#endif

/**
 * @page chase_dist_matrix_namespace chase::distMatrix Namespace
 * The `chase::distMatrix` namespace contains a set of matrix types designed for distributed memory environments. 
 * These matrices are meant to support different data distributions across multiple MPI processes.
 */

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

/**
 * @defgroup dist_matrix_classes Distributed Matrix Classes
 * @brief Classes and utilities for distributed matrix operations.

 * The `distMatrix` module provides several matrix types that are distributed across MPI ranks. These matrix types are:
 * - `RedundantMatrix`: A matrix where all elements are replicated across MPI processes.
 * - `BlockBlockMatrix`: A matrix distributed in block-wise fashion.
 * - `BlockCyclicMatrix`: A matrix distributed cyclically in blocks.
 * 
 * These matrix classes are templated on the element type (`T`) and the platform (`Platform`), where `Platform` is typically `chase::platform::CPU` or `chase::platform::GPU`. 
 * @{
 */   
namespace distMatrix {

/**
* @brief Represents the BlockBlock Matrix.
*/
struct BlockBlock {}; 
/**
* @brief Represents the Redundant Matrix.
*/
struct Redundant {}; 
/**
* @brief Represents the BlockCyclic Matrix.
*/
struct BlockCyclic {}; 

template<typename T, typename Platform>
class RedundantMatrix;

template<typename T, typename Platform>
class BlockBlockMatrix;

template<typename T, typename Platform>
class QuasiHermitianBlockBlockMatrix;

template<typename T, typename Platform>
class BlockCyclicMatrix;

/**
 * @class AbstractDistMatrix
 * @brief Abstract base class for distributed matrices
 * 
 * `AbstractDistMatrix` is the base class for all distributed matrix types, providing basic functionality for matrix distribution and communication.
 * It provides methods for enabling and disabling single precision matrices, as well as memory management for both CPU and GPU platforms.
 * 
 * @tparam T Element type of the matrix.
 * @tparam Derived Derived matrix type (RedundantMatrix, BlockBlockMatrix, etc.).
 * @tparam Platform Platform type (CPU, GPU).
 */
template <typename T, template <typename, typename> class Derived, typename Platform = chase::platform::CPU>
class AbstractDistMatrix {
protected:
    // Single precision matrix
#ifdef ENABLE_MIXED_PRECISION       
    using SinglePrecisionType = typename chase::ToSinglePrecisionTrait<T>::Type; /**< Single precision type */
    using SinglePrecisionDerived = Derived<SinglePrecisionType, Platform>; /**< Derived class in single precision */
    std::unique_ptr<SinglePrecisionDerived> single_precision_matrix_; /**< Pointer to single precision distMatrix */
    bool is_single_precision_enabled_ = false;  /**< Flag indicating if single precision is enabled */
    std::chrono::high_resolution_clock::time_point start, end; /**< Timing points for performance measurement */
    std::chrono::duration<double> elapsed; /**< Duration for single precision operations */
#endif

public:
    /**
     * @brief Virtual destructor
     */
    virtual ~AbstractDistMatrix() = default;
    /**
     * @brief Get the 2D MPI grid associated with the matrix.
     * 
     * @return Pointer to the MPI grid object.
     */    
    virtual chase::grid::MpiGrid2DBase* getMpiGrid() const = 0;   
    /**
     * @brief Get a shared pointer to the 2D MPI grid.
     * 
     * @return Shared pointer to the MPI grid object.
     */    
    virtual std::shared_ptr<chase::grid::MpiGrid2DBase> getMpiGrid_shared_ptr() const = 0;
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
    virtual T *         l_data() = 0;
    //virtual typename chase::platform::MatrixTypePlatform<T, Platform>::type& loc_matrix() = 0;
    /**
     * @brief Get the local matrix object.
     * @return Reference to Matrix object representing local matrix data.
     */     
    virtual chase::matrix::Matrix<T, Platform>& loc_matrix() = 0;
    /**
     * @brief Get the block size for block-cyclic distribution.
     * @return Block size for distribution.
     */     
    virtual std::size_t mb() const = 0;
    /**
     * @brief Get the block size for block-cyclic distribution.
     * @return Block size for distribution.
     */     
    virtual std::size_t nb() const = 0;  
    //virtual std::unique_ptr<SinglePrecisionDerived> createSinglePrecisionMatrix() = 0;
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
        auto& loc_matrix = this->loc_matrix();
        if constexpr (std::is_same<Platform, chase::platform::GPU>::value)
        {
            loc_matrix.D2H();
        }else
        {
            throw std::runtime_error("[DistMatrix]: CPU type of matrix do not support D2H operation");
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
        auto& loc_matrix = this->loc_matrix();
        if constexpr (std::is_same<Platform, chase::platform::GPU>::value)
        {
            loc_matrix.H2D();
        }else
        {
            throw std::runtime_error("[DistMatrix]: CPU type of matrix do not support H2D operation");
        }
    }
    /**
     * @brief Allocate memory for local data on the CPU.
     */
    void allocate_cpu_data()
    {
        auto& loc_matrix = this->loc_matrix();
        loc_matrix.allocate_cpu_data();   
    }
#endif
    /**
     * @brief Provides a pointer to the CPU data.
     * 
     * @return Pointer to the CPU data.
     */
    T *cpu_data()
    {
        auto& loc_matrix = this->loc_matrix();
        return loc_matrix.cpu_data();      
    }

    /**
     * @brief Retrieves the leading dimension for the CPU data.
     * 
     * @return Leading dimension of the CPU data.
     */
    std::size_t cpu_ld()
    {
        auto& loc_matrix = this->loc_matrix();
        return loc_matrix.cpu_ld();          
    }

#ifdef ENABLE_MIXED_PRECISION       
    // Enable single precision for double types (and complex<double>)
    /**
     * @brief Enable single precision for double types.
     */      
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
    /**
     * @brief Disable single precision and optionally copy data back.
     * @param copyback Whether to copy data back to double precision.
     */      
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
    /**
     * @brief Check if single precision is enabled.
     * @return True if single precision is enabled, false otherwise.
     */   
    template <typename U = T, typename std::enable_if<std::is_same<U, double>::value || std::is_same<U, std::complex<double>>::value, int>::type = 0>
    bool isSinglePrecisionEnabled() const {
        return is_single_precision_enabled_;
    }

    // Get the single precision matrix itself
    /**
     * @brief Get the single precision multi-vector.
     * @return Pointer to the single precision multi-vector.
     */
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

/**
 * @class RedundantMatrix
 * @brief A class representing a distributed matrix with redundant storage.
 *
 * This class inherits from `AbstractDistMatrix` and represents a matrix
 * that is distributed across an MPI 2D grid. It supports operations like
 * redistribution to other matrix types (BlockBlock, BlockCyclic), and provides
 * access to the matrix's local data and MPI grid.
 *
 * @tparam T The element type of the matrix.
 * @tparam Platform The platform on which the matrix is hosted (defaults to CPU).
 */
template<typename T, typename Platform = chase::platform::CPU> 
class RedundantMatrix : public AbstractDistMatrix<T, RedundantMatrix, Platform>
{

public:
    using platform_type = Platform; ///< Alias for the platform type
    using value_type = T;  ///< Alias for the element type of the matrix
    using matrix_type = Redundant; ///< Alias for the matrix type (Redundant)
    /**
     * @brief Destructor for RedundantMatrix.
     *
     */
    ~RedundantMatrix() override {};
    /**
     * @brief Default constructor for RedundantMatrix.
     */    
    RedundantMatrix();
    /**
     * @brief Constructor initializing a RedundantMatrix with given dimensions and MPI grid.
     *
     * @param m Number of rows for the local matrix.
     * @param n Number of columns for the local matrix.
     * @param mpi_grid A shared pointer to the MPI grid for the distribution.
     */    
    RedundantMatrix(std::size_t m, std::size_t n,
                    std::shared_ptr<chase::grid::MpiGrid2DBase> mpi_grid)
                    :m_(m), n_(n), ld_(m_), mpi_grid_(mpi_grid)
    {
        M_ = m_;
        N_ = n_;
        local_matrix_ = chase::matrix::Matrix<T, Platform>(m_, n_); 
    }
    /**
     * @brief Constructor initializing a RedundantMatrix with given data, dimensions, and MPI grid.
     *
     * @param m Number of rows for the local matrix.
     * @param n Number of columns for the local matrix.
     * @param ld Leading dimension of the local matrix.
     * @param data Pointer to the data for initializing the matrix.
     * @param mpi_grid A shared pointer to the MPI grid for the distribution.
     */
    RedundantMatrix(std::size_t m, std::size_t n, std::size_t ld, T *data, 
                    std::shared_ptr<chase::grid::MpiGrid2DBase> mpi_grid)
                    :m_(m), n_(n), ld_(ld), mpi_grid_(mpi_grid)
    {
        M_ = m_;
        N_ = n_;        
        local_matrix_ = chase::matrix::Matrix<T, Platform>(m_, n_, ld_, data);
        
        if constexpr (std::is_same<Platform, chase::platform::GPU>::value)
        {
            ld_ = local_matrix_.ld();
        }    
    }
    /**
     * @brief Get the global number of rows.
     * 
     * @return Global number of rows in the matrix.
     */
    std::size_t g_rows() const override { return M_;}
    /**
     * @brief Get the global number of columns.
     * 
     * @return Global number of columns in the matrix.
     */    
    std::size_t g_cols() const override { return N_;}
    /**
     * @brief Get the local number of rows.
     * 
     * @return Local number of rows in the matrix.
     */    
    std::size_t l_rows() const override { return m_;}
    /**
     * @brief Get the local number of columns.
     * 
     * @return Local number of columns in the matrix.
     */    
    std::size_t l_cols() const override { return n_;}
    /**
     * @brief Get the local leading dimension (stride).
     * 
     * @return Local leading dimension of the matrix.
     */    
    std::size_t l_ld() const override { return ld_;}
    /**
     * @brief Get the block size in the row direction (unused for redundant matrices).
     * 
     * @return 0 as the block size is not applicable.
     */    
    std::size_t mb() const override { return 0;}
    /**
     * @brief Get the block size in the column direction (unused for redundant matrices).
     * 
     * @return 0 as the block size is not applicable.
     */    
    std::size_t nb() const override { return 0;} 
    /**
     * @brief Get the pointer to the local matrix data.
     *
     * @return A pointer to the local matrix data.
     */
    T *         l_data() override { 
        return local_matrix_.data();       
    }
    /**
     * @brief Get a reference to the local matrix.
     * 
     * @return A reference to the local matrix object.
     */    
    chase::matrix::Matrix<T, Platform>& loc_matrix() override { return local_matrix_;}


    // Accessors for MPI grid
    /**
     * @brief Get the MPI grid associated with this matrix.
     *
     * @return A raw pointer to the MPI grid.
     */    
    chase::grid::MpiGrid2DBase* getMpiGrid() const override {
        return mpi_grid_.get();
    }
    /**
     * @brief Get the shared pointer to the MPI grid.
     *
     * @return A shared pointer to the MPI grid.
     */
    std::shared_ptr<chase::grid::MpiGrid2DBase> getMpiGrid_shared_ptr() const override
    {
        return mpi_grid_;
    }
    /**
     * @brief Map the matrix to a new MPI grid.
     * 
     * This method updates the MPI grid of the matrix to a new one.
     * 
     * @param new_mpi_grid A shared pointer to the new MPI grid.
     */
    void mapToNewMpiGrid(std::shared_ptr<chase::grid::MpiGrid2DBase> new_mpi_grid)
    {
        mpi_grid_ = new_mpi_grid;
    }

    //here the startrow/col indices should be the global indices
    /**
     * @brief Redistribute the matrix to another matrix type.
     * 
     * This method supports redistribution to BlockBlock, BlockCyclic, or other supported types.
     * 
     * @tparam targetType The target matrix type to redistribute to.
     * @param targetMatrix A pointer to the target matrix to redistribute the data to.
     * @param startRow Starting row index for the redistribution (global index).
     * @param subRows Number of rows to redistribute.
     * @param startCol Starting column index for the redistribution (global index).
     * @param subCols Number of columns to redistribute.
     * 
     * @throws std::runtime_error If the global dimensions do not match between matrices.
     */    
    template<template <typename, typename> class targetType>
    void redistributeImpl(targetType<T, Platform>* targetMatrix,
                          std::size_t startRow, std::size_t subRows, std::size_t startCol, std::size_t subCols)
    {
        if(M_ != targetMatrix->g_rows() || N_ != targetMatrix->g_cols() )
        {
            throw std::runtime_error("[RedundantMatrix]: redistribution requires original and target matrices have same global size");
        }

        else if constexpr (std::is_same<typename targetType<T, Platform>::matrix_type, BlockBlock>::value)
        {
            redistributeToBlockBlock(targetMatrix, startRow, subRows, startCol, subCols);
        }
        else if constexpr (std::is_same<typename targetType<T, Platform>::matrix_type, BlockCyclic>::value) 
        {
            redistributeToBlockCyclic(targetMatrix, startRow, subRows, startCol, subCols);
        }        
        else if constexpr (std::is_same<typename targetType<T, Platform>::matrix_type, Redundant>::value) 
        {
            throw std::runtime_error("[RedundantMatrix]: no need to redistribute from redundant to redundant");
        }else
        {
            throw std::runtime_error("[RedundantMatrix]: no support for redistribution from redundant to othertypes yet");
        }      
    }   
    /**
     * @brief Redistribute the matrix to another matrix type with full global dimensions.
     * 
     * This method redistributes the entire matrix without specifying a sub-region.
     * 
     * @tparam targetType The target matrix type to redistribute to.
     * @param targetMatrix A pointer to the target matrix to redistribute the data to.
     */
    template<template <typename, typename> class targetType>
    void redistributeImpl(targetType<T, Platform>* targetMatrix)
    {
        this->redistributeImpl(targetMatrix, 0, this->g_rows(), 0, this->g_cols());
    }
    /**
     * @brief Redistribute the matrix to another matrix type with extended logic.
     * 
     * Similar to `redistributeImpl`, but includes additional logic to handle
     * more complex cases.
     *
     * @tparam TargetMatrixType The target matrix type to redistribute to.
     * @param targetMatrix A pointer to the target matrix.
     * @param startRow Starting row index (global index).
     * @param subRows Number of rows to redistribute.
     * @param startCol Starting column index (global index).
     * @param subCols Number of columns to redistribute.
     * 
     * @throws std::runtime_error If global dimensions do not match.
     */
    template <typename TargetMatrixType>
    void redistributeImpl_2(TargetMatrixType* targetMatrix,
                        std::size_t startRow, std::size_t subRows, std::size_t startCol, std::size_t subCols)
    {
        // Check if global sizes match
        if (M_ != targetMatrix->g_rows() || N_ != targetMatrix->g_cols())
        {
            throw std::runtime_error("[RedundantMatrix]: Redistribution requires both matrices to have the same global size.");
        }

        // Dispatch based on matrix type using TargetMatrixType's `matrix_type` alias
        if constexpr (std::is_same<typename TargetMatrixType::matrix_type, BlockBlock>::value)
        {
            redistributeToBlockBlock_2(targetMatrix, startRow, subRows, startCol, subCols);
        }
        else if constexpr (std::is_same<typename TargetMatrixType::matrix_type, BlockCyclic>::value) 
        {
            redistributeToBlockCyclic_2(targetMatrix, startRow, subRows, startCol, subCols);
        }
        else if constexpr (std::is_same<typename TargetMatrixType::matrix_type, Redundant>::value) 
        {
            throw std::runtime_error("[RedundantMatrix]: No need to redistribute from redundant to redundant.");
        }
        else
        {
            throw std::runtime_error("[RedundantMatrix]: Unsupported redistribution target type.");
        }
    }
    /**
     * @brief Redistribute the matrix to another matrix type with full global dimensions.
     *
     * Similar to `redistributeImpl_2`, but without specifying a sub-region.
     *
     * @tparam TargetMatrixType The target matrix type to redistribute to.
     * @param targetMatrix A pointer to the target matrix.
     */
    template <typename TargetMatrixType>
    void redistributeImpl_2(TargetMatrixType* targetMatrix)
    {
        this->redistributeImpl_2(targetMatrix, 0, this->g_rows(), 0, this->g_cols());
    }

private:
    std::size_t M_; ///< Global number of rows in the matrix
    std::size_t N_; ///< Global number of columns in the matrix
    std::size_t m_; ///< Local number of rows in the matrix
    std::size_t n_; ///< Local number of columns in the matrix
    std::size_t ld_; ///< Local leading dimension (stride)

    chase::matrix::Matrix<T, Platform> local_matrix_; ///< Local matrix holding the data
    std::shared_ptr<chase::grid::MpiGrid2DBase> mpi_grid_; ///< Shared pointer to the MPI grid   

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
                        targetMatrix->cpu_data()[x + targetMatrix->cpu_ld() * y] = this->cpu_data()[(g_offs[1] + y) * this->cpu_ld() + (g_offs[0] + x)];
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

    template <typename TargetMatrixType>
    void redistributeToBlockBlock_2(TargetMatrixType* targetMatrix,
                                   std::size_t startRow, std::size_t subRows, std::size_t startCol, std::size_t subCols)
    {
        using TargetPlatform = typename TargetMatrixType::platform_type; // Extract platform type from TargetMatrixType
        //later should check, if this->platform and otherplatform = GPU, copy from GPU to GPU
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
                                targetMatrix->cpu_data()[(q + n_contiguous_local_offs[j]) * targetMatrix->cpu_ld() + p + m_contiguous_local_offs[i]]
                                    = this->cpu_data()[(q + n_contiguous_global_offs[j]) * this->cpu_ld() + p + m_contiguous_global_offs[i]];
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
        }
#endif        
    }

    template <typename TargetMatrixType>
    void redistributeToBlockCyclic_2(TargetMatrixType* targetMatrix,
                                   std::size_t startRow, std::size_t subRows, std::size_t startCol, std::size_t subCols)
    {
        using TargetPlatform = typename TargetMatrixType::platform_type; // Extract platform type from TargetMatrixType

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
    }   

};

/**
 * @brief Class representing a distributed block-block matrix.
 *
 * This class implements a matrix that is distributed across 2D MPI grids. It inherits from
 * `AbstractDistMatrix` and provides methods for handling matrix data in distributed memory 
 * environments, including matrix manipulation, serialization, and redistribution.
 *
 * @tparam T The element type of the matrix (e.g., float, double).
 * @tparam Platform The platform on which the matrix will be operated (e.g., CPU or GPU).
 */
template<typename T, typename Platform = chase::platform::CPU> 
class BlockBlockMatrix : public AbstractDistMatrix<T, BlockBlockMatrix, Platform>
{

public:
    using platform_type = Platform; ///< Platform type (e.g., CPU or GPU).
    using value_type = T;           ///< Alias for the element type.
    using matrix_type = BlockBlock; ///< Alias for the matrix type.
    /**
     * @brief Destructor for BlockBlockMatrix.
     */
    ~BlockBlockMatrix() override {};
    /**
     * @brief Default constructor for BlockBlockMatrix.
     */    
    BlockBlockMatrix();
    /**
     * @brief Constructor for BlockBlockMatrix with given dimensions and MPI grid.
     *
     * Initializes the matrix with global dimensions M and N and sets up MPI communication grid.
     * Calculates the local matrix dimensions based on the MPI grid configuration.
     *
     * @param M Global number of rows of the matrix.
     * @param N Global number of columns of the matrix.
     * @param mpi_grid Shared pointer to the MPI grid configuration.
     */    
    BlockBlockMatrix(std::size_t M, std::size_t N,
                    std::shared_ptr<chase::grid::MpiGrid2DBase> mpi_grid)
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
        local_matrix_ = chase::matrix::Matrix<T, Platform>(m_, n_);     
    }
    /**
     * @brief Constructor for BlockBlockMatrix with local data.
     *
     * Initializes the matrix with local dimensions m and n, and loads the matrix data.
     * Also sets up the global matrix size using MPI operations.
     *
     * @param m Local number of rows of the matrix.
     * @param n Local number of columns of the matrix.
     * @param ld Leading dimension of the matrix.
     * @param data Pointer to the data buffer holding the matrix data.
     * @param mpi_grid Shared pointer to the MPI grid configuration.
     */
    BlockBlockMatrix(std::size_t m, std::size_t n, std::size_t ld, T *data,
                    std::shared_ptr<chase::grid::MpiGrid2DBase> mpi_grid)
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

        local_matrix_ = chase::matrix::Matrix<T, Platform>(m_, n_, ld_, data);

        if constexpr (std::is_same<Platform, chase::platform::GPU>::value)
        {
            ld_ = local_matrix_.ld();
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
    /**
     * @brief Get the global number of rows in the matrix.
     *
     * @return Global number of rows.
     */
    std::size_t g_rows() const override { return M_;}
    /**
     * @brief Get the global number of columns in the matrix.
     *
     * @return Global number of columns.
     */    
    std::size_t g_cols() const override { return N_;}
    /**
     * @brief Get the leading dimension of the local matrix.
     *
     * @return Leading dimension of the local matrix.
     */    
    std::size_t l_ld() const override { return ld_;}
    /**
     * @brief Get the number of rows in the local matrix.
     *
     * @return Number of rows in the local matrix.
     */    
    std::size_t l_rows() const override { return m_;}
    /**
     * @brief Get the number of columns in the local matrix.
     *
     * @return Number of columns in the local matrix.
     */    
    std::size_t l_cols() const override { return n_;}
    /**
     * @brief Get the global offsets for this matrix.
     *
     * @return Pointer to the array holding global offsets for the matrix.
     */    
    std::size_t *g_offs() { return g_offs_;}
    /**
     * @brief Get the block size in the row direction (not implemented).
     *
     * @return Always returns 0.
     */    
    std::size_t mb() const override { return 0;}
    /**
     * @brief Get the block size in the column direction (not implemented).
     *
     * @return Always returns 0.
     */    
    std::size_t nb() const override { return 0;}    
    /**
     * @brief Get the local data pointer for the matrix.
     *
     * @return Pointer to the local data of the matrix.
     */    
    T *         l_data() override { 
        return local_matrix_.data();       
    }
    /**
     * @brief Get the local matrix object.
     *
     * @return Reference to the local matrix.
     */    
    chase::matrix::Matrix<T, Platform>& loc_matrix() override { return local_matrix_;}

    // Accessors for MPI grid
    /**
     * @brief Get the MPI grid configuration.
     *
     * @return Pointer to the MPI grid configuration.
     */    
    chase::grid::MpiGrid2DBase* getMpiGrid() const override {
        return mpi_grid_.get();
    }
    /**
     * @brief Get the shared pointer to the MPI grid configuration.
     *
     * @return Shared pointer to the MPI grid configuration.
     */
    std::shared_ptr<chase::grid::MpiGrid2DBase> getMpiGrid_shared_ptr() const override
    {
        return mpi_grid_;
    }
    /**
     * @brief Clone the matrix as a multi-vector of a different type.
     *
     * @tparam CloneVectorType Type of the vector to be cloned.
     * @param g_M Global number of rows in the cloned vector.
     * @param g_N Global number of columns in the cloned vector.
     * @return The cloned vector of the given type.
     */
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
    /**
     * @brief Save the matrix data to a binary file.
     *
     * Saves the matrix data to a binary file, with special handling for CPU and GPU platforms.
     * Only CPU buffer data is saved, and GPU data is transferred to CPU before saving.
     *
     * @param filename Name of the file where the matrix data is saved.
     * @throws std::runtime_error If the matrix data is not initialized.
     */    
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
    /**
     * @brief Read matrix data from a binary file.
     *
     * Reads the matrix data from a binary file. If the matrix is on a GPU, data is transferred to
     * the CPU first before reading. For MPI I/O, MPI is used to read the matrix.
     *
     * @param filename Name of the file from which to read the matrix data.
     * @throws std::runtime_error If the matrix data is not initialized or if the file cannot be opened.
     */    
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
    /**
     * @brief Get the ScaLAPACK descriptor for the matrix.
     *
     * @return Pointer to the ScaLAPACK descriptor.
     */
    std::size_t *get_scalapack_desc(){ return desc_; }
#endif

#ifdef HAS_SCALAPACK
    /**
     * @brief Initialize the ScaLAPACK descriptor for the matrix.
     *
     * Initializes the ScaLAPACK descriptor based on the matrix dimensions and MPI grid configuration.
     *
     * @return Pointer to the initialized ScaLAPACK descriptor.
     */
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
    /**
     * @brief Redistribute the matrix to another type of matrix.
     *
     * Redistributes the matrix data to another matrix of a different type (e.g., from BlockBlockMatrix
     * to another matrix type).
     *
     * @tparam targetType Type of the target matrix to redistribute data to.
     * @param targetMatrix Pointer to the target matrix.
     */
    template<template <typename, typename> class targetType>
    void redistributeImpl(targetType<T, Platform>* targetMatrix)//,
                            //std::size_t offset, std::size_t subsetSize)
    {
        if(M_ != targetMatrix->g_rows() || N_ != targetMatrix->g_cols() )
        {
            throw std::runtime_error("[BlockBlockMatrix]: redistribution requires original and target matrices have same global size");
        }

        if constexpr (std::is_same<typename targetType<T, Platform>::matrix_type, Redundant>::value) 
        {
            redistributeToRedundant(targetMatrix);
        }
        else if constexpr (std::is_same<typename targetType<T, Platform>::matrix_type, BlockBlock>::value) 
        {
            throw std::runtime_error("[BlockBlockMatrix]: no need to redistribute from BlockBlock to BlockBlock");
        }else
        {
            throw std::runtime_error("[BlockBlockMatrix]:  no support for redistribution from redundant to othertypes yet");
        }
    }

private:
    std::size_t M_; ///< Global number of rows in the matrix
    std::size_t N_; ///< Global number of columns in the matrix
    std::size_t m_; ///< Local number of rows in the matrix
    std::size_t n_; ///< Local number of columns in the matrix
    std::size_t ld_; ///< Local leading dimension (stride)
    std::size_t g_offs_[2];

    chase::matrix::Matrix<T, Platform> local_matrix_;  ///< Local matrix holding the data
    std::shared_ptr<chase::grid::MpiGrid2DBase> mpi_grid_; ///< Shared pointer to the MPI grid  

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

template <typename T, typename Platform = chase::platform::CPU>
class QuasiHermitianBlockBlockMatrix;

template <typename T>
class QuasiHermitianBlockBlockMatrix<T, chase::platform::CPU> : public BlockBlockMatrix<T, chase::platform::CPU> {
public:
    // Default constructor
    QuasiHermitianBlockBlockMatrix() : BlockBlockMatrix<T, chase::platform::CPU>() {}
    
    // Constructor with dimensions
    QuasiHermitianBlockBlockMatrix(std::size_t M, std::size_t N, std::shared_ptr<chase::grid::MpiGrid2DBase> mpi_grid) 
        : BlockBlockMatrix<T, chase::platform::CPU>(M, N, mpi_grid) {}
    
    // Constructor with external data
    QuasiHermitianBlockBlockMatrix(std::size_t m, std::size_t n, std::size_t ld, T* data, std::shared_ptr<chase::grid::MpiGrid2DBase> mpi_grid)
        : BlockBlockMatrix<T, chase::platform::CPU>(m, n, ld, data, mpi_grid) {}
};

#ifdef HAS_CUDA
template <typename T>
class QuasiHermitianBlockBlockMatrix<T, chase::platform::GPU> : public BlockBlockMatrix<T, chase::platform::GPU> {
public:
    // Default constructor
    QuasiHermitianBlockBlockMatrix() : BlockBlockMatrix<T, chase::platform::GPU>() {}
    
    // Constructor with dimensions
    QuasiHermitianBlockBlockMatrix(std::size_t M, std::size_t N, std::shared_ptr<chase::grid::MpiGrid2DBase> mpi_grid) 
        : BlockBlockMatrix<T, chase::platform::GPU>(M, N, mpi_grid) {}
    
    // Constructor with external data
    QuasiHermitianBlockBlockMatrix(std::size_t m, std::size_t n, std::size_t ld, T* data, std::shared_ptr<chase::grid::MpiGrid2DBase> mpi_grid)
        : BlockBlockMatrix<T, chase::platform::GPU>(m, n, ld, data, mpi_grid) {}

};
#endif


/**
 * @brief A class representing a block-cyclic distributed matrix.
 *
 * This class inherits from `AbstractDistMatrix` and provides functionality for a block-cyclic
 * distributed matrix, which is partitioned into blocks that are distributed across multiple MPI processes.
 * The matrix is represented locally on each process, and global matrix properties are derived from the
 * local data and distribution.
 *
 * @tparam T The data type of the matrix elements (e.g., `float`, `double`).
 * @tparam Platform The platform type (e.g., CPU, GPU) specifying the hardware on which the matrix is stored.
 */
template<typename T, typename Platform = chase::platform::CPU> 
class BlockCyclicMatrix : public AbstractDistMatrix<T, BlockCyclicMatrix, Platform>
{
public:
    using platform_type = Platform; ///< Alias for platform type.
    using value_type = T;  ///< Alias for element type.
    using matrix_type = BlockCyclic; ///< Alias for matrix type.
    
    /**
     * @brief Destructor for `BlockCyclicMatrix`.
     */    
    ~BlockCyclicMatrix() override {};
    /**
     * @brief Default constructor for `BlockCyclicMatrix`.
     */    
    BlockCyclicMatrix();
    /**
     * @brief Constructor for `BlockCyclicMatrix` given global matrix dimensions, block sizes, and MPI grid.
     *
     * This constructor initializes the matrix with the given global dimensions, block sizes, and MPI grid
     * for the 2D block-cyclic distribution.
     *
     * @param M The global number of rows in the matrix.
     * @param N The global number of columns in the matrix.
     * @param mb The block size for rows.
     * @param nb The block size for columns.
     * @param mpi_grid A shared pointer to the `MpiGrid2DBase` object representing the MPI grid for the distribution.
     */    
    BlockCyclicMatrix(std::size_t M, std::size_t N, std::size_t mb, std::size_t nb,
                    std::shared_ptr<chase::grid::MpiGrid2DBase> mpi_grid)
                    :M_(M), N_(N), mpi_grid_(mpi_grid), mb_(mb), nb_(nb)
    {
        int *dims_ = mpi_grid_.get()->get_dims();
        int *coord_ = mpi_grid_.get()->get_coords();
        std::tie(m_, mblocks_) = numroc(M_, mb_, coord_[0], dims_[0]);
        std::tie(n_, nblocks_) = numroc(N_, nb_, coord_[1], dims_[1]);   
        ld_ = m_;
        local_matrix_ = chase::matrix::Matrix<T, Platform>(m_, n_);      

        init_contiguous_buffer_info();

    }
    /**
     * @brief Constructor for `BlockCyclicMatrix` with pre-allocated data.
     *
     * This constructor initializes the matrix with pre-allocated data and validates that the local matrix
     * dimensions match the block-cyclic distribution.
     *
     * @param M The global number of rows in the matrix.
     * @param N The global number of columns in the matrix.
     * @param m The local number of rows on this process.
     * @param n The local number of columns on this process.
     * @param mb The block size for rows.
     * @param nb The block size for columns.
     * @param ld The leading dimension of the local matrix.
     * @param data Pointer to the pre-allocated data for the local matrix.
     * @param mpi_grid A shared pointer to the `MpiGrid2DBase` object representing the MPI grid for the distribution.
     */
    BlockCyclicMatrix(std::size_t M, std::size_t N, 
                      std::size_t m, std::size_t n,
                      std::size_t mb, std::size_t nb,
                      std::size_t ld, T *data,
                      std::shared_ptr<chase::grid::MpiGrid2DBase> mpi_grid)
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

        local_matrix_ = chase::matrix::Matrix<T, Platform>(m_, n_, ld_, data);        

        if constexpr (std::is_same<Platform, chase::platform::GPU>::value)
        {
            ld_ = local_matrix_.ld();
        }
        
        init_contiguous_buffer_info();
           
    }
    /** 
     * @brief Returns the global number of rows in the matrix.
     * 
     * @return The global number of rows.
     */
    std::size_t g_rows() const override { return M_; }
    /** 
     * @brief Returns the global number of columns in the matrix.
     * 
     * @return The global number of columns.
     */    
    std::size_t g_cols() const override { return N_; }
    /** 
     * @brief Returns the local leading dimension.
     * 
     * @return The local leading dimension.
     */    
    std::size_t l_ld() const override { return ld_; }
    /** 
     * @brief Returns the local number of rows.
     * 
     * @return The local number of rows.
     */    
    std::size_t l_rows() const override { return m_; }
    /** 
     * @brief Returns the local number of columns.
     * 
     * @return The local number of columns.
     */    
    std::size_t l_cols() const override { return n_;}
    /** 
     * @brief Returns the block size for rows.
     * 
     * @return The block size for rows.
     */    
    std::size_t mb() const override { return mb_;}
    /** 
     * @brief Returns the block size for columns.
     * 
     * @return The block size for columns.
     */    
    std::size_t nb() const override { return nb_;}
     /** 
     * @brief Returns the number of row blocks.
     * 
     * @return The number of row blocks.
     */   
    std::size_t mblocks() {return mblocks_; }
    /** 
     * @brief Returns the number of column blocks.
     * 
     * @return The number of column blocks.
     */    
    std::size_t nblocks() {return nblocks_; }
    std::vector<std::size_t> m_contiguous_global_offs() { return m_contiguous_global_offs_; }
    std::vector<std::size_t> n_contiguous_global_offs() { return n_contiguous_global_offs_; }
    std::vector<std::size_t> m_contiguous_local_offs() { return m_contiguous_local_offs_; }
    std::vector<std::size_t> n_contiguous_local_offs() { return n_contiguous_local_offs_; }
    std::vector<std::size_t> m_contiguous_lens() { return m_contiguous_lens_; }
    std::vector<std::size_t> n_contiguous_lens() {return n_contiguous_lens_; }
    
    /** 
     * @brief Returns a pointer to the local matrix data.
     * 
     * @return A pointer to the local matrix data.
     */
    T *         l_data() override { 
        return local_matrix_.data();
    }
    /** 
     * @brief Returns the local matrix object.
     * 
     * @return A reference to the local matrix.
     */    
    chase::matrix::Matrix<T, Platform>& loc_matrix() override { return local_matrix_;}

    // Accessors for MPI grid
    /** 
     * @brief Returns a pointer to the MPI grid.
     * 
     * @return A pointer to the MPI grid.
     */    
    chase::grid::MpiGrid2DBase* getMpiGrid() const override {
        return mpi_grid_.get();
    }
    /** 
     * @brief Returns a shared pointer to the MPI grid.
     * 
     * @return A shared pointer to the MPI grid.
     */
    std::shared_ptr<chase::grid::MpiGrid2DBase> getMpiGrid_shared_ptr() const override
    {
        return mpi_grid_;
    }
    /** 
     * @brief Clones a multi-vector for the given global dimensions.
     * 
     * @tparam CloneVectorType The type of the vector to clone.
     * @param g_M The global number of rows for the clone.
     * @param g_N The global number of columns for the clone.
     * @return A new multi-vector of the given type.
     */
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
    /** 
     * @brief Saves the matrix data to a binary file.
     *
     * This method saves the matrix data from the CPU buffer to a binary file using MPI I/O.
     * For GPU data, it must first be copied to the CPU.
     *
     * @param filename The name of the file to save the matrix data.
     * @throws std::runtime_error If the data cannot be saved.
     */    
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
    /** 
     * @brief Reads matrix data from a binary file.
     *
     * This method reads the matrix data from a binary file using MPI I/O.
     * For GPU data, it will copy the data to the CPU buffer first.
     *
     * @param filename The name of the file to read the matrix data.
     * @throws std::runtime_error If the data cannot be read.
     */    
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
    /** 
     * @brief Returns the Scalapack descriptor for the matrix.
     * 
     * @return A pointer to the Scalapack descriptor.
     */
    std::size_t *get_scalapack_desc(){ return desc_; }
#endif

#ifdef HAS_SCALAPACK
    /**
     * @brief Initialize the ScaLAPACK descriptor for the matrix.
     *
     * Initializes the ScaLAPACK descriptor based on the matrix dimensions and MPI grid configuration.
     *
     * @return Pointer to the initialized ScaLAPACK descriptor.
     */
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
    /**
     * @brief Redistribute the matrix to another type of matrix.
     *
     * Redistributes the matrix data to another matrix of a different type (e.g., from BlockCyclicMatrix
     * to another matrix type).
     *
     * @tparam targetType Type of the target matrix to redistribute data to.
     * @param targetMatrix Pointer to the target matrix.
     */
    template<template <typename, typename> class targetType>
    void redistributeImpl(targetType<T, Platform>* targetMatrix)//,
                            //std::size_t offset, std::size_t subsetSize)
    {
        if(M_ != targetMatrix->g_rows() || N_ != targetMatrix->g_cols() )
        {
            throw std::runtime_error("[BlockCyclicMatrix]: redistribution requires original and target matrices have same global size");
        }

        if constexpr (std::is_same<typename targetType<T, Platform>::matrix_type, Redundant>::value) 
        {
            redistributeToRedundant(targetMatrix);
        }
        else if constexpr (std::is_same<typename targetType<T, Platform>::matrix_type, BlockCyclic>::value) 
        {
            throw std::runtime_error("[BlockCyclicMatrix]: no need to redistribute from BlockCyclic to BlockCyclic");
        }else
        {
            throw std::runtime_error("[BlockCyclicMatrix]:  no support for redistribution from redundant to othertypes yet");
        }        
    }

#ifdef HAS_NCCL
    template<template <typename, typename> class targetType>
    void redistributeImplAsync(targetType<T, Platform>* targetMatrix, cudaStream_t* stream_ = nullptr)//,
                            //std::size_t offset, std::size_t subsetSize)
    {
        cudaStream_t usedStream = (stream_ == nullptr) ? 0 : *stream_;

        if(M_ != targetMatrix->g_rows() || N_ != targetMatrix->g_cols() )
        {
            throw std::runtime_error("[BlockCyclicMatrix]: redistribution requires original and target matrices have same global size");
        }

        if constexpr (std::is_same<typename targetType<T, Platform>::matrix_type, Redundant>::value) 
        {
            redistributeToRedundantAsync(targetMatrix, usedStream);
        }
        else if constexpr (std::is_same<typename targetType<T, Platform>::matrix_type, BlockCyclic>::value) 
        {
            throw std::runtime_error("[BlockCyclicMatrix]: no need to redistribute from BlockCyclic to BlockCyclic");
        }else
        {
            throw std::runtime_error("[BlockCyclicMatrix]:  no support for redistribution from redundant to othertypes yet");
        }        
    }
#endif
private:
    std::size_t M_; ///< Global number of rows in the matrix
    std::size_t N_; ///< Global number of columns in the matrix
    std::size_t m_; ///< Local number of rows in the matrix
    std::size_t n_; ///< Local number of columns in the matrix
    std::size_t ld_; ///< Local leading dimension (stride)
    std::size_t mb_; ///< Block size for the row partitioning in the block-cyclic distribution
    std::size_t nb_; ///< Block size for the column partitioning in the block-cyclic distribution
    std::size_t mblocks_;
    std::size_t nblocks_;    
    std::vector<std::size_t> m_contiguous_global_offs_;
    std::vector<std::size_t> n_contiguous_global_offs_;
    std::vector<std::size_t> m_contiguous_local_offs_;
    std::vector<std::size_t> n_contiguous_local_offs_;
    std::vector<std::size_t> m_contiguous_lens_;
    std::vector<std::size_t> n_contiguous_lens_;

    chase::matrix::Matrix<T, Platform> local_matrix_;  ///< Local matrix holding the data
    std::shared_ptr<chase::grid::MpiGrid2DBase> mpi_grid_; ///< Shared pointer to the MPI grid  

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
    {
        int *coords =  mpi_grid_.get()->get_coords();
        int *dims = mpi_grid_.get()->get_dims();  

        std::vector<std::size_t> m_locs;
        std::vector<std::size_t> n_locs;

        for(auto i = 0; i < dims[0]; i++)
        {
            std::size_t l;
            std::tie(l, std::ignore) = numroc(M_, mb_, i, dims[0]);
            m_locs.push_back(l);
        }

        for(auto i = 0; i < dims[1]; i++)
        {
            std::size_t l;
            std::tie(l, std::ignore) = numroc(N_, nb_, i, dims[1]);
            n_locs.push_back(l);
        }

        std::vector<std::vector<std::vector<std::size_t>>> m_contiguous_global_offs_2;
        std::vector<std::vector<std::vector<std::size_t>>> n_contiguous_global_offs_2;
        std::vector<std::vector<std::vector<std::size_t>>> m_contiguous_lens_2;
        std::vector<std::vector<std::vector<std::size_t>>> n_contiguous_lens_2;
      
        m_contiguous_global_offs_2.resize(dims[0]);
        n_contiguous_global_offs_2.resize(dims[0]);
        m_contiguous_lens_2.resize(dims[0]);
        n_contiguous_lens_2.resize(dims[0]);
        for(auto i = 0; i < dims[0]; i++)
        {
            m_contiguous_global_offs_2[i].resize(dims[1]);
            n_contiguous_global_offs_2[i].resize(dims[1]);
            m_contiguous_lens_2[i].resize(dims[1]);
            n_contiguous_lens_2[i].resize(dims[1]);
        }

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
            for(auto i = 0; i < dims[1]; i++)
            {
                m_contiguous_global_offs_2[sendr][i].push_back(r);
                m_contiguous_lens_2[sendr][i].push_back(nr);
            }
        }

        for (std::size_t c = 0; c < N_; c += nb_, sendc = (sendc + 1) % dims[1])
        {
            nc = nb_;
            if(N_ - c < nb_)
            {
                nc = N_ - c;
            }

            for(auto i = 0; i < dims[0]; i++)
            {
                n_contiguous_global_offs_2[i][sendc].push_back(c);
                n_contiguous_lens_2[i][sendc].push_back(nc);
            }
        }

        std::size_t max_lcols = *std::max_element(n_locs.begin(), n_locs.end());
        std::size_t max_lrows = *std::max_element(m_locs.begin(), m_locs.end());

        std::unique_ptr<chase::distMatrix::RedundantMatrix<T, Platform>> buff = std::make_unique<chase::distMatrix::RedundantMatrix<T, Platform>>(M_, max_lcols, mpi_grid_);

        T *buff_data = buff->l_data();

        //first bcast within column
        for(auto src = 0; src < dims[0]; src++)
        {
            if(coords[0] == src){
                if constexpr (std::is_same<Platform, chase::platform::CPU>::value){
                    chase::linalg::lapackpp::t_lacpy('A', this->l_rows(), this->l_cols(), this->l_data(), this->l_ld(), buff_data, m_locs[coords[0]]);
                }
#ifdef HAS_CUDA
                else
                {
                    chase::linalg::internal::cuda::t_lacpy('A', this->l_rows(), this->l_cols(), this->l_data(), this->l_ld(), buff_data, m_locs[coords[0]]);
                }
#endif
            }
#ifdef HAS_CUDA
            cudaDeviceSynchronize();
#endif
            MPI_Bcast(buff_data, n_locs[coords[1]] *  m_locs[src], chase::mpi::getMPI_Type<T>(), src, mpi_grid_.get()->get_col_comm());

            for(auto p = 0; p < m_contiguous_global_offs_2[src][coords[1]].size(); p++)
            {
                for(auto q = 0; q < n_contiguous_global_offs_2[src][coords[1]].size(); q++)
                {
                    if constexpr (std::is_same<Platform, chase::platform::CPU>::value){
                        chase::linalg::lapackpp::t_lacpy('A', 
                                            m_contiguous_lens_2[src][coords[1]][p], 
                                            n_contiguous_lens_2[src][coords[1]][q], 
                                            buff_data + this->mb() * p + this->nb() * q *  m_locs[src], 
                                            m_locs[src], 
                                            targetMatrix->l_data() + m_contiguous_global_offs_2[src][coords[1]][p] + targetMatrix->l_ld() * n_contiguous_global_offs_2[src][coords[1]][q], 
                                            targetMatrix->l_ld());
                    }
#ifdef HAS_CUDA
                    else
                    {
                        chase::linalg::internal::cuda::t_lacpy('A', 
                                            m_contiguous_lens_2[src][coords[1]][p], 
                                            n_contiguous_lens_2[src][coords[1]][q], 
                                            buff_data + this->mb() * p + this->nb() * q *  m_locs[src], 
                                            m_locs[src], 
                                            targetMatrix->l_data() + m_contiguous_global_offs_2[src][coords[1]][p] + targetMatrix->l_ld() * n_contiguous_global_offs_2[src][coords[1]][q], 
                                            targetMatrix->l_ld());                        
                    }
#endif

                                        
                }   
            }        
        }
        //bcast within row
        for(auto src = 0; src < dims[1]; src++)
        {
            for(auto q = 0; q < n_contiguous_global_offs_2[coords[0]][src].size(); q++)
            {
                if(coords[1] == src)
                {
                    if constexpr (std::is_same<Platform, chase::platform::CPU>::value){
                        chase::linalg::lapackpp::t_lacpy('A', M_, n_contiguous_lens_2[coords[0]][src][q], targetMatrix->l_data() + n_contiguous_global_offs_2[coords[0]][src][q] * targetMatrix->l_ld(), targetMatrix->l_ld(), buff_data, M_);
                    }
#ifdef HAS_CUDA
                    else
                    {
                        chase::linalg::internal::cuda::t_lacpy('A', M_, n_contiguous_lens_2[coords[0]][src][q], targetMatrix->l_data() + n_contiguous_global_offs_2[coords[0]][src][q] * targetMatrix->l_ld(), targetMatrix->l_ld(), buff_data, M_);
                    }
#endif
                }
#ifdef HAS_CUDA
                cudaDeviceSynchronize();
#endif
                MPI_Bcast(buff_data, M_ * n_contiguous_lens_2[coords[0]][src][q], chase::mpi::getMPI_Type<T>(), src, mpi_grid_.get()->get_row_comm());
   
                if constexpr (std::is_same<Platform, chase::platform::CPU>::value){
                    chase::linalg::lapackpp::t_lacpy('A', 
                                        M_, 
                                        n_contiguous_lens_2[coords[0]][src][q], 
                                        buff_data, 
                                        M_, 
                                        targetMatrix->l_data() + targetMatrix->l_ld() * n_contiguous_global_offs_2[coords[0]][src][q], 
                                        targetMatrix->l_ld()); 
                }
#ifdef HAS_CUDA
                else
                {
                    chase::linalg::internal::cuda::t_lacpy('A', 
                                        M_, 
                                        n_contiguous_lens_2[coords[0]][src][q], 
                                        buff_data, 
                                        M_, 
                                        targetMatrix->l_data() + targetMatrix->l_ld() * n_contiguous_global_offs_2[coords[0]][src][q], 
                                        targetMatrix->l_ld());                     
                }
#endif                                        
            }
        }
    }

#ifdef HAS_NCCL
    void redistributeToRedundantAsync(RedundantMatrix<T, chase::platform::GPU>* targetMatrix, cudaStream_t stream)
    {
        int *coords =  mpi_grid_.get()->get_coords();
        int *dims = mpi_grid_.get()->get_dims();  

        std::vector<std::size_t> m_locs;
        std::vector<std::size_t> n_locs;

        for(auto i = 0; i < dims[0]; i++)
        {
            std::size_t l;
            std::tie(l, std::ignore) = numroc(M_, mb_, i, dims[0]);
            m_locs.push_back(l);
        }

        for(auto i = 0; i < dims[1]; i++)
        {
            std::size_t l;
            std::tie(l, std::ignore) = numroc(N_, nb_, i, dims[1]);
            n_locs.push_back(l);
        }
        
        std::vector<std::vector<std::vector<std::size_t>>> m_contiguous_global_offs_2;
        std::vector<std::vector<std::vector<std::size_t>>> n_contiguous_global_offs_2;
        std::vector<std::vector<std::vector<std::size_t>>> m_contiguous_lens_2;
        std::vector<std::vector<std::vector<std::size_t>>> n_contiguous_lens_2;
      
        m_contiguous_global_offs_2.resize(dims[0]);
        n_contiguous_global_offs_2.resize(dims[0]);
        m_contiguous_lens_2.resize(dims[0]);
        n_contiguous_lens_2.resize(dims[0]);
        for(auto i = 0; i < dims[0]; i++)
        {
            m_contiguous_global_offs_2[i].resize(dims[1]);
            n_contiguous_global_offs_2[i].resize(dims[1]);
            m_contiguous_lens_2[i].resize(dims[1]);
            n_contiguous_lens_2[i].resize(dims[1]);
        }

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
            for(auto i = 0; i < dims[1]; i++)
            {
                m_contiguous_global_offs_2[sendr][i].push_back(r);
                m_contiguous_lens_2[sendr][i].push_back(nr);
            }
        }

        for (std::size_t c = 0; c < N_; c += nb_, sendc = (sendc + 1) % dims[1])
        {
            nc = nb_;
            if(N_ - c < nb_)
            {
                nc = N_ - c;
            }

            for(auto i = 0; i < dims[0]; i++)
            {
                n_contiguous_global_offs_2[i][sendc].push_back(c);
                n_contiguous_lens_2[i][sendc].push_back(nc);
            }
        }

        std::size_t max_lcols = *std::max_element(n_locs.begin(), n_locs.end());
        std::size_t max_lrows = *std::max_element(m_locs.begin(), m_locs.end());

        std::unique_ptr<chase::distMatrix::RedundantMatrix<T, chase::platform::GPU>> buff = std::make_unique<chase::distMatrix::RedundantMatrix<T, chase::platform::GPU>>(M_, max_lcols, mpi_grid_);

        T *buff_data = buff->l_data();

        //first bcast within column
        for(auto src = 0; src < dims[0]; src++)
        {
            if(coords[0] == src){
               chase::linalg::internal::cuda::t_lacpy('A', this->l_rows(), this->l_cols(), this->l_data(), this->l_ld(), buff_data, m_locs[coords[0]]);
            }

            CHECK_NCCL_ERROR(chase::nccl::ncclBcastWrapper(buff_data, 
                                                        n_locs[coords[1]] *  m_locs[src], 
                                                        src, 
                                                        this->mpi_grid_->get_nccl_col_comm(), 
                                                        &stream));

            for(auto p = 0; p < m_contiguous_global_offs_2[src][coords[1]].size(); p++)
            {
                for(auto q = 0; q < n_contiguous_global_offs_2[src][coords[1]].size(); q++)
                {
                    chase::linalg::internal::cuda::t_lacpy('A', 
                                        m_contiguous_lens_2[src][coords[1]][p], 
                                        n_contiguous_lens_2[src][coords[1]][q], 
                                        buff_data + this->mb() * p + this->nb() * q *  m_locs[src], 
                                        m_locs[src], 
                                        targetMatrix->l_data() + m_contiguous_global_offs_2[src][coords[1]][p] + targetMatrix->l_ld() * n_contiguous_global_offs_2[src][coords[1]][q], 
                                        targetMatrix->l_ld());
                                        
                }   
            }        
        }
        //bcast within row
        for(auto src = 0; src < dims[1]; src++)
        {
            for(auto q = 0; q < n_contiguous_global_offs_2[coords[0]][src].size(); q++)
            {
                if(coords[1] == src)
                {
                    chase::linalg::internal::cuda::t_lacpy('A', M_, n_contiguous_lens_2[coords[0]][src][q], targetMatrix->l_data() + n_contiguous_global_offs_2[coords[0]][src][q] * targetMatrix->l_ld(), targetMatrix->l_ld(), buff_data, M_);
                }

                CHECK_NCCL_ERROR(chase::nccl::ncclBcastWrapper(buff_data, 
                                                               M_ * n_contiguous_lens_2[coords[0]][src][q], 
                                                               src, 
                                                               this->mpi_grid_->get_nccl_row_comm(), 
                                                               &stream));
   
                chase::linalg::internal::cuda::t_lacpy('A', 
                                    M_, 
                                    n_contiguous_lens_2[coords[0]][src][q], 
                                    buff_data, 
                                    M_, 
                                    targetMatrix->l_data() + targetMatrix->l_ld() * n_contiguous_global_offs_2[coords[0]][src][q], 
                                    targetMatrix->l_ld());  
                                        
            }
        }
        
    }
#endif

#ifdef HAS_SCALAPACK
    std::size_t desc_[9];
#endif
};

/** @} */  // End of dist_matrix_classes group

}
}

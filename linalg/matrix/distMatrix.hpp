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
#include "Impl/mpi/mpiGrid2D.hpp"
#include "Impl/mpi/mpiTypes.hpp"

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

template <typename T>
struct MatrixTypeTrait<RedundantMatrix<T> *> {
    static constexpr MatrixType value = MatrixType::Redundant;
};

// Specialize for BlockBlockMatrix
template <typename T>
struct MatrixTypeTrait<BlockBlockMatrix<T> *> {
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
    std::chrono::high_resolution_clock::time_point start, end;
    std::chrono::duration<double> elapsed;
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
    int grank()
    {
        int grank = 0;
        MPI_Comm_rank(this->getMpiGrid()->get_comm(), &grank);
        return grank;
    }
#ifdef ENABLE_MIXED_PRECISION       
    // Enable single precision for double types (and complex<double>)
    template <typename U = T, typename std::enable_if<std::is_same<U, double>::value || std::is_same<U, std::complex<double>>::value, int>::type = 0>
    void enableSinglePrecision() {
        if (!single_precision_matrix_) {
            start = std::chrono::high_resolution_clock::now();
            single_precision_matrix_ = std::make_unique<SinglePrecisionDerived>(this->g_rows(), this->g_cols(), this->getMpiGrid_shared_ptr());
            #pragma omp parallel for
            for (std::size_t j = 0; j < this->l_cols(); ++j) {
                for (std::size_t i = 0; i < this->l_rows(); ++i) {
                    single_precision_matrix_->l_data()[j * single_precision_matrix_.get()->l_ld() + i] 
                                        = chase::convertToSinglePrecision(this->l_data()[j * this->l_ld() + i]);
                }
            }
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
                #pragma omp parallel for
                for (std::size_t j = 0; j < this->l_cols(); ++j) {
                    for (std::size_t i = 0; i < this->l_rows(); ++i) {
                        this->l_data()[j * this->l_ld() + i] = 
                                chase::convertToDoublePrecision<T>(single_precision_matrix_->l_data()[j * single_precision_matrix_.get()->l_ld() + i]);
                    }
                }
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

    void mapToNewMpiGrid(std::shared_ptr<chase::Impl::mpi::MpiGrid2DBase> new_mpi_grid)
    {
        mpi_grid_ = new_mpi_grid;
    }

    //here the startrow/col indices should be the global indices
    template<MatrixType TargetType>
    void redistributeImpl(typename MatrixConstructorTrait<TargetType, T>::type* targetMatrix,
                            std::size_t startRow, std::size_t subRows, std::size_t startCol, std::size_t subCols)
    {
        if(M_ != targetMatrix->g_rows() || N_ != targetMatrix->g_cols() )
        {
            throw std::runtime_error("[RedundantMatrix]: redistribution requires original and target matrices have same global size");
        }

        if constexpr (TargetType == MatrixType::BlockBlock)
        {
            redistributeToBlockBlock(targetMatrix, startRow, subRows, startCol, subCols);
        }else if constexpr (TargetType == MatrixType::Redundant)
        {
            throw std::runtime_error("[RedundantMatrix]: no need to redistribute from redundant to redundant");
        }else if constexpr (TargetType == MatrixType::BlockCyclic)
        {
            throw std::runtime_error("[RedundantMatrix]: no support for redistribution from redundant to block-cyclic yet");
        }
    }

    template<MatrixType TargetType>
    void redistributeImpl(typename MatrixConstructorTrait<TargetType, T>::type* targetMatrix)
    {
        using type = typename std::remove_pointer<typename std::remove_reference<decltype(*targetMatrix)>::type>::type;
        this->template redistributeImpl<chase::distMatrix::MatrixTypeTrait<type>::value>(targetMatrix, 0, this->g_rows(), 0, this->g_cols());
    }   

private:
    std::size_t M_;
    std::size_t N_;
    std::size_t m_;
    std::size_t n_;
    std::size_t ld_;

    chase::matrix::MatrixCPU<T> local_matrix_;
    std::shared_ptr<chase::Impl::mpi::MpiGrid2DBase> mpi_grid_;    

    void redistributeToBlockBlock(BlockBlockMatrix<T>* targetMatrix,
                                   std::size_t startRow, std::size_t subRows, std::size_t startCol, std::size_t subCols)
    {
        std::size_t *g_offs = targetMatrix->g_offs();
        std::size_t l_cols = targetMatrix->l_cols();
        std::size_t l_rows = targetMatrix->l_rows();
        #pragma omp parallel for
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

};


template<typename T> 
class BlockBlockMatrix : public AbstractDistMatrix<T, BlockBlockMatrix>
{

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
#ifdef HAS_SCALAPACK        
        scalapack_descriptor_init();
#endif        
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
#ifdef HAS_SCALAPACK               
        scalapack_descriptor_init();
#endif       
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

    void saveToBinaryFile(const std::string& filename) {
    	MPI_File fileHandle;
        MPI_Status status;

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
        MPI_File_write_all(fileHandle, this->l_data(), count_write, chase::mpi::getMPI_Type<T>(), &status);

        MPI_Type_free(&subarray);

    	if (MPI_File_close(&fileHandle) != MPI_SUCCESS)
    	{
            MPI_Abort(this->mpi_grid_.get()->get_comm(), EXIT_FAILURE);
        }
    }

    // Read matrix data from a binary file
    void readFromBinaryFile(const std::string& filename) {
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
        MPI_File_read_all(fileHandle, this->l_data(), count_read, chase::mpi::getMPI_Type<T>(), &status);

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
            input.read(reinterpret_cast<char*>(this->l_data() + ld_ * y), m_ * sizeof(T));
        }
        
        input.close();
#endif
    }

#ifdef HAS_SCALAPACK
    std::size_t *get_scalapack_desc(){ return desc_; }
#endif

    template<MatrixType TargetType>
    void redistributeImpl(typename MatrixConstructorTrait<TargetType, T>::type* targetMatrix)//,
                            //std::size_t offset, std::size_t subsetSize)
    {
        if(M_ != targetMatrix->g_rows() || N_ != targetMatrix->g_cols() )
        {
            throw std::runtime_error("[BlockBlockMatrix]: redistribution requires original and target matrices have same global size");
        }

        if constexpr (TargetType == MatrixType::Redundant)
        {
            redistributeToRedundant(targetMatrix);
        }else if constexpr (TargetType == MatrixType::BlockBlock)
        {
            throw std::runtime_error("[BlockBlockMatrix]: no need to redistribute from BlockBlock to BlockBlock");
        }else if constexpr (TargetType == MatrixType::BlockCyclic)
        {
            throw std::runtime_error("[BlockBlockMatrix]: no support for redistribution from redundant to block-cyclic yet");
        }
    }

private:
    std::size_t M_;
    std::size_t N_;
    std::size_t m_;
    std::size_t n_;
    std::size_t ld_;
    std::size_t g_offs_[2];

    chase::matrix::MatrixCPU<T> local_matrix_;
    std::shared_ptr<chase::Impl::mpi::MpiGrid2DBase> mpi_grid_;

    void redistributeToRedundant(RedundantMatrix<T>* targetMatrix)
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

#ifdef HAS_SCALAPACK
    std::size_t desc_[9];

    void scalapack_descriptor_init()
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
                                               &ld_, 
                                               &info);  


    }
#endif

};


template<typename T> 
class BlockCyclicMatrix : public AbstractDistMatrix<T, BlockCyclicMatrix>
{
    //impl later
};


}
}
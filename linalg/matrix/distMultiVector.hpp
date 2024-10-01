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
#include "Impl/mpi/mpiTypes.hpp"
#include "linalg/lapackpp/lapackpp.hpp"

namespace chase
{
namespace distMultiVector
{
enum class DistributionType {
    Block,
    BlockCylic
};

enum class CommunicatorType{
    row,
    column,
    all
};


// Type trait to map enum CommunicatorType to its opposite
template <CommunicatorType>
struct OutputCommType;

template <>
struct OutputCommType<CommunicatorType::row> {
    static constexpr CommunicatorType value = CommunicatorType::column;  // Map 'row' to 'column'
};

template <>
struct OutputCommType<CommunicatorType::column> {
    static constexpr CommunicatorType value = CommunicatorType::row;  // Map 'column' to 'row'
};

template<typename T, CommunicatorType comm_type> 
class DistMultiVector1D;

template<typename T>
struct ExtractFirstTemplateParameter;

template<typename T, chase::distMultiVector::CommunicatorType CommType>
struct ExtractFirstTemplateParameter<chase::distMultiVector::DistMultiVector1D<T, CommType>> {
    using type = T;  // Extracts the first template parameter (T)
};

template <typename T, CommunicatorType comm_type, template <typename, CommunicatorType> class Derived>
class AbstractDistMultiVector {
protected:
    // Single precision matrix
#ifdef ENABLE_MIXED_PRECISION       
    using SinglePrecisionType = typename chase::ToSinglePrecisionTrait<T>::Type;
    using SinglePrecisionDerived = Derived<SinglePrecisionType, comm_type>;
    std::unique_ptr<SinglePrecisionDerived> single_precision_multivec_;
    bool is_single_precision_enabled_ = false; 
    std::chrono::high_resolution_clock::time_point start, end;
    std::chrono::duration<double> elapsed;    
#endif    
public:
    virtual ~AbstractDistMultiVector() = default;
    virtual DistributionType getMultiVectorDistributionType() const = 0;
    virtual CommunicatorType getMultiVectorCommunicatorType() const = 0;
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
        if (!single_precision_multivec_) {
            start = std::chrono::high_resolution_clock::now();
            single_precision_multivec_ = std::make_unique<SinglePrecisionDerived>(this->g_rows(), this->g_cols(), this->getMpiGrid_shared_ptr());
            #pragma omp parallel for collapse(2) schedule(static, 16)
            for (std::size_t j = 0; j < this->l_cols(); ++j) {
                for (std::size_t i = 0; i < this->l_rows(); ++i) {
                    single_precision_multivec_->l_data()[j * single_precision_multivec_.get()->l_ld() + i] 
                                        = chase::convertToSinglePrecision(this->l_data()[j * this->l_ld() + i]);
                }
            }
    
            is_single_precision_enabled_ = true;
            end = std::chrono::high_resolution_clock::now();
            elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

            if(this->grank() == 0)
                std::cout << "Single precision matrix enabled in AbstractDistMultiVector in " << elapsed.count() << " s\n";
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
            if (single_precision_multivec_) {
                #pragma omp parallel for collapse(2) schedule(static, 16)
                for (std::size_t j = 0; j < this->l_cols(); ++j) {
                    for (std::size_t i = 0; i < this->l_rows(); ++i) {
                        this->l_data()[j * this->l_ld() + i] = 
                                chase::convertToDoublePrecision<T>(single_precision_multivec_->l_data()[j * single_precision_multivec_.get()->l_ld() + i]);
                    }
                }
            } else {
                throw std::runtime_error("Single precision is not enabled.");
            }
        }

        end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

        if(this->grank() == 0)
            std::cout << "Single precision matrix disabled in AbstractDistMultiVector in " << elapsed.count() << " s\n"; 
               
        single_precision_multivec_.reset();  // Free the single precision memory
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
            return single_precision_multivec_.get();
        } else {
            throw std::runtime_error("Single precision is not enabled.");
        }
    }

    // If T is already single precision, these methods should not be available
    template <typename U = T, typename std::enable_if<!std::is_same<U, double>::value && !std::is_same<U, std::complex<double>>::value, int>::type = 0>
    void enableSinglePrecision() {
        throw std::runtime_error("[DistMultiVector]: Single precision operations not supported for this type.");
    }

    template <typename U = T, typename std::enable_if<!std::is_same<U, double>::value && !std::is_same<U, std::complex<double>>::value, int>::type = 0>
    void disableSinglePrecision() {
        throw std::runtime_error("[DistMultiVector]: Single precision operations not supported for this type.");
    }
#endif    
};

template<typename T, CommunicatorType comm_type> 
class DistMultiVector1D : public AbstractDistMultiVector<T, comm_type, DistMultiVector1D> //distribute either within row or column communicator of 2D MPI grid
{
public:
    ~DistMultiVector1D() override {};
    DistMultiVector1D();    
    DistMultiVector1D(std::size_t M, std::size_t N,
                    std::shared_ptr<chase::Impl::mpi::MpiGrid2DBase> mpi_grid)
                    :M_(M), N_(N), mpi_grid_(mpi_grid)
    {
        int *dims_ = mpi_grid_.get()->get_dims();
        int *coord_ = mpi_grid_.get()->get_coords();
        std::size_t len;
        n_ = N_;
        int dim, coord;

        if constexpr (comm_type == chase::distMultiVector::CommunicatorType::row) //distributed within row communicator
        {
            dim = dims_[1];
            coord = coord_[1];
        } else if constexpr (comm_type == chase::distMultiVector::CommunicatorType::column)
        {
            dim = dims_[0];
            coord = coord_[0];
        }else
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
        
        ld_ = m_;

        local_matrix_ = chase::matrix::MatrixCPU<T>(m_, n_);
#ifdef HAS_SCALAPACK
        scalapack_descriptor_init();
#endif          
    }

    DistMultiVector1D(std::size_t m, std::size_t n, std::size_t ld, T *data,
                    std::shared_ptr<chase::Impl::mpi::MpiGrid2DBase> mpi_grid)
                    :m_(m), n_(n),ld_(ld), mpi_grid_(mpi_grid)
    {
        N_ = n_;
        uint64_t lv = static_cast<uint64_t>(m_);
        uint64_t res = 0; 

        MPI_Comm comm;

        if constexpr (comm_type == chase::distMultiVector::CommunicatorType::row) //distributed within row communicator
        {
            comm = mpi_grid_.get()->get_row_comm();
        }
        else if constexpr (comm_type == chase::distMultiVector::CommunicatorType::column)
        {
            comm = mpi_grid_.get()->get_col_comm();
        }else
        {
            std::runtime_error("no CommunicatorType supported");
        }

        MPI_Allreduce(&lv, &res, 1, MPI_UINT64_T, MPI_SUM, comm);
        M_ = static_cast<std::size_t>(res);

        local_matrix_ = chase::matrix::MatrixCPU<T>(m_, n_, ld_, data);
#ifdef HAS_SCALAPACK
        scalapack_descriptor_init();
#endif        
    }    

    DistributionType getMultiVectorDistributionType() const override {
        return DistributionType::Block;
    }
    
    CommunicatorType getMultiVectorCommunicatorType() const override {
        return comm_type;
    }

    // Accessors for MPI grid
    chase::Impl::mpi::MpiGrid2DBase* getMpiGrid() const override {
        return mpi_grid_.get();
    }

    std::shared_ptr<chase::Impl::mpi::MpiGrid2DBase> getMpiGrid_shared_ptr() const override
    {
        return mpi_grid_;
    }

    template <CommunicatorType OtherCommType>
    void swap(DistMultiVector1D<T, OtherCommType>& other) 
    {
        // Check if the communicator types are the same
        if constexpr (comm_type != OtherCommType) {
            throw std::runtime_error("Cannot swap: Communicator types do not match.");
        }

        // Ensure both objects have the same MPI grid
        if (mpi_grid_.get() != other.mpi_grid_.get()) {
            throw std::runtime_error("Cannot swap: MPI grids do not match.");
        }

        std::swap(M_, other.M_);
        std::swap(N_, other.N_);
        std::swap(m_, other.m_);
        std::swap(n_, other.n_);
        std::swap(ld_, other.ld_);
        local_matrix_.swap(other.local_matrix_);
#ifdef ENABLE_MIXED_PRECISION
        std::swap(this->is_single_precision_enabled_, other.is_single_precision_enabled_);
        std::swap(this->single_precision_multivec_, other.single_precision_multivec_);
#endif
    }
    //swap column i with j
    void swap_ij(std::size_t i, std::size_t j)
    {
        std::vector<T> tmp(m_);
        chase::linalg::lapackpp::t_lacpy('A',
                                         m_,
                                         1,
                                         this->l_data() + i * ld_,
                                         1,
                                         tmp.data(),
                                         1);
        chase::linalg::lapackpp::t_lacpy('A',
                                         m_,
                                         1,
                                         this->l_data() + j * ld_,
                                         1,
                                         this->l_data() + i * ld_,
                                         1);    
        chase::linalg::lapackpp::t_lacpy('A',
                                         m_,
                                         1,
                                         tmp.data(),
                                         1,
                                         this->l_data() + j * ld_,
                                         1);      
    }

    template<CommunicatorType target_comm_type>
    void redistributeImpl(DistMultiVector1D<T, target_comm_type>* targetMultiVector,
                            std::size_t offset, std::size_t subsetSize) {
        // Validate the subset range
        if (offset + subsetSize > this->g_cols() || subsetSize > targetMultiVector->g_cols()) {
            throw std::invalid_argument("Invalid subset range");
        }   
        // Check if the target matrix's communicator type matches the allowed types
        if constexpr (comm_type == CommunicatorType::row && target_comm_type == CommunicatorType::column) {
            // Implement redistribution from row to column
            redistributeRowToColumn(targetMultiVector, offset, subsetSize);
        } else if constexpr (comm_type == CommunicatorType::column && target_comm_type == CommunicatorType::row) {
            // Implement redistribution from column to row
            redistributeColumnToRow(targetMultiVector, offset, subsetSize);
        } else {
            throw std::runtime_error("Invalid redistribution between matrix types");
        }
        
    }

    template<CommunicatorType target_comm_type>
    void redistributeImpl(DistMultiVector1D<T, target_comm_type>* targetMultiVector) 
    {
        this->redistributeImpl(targetMultiVector, 0, this->n_);
    }

    std::size_t g_rows() const override { return M_;}
    std::size_t g_cols() const override { return N_;}
    std::size_t l_rows() const override { return m_;}
    std::size_t l_cols() const override { return n_;}
    std::size_t l_ld() const override { return ld_;}
    T *         l_data() override { return local_matrix_.data();}
    chase::matrix::MatrixCPU<T> loc_matrix() { return local_matrix_;}
#ifdef HAS_SCALAPACK
    std::size_t *get_scalapack_desc(){ return desc_; }
#endif    

private:
    std::size_t M_;
    std::size_t N_;
    std::size_t m_;
    std::size_t n_;
    std::size_t ld_;
    chase::matrix::MatrixCPU<T> local_matrix_;
    std::shared_ptr<chase::Impl::mpi::MpiGrid2DBase> mpi_grid_;
#ifdef HAS_SCALAPACK
    std::size_t desc_[9];
    
    void scalapack_descriptor_init()
    {
        if constexpr (comm_type ==  CommunicatorType::column)
        {
            std::size_t mb = m_;
            int *coords = mpi_grid_.get()->get_coords();
            int *dims = mpi_grid_.get()->get_dims();

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
            chase::linalg::scalapackpp::t_descinit(desc_, 
                                                  &M_, 
                                                  &N_, 
                                                  &mb, 
                                                  &nb, 
                                                  &zero, 
                                                  &zero,
                                                  &colcomm1D_ctxt, 
                                                  &ld_, 
                                                  &info);


        }else
        {
            //row based will be implemented later
        }
    }

#endif
    void redistributeRowToColumn(DistMultiVector1D<T, CommunicatorType::column>* targetMultiVector,
                                    std::size_t offset, std::size_t subsetSize) {
        // Ensure the dimensions are compatible
        if (this->M_ != targetMultiVector->g_rows() || this->N_ != targetMultiVector->g_cols()) {
            throw std::runtime_error("Dimension mismatch during redistribution");
        }
        
        if(this->mpi_grid_.get() != targetMultiVector->getMpiGrid())
        {
            throw std::runtime_error("MPI Grid mismatch during redistribution");
        }
        
        int* dims = this->mpi_grid_->get_dims();
        int* coords = this->mpi_grid_->get_coords();
        if(dims[0] == dims[1]) //squared grid
        {
            for(auto i = 0; i < dims[1]; i++)
            {
                if(coords[0] == i)
                {
                    if(coords[1] == i)
                    {
                        MPI_Bcast(this->l_data() + offset * this->ld_, this->m_ * subsetSize, chase::mpi::getMPI_Type<T>(), i, this->mpi_grid_->get_row_comm());
                    }
                    else
                    {
                        MPI_Bcast(targetMultiVector->l_data() + offset * targetMultiVector->l_ld(), targetMultiVector->l_rows() * subsetSize, 
                                        chase::mpi::getMPI_Type<T>(), i, this->mpi_grid_->get_row_comm());
                    }
                }
            }

            for(auto i = 0; i < dims[1]; i++)
            {
                if(coords[0] == coords[1])
                {
                    chase::linalg::lapackpp::t_lacpy('A', this->m_, subsetSize, this->l_data() + offset * this->ld_, this->ld_, 
                                                        targetMultiVector->l_data() + offset * targetMultiVector->l_ld(), targetMultiVector->l_ld());
                }
            }
        }
    }

    void redistributeColumnToRow(DistMultiVector1D<T, CommunicatorType::row>* targetMultiVector,
                                    std::size_t offset, std::size_t subsetSize) {
        // Ensure the dimensions are compatible
        if (this->M_ != targetMultiVector->g_rows() || this->N_ != targetMultiVector->g_cols()) {
            throw std::runtime_error("Dimension mismatch during redistribution");
        }

        if(this->mpi_grid_.get() != targetMultiVector->getMpiGrid())
        {
            throw std::runtime_error("MPI Grid mismatch during redistribution");
        }

        int* dims = this->mpi_grid_->get_dims();
        int* coords = this->mpi_grid_->get_coords();
        if(dims[0] == dims[1]) //squared grid
        {
            for(auto i = 0; i < dims[0]; i++)
            {
                if(coords[1] == i)
                {
                    if(coords[0] == i)
                    {
                        MPI_Bcast(this->l_data() + offset * this->ld_, this->m_ * subsetSize, chase::mpi::getMPI_Type<T>(), i, this->mpi_grid_->get_col_comm());
                    }
                    else
                    {
                        MPI_Bcast(targetMultiVector->l_data() + offset * targetMultiVector->l_ld(), targetMultiVector->l_rows() * subsetSize, 
                                        chase::mpi::getMPI_Type<T>(), i, this->mpi_grid_->get_col_comm());
                    }
                }
            }

            for(auto i = 0; i < dims[0]; i++)
            {
                if(coords[0] == coords[1])
                {
                    chase::linalg::lapackpp::t_lacpy('A', this->m_, subsetSize, this->l_data() + offset * this->ld_, this->ld_, 
                                                        targetMultiVector->l_data() + offset * targetMultiVector->l_ld(), targetMultiVector->l_ld());
                }
            }
        }

    }
};


}    
}
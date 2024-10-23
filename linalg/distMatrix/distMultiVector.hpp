#pragma once

#include <iostream>                // For std::cout
#include <memory>                  // For std::unique_ptr, std::shared_ptr
#include <complex>                 // For std::complex types
#include <stdexcept>               // For throwing runtime errors
#include <omp.h>                   // For OpenMP parallelization
#include <mpi.h>

#include "algorithm/types.hpp"
#include "linalg/matrix/matrix.hpp"
#include "grid/mpiGrid2D.hpp"
#include "grid/mpiTypes.hpp"
#include "external/lapackpp/lapackpp.hpp"
#include "linalg/distMatrix/distMatrix.hpp"

#ifdef HAS_CUDA
#include <cuda_runtime.h>
#include "Impl/cuda/cuda_utils.hpp"
#include "linalg/internal/cuda/lacpy.hpp"
#include "external/cublaspp/cublaspp.hpp"
#include "linalg/internal/cuda/precision_conversion.cuh"
#endif

#include "Impl/cuda/nvtx.hpp"

namespace chase
{
namespace distMultiVector
{
enum class DistributionType {
    Block,
    BlockCyclic
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

template<typename T, CommunicatorType comm_type, typename Platform> 
class DistMultiVector1D;

template<typename T, CommunicatorType comm_type, typename Platform> 
class DistMultiVectorBlockCyclic1D;

template <typename T, CommunicatorType comm_type, template <typename, CommunicatorType, typename> class Derived, typename Platform = chase::platform::CPU>
class AbstractDistMultiVector {
protected:
    // Single precision matrix
#ifdef ENABLE_MIXED_PRECISION       
    using SinglePrecisionType = typename chase::ToSinglePrecisionTrait<T>::Type;
    using SinglePrecisionDerived = Derived<SinglePrecisionType, comm_type, Platform>;
    std::unique_ptr<SinglePrecisionDerived> single_precision_multivec_;
    bool is_single_precision_enabled_ = false; 
    std::chrono::high_resolution_clock::time_point start, end;
    std::chrono::duration<double> elapsed;    
#endif    
public:
    virtual ~AbstractDistMultiVector() = default;
    virtual DistributionType getMultiVectorDistributionType() const = 0;
    virtual CommunicatorType getMultiVectorCommunicatorType() const = 0;
    virtual chase::grid::MpiGrid2DBase* getMpiGrid() const = 0;   
    virtual std::shared_ptr<chase::grid::MpiGrid2DBase> getMpiGrid_shared_ptr() const = 0;    
    virtual std::size_t g_rows() const = 0;
    virtual std::size_t g_cols() const = 0;
    virtual std::size_t l_rows() const = 0;
    virtual std::size_t l_cols() const = 0;
    virtual std::size_t l_ld() const = 0;
    virtual T *         l_data() = 0;
    virtual std::size_t mb() const = 0;  

    virtual typename chase::platform::MatrixTypePlatform<T, Platform>::type& loc_matrix() = 0;

    int grank()
    {
        int grank = 0;
        MPI_Comm_rank(this->getMpiGrid()->get_comm(), &grank);
        return grank;
    }

    void allocate_cpu_data()
    {
        auto& loc_matrix = this->loc_matrix();
        loc_matrix.allocate_cpu_data();   
    }    
#ifdef ENABLE_MIXED_PRECISION       
    // Enable single precision for double types (and complex<double>)
    template <typename U = T, typename std::enable_if<std::is_same<U, double>::value || std::is_same<U, std::complex<double>>::value, int>::type = 0>
    void enableSinglePrecision() {
        if (!single_precision_multivec_) {
            start = std::chrono::high_resolution_clock::now();
            if constexpr(std::is_same<Derived<T, comm_type, Platform>, chase::distMultiVector::DistMultiVectorBlockCyclic1D<T, comm_type, Platform>>::value)
            {
                single_precision_multivec_ = std::make_unique<SinglePrecisionDerived>(this->g_rows(), this->g_cols(), this->mb(), this->getMpiGrid_shared_ptr());
            }else
            {
                single_precision_multivec_ = std::make_unique<SinglePrecisionDerived>(this->g_rows(), this->g_cols(), this->getMpiGrid_shared_ptr());
            }
            //
            if constexpr (std::is_same<Platform, chase::platform::CPU>::value) {
                #pragma omp parallel for collapse(2) schedule(static, 16)
                for (std::size_t j = 0; j < this->l_cols(); ++j) {
                    for (std::size_t i = 0; i < this->l_rows(); ++i) {
                        single_precision_multivec_->l_data()[j * single_precision_multivec_.get()->l_ld() + i] 
                                            = chase::convertToSinglePrecision(this->l_data()[j * this->l_ld() + i]);
                    }
                }
            }else
#ifdef HAS_CUDA            
            {

                chase::linalg::internal::cuda::convert_DP_TO_SP_GPU(this->l_data(), single_precision_multivec_->l_data(), this->l_cols() * this->l_rows());
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
                if constexpr (std::is_same<Platform, chase::platform::CPU>::value) {

                    #pragma omp parallel for collapse(2) schedule(static, 16)
                    for (std::size_t j = 0; j < this->l_cols(); ++j) {
                        for (std::size_t i = 0; i < this->l_rows(); ++i) {
                            this->l_data()[j * this->l_ld() + i] = 
                                    chase::convertToDoublePrecision<T>(single_precision_multivec_->l_data()[j * single_precision_multivec_.get()->l_ld() + i]);
                        }
                    }
                }else
    #ifdef HAS_CUDA            
                {

                    chase::linalg::internal::cuda::convert_SP_TO_DP_GPU(single_precision_multivec_->l_data(), this->l_data(), this->l_cols() * this->l_rows());
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

template<typename T, CommunicatorType comm_type, typename Platform = chase::platform::CPU> 
class DistMultiVector1D : public AbstractDistMultiVector<T, comm_type, DistMultiVector1D, Platform> //distribute either within row or column communicator of 2D MPI grid
{
public:
    using platform_type = Platform;
    using value_type = T;  // Alias for element type

    ~DistMultiVector1D() override {};
    DistMultiVector1D();    
    DistMultiVector1D(std::size_t M, std::size_t N,
                    std::shared_ptr<chase::grid::MpiGrid2DBase> mpi_grid)
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

        local_matrix_ =  typename chase::platform::MatrixTypePlatform<T, Platform>::type(m_, n_);  

        mb_ = m_;

        if (coord == dim - 1 && dim != 1)
        {
            mb_ = (M_ - m_) / (dim - 1);
        }    
    }

    DistMultiVector1D(std::size_t m, std::size_t n, std::size_t ld, T *data,
                    std::shared_ptr<chase::grid::MpiGrid2DBase> mpi_grid)
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

        local_matrix_ = typename chase::platform::MatrixTypePlatform<T, Platform>::type(m_, n_, ld_, data);

        if constexpr (std::is_same<Platform, chase::platform::GPU>::value)
        {
            ld_ = local_matrix_.gpu_ld();
        }

        int *dims_ = mpi_grid_.get()->get_dims();
        int *coord_ = mpi_grid_.get()->get_coords();

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

        mb_ = m_;

        if (coord == dim - 1 && dim != 1)
        {
            mb_ = (M_ - m_) / (dim - 1);
        }
        
    }    

    DistributionType getMultiVectorDistributionType() const override {
        return DistributionType::Block;
    }
    
    CommunicatorType getMultiVectorCommunicatorType() const override {
        return comm_type;
    }

    // Accessors for MPI grid
    chase::grid::MpiGrid2DBase* getMpiGrid() const override {
        return mpi_grid_.get();
    }

    std::shared_ptr<chase::grid::MpiGrid2DBase> getMpiGrid_shared_ptr() const override
    {
        return mpi_grid_;
    }

    template<typename CloneType>
    CloneType clone()
    {
        static_assert(
            std::is_same_v<T, typename CloneType::value_type>,
            "Cloned type must have the same value_type"
        );
        ///using NewCommType = typename CloneType::communicator_type;
        return CloneType(M_, N_, mpi_grid_);        
    }

    template<typename CloneType>
    CloneType clone(std::size_t g_M, std::size_t g_N)
    {
        static_assert(
            std::is_same_v<T, typename CloneType::value_type>,
            "Cloned type must have the same value_type"
        );
        ///using NewCommType = typename CloneType::communicator_type;
        return CloneType(g_M, g_N, mpi_grid_);        
    }

    template<typename CloneType>
    std::unique_ptr<CloneType> clone2()
    {
        static_assert(
            std::is_same_v<T, typename CloneType::value_type>,
            "Cloned type must have the same value_type"
        );
        ///using NewCommType = typename CloneType::communicator_type;
        return std::make_unique<CloneType>(M_, N_, mpi_grid_);        
    }

    template<typename CloneType>
    std::unique_ptr<CloneType> clone2(std::size_t g_M, std::size_t g_N)
    {
        static_assert(
            std::is_same_v<T, typename CloneType::value_type>,
            "Cloned type must have the same value_type"
        );
        ///using NewCommType = typename CloneType::communicator_type;
        return std::make_unique<CloneType>(g_M, g_N, mpi_grid_);        
    }

    template <CommunicatorType OtherCommType, typename OtherPlatform>
    void swap(DistMultiVector1D<T, OtherCommType, OtherPlatform>& other) 
    {

        SCOPED_NVTX_RANGE();

        // Check if the communicator types are the same
        if constexpr (comm_type != OtherCommType) {
            throw std::runtime_error("Cannot swap: Communicator types do not match.");
        }

        if constexpr (!std::is_same<Platform, OtherPlatform>::value) {
            throw std::runtime_error("Cannot swap: Platform types do not match.");
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
        std::swap(mb_, other.mb_);
        local_matrix_.swap(other.local_matrix_);
#ifdef ENABLE_MIXED_PRECISION
        std::swap(this->is_single_precision_enabled_, other.is_single_precision_enabled_);
        std::swap(this->single_precision_multivec_, other.single_precision_multivec_);
#endif
    }
    //swap column i with j
    void swap_ij(std::size_t i, std::size_t j)
    {
        if constexpr (std::is_same<Platform, chase::platform::CPU>::value){
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
#ifdef HAS_CUDA        
        else
        {
            T *tmp;
            CHECK_CUDA_ERROR(cudaMalloc(&tmp, m_ * sizeof(T)));
            chase::linalg::internal::cuda::t_lacpy('A',
                                            m_,
                                            1,
                                            this->l_data() + i * ld_,
                                            1,
                                            tmp,
                                            1);
            chase::linalg::internal::cuda::t_lacpy('A',
                                            m_,
                                            1,
                                            this->l_data() + j * ld_,
                                            1,
                                            this->l_data() + i * ld_,
                                            1);    
            chase::linalg::internal::cuda::t_lacpy('A',
                                            m_,
                                            1,
                                            tmp,
                                            1,
                                            this->l_data() + j * ld_,
                                            1);   
            cudaFree(tmp);
        }
#endif        
    }

#ifdef HAS_CUDA
    void D2H()
    {
        if constexpr (std::is_same<Platform, chase::platform::GPU>::value)
        {
            local_matrix_.D2H();
        }else
        {
            throw std::runtime_error("[DistMultiVector]: CPU type of matrix do not support D2H operation");
        }
    }

    void H2D()
    {
        if constexpr (std::is_same<Platform, chase::platform::GPU>::value)
        {
            local_matrix_.H2D();
        }else
        {
            throw std::runtime_error("[DistMultiVector]: CPU type of matrix do not support H2D operation");
        }
    }
#endif
    T *cpu_data()
    {
        if constexpr (std::is_same<Platform, chase::platform::GPU>::value)
        {
            return local_matrix_.cpu_data();
        }else
        {
            return local_matrix_.data();
        }        
    }

    std::size_t cpu_ld()
    {
        if constexpr (std::is_same<Platform, chase::platform::GPU>::value)
        {
            return local_matrix_.cpu_ld();
        }else
        {
            return local_matrix_.ld();
        }           
    }

    template<CommunicatorType target_comm_type, typename OtherPlatform>
    void redistributeImpl(DistMultiVector1D<T, target_comm_type, OtherPlatform>* targetMultiVector,
                            std::size_t offset, std::size_t subsetSize) {
        // Validate the subset range
        if (offset + subsetSize > this->g_cols() || subsetSize > targetMultiVector->g_cols()) {
            throw std::invalid_argument("Invalid subset range");
        }   

        if constexpr (!std::is_same<Platform, OtherPlatform>::value) {
            throw std::runtime_error("Cannot redistribute: Platform types do not match.");
        }

        // Check if the target matrix's communicator type matches the allowed types
        if constexpr (comm_type == CommunicatorType::row && target_comm_type == CommunicatorType::column) {
            // Implement redistribution from row to column
            redistributeRowToColumn<OtherPlatform>(targetMultiVector, offset, subsetSize);
        } else if constexpr (comm_type == CommunicatorType::column && target_comm_type == CommunicatorType::row) {
            // Implement redistribution from column to row
            redistributeColumnToRow<OtherPlatform>(targetMultiVector, offset, subsetSize);
        } else {
            throw std::runtime_error("Invalid redistribution between matrix types");
        }
    }

    template<CommunicatorType target_comm_type, typename OtherPlatform>
    void redistributeImpl(DistMultiVector1D<T, target_comm_type, OtherPlatform>* targetMultiVector) 
    {
        this->redistributeImpl(targetMultiVector, 0, this->n_);
    }

#ifdef HAS_NCCL
    template<CommunicatorType target_comm_type>
    void redistributeImplAsync(DistMultiVector1D<T, target_comm_type, chase::platform::GPU>* targetMultiVector,
                            std::size_t offset, std::size_t subsetSize, cudaStream_t* stream_ = nullptr) {
        
        cudaStream_t usedStream = (stream_ == nullptr) ? 0 : *stream_;

        // Validate the subset range
        if (offset + subsetSize > this->g_cols() || subsetSize > targetMultiVector->g_cols()) {
            throw std::invalid_argument("Invalid subset range");
        }   

        if constexpr (!std::is_same<Platform, chase::platform::GPU>::value) {
            throw std::runtime_error("NCCL based redistribution support only GPU.");
        }

        // Check if the target matrix's communicator type matches the allowed types
        if constexpr (comm_type == CommunicatorType::row && target_comm_type == CommunicatorType::column) {
            // Implement redistribution from row to column
            redistributeRowToColumnAsync(targetMultiVector, offset, subsetSize, usedStream);
        } else if constexpr (comm_type == CommunicatorType::column && target_comm_type == CommunicatorType::row) {
            // Implement redistribution from column to row
            redistributeColumnToRowAsync(targetMultiVector, offset, subsetSize, usedStream);
        } else {
            throw std::runtime_error("Invalid redistribution between matrix types");
        }
    }

    template<CommunicatorType target_comm_type>
    void redistributeImplAsync(DistMultiVector1D<T, target_comm_type,  chase::platform::GPU>* targetMultiVector, cudaStream_t* stream_ = nullptr) 
    {        
        this->redistributeImplAsync(targetMultiVector, 0, this->n_, stream_);
    }

#endif
    std::size_t g_rows() const override { return M_;}
    std::size_t g_cols() const override { return N_;}
    std::size_t l_rows() const override { return m_;}
    std::size_t l_cols() const override { return n_;}
    std::size_t l_ld() const override { return ld_;}
    std::size_t mb() const override { return mb_;}    

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
#ifdef HAS_SCALAPACK
    std::size_t *get_scalapack_desc(){ return desc_; }

    std::size_t * scalapack_descriptor_init()
    {
        if constexpr (std::is_same<Platform, chase::platform::GPU>::value)
        {
            if(local_matrix_.cpu_data() == nullptr)
            {
                local_matrix_.allocate_cpu_data();
            }
        }

        std::size_t ldd = this->cpu_ld();
     
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
                                                  &ldd, 
                                                  &info);


        }else
        {
            //row based will be implemented later
        }

        return desc_;
    }
#endif    


private:
    std::size_t M_;
    std::size_t N_;
    std::size_t m_;
    std::size_t n_;
    std::size_t ld_;
    std::size_t mb_;
    typename chase::platform::MatrixTypePlatform<T, Platform>::type local_matrix_;
    std::shared_ptr<chase::grid::MpiGrid2DBase> mpi_grid_;    
#ifdef HAS_SCALAPACK
    std::size_t desc_[9];
#endif
    //data for redistribution
    std::vector<std::size_t> orig_dests;
    std::vector<std::size_t> orig_srcs;
    std::vector<std::size_t> orig_lens;
    std::vector<std::size_t> target_disps;
    std::vector<std::size_t> orig_disps;

    template<typename OtherPlatform, chase::distMultiVector::CommunicatorType OtherCommType>
    void init_redistribution(DistMultiVector1D<T, OtherCommType, OtherPlatform>* targetMultiVector)
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
        int *coords = mpi_grid_.get()->get_coords();
        int *dims = mpi_grid_.get()->get_dims();
        int dim0, dim1;
        if constexpr (comm_type == chase::distMultiVector::CommunicatorType::column)
        {
            dim0 = dims[0];
            dim1 = dims[1];
        }else if constexpr (comm_type == chase::distMultiVector::CommunicatorType::row)
        {
            dim0 = dims[1];
            dim1 = dims[0];            
        }

        for(auto i = 1; i < M_; i++)
        {
            auto src_tmp = (i / mb) % dim0;
            auto dest_tmp = (i / nb) % dim1;
            if (dest_tmp == orig_dest && src_tmp == orig_src)
            {
                len += 1;
            }else
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

    template<typename OtherPlatform>
    void redistributeRowToColumn(DistMultiVector1D<T, CommunicatorType::column, OtherPlatform>* targetMultiVector,
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
                    if constexpr (std::is_same<Platform, chase::platform::CPU>::value)
                    {                    
                        chase::linalg::lapackpp::t_lacpy('A', this->m_, subsetSize, this->l_data() + offset * this->ld_, this->ld_, 
                                                            targetMultiVector->l_data() + offset * targetMultiVector->l_ld(), targetMultiVector->l_ld());
                    }
#ifdef HAS_CUDA
                    else
                    {
                        chase::linalg::internal::cuda::t_lacpy('A', this->m_, subsetSize, this->l_data() + offset * this->ld_, this->ld_, 
                                                            targetMultiVector->l_data() + offset * targetMultiVector->l_ld(), targetMultiVector->l_ld());   
                    }
#endif
                }
            }
        }else
        {
            init_redistribution<OtherPlatform, chase::distMultiVector::CommunicatorType::column>(targetMultiVector);

            for(auto i = 0; i < orig_lens.size(); i++)
            {
                if(coords[0] == orig_dests[i])
                {
                    if constexpr (std::is_same<Platform, chase::platform::CPU>::value)
                    {
                        auto max_c_len = *max_element(orig_lens.begin(), orig_lens.end());
                        std::unique_ptr<chase::matrix::MatrixCPU<T>> buff = std::make_unique<chase::matrix::MatrixCPU<T>>(max_c_len, subsetSize);
                        chase::linalg::lapackpp::t_lacpy('A', orig_lens[i], subsetSize, this->l_data() + offset * this->ld_ + orig_disps[i], this->ld_, 
                                                           buff->data(), orig_lens[i]);    

                        MPI_Bcast(buff->data(), orig_lens[i] * subsetSize, chase::mpi::getMPI_Type<T>(), orig_srcs[i], this->mpi_grid_->get_row_comm());
                        chase::linalg::lapackpp::t_lacpy('A', orig_lens[i], subsetSize, buff->data(), orig_lens[i], 
                                                           targetMultiVector->l_data() + target_disps[i] + offset * targetMultiVector->l_ld(), targetMultiVector->l_ld());   

                    }
#ifdef HAS_CUDA                                    
                    else
                    {
                        auto max_c_len = *max_element(orig_lens.begin(), orig_lens.end());
                        std::unique_ptr<chase::matrix::MatrixGPU<T>> buff = std::make_unique<chase::matrix::MatrixGPU<T>>(max_c_len, subsetSize);
                        chase::linalg::internal::cuda::t_lacpy('A', orig_lens[i], subsetSize, this->l_data() + offset * this->ld_ + orig_disps[i], this->ld_, 
                                                           buff->gpu_data(), orig_lens[i]);    

                        MPI_Bcast(buff->gpu_data(), orig_lens[i] * subsetSize, chase::mpi::getMPI_Type<T>(), orig_srcs[i], this->mpi_grid_->get_row_comm());
                        chase::linalg::internal::cuda::t_lacpy('A', orig_lens[i], subsetSize, buff->gpu_data(), orig_lens[i], 
                                                           targetMultiVector->l_data() + target_disps[i] + offset * targetMultiVector->l_ld(), targetMultiVector->l_ld());                         
                    }
#endif                    
                }
                
            }
        }
    }

    template<typename OtherPlatform>
    void redistributeColumnToRow(DistMultiVector1D<T, CommunicatorType::row, OtherPlatform>* targetMultiVector,
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
                    if constexpr (std::is_same<Platform, chase::platform::CPU>::value)
                    {
                        chase::linalg::lapackpp::t_lacpy('A', this->m_, subsetSize, this->l_data() + offset * this->ld_, this->ld_, 
                                                            targetMultiVector->l_data() + offset * targetMultiVector->l_ld(), targetMultiVector->l_ld());                        
                    }
    #ifdef HAS_CUDA                
                    else
                    {
                        chase::linalg::internal::cuda::t_lacpy('A', this->m_, subsetSize, this->l_data() + offset * this->ld_, this->ld_, 
                                                            targetMultiVector->l_data() + offset * targetMultiVector->l_ld(), targetMultiVector->l_ld());                         
                    }
    #endif                                    

                }
            }
        }else
        {
            init_redistribution<OtherPlatform, chase::distMultiVector::CommunicatorType::row>(targetMultiVector);

            for(auto i = 0; i < orig_lens.size(); i++)
            {
                if(coords[1] == orig_dests[i])
                {
                    if constexpr (std::is_same<Platform, chase::platform::CPU>::value)
                    {
                        auto max_c_len = *max_element(orig_lens.begin(), orig_lens.end());
                        std::unique_ptr<chase::matrix::MatrixCPU<T>> buff = std::make_unique<chase::matrix::MatrixCPU<T>>(max_c_len, subsetSize);
                        chase::linalg::lapackpp::t_lacpy('A', orig_lens[i], subsetSize, this->l_data() + offset * this->ld_ + orig_disps[i], this->ld_, 
                                                           buff->data(), orig_lens[i]);    

                        MPI_Bcast(buff->data(), orig_lens[i] * subsetSize, chase::mpi::getMPI_Type<T>(), orig_srcs[i], this->mpi_grid_->get_col_comm());
                        chase::linalg::lapackpp::t_lacpy('A', orig_lens[i], subsetSize, buff->data(), orig_lens[i], 
                                                           targetMultiVector->l_data() + target_disps[i] + offset * targetMultiVector->l_ld(), targetMultiVector->l_ld());   

                    }
#ifdef HAS_CUDA                                    
                    else
                    {
                        auto max_c_len = *max_element(orig_lens.begin(), orig_lens.end());
                        std::unique_ptr<chase::matrix::MatrixGPU<T>> buff = std::make_unique<chase::matrix::MatrixGPU<T>>(max_c_len, subsetSize);
                        chase::linalg::internal::cuda::t_lacpy('A', orig_lens[i], subsetSize, this->l_data() + offset * this->ld_ + orig_disps[i], this->ld_, 
                                                           buff->gpu_data(), orig_lens[i]);    

                        MPI_Bcast(buff->gpu_data(), orig_lens[i] * subsetSize, chase::mpi::getMPI_Type<T>(), orig_srcs[i], this->mpi_grid_->get_col_comm());
                        chase::linalg::internal::cuda::t_lacpy('A', orig_lens[i], subsetSize, buff->gpu_data(), orig_lens[i], 
                                                           targetMultiVector->l_data() + target_disps[i] + offset * targetMultiVector->l_ld(), targetMultiVector->l_ld());                         
                    }
#endif                    
                }
                
            }
        }
    }
#ifdef HAS_CUDA
#ifdef HAS_NCCL
    void redistributeRowToColumnAsync(DistMultiVector1D<T, CommunicatorType::column, chase::platform::GPU>* targetMultiVector,
                                    std::size_t offset, std::size_t subsetSize, cudaStream_t stream) {
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
                        CHECK_NCCL_ERROR(chase::nccl::ncclBcastWrapper(this->l_data() + offset * this->ld_, 
                                                                             this->m_ * subsetSize, 
                                                                             i, 
                                                                             this->mpi_grid_->get_nccl_row_comm(), 
                                                                             &stream));
                    }
                    else
                    {
                        CHECK_NCCL_ERROR(chase::nccl::ncclBcastWrapper(targetMultiVector->l_data() + offset * targetMultiVector->l_ld(), 
                                                                             targetMultiVector->l_rows() * subsetSize, 
                                                                             i, 
                                                                             this->mpi_grid_->get_nccl_row_comm(), 
                                                                             &stream));                        
                    }
                }
            }

            for(auto i = 0; i < dims[1]; i++)
            {
                if(coords[0] == coords[1])
                {

                    chase::linalg::internal::cuda::t_lacpy('A', this->m_, subsetSize, this->l_data() + offset * this->ld_, this->ld_, 
                                                        targetMultiVector->l_data() + offset * targetMultiVector->l_ld(), targetMultiVector->l_ld());   
                    
                }
            }
        }else
        {
            init_redistribution<chase::platform::GPU, chase::distMultiVector::CommunicatorType::column>(targetMultiVector);

            for(auto i = 0; i < orig_lens.size(); i++)
            {
                if(coords[0] == orig_dests[i])
                {
                    auto max_c_len = *max_element(orig_lens.begin(), orig_lens.end());
                    std::unique_ptr<chase::matrix::MatrixGPU<T>> buff = std::make_unique<chase::matrix::MatrixGPU<T>>(max_c_len, subsetSize);
                    chase::linalg::internal::cuda::t_lacpy('A', orig_lens[i], subsetSize, this->l_data() + offset * this->ld_ + orig_disps[i], this->ld_, 
                                                        buff->gpu_data(), orig_lens[i]);    

                    CHECK_NCCL_ERROR(chase::nccl::ncclBcastWrapper(buff->gpu_data(), 
                                                        orig_lens[i] * subsetSize, 
                                                        orig_srcs[i], 
                                                        this->mpi_grid_->get_nccl_row_comm(), 
                                                        &stream));
                                                        
                    chase::linalg::internal::cuda::t_lacpy('A', orig_lens[i], subsetSize, buff->gpu_data(), orig_lens[i], 
                                                        targetMultiVector->l_data() + target_disps[i] + offset * targetMultiVector->l_ld(), targetMultiVector->l_ld());   
                                     
                }   
            }
        }        
    }

    void redistributeColumnToRowAsync(DistMultiVector1D<T, CommunicatorType::row, chase::platform::GPU>* targetMultiVector,
                                    std::size_t offset, std::size_t subsetSize, cudaStream_t stream) {
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
                        CHECK_NCCL_ERROR(chase::nccl::ncclBcastWrapper(this->l_data() + offset * this->ld_, 
                                                                             this->m_ * subsetSize, 
                                                                             i, 
                                                                             this->mpi_grid_->get_nccl_col_comm(), 
                                                                             &stream));
                    }
                    else
                    {
                        CHECK_NCCL_ERROR(chase::nccl::ncclBcastWrapper(targetMultiVector->l_data() + offset * targetMultiVector->l_ld(), 
                                                                             targetMultiVector->l_rows() * subsetSize, 
                                                                             i, 
                                                                             this->mpi_grid_->get_nccl_col_comm(), 
                                                                             &stream));     
                    }
                }
            }

            for(auto i = 0; i < dims[0]; i++)
            {
                if(coords[0] == coords[1])
                {
                    chase::linalg::internal::cuda::t_lacpy('A', this->m_, subsetSize, this->l_data() + offset * this->ld_, this->ld_, 
                                                        targetMultiVector->l_data() + offset * targetMultiVector->l_ld(), targetMultiVector->l_ld());                         
                }
            }
        }else
        {
            init_redistribution<chase::platform::GPU, chase::distMultiVector::CommunicatorType::row>(targetMultiVector);

            for(auto i = 0; i < orig_lens.size(); i++)
            {
                if(coords[1] == orig_dests[i])
                {
                    auto max_c_len = *max_element(orig_lens.begin(), orig_lens.end());
                    std::unique_ptr<chase::matrix::MatrixGPU<T>> buff = std::make_unique<chase::matrix::MatrixGPU<T>>(max_c_len, subsetSize);
                    chase::linalg::internal::cuda::t_lacpy('A', orig_lens[i], subsetSize, this->l_data() + offset * this->ld_ + orig_disps[i], this->ld_, 
                                                        buff->gpu_data(), orig_lens[i]);    

                    CHECK_NCCL_ERROR(chase::nccl::ncclBcastWrapper(buff->gpu_data(), 
                                                        orig_lens[i] * subsetSize, 
                                                        orig_srcs[i], 
                                                        this->mpi_grid_->get_nccl_col_comm(), 
                                                        &stream));
                                                        
                    chase::linalg::internal::cuda::t_lacpy('A', orig_lens[i], subsetSize, buff->gpu_data(), orig_lens[i], 
                                                        targetMultiVector->l_data() + target_disps[i] + offset * targetMultiVector->l_ld(), targetMultiVector->l_ld());   
                                     
                }   
            }
        }
    }
    
#endif
#endif

};

template<typename T, CommunicatorType comm_type, typename Platform = chase::platform::CPU> 
class DistMultiVectorBlockCyclic1D : public AbstractDistMultiVector<T, comm_type, DistMultiVectorBlockCyclic1D, Platform> //distribute either within row or column communicator of 2D MPI grid
{
public:
    using platform_type = Platform;
    using value_type = T;  // Alias for element type

    ~DistMultiVectorBlockCyclic1D() override {};
    DistMultiVectorBlockCyclic1D(); 

    DistMultiVectorBlockCyclic1D(std::size_t M, std::size_t N, std::size_t mb,
                    std::shared_ptr<chase::grid::MpiGrid2DBase> mpi_grid)
                    :M_(M), N_(N), mpi_grid_(mpi_grid), mb_(mb)
    {
        int *dims_ = mpi_grid_.get()->get_dims();
        int *coord_ = mpi_grid_.get()->get_coords();
        int dim, coord;
        if constexpr (comm_type == CommunicatorType::column) 
        {
            dim = dims_[0];
            coord = coord_[0];
        }else
        {
            dim = dims_[1];
            coord = coord_[1];
        }

        std::tie(m_, mblocks_) = numroc(M_, mb_, coord, dim);
        n_ = N_;
        ld_ = m_;
        local_matrix_ = typename chase::platform::MatrixTypePlatform<T, Platform>::type(m_, n_);  
    }

    DistMultiVectorBlockCyclic1D(std::size_t M, std::size_t m, 
                                 std::size_t n, std::size_t mb, 
                                 std::size_t ld, T *data,
                                 std::shared_ptr<chase::grid::MpiGrid2DBase> mpi_grid)
                                 : M_(M), N_(n), n_(n), mb_(mb), mpi_grid_(mpi_grid), ld_(ld)
    {
        int *dims_ = mpi_grid_.get()->get_dims();
        int *coord_ = mpi_grid_.get()->get_coords();

        int dim, coord;
        if constexpr (comm_type == CommunicatorType::column) 
        {
            dim = dims_[0];
            coord = coord_[0];
        }else
        {
            dim = dims_[1];
            coord = coord_[1];
        }

        std::tie(m_, mblocks_) = numroc(M_, mb_, coord, dim);      

        if(m_ != m)
        {
            throw std::runtime_error("the local row number of input matrix is not correctly matching the given block-cyclic distribution");
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
    }


    DistributionType getMultiVectorDistributionType() const override {
        return DistributionType::BlockCyclic;
    }
    
    CommunicatorType getMultiVectorCommunicatorType() const override {
        return comm_type;
    }

    // Accessors for MPI grid
    chase::grid::MpiGrid2DBase* getMpiGrid() const override {
        return mpi_grid_.get();
    }

    std::shared_ptr<chase::grid::MpiGrid2DBase> getMpiGrid_shared_ptr() const override
    {
        return mpi_grid_;
    }


    template<typename CloneType>
    CloneType clone()
    {
        static_assert(
            std::is_same_v<T, typename CloneType::value_type>,
            "Cloned type must have the same value_type"
        );
        ///using NewCommType = typename CloneType::communicator_type;
        return CloneType(M_, N_, mb_, mpi_grid_);        
    }

    template<typename CloneType>
    CloneType clone(std::size_t g_M, std::size_t g_N)
    {
        static_assert(
            std::is_same_v<T, typename CloneType::value_type>,
            "Cloned type must have the same value_type"
        );
        ///using NewCommType = typename CloneType::communicator_type;
        return CloneType(g_M, g_N, mb_, mpi_grid_);        
    }

    template<typename CloneType>
    std::unique_ptr<CloneType> clone2()
    {
        static_assert(
            std::is_same_v<T, typename CloneType::value_type>,
            "Cloned type must have the same value_type"
        );
        ///using NewCommType = typename CloneType::communicator_type;
        return std::make_unique<CloneType>(M_, N_, mb_, mpi_grid_);        
    }

    template<typename CloneType>
    std::unique_ptr<CloneType> clone2(std::size_t g_M, std::size_t g_N)
    {
        static_assert(
            std::is_same_v<T, typename CloneType::value_type>,
            "Cloned type must have the same value_type"
        );
        ///using NewCommType = typename CloneType::communicator_type;
        return std::make_unique<CloneType>(g_M, g_N, mb_, mpi_grid_);        
    }

    template <CommunicatorType OtherCommType, typename OtherPlatform>
    void swap(DistMultiVectorBlockCyclic1D<T, OtherCommType, OtherPlatform>& other) 
    {
        // Check if the communicator types are the same
        if constexpr (comm_type != OtherCommType) {
            throw std::runtime_error("Cannot swap: Communicator types do not match.");
        }

        if constexpr (!std::is_same<Platform, OtherPlatform>::value) {
            throw std::runtime_error("Cannot swap: Platform types do not match.");
        }
        // Ensure both objects have the same MPI grid
        if (mpi_grid_.get() != other.mpi_grid_.get()) {
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
#ifdef ENABLE_MIXED_PRECISION
        std::swap(this->is_single_precision_enabled_, other.is_single_precision_enabled_);
        std::swap(this->single_precision_multivec_, other.single_precision_multivec_);
#endif
    }

    //swap column i with j
    void swap_ij(std::size_t i, std::size_t j)
    {
        if constexpr (std::is_same<Platform, chase::platform::CPU>::value){
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
#ifdef HAS_CUDA        
        else
        {
            T *tmp;
            CHECK_CUDA_ERROR(cudaMalloc(&tmp, m_ * sizeof(T)));
            chase::linalg::internal::cuda::t_lacpy('A',
                                            m_,
                                            1,
                                            this->l_data() + i * ld_,
                                            1,
                                            tmp,
                                            1);
            chase::linalg::internal::cuda::t_lacpy('A',
                                            m_,
                                            1,
                                            this->l_data() + j * ld_,
                                            1,
                                            this->l_data() + i * ld_,
                                            1);    
            chase::linalg::internal::cuda::t_lacpy('A',
                                            m_,
                                            1,
                                            tmp,
                                            1,
                                            this->l_data() + j * ld_,
                                            1);   
            cudaFree(tmp);
        }
#endif        
    }

#ifdef HAS_CUDA
    void D2H()
    {
        if constexpr (std::is_same<Platform, chase::platform::GPU>::value)
        {
            local_matrix_.D2H();
        }else
        {
            throw std::runtime_error("[DistMultiVector]: CPU type of matrix do not support D2H operation");
        }
    }

    void H2D()
    {
        if constexpr (std::is_same<Platform, chase::platform::GPU>::value)
        {
            local_matrix_.H2D();
        }else
        {
            throw std::runtime_error("[DistMultiVector]: CPU type of matrix do not support H2D operation");
        }
    }
#endif
    T *cpu_data()
    {
        if constexpr (std::is_same<Platform, chase::platform::GPU>::value)
        {
            return local_matrix_.cpu_data();
        }else
        {
            return local_matrix_.data();
        }        
    }

    std::size_t cpu_ld()
    {
        if constexpr (std::is_same<Platform, chase::platform::GPU>::value)
        {
            return local_matrix_.cpu_ld();
        }else
        {
            return local_matrix_.ld();
        }           
    }

    template<CommunicatorType target_comm_type, typename OtherPlatform>
    void redistributeImpl(DistMultiVectorBlockCyclic1D<T, target_comm_type, OtherPlatform>* targetMultiVector,
                            std::size_t offset, std::size_t subsetSize) {
        // Validate the subset range
        if (offset + subsetSize > this->g_cols() || subsetSize > targetMultiVector->g_cols()) {
            throw std::invalid_argument("Invalid subset range");
        }   

        if constexpr (!std::is_same<Platform, OtherPlatform>::value) {
            throw std::runtime_error("Cannot redistribute: Platform types do not match.");
        }

        // Check if the target matrix's communicator type matches the allowed types
        if constexpr (comm_type == CommunicatorType::row && target_comm_type == CommunicatorType::column) {
            // Implement redistribution from row to column
            redistributeRowToColumn<OtherPlatform>(targetMultiVector, offset, subsetSize);
        } else if constexpr (comm_type == CommunicatorType::column && target_comm_type == CommunicatorType::row) {
            // Implement redistribution from column to row
            redistributeColumnToRow<OtherPlatform>(targetMultiVector, offset, subsetSize);
        } else {
            throw std::runtime_error("Invalid redistribution between matrix types");
        }
    }

    template<CommunicatorType target_comm_type, typename OtherPlatform>
    void redistributeImpl(DistMultiVectorBlockCyclic1D<T, target_comm_type, OtherPlatform>* targetMultiVector) 
    {
        this->redistributeImpl(targetMultiVector, 0, this->n_);
    }

#ifdef HAS_NCCL
    template<CommunicatorType target_comm_type>
    void redistributeImplAsync(DistMultiVectorBlockCyclic1D<T, target_comm_type, chase::platform::GPU>* targetMultiVector,
                            std::size_t offset, std::size_t subsetSize, cudaStream_t* stream_ = nullptr) {
        
        cudaStream_t usedStream = (stream_ == nullptr) ? 0 : *stream_;

        // Validate the subset range
        if (offset + subsetSize > this->g_cols() || subsetSize > targetMultiVector->g_cols()) {
            throw std::invalid_argument("Invalid subset range");
        }   

        if constexpr (!std::is_same<Platform, chase::platform::GPU>::value) {
            throw std::runtime_error("NCCL based redistribution support only GPU.");
        }

        // Check if the target matrix's communicator type matches the allowed types
        if constexpr (comm_type == CommunicatorType::row && target_comm_type == CommunicatorType::column) {
            // Implement redistribution from row to column
            redistributeRowToColumnAsync(targetMultiVector, offset, subsetSize, usedStream);
        } else if constexpr (comm_type == CommunicatorType::column && target_comm_type == CommunicatorType::row) {
            // Implement redistribution from column to row
            redistributeColumnToRowAsync(targetMultiVector, offset, subsetSize, usedStream);
        } else {
            throw std::runtime_error("Invalid redistribution between matrix types");
        }
    }

    template<CommunicatorType target_comm_type>
    void redistributeImplAsync(DistMultiVectorBlockCyclic1D<T, target_comm_type,  chase::platform::GPU>* targetMultiVector, cudaStream_t* stream_ = nullptr) 
    {        
        this->redistributeImplAsync(targetMultiVector, 0, this->n_, stream_);
    }
#endif
    std::size_t g_rows() const override { return M_;}
    std::size_t g_cols() const override { return N_;}
    std::size_t l_rows() const override { return m_;}
    std::size_t l_cols() const override { return n_;}
    std::size_t l_ld() const override { return ld_;}
    std::size_t mb() const override { return mb_;}    
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

#ifdef HAS_SCALAPACK
    std::size_t *get_scalapack_desc(){ return desc_; }

    std::size_t * scalapack_descriptor_init()
    {
        if constexpr (std::is_same<Platform, chase::platform::GPU>::value)
        {
            if(local_matrix_.cpu_data() == nullptr)
            {
                local_matrix_.allocate_cpu_data();
            }
        }

        std::size_t ldd = this->cpu_ld();
     
        if constexpr (comm_type ==  CommunicatorType::column)
        {
            int *coords = mpi_grid_.get()->get_coords();
            int *dims = mpi_grid_.get()->get_dims();

            std::size_t default_blocksize = 64;
            std::size_t nb = std::min(n_, default_blocksize);
            int zero = 0;
            int one = 1;
            int info;
            int colcomm1D_ctxt = mpi_grid_.get()->get_blacs_colcomm_ctxt();
            chase::linalg::scalapackpp::t_descinit(desc_, 
                                                  &M_, 
                                                  &N_, 
                                                  &mb_, 
                                                  &nb, 
                                                  &zero, 
                                                  &zero,
                                                  &colcomm1D_ctxt, 
                                                  &ldd, 
                                                  &info);


        }else
        {
            //row based will be implemented later
        }

        return desc_;
    }
#endif    

private:
    std::size_t M_;
    std::size_t N_;
    std::size_t m_;
    std::size_t n_;
    std::size_t ld_;
    std::size_t mb_;
    std::size_t mblocks_;
    typename chase::platform::MatrixTypePlatform<T, Platform>::type local_matrix_;
    std::shared_ptr<chase::grid::MpiGrid2DBase> mpi_grid_;
#ifdef HAS_SCALAPACK
    std::size_t desc_[9];
#endif
    //data for redistribution
    std::vector<std::size_t> orig_dests;
    std::vector<std::size_t> orig_srcs;
    std::vector<std::size_t> orig_lens;
    std::vector<std::size_t> target_disps;
    std::vector<std::size_t> orig_disps;

    template<typename OtherPlatform, chase::distMultiVector::CommunicatorType OtherCommType>
    void init_redistribution(DistMultiVectorBlockCyclic1D<T, OtherCommType, OtherPlatform>* targetMultiVector)
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
        int *coords = mpi_grid_.get()->get_coords();
        int *dims = mpi_grid_.get()->get_dims();
        int dim0, dim1;
        if constexpr (comm_type == chase::distMultiVector::CommunicatorType::column)
        {
            dim0 = dims[0];
            dim1 = dims[1];
        }else if constexpr (comm_type == chase::distMultiVector::CommunicatorType::row)
        {
            dim0 = dims[1];
            dim1 = dims[0];            
        }

        for(auto i = 1; i < M_; i++)
        {
            auto src_tmp = (i / mb) % dim0;
            auto dest_tmp = (i / nb) % dim1;
            if (dest_tmp == orig_dest && src_tmp == orig_src)
            {
                len += 1;
            }else
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

    template<typename OtherPlatform>
    void redistributeRowToColumn(DistMultiVectorBlockCyclic1D<T, CommunicatorType::column, OtherPlatform>* targetMultiVector,
                                    std::size_t offset, std::size_t subsetSize) {
        // Ensure the dimensions are compatible
        if (this->M_ != targetMultiVector->g_rows() || this->N_ != targetMultiVector->g_cols()) {
            throw std::runtime_error("Dimension mismatch during redistribution");
        }
        
        if(this->mpi_grid_.get() != targetMultiVector->getMpiGrid())
        {
            throw std::runtime_error("MPI Grid mismatch during redistribution");
        }

        if(this->mb_ != targetMultiVector->mb())
        {
            throw std::runtime_error("Blocksize of the original and target matrices mismatch during redistribution");
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
                    if constexpr (std::is_same<Platform, chase::platform::CPU>::value)
                    {                    
                        chase::linalg::lapackpp::t_lacpy('A', this->m_, subsetSize, this->l_data() + offset * this->ld_, this->ld_, 
                                                            targetMultiVector->l_data() + offset * targetMultiVector->l_ld(), targetMultiVector->l_ld());
                    }
#ifdef HAS_CUDA
                    else
                    {
                        chase::linalg::internal::cuda::t_lacpy('A', this->m_, subsetSize, this->l_data() + offset * this->ld_, this->ld_, 
                                                            targetMultiVector->l_data() + offset * targetMultiVector->l_ld(), targetMultiVector->l_ld());   
                    }
#endif
                }
            }
        }else
        {
            init_redistribution<OtherPlatform, chase::distMultiVector::CommunicatorType::column>(targetMultiVector);

            for(auto i = 0; i < orig_lens.size(); i++)
            {
                if(coords[0] == orig_dests[i])
                {
                    if constexpr (std::is_same<Platform, chase::platform::CPU>::value)
                    {
                        auto max_c_len = *max_element(orig_lens.begin(), orig_lens.end());
                        std::unique_ptr<chase::matrix::MatrixCPU<T>> buff = std::make_unique<chase::matrix::MatrixCPU<T>>(max_c_len, subsetSize);
                        chase::linalg::lapackpp::t_lacpy('A', orig_lens[i], subsetSize, this->l_data() + offset * this->ld_ + orig_disps[i], this->ld_, 
                                                           buff->data(), orig_lens[i]);    

                        MPI_Bcast(buff->data(), orig_lens[i] * subsetSize, chase::mpi::getMPI_Type<T>(), orig_srcs[i], this->mpi_grid_->get_row_comm());
                        chase::linalg::lapackpp::t_lacpy('A', orig_lens[i], subsetSize, buff->data(), orig_lens[i], 
                                                           targetMultiVector->l_data() + target_disps[i] + offset * targetMultiVector->l_ld(), targetMultiVector->l_ld());   

                    }
#ifdef HAS_CUDA                                    
                    else
                    {
                        auto max_c_len = *max_element(orig_lens.begin(), orig_lens.end());
                        std::unique_ptr<chase::matrix::MatrixGPU<T>> buff = std::make_unique<chase::matrix::MatrixGPU<T>>(max_c_len, subsetSize);
                        chase::linalg::internal::cuda::t_lacpy('A', orig_lens[i], subsetSize, this->l_data() + offset * this->ld_ + orig_disps[i], this->ld_, 
                                                           buff->gpu_data(), orig_lens[i]);    

                        MPI_Bcast(buff->gpu_data(), orig_lens[i] * subsetSize, chase::mpi::getMPI_Type<T>(), orig_srcs[i], this->mpi_grid_->get_row_comm());
                        chase::linalg::internal::cuda::t_lacpy('A', orig_lens[i], subsetSize, buff->gpu_data(), orig_lens[i], 
                                                           targetMultiVector->l_data() + target_disps[i] + offset * targetMultiVector->l_ld(), targetMultiVector->l_ld());                         
                    }
#endif                    
                }
                
            }
        }
    }


    template<typename OtherPlatform>
    void redistributeColumnToRow(DistMultiVectorBlockCyclic1D<T, CommunicatorType::row, OtherPlatform>* targetMultiVector,
                                    std::size_t offset, std::size_t subsetSize) {
        // Ensure the dimensions are compatible
        if (this->M_ != targetMultiVector->g_rows() || this->N_ != targetMultiVector->g_cols()) {
            throw std::runtime_error("Dimension mismatch during redistribution");
        }

        if(this->mpi_grid_.get() != targetMultiVector->getMpiGrid())
        {
            throw std::runtime_error("MPI Grid mismatch during redistribution");
        }

        if(this->mb_ != targetMultiVector->mb())
        {
            throw std::runtime_error("Blocksize of the original and target matrices mismatch during redistribution");
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
                    if constexpr (std::is_same<Platform, chase::platform::CPU>::value)
                    {
                        chase::linalg::lapackpp::t_lacpy('A', this->m_, subsetSize, this->l_data() + offset * this->ld_, this->ld_, 
                                                            targetMultiVector->l_data() + offset * targetMultiVector->l_ld(), targetMultiVector->l_ld());                        
                    }
    #ifdef HAS_CUDA                
                    else
                    {
                        chase::linalg::internal::cuda::t_lacpy('A', this->m_, subsetSize, this->l_data() + offset * this->ld_, this->ld_, 
                                                            targetMultiVector->l_data() + offset * targetMultiVector->l_ld(), targetMultiVector->l_ld());                         
                    }
    #endif                                    

                }
            }
        }
        else
        {
            init_redistribution<OtherPlatform, chase::distMultiVector::CommunicatorType::row>(targetMultiVector);

            for(auto i = 0; i < orig_lens.size(); i++)
            {
                if(coords[1] == orig_dests[i])
                {
                    if constexpr (std::is_same<Platform, chase::platform::CPU>::value)
                    {
                        auto max_c_len = *max_element(orig_lens.begin(), orig_lens.end());
                        std::unique_ptr<chase::matrix::MatrixCPU<T>> buff = std::make_unique<chase::matrix::MatrixCPU<T>>(max_c_len, subsetSize);
                        chase::linalg::lapackpp::t_lacpy('A', orig_lens[i], subsetSize, this->l_data() + offset * this->ld_ + orig_disps[i], this->ld_, 
                                                           buff->data(), orig_lens[i]);    

                        MPI_Bcast(buff->data(), orig_lens[i] * subsetSize, chase::mpi::getMPI_Type<T>(), orig_srcs[i], this->mpi_grid_->get_col_comm());
                        chase::linalg::lapackpp::t_lacpy('A', orig_lens[i], subsetSize, buff->data(), orig_lens[i], 
                                                           targetMultiVector->l_data() + target_disps[i] + offset * targetMultiVector->l_ld(), targetMultiVector->l_ld());   

                    }
#ifdef HAS_CUDA                                    
                    else
                    {
                        auto max_c_len = *max_element(orig_lens.begin(), orig_lens.end());
                        std::unique_ptr<chase::matrix::MatrixGPU<T>> buff = std::make_unique<chase::matrix::MatrixGPU<T>>(max_c_len, subsetSize);
                        chase::linalg::internal::cuda::t_lacpy('A', orig_lens[i], subsetSize, this->l_data() + offset * this->ld_ + orig_disps[i], this->ld_, 
                                                           buff->gpu_data(), orig_lens[i]);    

                        MPI_Bcast(buff->gpu_data(), orig_lens[i] * subsetSize, chase::mpi::getMPI_Type<T>(), orig_srcs[i], this->mpi_grid_->get_col_comm());
                        chase::linalg::internal::cuda::t_lacpy('A', orig_lens[i], subsetSize, buff->gpu_data(), orig_lens[i], 
                                                           targetMultiVector->l_data() + target_disps[i] + offset * targetMultiVector->l_ld(), targetMultiVector->l_ld());                         
                    }
#endif                    
                }
                
            }
        }

    }


#ifdef HAS_CUDA
#ifdef HAS_NCCL
    void redistributeRowToColumnAsync(DistMultiVectorBlockCyclic1D<T, CommunicatorType::column, chase::platform::GPU>* targetMultiVector,
                                    std::size_t offset, std::size_t subsetSize, cudaStream_t stream) {
        // Ensure the dimensions are compatible
        if (this->M_ != targetMultiVector->g_rows() || this->N_ != targetMultiVector->g_cols()) {
            throw std::runtime_error("Dimension mismatch during redistribution");
        }
        
        if(this->mpi_grid_.get() != targetMultiVector->getMpiGrid())
        {
            throw std::runtime_error("MPI Grid mismatch during redistribution");
        }
        
        if(this->mb_ != targetMultiVector->mb())
        {
            throw std::runtime_error("Blocksize of the original and target matrices mismatch during redistribution");
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
                        CHECK_NCCL_ERROR(chase::nccl::ncclBcastWrapper(this->l_data() + offset * this->ld_, 
                                                                             this->m_ * subsetSize, 
                                                                             i, 
                                                                             this->mpi_grid_->get_nccl_row_comm(), 
                                                                             &stream));
                    }
                    else
                    {
                        CHECK_NCCL_ERROR(chase::nccl::ncclBcastWrapper(targetMultiVector->l_data() + offset * targetMultiVector->l_ld(), 
                                                                             targetMultiVector->l_rows() * subsetSize, 
                                                                             i, 
                                                                             this->mpi_grid_->get_nccl_row_comm(), 
                                                                             &stream));                        
                    }
                }
            }

            for(auto i = 0; i < dims[1]; i++)
            {
                if(coords[0] == coords[1])
                {

                    chase::linalg::internal::cuda::t_lacpy('A', this->m_, subsetSize, this->l_data() + offset * this->ld_, this->ld_, 
                                                        targetMultiVector->l_data() + offset * targetMultiVector->l_ld(), targetMultiVector->l_ld());   
                    
                }
            }
        }else
        {
            init_redistribution<chase::platform::GPU, chase::distMultiVector::CommunicatorType::column>(targetMultiVector);

            for(auto i = 0; i < orig_lens.size(); i++)
            {
                if(coords[0] == orig_dests[i])
                {
                    auto max_c_len = *max_element(orig_lens.begin(), orig_lens.end());
                    std::unique_ptr<chase::matrix::MatrixGPU<T>> buff = std::make_unique<chase::matrix::MatrixGPU<T>>(max_c_len, subsetSize);
                    chase::linalg::internal::cuda::t_lacpy('A', orig_lens[i], subsetSize, this->l_data() + offset * this->ld_ + orig_disps[i], this->ld_, 
                                                        buff->gpu_data(), orig_lens[i]);    

                    CHECK_NCCL_ERROR(chase::nccl::ncclBcastWrapper(buff->gpu_data(), 
                                                        orig_lens[i] * subsetSize, 
                                                        orig_srcs[i], 
                                                        this->mpi_grid_->get_nccl_row_comm(), 
                                                        &stream));
                                                        
                    chase::linalg::internal::cuda::t_lacpy('A', orig_lens[i], subsetSize, buff->gpu_data(), orig_lens[i], 
                                                        targetMultiVector->l_data() + target_disps[i] + offset * targetMultiVector->l_ld(), targetMultiVector->l_ld());   
                                     
                }   
            }
        }  
    }

    void redistributeColumnToRowAsync(DistMultiVectorBlockCyclic1D<T, CommunicatorType::row, chase::platform::GPU>* targetMultiVector,
                                    std::size_t offset, std::size_t subsetSize, cudaStream_t stream) {
        // Ensure the dimensions are compatible
        if (this->M_ != targetMultiVector->g_rows() || this->N_ != targetMultiVector->g_cols()) {
            throw std::runtime_error("Dimension mismatch during redistribution");
        }

        if(this->mpi_grid_.get() != targetMultiVector->getMpiGrid())
        {
            throw std::runtime_error("MPI Grid mismatch during redistribution");
        }

        if(this->mb_ != targetMultiVector->mb())
        {
            throw std::runtime_error("Blocksize of the original and target matrices mismatch during redistribution");
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
                        CHECK_NCCL_ERROR(chase::nccl::ncclBcastWrapper(this->l_data() + offset * this->ld_, 
                                                                             this->m_ * subsetSize, 
                                                                             i, 
                                                                             this->mpi_grid_->get_nccl_col_comm(), 
                                                                             &stream));
                    }
                    else
                    {
                        CHECK_NCCL_ERROR(chase::nccl::ncclBcastWrapper(targetMultiVector->l_data() + offset * targetMultiVector->l_ld(), 
                                                                             targetMultiVector->l_rows() * subsetSize, 
                                                                             i, 
                                                                             this->mpi_grid_->get_nccl_col_comm(), 
                                                                             &stream));     
                    }
                }
            }

            for(auto i = 0; i < dims[0]; i++)
            {
                if(coords[0] == coords[1])
                {
                    chase::linalg::internal::cuda::t_lacpy('A', this->m_, subsetSize, this->l_data() + offset * this->ld_, this->ld_, 
                                                        targetMultiVector->l_data() + offset * targetMultiVector->l_ld(), targetMultiVector->l_ld());                         
                }
            }
        }else
        {
            init_redistribution<chase::platform::GPU, chase::distMultiVector::CommunicatorType::row>(targetMultiVector);

            for(auto i = 0; i < orig_lens.size(); i++)
            {
                if(coords[1] == orig_dests[i])
                {
                    auto max_c_len = *max_element(orig_lens.begin(), orig_lens.end());
                    std::unique_ptr<chase::matrix::MatrixGPU<T>> buff = std::make_unique<chase::matrix::MatrixGPU<T>>(max_c_len, subsetSize);
                    chase::linalg::internal::cuda::t_lacpy('A', orig_lens[i], subsetSize, this->l_data() + offset * this->ld_ + orig_disps[i], this->ld_, 
                                                        buff->gpu_data(), orig_lens[i]);    

                    CHECK_NCCL_ERROR(chase::nccl::ncclBcastWrapper(buff->gpu_data(), 
                                                        orig_lens[i] * subsetSize, 
                                                        orig_srcs[i], 
                                                        this->mpi_grid_->get_nccl_col_comm(), 
                                                        &stream));
                                                        
                    chase::linalg::internal::cuda::t_lacpy('A', orig_lens[i], subsetSize, buff->gpu_data(), orig_lens[i], 
                                                        targetMultiVector->l_data() + target_disps[i] + offset * targetMultiVector->l_ld(), targetMultiVector->l_ld());   
                                     
                }   
            }
        }
    }
    
#endif
#endif


};


}    
}
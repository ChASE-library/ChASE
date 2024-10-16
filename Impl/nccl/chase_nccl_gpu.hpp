#pragma once

#include <cstring>
#include <memory>
#include <random>
#include <vector>
#include "algorithm/chaseBase.hpp"
#include "linalg/matrix/matrix.hpp"
#include "linalg/lapackpp/lapackpp.hpp"
#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"
#include "Impl/grid/mpiGrid2D.hpp"
#include "linalg/internal/nccl/cholqr.hpp"
#include "linalg/internal/nccl/lanczos.hpp"
#include "linalg/internal/nccl/residuals.hpp"
#include "linalg/internal/nccl/rayleighRitz.hpp"
#include "linalg/internal/nccl/shiftDiagonal.hpp"
#include "linalg/internal/cuda/random_normal_distribution.cuh"
#ifdef HAS_SCALAPACK
#include "linalg/scalapackpp/scalapackpp.hpp"
#endif
#include "linalg/internal/nccl/symOrHerm.hpp"
#include "algorithm/types.hpp"

#include "Impl/config/config.hpp"

#include "Impl/cuda/nvtx.hpp"

using namespace chase::linalg;

namespace chase
{
namespace Impl
{
template <class T>
class ChaseNCCLGPU : public ChaseBase<T>
{
public:
    ChaseNCCLGPU(std::size_t nev,
                 std::size_t nex,
                 chase::distMatrix::BlockBlockMatrix<T, chase::platform::GPU> *H,
                 chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::column, chase::platform::GPU> *V,
                 chase::Base<T> *ritzv): nev_(nev), nex_(nex), nevex_(nev + nex), config_(H->g_rows(), nev, nex), N_(H->g_rows())
    {
        SCOPED_NVTX_RANGE();

        if(H->g_rows() != H->g_cols())
        {
            std::runtime_error("ChASE requires the matrix solved to be squared");
        }

        if( H->getMpiGrid() != V->getMpiGrid())
        {   
            std::runtime_error("ChASE requires the matrix and eigenvectors mapped to same MPI grid");
        }    

        N_ = H->g_rows();
        Hmat_ = H;
        V1_ = V;
        V2_ = new chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::column, chase::platform::GPU>(Hmat_->g_rows(), nevex_, Hmat_->getMpiGrid_shared_ptr());
        W1_ = new chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::row, chase::platform::GPU>(Hmat_->g_rows(), nevex_, Hmat_->getMpiGrid_shared_ptr());
        W2_ = new chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::row, chase::platform::GPU>(Hmat_->g_rows(), nevex_, Hmat_->getMpiGrid_shared_ptr());
        ritzv_ = new chase::distMatrix::RedundantMatrix<chase::Base<T>, chase::platform::GPU>(nevex_, 1, nevex_, ritzv, Hmat_->getMpiGrid_shared_ptr());
        resid_ = new chase::distMatrix::RedundantMatrix<chase::Base<T>, chase::platform::GPU>(nevex_, 1, Hmat_->getMpiGrid_shared_ptr());
        A_ = new chase::distMatrix::RedundantMatrix<T, chase::platform::GPU>(nevex_, nevex_, Hmat_->getMpiGrid_shared_ptr());
    
        MPI_Comm_rank(Hmat_->getMpiGrid()->get_comm(), &my_rank_);
        MPI_Comm_size(Hmat_->getMpiGrid()->get_comm(), &nprocs_);
        coords_ = Hmat_->getMpiGrid()->get_coords();
        dims_ = Hmat_->getMpiGrid()->get_dims();

        CHECK_CUBLAS_ERROR(cublasCreate(&cublasH_));
        CHECK_CUSOLVER_ERROR(cusolverDnCreate(&cusolverH_));
        //CHECK_CUDA_ERROR(cudaStreamCreate(&stream_));
        //CHECK_CUBLAS_ERROR(cublasSetStream(cublasH_, stream_));
        //CHECK_CUSOLVER_ERROR(cusolverDnSetStream(cusolverH_, stream_));

        CHECK_CUDA_ERROR(cudaMalloc((void**)&devInfo_, sizeof(int)));
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_return_, sizeof(T) * nevex_));

        int lwork_geqrf = 0;
        int lwork_orgqr = 0;

        CHECK_CUSOLVER_ERROR(chase::linalg::cusolverpp::cusolverDnTgeqrf_bufferSize(
                                                            cusolverH_, 
                                                            N_, 
                                                            nevex_, 
                                                            V1_->l_data(), 
                                                            V1_->l_ld(), 
                                                            &lwork_geqrf));

        CHECK_CUSOLVER_ERROR(chase::linalg::cusolverpp::cusolverDnTgqr_bufferSize(
                                                            cusolverH_, 
                                                            N_, 
                                                            nevex_, 
                                                            nevex_,
                                                            V1_->l_data(), 
                                                            V1_->l_ld(),  
                                                            d_return_, 
                                                            &lwork_orgqr));

        lwork_ = (lwork_geqrf > lwork_orgqr) ? lwork_geqrf : lwork_orgqr;

        int lwork_heevd = 0;

        CHECK_CUSOLVER_ERROR(chase::linalg::cusolverpp::cusolverDnTheevd_bufferSize(
                                                            cusolverH_, 
                                                            CUSOLVER_EIG_MODE_VECTOR, 
                                                            CUBLAS_FILL_MODE_LOWER,
                                                            nevex_, 
                                                            A_->l_data(), 
                                                            A_->l_ld(), 
                                                            ritzv_->l_data(), 
                                                            &lwork_heevd));
        if (lwork_heevd > lwork_)
        {
            lwork_ = lwork_heevd;
        }

        int lwork_potrf = 0;

        CHECK_CUSOLVER_ERROR(chase::linalg::cusolverpp::cusolverDnTpotrf_bufferSize(
                                                            cusolverH_, 
                                                            CUBLAS_FILL_MODE_UPPER, 
                                                            nevex_, 
                                                            A_->l_data(), 
                                                            A_->l_ld(),
                                                            &lwork_potrf));
        if (lwork_potrf > lwork_)
        {
            lwork_ = lwork_potrf;
        }
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_work_, sizeof(T) * lwork_));    

        CHECK_CUDA_ERROR(cudaMalloc((void**)&states_,
                             sizeof(curandStatePhilox4_32_10_t) * (256 * 32)));


        std::vector<std::size_t> diag_xoffs, diag_yoffs;

        std::size_t *g_offs = Hmat_->g_offs();

        for(auto j = 0; j < Hmat_->l_cols(); j++)
        {
            for(auto i = 0; i < Hmat_->l_rows(); i++)
            {
                if(g_offs[0] + i == g_offs[1] + j)
                {
                    diag_xoffs.push_back(i);
                    diag_yoffs.push_back(j);
                }
            }
        }

        diag_cnt = diag_xoffs.size();

        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_diag_xoffs, sizeof(std::size_t) * diag_cnt));    
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_diag_yoffs, sizeof(std::size_t) * diag_cnt));    

        CHECK_CUDA_ERROR(cudaMemcpy(d_diag_xoffs, diag_xoffs.data(), sizeof(std::size_t) * diag_cnt , cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_diag_yoffs, diag_yoffs.data(), sizeof(std::size_t) * diag_cnt , cudaMemcpyHostToDevice));
        
    }

    ChaseNCCLGPU(const ChaseNCCLGPU&) = delete;

    ~ChaseNCCLGPU() 
    {
        SCOPED_NVTX_RANGE();
        
        if (cublasH_)
            CHECK_CUBLAS_ERROR(cublasDestroy(cublasH_));
        if (cusolverH_)
            CHECK_CUSOLVER_ERROR(cusolverDnDestroy(cusolverH_));
        if (d_work_)
            CHECK_CUDA_ERROR(cudaFree(d_work_));
        if (devInfo_)
            CHECK_CUDA_ERROR(cudaFree(devInfo_));
        if (d_return_)
            CHECK_CUDA_ERROR(cudaFree(d_return_));  
        if (states_)
            CHECK_CUDA_ERROR(cudaFree(states_));            
    }

    std::size_t GetN() const override { return N_; }

    std::size_t GetNev() override { return nev_; }
    
    std::size_t GetNex() override { return nex_;}

    chase::Base<T>* GetRitzv() override { return ritzv_->cpu_data(); }
    chase::Base<T>* GetResid() override { resid_->allocate_cpu_data(); return resid_->cpu_data(); }
    ChaseConfig<T>& GetConfig() override { return config_; }
    int get_nprocs() override { return nprocs_; }

    void loadProblemFromFile(std::string filename)
    {
       SCOPED_NVTX_RANGE();
       Hmat_->readFromBinaryFile(filename);
    }

#ifdef CHASE_OUTPUT
    //! Print some intermediate infos during the solving procedure
    void Output(std::string str) override
    {
        if(my_rank_ == 0)
        {
            std::cout << str;
        }        
    }
#endif
    bool checkSymmetryEasy() override
    {
        SCOPED_NVTX_RANGE();

        is_sym_ = chase::linalg::internal::nccl::checkSymmetryEasy(cublasH_, *Hmat_);  
        return is_sym_;
    }

    bool isSym() { return is_sym_; }

    void symOrHermMatrix(char uplo) override
    {
        SCOPED_NVTX_RANGE();

        chase::linalg::internal::nccl::symOrHermMatrix(uplo, *Hmat_);   
    }

    void Start() override
    {
        locked_ = 0;
    }

    void initVecs(bool random) override
    {
        SCOPED_NVTX_RANGE();

        if (random)
        {
            int mpi_col_rank;
            MPI_Comm_rank(Hmat_->getMpiGrid()->get_col_comm(), &mpi_col_rank);
            unsigned long long seed = 1337 + mpi_col_rank;

            chase::linalg::internal::cuda::chase_rand_normal(seed, states_, V1_->l_data(), V1_->l_ld() * V1_->l_cols(),
                            (cudaStream_t)0);
        }        
        
        chase::linalg::internal::cuda::t_lacpy('A', 
                                                V1_->l_rows(), 
                                                V1_->l_cols(), 
                                                V1_->l_data(), 
                                                V1_->l_ld(), 
                                                V2_->l_data(), 
                                                V2_->l_ld());

        Hmat_->H2D();
        next_ = NextOp::bAc;
    }

    void Lanczos(std::size_t m, chase::Base<T>* upperb) override 
    {   
        SCOPED_NVTX_RANGE();

        chase::linalg::internal::nccl::lanczos(cublasH_,
                                              m, 
                                              *Hmat_, 
                                              *V1_, 
                                              upperb);    
    }

    void Lanczos(std::size_t M, std::size_t numvec, chase::Base<T>* upperb,
                         chase::Base<T>* ritzv, chase::Base<T>* Tau, chase::Base<T>* ritzV) override
    {
        SCOPED_NVTX_RANGE();

        chase::linalg::internal::nccl::lanczos(cublasH_,
                                              M, 
                                              numvec, 
                                              *Hmat_, 
                                              *V1_, 
                                              upperb, 
                                              ritzv, 
                                              Tau, 
                                              ritzV);
    }

    void LanczosDos(std::size_t idx, std::size_t m, T* ritzVc) override
    {
        SCOPED_NVTX_RANGE();

        T One = T(1.0);
        T Zero = T(0.0);

        std::unique_ptr<T, chase::cuda::utils::CudaDeleter> d_ritzVc_ptr = nullptr;
        T *d_ritzVc_;
        CHECK_CUDA_ERROR(cudaMalloc(&d_ritzVc_, m * idx * sizeof(T))); 
        d_ritzVc_ptr.reset(d_ritzVc_);
        d_ritzVc_ = d_ritzVc_ptr.get();
        CHECK_CUDA_ERROR(cudaMemcpy(d_ritzVc_, ritzVc, m * idx * sizeof(T),
                             cudaMemcpyHostToDevice));
        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(cublasH_,
                                                      CUBLAS_OP_N,
                                                      CUBLAS_OP_N,
                                                      V1_->l_rows(),
                                                      idx,
                                                      m,
                                                      &One,
                                                      V1_->l_data(),
                                                      V1_->l_ld(),
                                                      d_ritzVc_,
                                                      m,
                                                      &Zero,
                                                      V2_->l_data(),
                                                      V2_->l_ld()));

        chase::linalg::internal::cuda::t_lacpy('A',
                                         V2_->l_rows(),
                                         m,
                                         V2_->l_data(),
                                         V2_->l_ld(),
                                         V1_->l_data(),
                                         V1_->l_ld());    
    }

    void Shift(T c, bool isunshift = false) override 
    {
        SCOPED_NVTX_RANGE();

        if(isunshift)
        {
            next_ = NextOp::bAc;
        }        
        chase::linalg::internal::nccl::shiftDiagonal(*Hmat_, d_diag_xoffs, d_diag_yoffs, diag_cnt, std::real(c));

#ifdef ENABLE_MIXED_PRECISION
        if constexpr (std::is_same<T, double>::value || std::is_same<T, std::complex<double>>::value)
        {
            auto min = *std::min_element(resid_->cpu_data() + locked_, resid_->cpu_data() + nev_);
            bool shouldEnableSP = (min > 1e-3 && !isunshift);
            auto updatePrecision = [&](auto& mat, bool copyback = false) {
                if (shouldEnableSP) {
                    mat->enableSinglePrecision();
                } else if (mat->isSinglePrecisionEnabled()) {
                    mat->disableSinglePrecision(copyback);
                }
            };

            // Update precision for all matrices
            updatePrecision(Hmat_);
            updatePrecision(V1_, true);  // Special case for V1_
            updatePrecision(W1_);

            // Message on enabling single precision
            if (shouldEnableSP && my_rank_ == 0 && !isunshift) {
                std::cout << "Enable Single Precision in Filter" << std::endl;
            }
            
        }
#endif
         
    }

    void HEMM(std::size_t block, T alpha, T beta, std::size_t offset) override 
    {
#ifdef ENABLE_MIXED_PRECISION
        if constexpr (std::is_same<T, double>::value || std::is_same<T, std::complex<double>>::value)
        {
            using singlePrecisionT = typename chase::ToSinglePrecisionTrait<T>::Type;
            auto min = *std::min_element(resid_->cpu_data() + locked_, resid_->cpu_data() + nev_);
            
            if(min > 1e-3)
            {
                auto Hmat_sp = Hmat_->getSinglePrecisionMatrix();
                auto V1_sp = V1_->getSinglePrecisionMatrix();
                auto W1_sp = W1_->getSinglePrecisionMatrix();
                singlePrecisionT alpha_sp = static_cast<singlePrecisionT>(alpha);
                singlePrecisionT beta_sp = static_cast<singlePrecisionT>(beta);  

                if (next_ == NextOp::bAc)
                {
                    chase::linalg::internal::nccl::MatrixMultiplyMultiVectors<singlePrecisionT>(cublasH_,
                                                                                &alpha_sp, 
                                                                                *Hmat_sp, 
                                                                                *V1_sp, 
                                                                                &beta_sp, 
                                                                                *W1_sp, 
                                                                                offset + locked_, 
                                                                                block);
                    next_ = NextOp::cAb;
                }
                else
                {
                    chase::linalg::internal::nccl::MatrixMultiplyMultiVectors<singlePrecisionT>(cublasH_,
                                                                                &alpha_sp, 
                                                                                *Hmat_sp, 
                                                                                *W1_sp, 
                                                                                &beta_sp, 
                                                                                *V1_sp, 
                                                                                offset + locked_, 
                                                                                block);            
                    next_ = NextOp::bAc;

                }                              
            }
            else
            {
                if (next_ == NextOp::bAc)
                {
                    chase::linalg::internal::nccl::MatrixMultiplyMultiVectors(cublasH_,
                                                                                &alpha, 
                                                                                *Hmat_, 
                                                                                *V1_, 
                                                                                &beta, 
                                                                                *W1_, 
                                                                                offset + locked_, 
                                                                                block);
                    next_ = NextOp::cAb;
                }
                else
                {
                    chase::linalg::internal::nccl::MatrixMultiplyMultiVectors(cublasH_,
                                                                                    &alpha, 
                                                                                    *Hmat_, 
                                                                                    *W1_, 
                                                                                    &beta, 
                                                                                    *V1_, 
                                                                                    offset + locked_, 
                                                                                    block);              
                    next_ = NextOp::bAc;

                }                
            }
        }        
        else
#endif
        
        {
            if (next_ == NextOp::bAc)
            {
                chase::linalg::internal::nccl::MatrixMultiplyMultiVectors(cublasH_,
                                                                            &alpha, 
                                                                            *Hmat_, 
                                                                            *V1_, 
                                                                            &beta, 
                                                                            *W1_, 
                                                                            offset + locked_, 
                                                                            block);
                next_ = NextOp::cAb;
            }
            else
            {
                chase::linalg::internal::nccl::MatrixMultiplyMultiVectors(cublasH_,
                                                                            &alpha, 
                                                                            *Hmat_, 
                                                                            *W1_, 
                                                                            &beta, 
                                                                            *V1_, 
                                                                            offset + locked_, 
                                                                            block);            
                next_ = NextOp::bAc;

            }                                    
        }            
    }

    void QR(std::size_t fixednev, chase::Base<T> cond) override 
    {
        SCOPED_NVTX_RANGE();

        int disable = config_.DoCholQR() ? 0 : 1;
        char* cholddisable = getenv("CHASE_DISABLE_CHOLQR");
        if (cholddisable) {
            disable = std::atoi(cholddisable);
        }

        int info = 1;

        if (disable == 1)
        {
#ifdef HAS_SCALAPACK
            chase::linalg::internal::nccl::houseHoulderQR(*V1_);
#else
        std::runtime_error("For ChASE-MPI, distributed Householder QR requires ScaLAPACK, which is not detected\n");
#endif
        }else if(nevex_ >= MINIMAL_N_INVOKE_MODIFIED_GRAM_SCHMIDT_QR_GPU_NCCL)
        {
            info = chase::linalg::internal::nccl::modifiedGramSchmidtCholQR(cublasH_,
                                                            cusolverH_,
                                                            V1_->l_rows(), 
                                                            V1_->l_cols(), 
                                                            locked_,
                                                            V1_->l_data(),  
                                                            V1_->l_ld(), 
                                                            V1_->getMpiGrid()->get_nccl_col_comm(),
                                                            d_work_,
                                                            lwork_,
                                                            A_->l_data());

#ifdef CHASE_OUTPUT
            if(my_rank_ == 0){
                std::cout << "NEV+NEX is larger than: " << MINIMAL_N_INVOKE_MODIFIED_GRAM_SCHMIDT_QR_GPU_NCCL << ", use modifiedGramSchmidtCholQR" << std::endl;
            }
#endif
            if(info != 0)
            {
                chase::linalg::internal::cuda::t_lacpy('A',
                                                V2_->l_rows(),
                                                V2_->l_cols(),
                                                V2_->l_data(),
                                                V2_->l_ld(),
                                                V1_->l_data(),
                                                V1_->l_ld()); 

                if(my_rank_ == 0){
                    std::cout << "modifiedGramSchmidtCholQR doesn't work, try with shiftedcholQR2." << std::endl;
                }


                info = chase::linalg::internal::nccl::shiftedcholQR2(cublasH_,
                                                                cusolverH_,
                                                                V1_->g_rows(),
                                                                V1_->l_rows(), 
                                                                V1_->l_cols(), 
                                                                V1_->l_data(),  
                                                                V1_->l_ld(), 
                                                                V1_->getMpiGrid()->get_nccl_col_comm(),
                                                                d_work_,
                                                                lwork_,
                                                                A_->l_data());   

                if(info != 0)
                {
#ifdef HAS_SCALAPACK
#ifdef CHASE_OUTPUT
                    if(my_rank_ == 0){
                        std::cout << "CholeskyQR doesn't work, Househoulder QR will be used." << std::endl;
                    }
#endif
                    chase::linalg::internal::nccl::houseHoulderQR(*V1_);
#else
                    std::runtime_error("For ChASE-MPI, distributed Householder QR requires ScaLAPACK, which is not detected\n");
#endif      
                }                                          
                                                               
            }
        }else
        {
            Base<T> cond_threshold_upper = (sizeof(Base<T>) == 8) ? 1e8 : 1e4;
            Base<T> cond_threshold_lower = (sizeof(Base<T>) == 8) ? 2e1 : 1e1;

            char* chol_threshold = getenv("CHASE_CHOLQR1_THLD");
            if (chol_threshold)
            {
                cond_threshold_lower = std::atof(chol_threshold);
            }

#ifdef CHASE_OUTPUT
            if(my_rank_ == 0){
                std::cout << std::setprecision(2) << "cond(V): " << cond << std::endl;
            }
#endif

            int info = 1;

            if (cond > cond_threshold_upper)
            {
                info = chase::linalg::internal::nccl::shiftedcholQR2(cublasH_,
                                                                cusolverH_,
                                                                V1_->g_rows(),
                                                                V1_->l_rows(), 
                                                                V1_->l_cols(), 
                                                                V1_->l_data(),  
                                                                V1_->l_ld(), 
                                                                V1_->getMpiGrid()->get_nccl_col_comm(),
                                                                d_work_,
                                                                lwork_,
                                                                A_->l_data());                                                
            }
            else if(cond < cond_threshold_lower)
            {
                info = chase::linalg::internal::nccl::cholQR1(cublasH_,
                                                                cusolverH_,
                                                                V1_->l_rows(), 
                                                                V1_->l_cols(), 
                                                                V1_->l_data(),  
                                                                V1_->l_ld(), 
                                                                V1_->getMpiGrid()->get_nccl_col_comm(),
                                                                d_work_,
                                                                lwork_,
                                                                A_->l_data());  
                                                            
            }
            else
            {                
                info = chase::linalg::internal::nccl::cholQR2(cublasH_,
                                                                cusolverH_,
                                                                V1_->l_rows(), 
                                                                V1_->l_cols(), 
                                                                V1_->l_data(),  
                                                                V1_->l_ld(), 
                                                                V1_->getMpiGrid()->get_nccl_col_comm(),
                                                                d_work_,
                                                                lwork_,
                                                                A_->l_data()); 
            }

            if (info != 0)
            {
#ifdef HAS_SCALAPACK
#ifdef CHASE_OUTPUT
                if(my_rank_ == 0){
                    std::cout << "CholeskyQR doesn't work, Househoulder QR will be used." << std::endl;
                }
#endif
                chase::linalg::internal::nccl::houseHoulderQR(*V1_);
#else
                std::runtime_error("For ChASE-MPI, distributed Householder QR requires ScaLAPACK, which is not detected\n");
#endif
            }

        }
        
        chase::linalg::internal::cuda::t_lacpy('A',
                                         V2_->l_rows(),
                                         locked_,
                                         V2_->l_data(),
                                         V2_->l_ld(),
                                         V1_->l_data(),
                                         V1_->l_ld());

        chase::linalg::internal::cuda::t_lacpy('A',
                                         V2_->l_rows(),
                                         nevex_ - locked_,
                                         V1_->l_data() + V1_->l_ld() * locked_,
                                         V1_->l_ld(),
                                         V2_->l_data() + V2_->l_ld() * locked_,
                                         V2_->l_ld());                                                                                                           
    }

    void RR(chase::Base<T>* ritzv, std::size_t block) override 
    {
        SCOPED_NVTX_RANGE();

        chase::linalg::internal::nccl::rayleighRitz(cublasH_,
                                                   cusolverH_,
                                                   *Hmat_, 
                                                   *V1_, 
                                                   *V2_, 
                                                   *W1_, 
                                                   *W2_, 
                                                   *ritzv_, 
                                                   locked_, 
                                                   block,
                                                   devInfo_,
                                                   d_work_,
                                                   lwork_,
                                                   A_);

        chase::linalg::internal::cuda::t_lacpy('A',
                                         V2_->l_rows(),
                                         block,
                                         V1_->l_data() + locked_ * V1_->l_ld(),
                                         V1_->l_ld(),
                                         V2_->l_data() + locked_ * V2_->l_ld(),
                                         V2_->l_ld());           
    }

    void Resd(chase::Base<T>* ritzv, chase::Base<T>* resd, std::size_t fixednev) override 
    {
        SCOPED_NVTX_RANGE();

        chase::linalg::internal::nccl::residuals(cublasH_,
                                                *Hmat_,
                                                *V1_,
                                                *V2_,
                                                *W1_,
                                                *W2_,
                                                ritzv_->loc_matrix(),
                                                resid_->loc_matrix(),
                                                locked_,
                                                nevex_ - locked_);         
    }

    void Swap(std::size_t i, std::size_t j) override 
    {
        V1_->swap_ij(i, j);
        V2_->swap_ij(i, j);        
    }

    void Lock(std::size_t new_converged) override 
    {
        SCOPED_NVTX_RANGE();

        locked_ += new_converged;
    }

    void End() override {         
        SCOPED_NVTX_RANGE();
        V1_->D2H(); 
    }

private:
    enum NextOp
    {
        cAb,
        bAc
    };
    NextOp next_; 

    bool is_sym_;
    std::size_t nev_;
    std::size_t nex_;
    std::size_t nevex_;
    std::size_t locked_;

    std::size_t N_; 

    int nprocs_;
    int my_rank_;
    int *coords_;
    int *dims_;

    chase::distMatrix::BlockBlockMatrix<T, chase::platform::GPU> *Hmat_;
    chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::column, chase::platform::GPU> *V1_;
    chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::column, chase::platform::GPU> *V2_;
    chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::row, chase::platform::GPU> *W1_;
    chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::row, chase::platform::GPU> *W2_;

    chase::distMatrix::RedundantMatrix<chase::Base<T>, chase::platform::GPU> *ritzv_;
    chase::distMatrix::RedundantMatrix<chase::Base<T>, chase::platform::GPU> *resid_;
    chase::distMatrix::RedundantMatrix<T, chase::platform::GPU> *A_;

    cudaStream_t stream_; 
    cublasHandle_t cublasH_;      
    cusolverDnHandle_t cusolverH_;
    curandStatePhilox4_32_10_t* states_ = NULL;

    int* devInfo_;
    T* d_return_;
    T* d_work_;
    int lwork_ = 0;

    std::size_t *d_diag_xoffs;
    std::size_t *d_diag_yoffs;
    std::size_t diag_cnt;
    chase::ChaseConfig<T> config_;
};

}
}
// This file is a part of ChASE.
// Copyright (c) 2015-2024, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include <cstring>
#include <memory>
#include <random>
#include <vector>
#include "algorithm/chaseBase.hpp"
#include "linalg/matrix/matrix.hpp"
#include "linalg/internal/cuda/cuda_kernels.hpp"
#include "linalg/internal/cpu/symOrHerm.hpp"
#include "linalg/internal/cpu/utils.hpp"
#include "algorithm/types.hpp"

#include "Impl/chase_gpu/nvtx.hpp"

using namespace chase::linalg;

namespace chase
{
namespace Impl
{
/**
 * @page ChASEGPU
 * 
 * @section intro_sec Introduction
 * This class implements the GPU-based sequential version of the Chase algorithm. It inherits from
 * `ChaseBase` and provides methods for solving generalized eigenvalue problems using Lanczos
 * iterations and various other matrix operations like QR factorization, HEMM, and Lanczos-based
 * algorithms. The computations are offloaded to the GPU using CUDA, CUBLAS, and CUSOLVER for efficient
 * matrix operations and eigenvalue solvers.
 * 
 * @section constructor_sec Constructors and Destructor
 * The constructor and destructor for the `ChASEGPU` class handle memory allocation, initialization
 * of matrices and associated data structures on the GPU, as well as resource cleanup. 
 * 
 * @subsection constructor_details Detailed Constructor
 * 
 * The constructor takes in several parameters to initialize the algorithm's matrices and configuration:
 * - `N`: The size of the matrix.
 * - `nev`: The number of eigenvalues to compute.
 * - `nex`: The number of extra vectors.
 * - `H`: A pointer to the matrix \( H \).
 * - `ldh`: The leading dimension of matrix \( H \).
 * - `V1`: A pointer to the matrix \( V_1 \).
 * - `ldv`: The leading dimension of matrix \( V_1 \).
 * - `ritzv`: A pointer to the Ritz values vector.
 * 
 * The constructor allocates memory for the GPU matrices, initializes CUDA streams, and sets up CUBLAS
 * and CUSOLVER handles. It also computes buffer sizes for various matrix operations.
 * 
 * @subsection destructor_details Destructor
 * 
 * The destructor ensures that all dynamically allocated memory and CUDA resources are properly freed
 * by calling the appropriate `cudaFree`, `cublasDestroy`, and `cusolverDnDestroy` functions.
 * 
 * @section members_sec Private Members
 * Private members are used to manage the algorithm's configuration and store matrices, as well as GPU
 * resources for matrix operations:
 * - `N_`: Size of the matrix.
 * - `locked_`: A counter for the number of converged eigenvalues.
 * - `nev_`: Number of eigenvalues to compute.
 * - `nex_`: Number of extra vectors.
 * - `nevex_`: The total number of eigenvalues and extra vectors.
 * - `ldh_`: Leading dimension of matrix \( H \).
 * - `ldv_`: Leading dimension of matrix \( V_1 \).
 * - `H_`, `V1_`, `ritzv_`: Pointers to the input data matrices.
 * - `tmp_`, `devInfo_`, `d_return_`, `d_work_`: Temporary buffers for GPU computations.
 * - `Hmat_`, `Vec1_`, `Vec2_`, `A_`, `ritzvs_`, `resid_`: GPU matrices for computations and results.
 * - `config_`: Configuration object for the Chase algorithm.
 * - `stream_`, `cublasH_`, `cusolverH_`: CUDA stream and handles for CUBLAS and CUSOLVER.
 * 
 * These members are initialized during the constructor to ensure the algorithm operates correctly on the GPU.
 */

/**
 * @brief GPU-based sequential Chase algorithm.
 * 
 * This class is responsible for solving generalized eigenvalue problems using GPU-based Lanczos
 * iterations and other matrix operations. It uses CUDA, CUBLAS, and CUSOLVER libraries for efficient
 * matrix operations and eigenvalue solvers.
 * 
 * @tparam T The data type (e.g., float, double).
 */    
template <class T, typename MatrixType = chase::matrix::Matrix<T, chase::platform::GPU>>
class ChASEGPU : public ChaseBase<T>
{
public:
    /**
     * @brief Constructor for the ChASEGPU class.
     * 
     * Initializes matrices and GPU resources, including CUDA streams, CUBLAS, and CUSOLVER handles.
     * Allocates memory for the GPU matrices and computes necessary buffer sizes for matrix operations.
     * 
     * @param N The size of the matrix.
     * @param nev The number of eigenvalues to compute.
     * @param nex The number of extra vectors.
     * @param H Pointer to the matrix \( H \).
     * @param ldh The leading dimension of matrix \( H \).
     * @param V1 Pointer to the matrix \( V_1 \).
     * @param ldv The leading dimension of matrix \( V_1 \).
     * @param ritzv Pointer to the Ritz values vector.
     */
    ChASEGPU(std::size_t N,
                std::size_t nev,
                std::size_t nex,
                T *H,
                std::size_t ldh,
                T *V1,
                std::size_t ldv,
                chase::Base<T> *ritzv)
                : N_(N),
                  H_(H),
                  V1_(V1),
                  ldh_(ldh),
                  ldv_(ldv),
                  ritzv_(ritzv),
                  nev_(nev),
                  nex_(nex),
                  nevex_(nev+nex),
                  config_(N, nev, nex)                  

    {
        SCOPED_NVTX_RANGE();

        Hmat_ = new MatrixType(N_, N_, ldh_, H_);
        Vec1_ = chase::matrix::Matrix<T, chase::platform::GPU>(N_, nevex_, ldv_, V1_);
        Vec2_ = chase::matrix::Matrix<T, chase::platform::GPU>(N_, nevex_);
        resid_ = chase::matrix::Matrix<chase::Base<T>, chase::platform::GPU>(nevex_, 1);
        ritzvs_ = chase::matrix::Matrix<chase::Base<T>, chase::platform::GPU>(nevex_, 1, nevex_, ritzv_);
        A_ = chase::matrix::Matrix<T, chase::platform::GPU>(nevex_, nevex_);

	    if constexpr (std::is_same<MatrixType, chase::matrix::QuasiHermitianMatrix<T, chase::platform::GPU>>::value)    
	    {
            is_sym_ = false;
            is_pseudoHerm_ = true;
            //Quasi Hermitian matrices require more space for the dual basis
            //A_ = chase::matrix::Matrix<T>(nevex_ + std::size_t(N/2), nevex_);
        }
        else
        {
            is_sym_ = true;
            is_pseudoHerm_ = false;
            //A_ = chase::matrix::Matrix<T>(nevex_, nevex_);
        }
        
	CUBLAS_INIT();

    }
            
    ChASEGPU(std::size_t N, 
             std::size_t nev, 
             std::size_t nex, 
             MatrixType *H, 
             T* V1, 
             std::size_t ldv,
             chase::Base<T>* ritzv)
             : N_(N), 
               H_(H->data()),
               V1_(V1),
               ldh_(H->ld()),
               ldv_(ldv),
               ritzv_(ritzv),
               nev_(nev), 
               nex_(nex), 
               nevex_(nev+nex),
               config_(N, nev, nex)
    {
        SCOPED_NVTX_RANGE();

	Hmat_ = H;
        Vec1_ = chase::matrix::Matrix<T, chase::platform::GPU>(N_, nevex_, ldv_, V1_);
        Vec2_ = chase::matrix::Matrix<T, chase::platform::GPU>(N_, nevex_);
        resid_ = chase::matrix::Matrix<chase::Base<T>, chase::platform::GPU>(nevex_, 1);
        ritzvs_ = chase::matrix::Matrix<chase::Base<T>, chase::platform::GPU>(nevex_, 1, nevex_, ritzv_);
        A_ = chase::matrix::Matrix<T, chase::platform::GPU>(nevex_, nevex_);
               
	if constexpr (std::is_same<MatrixType, chase::matrix::QuasiHermitianMatrix<T, chase::platform::GPU>>::value)    
	{
            is_sym_ = false;
            is_pseudoHerm_ = true;
            //Quasi Hermitian matrices require more space for the dual basis
            //A_ = chase::matrix::Matrix<T>(nevex_ + std::size_t(N/2), nevex_);
        }
        else
        {
            is_sym_ = true;
            is_pseudoHerm_ = false;
            //A_ = chase::matrix::Matrix<T>(nevex_, nevex_);
        }

	CUBLAS_INIT();
    }

    void CUBLAS_INIT(){

        CHECK_CUBLAS_ERROR(cublasCreate(&cublasH_));
        CHECK_CUSOLVER_ERROR(cusolverDnCreate(&cusolverH_));
        CHECK_CUDA_ERROR(cudaStreamCreate(&stream_));
        CHECK_CUBLAS_ERROR(cublasSetStream(cublasH_, stream_));
        CHECK_CUSOLVER_ERROR(cusolverDnSetStream(cusolverH_, stream_));

        CHECK_CUDA_ERROR(cudaMalloc((void**)&devInfo_, sizeof(int)));
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_return_, sizeof(T) * nevex_));
        CHECK_CUDA_ERROR(cudaMalloc((void**)&tmp_, N_ * sizeof(T)));

        int lwork_geqrf = 0;
        int lwork_orgqr = 0;

        CHECK_CUSOLVER_ERROR(chase::linalg::cusolverpp::cusolverDnTgeqrf_bufferSize(
                                                            cusolverH_, 
                                                            N_, 
                                                            nevex_, 
                                                            Vec1_.data(), 
                                                            Vec1_.ld(), 
                                                            &lwork_geqrf));

        CHECK_CUSOLVER_ERROR(chase::linalg::cusolverpp::cusolverDnTgqr_bufferSize(
                                                            cusolverH_, 
                                                            N_, 
                                                            nevex_, 
                                                            nevex_,
                                                            Vec1_.data(), 
                                                            Vec1_.ld(),  
                                                            d_return_, 
                                                            &lwork_orgqr));

        lwork_ = (lwork_geqrf > lwork_orgqr) ? lwork_geqrf : lwork_orgqr;

        int lwork_heevd = 0;

        CHECK_CUSOLVER_ERROR(chase::linalg::cusolverpp::cusolverDnTheevd_bufferSize(
                                                            cusolverH_, 
                                                            CUSOLVER_EIG_MODE_VECTOR, 
                                                            CUBLAS_FILL_MODE_LOWER,
                                                            nevex_, 
                                                            A_.data(), 
                                                            A_.ld(), 
                                                            ritzvs_.data(), 
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
                                                            A_.data(), 
                                                            A_.ld(),
                                                            &lwork_potrf));
        if (lwork_potrf > lwork_)
        {
            lwork_ = lwork_potrf;
        }
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_work_, sizeof(T) * lwork_));
    }

    ChASEGPU(const ChASEGPU&) = delete;

    /**
     * @brief Destructor for the ChASEGPU class.
     * 
     * Frees all dynamically allocated memory and destroys CUDA resources (CUBLAS, CUSOLVER, CUDA buffers).
     */
    ~ChASEGPU() 
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
        if (tmp_)
            CHECK_CUDA_ERROR(cudaFree(tmp_));

    }

    std::size_t GetN() const override { return N_; }

    std::size_t GetNev() override { return nev_; }
    
    std::size_t GetNex() override { return nex_; }

    chase::Base<T>* GetRitzv() override { return ritzv_; }
    chase::Base<T>* GetResid() override { resid_.allocate_cpu_data(); return resid_.cpu_data(); }
    ChaseConfig<T>& GetConfig() override { return config_; }
    int get_nprocs() override { return 1; }

    void loadProblemFromFile(std::string filename)
    {
        SCOPED_NVTX_RANGE();
        Hmat_->readFromBinaryFile(filename);
    }

#ifdef CHASE_OUTPUT
    //! Print some intermediate infos during the solving procedure
    void Output(std::string str) override
    {
        std::cout << str;
    }
#endif

    bool checkSymmetryEasy() override
    {
        SCOPED_NVTX_RANGE();
        is_sym_ = chase::linalg::internal::cpu::checkSymmetryEasy(N_, Hmat_->cpu_data(), Hmat_->cpu_ld());  
        return is_sym_;
    }

    bool isSym() override {return is_sym_;} 
    
    bool checkPseudoHermicityEasy() override
    {
        SCOPED_NVTX_RANGE();
	chase::linalg::internal::cuda::flipLowerHalfMatrixSign(Hmat_);
        is_pseudoHerm_ = chase::linalg::internal::cpu::checkSymmetryEasy(N_, Hmat_->cpu_data(), Hmat_->ld());  
	chase::linalg::internal::cuda::flipLowerHalfMatrixSign(Hmat_);
        return is_pseudoHerm_;
    }
    
    bool isPseudoHerm() override { return is_pseudoHerm_; }

    void symOrHermMatrix(char uplo) override
    {
        SCOPED_NVTX_RANGE();
        chase::linalg::internal::cpu::symOrHermMatrix(uplo, N_, Hmat_->cpu_data(), Hmat_->cpu_ld());
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
            chase::linalg::internal::cuda::init_random_vectors(Vec1_.data(), Vec1_.ld() * Vec1_.cols());
        }

        chase::linalg::internal::cuda::t_lacpy('A', 
                                          Vec1_.rows(), 
                                          Vec1_.cols(), 
                                          Vec1_.data(), 
                                          Vec1_.ld(),
                                          Vec2_.data(), 
                                          Vec2_.ld());  
        
        Hmat_->H2D();
    }

    void Lanczos(std::size_t M, chase::Base<T>* upperb) override
    {
        SCOPED_NVTX_RANGE();
        chase::linalg::internal::cuda::lanczos(cublasH_,
                                               M, 
                                               Hmat_,
                                               Vec1_,
                                               upperb);    
    }

    void Lanczos(std::size_t M, std::size_t numvec, chase::Base<T>* upperb,
                         chase::Base<T>* ritzv, chase::Base<T>* Tau, chase::Base<T>* ritzV) override
    {    
        SCOPED_NVTX_RANGE();
        chase::linalg::internal::cuda::lanczos(cublasH_,
                                               M, 
                                               numvec,
                                               Hmat_,
                                               Vec1_,
                                               upperb,
                                               ritzv,
                                               Tau,
                                               ritzV);
    }

    void LanczosDos(std::size_t idx, std::size_t m, T* ritzVc) override
    {
        SCOPED_NVTX_RANGE();
        T alpha = T(1.0);
        T beta = T(0.0);
        
        CHECK_CUBLAS_ERROR(cublasSetMatrix(m, 
                                        idx, 
                                        sizeof(T), 
                                        ritzVc, 
                                        m, 
                                        A_.data(), 
                                        A_.ld()));

        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(cublasH_, 
                                                                CUBLAS_OP_N, 
                                                                CUBLAS_OP_N, 
                                                                Vec1_.rows(), 
                                                                idx, 
                                                                m, 
                                                                &alpha,
                                                                Vec1_.data(), 
                                                                Vec1_.ld(), 
                                                                A_.data(), 
                                                                A_.ld(), 
                                                                &beta, 
                                                                Vec2_.data(), 
                                                                Vec2_.ld()));
        
        chase::linalg::internal::cuda::t_lacpy('A', 
                                    Vec2_.rows(), 
                                    m, 
                                    Vec2_.data(), 
                                    Vec2_.ld(),
                                    Vec1_.data(), 
                                    Vec1_.ld());  

    }

    void Shift(T c, bool isunshift = false) override
    {   
        SCOPED_NVTX_RANGE();
        chase::linalg::internal::cuda::shiftDiagonal(Hmat_, std::real(c));
/*
#ifdef ENABLE_MIXED_PRECISION
        //mixed precision
        if constexpr (std::is_same<T, double>::value || std::is_same<T, std::complex<double>>::value)
        {
            auto min = *std::min_element(resid_.cpu_data() + locked_, resid_.cpu_data() + nev_);
            
            if(min > 1e-3)
            {
                if(isunshift)
                {
                    if(Hmat_->isSinglePrecisionEnabled())
                    {
                        Hmat_->disableSinglePrecision();
                    }
                    if(Vec1_.isSinglePrecisionEnabled())
                    {
                        Vec1_.disableSinglePrecision(true);
                    }   
                    if(Vec2_.isSinglePrecisionEnabled())
                    {
                        Vec2_.disableSinglePrecision();
                    }                                       
                }else
                {
                    std::cout << "Enable Single Precision in Filter" << std::endl;
                    Hmat_->enableSinglePrecision();
                    Vec1_.enableSinglePrecision();
                    Vec2_.enableSinglePrecision();  
                }
  
            }else
            {
                if(Hmat_->isSinglePrecisionEnabled())
                {
                    Hmat_->disableSinglePrecision();
                }
                if(Vec1_.isSinglePrecisionEnabled())
                {
                    Vec1_.disableSinglePrecision(true);
                }   
                if(Vec2_.isSinglePrecisionEnabled())
                {
                    Vec2_.disableSinglePrecision();
                }  
            }
        }
#endif
*/
    }
    
    void HEMM(std::size_t block, T alpha, T beta, std::size_t offset) override
    {     
        SCOPED_NVTX_RANGE();
/*#ifdef ENABLE_MIXED_PRECISION
        if constexpr (std::is_same<T, double>::value || std::is_same<T, std::complex<double>>::value)
        {
            using singlePrecisionT = typename chase::ToSinglePrecisionTrait<T>::Type;
            auto min = *std::min_element(resid_.cpu_data() + locked_, resid_.cpu_data() + nev_);
            if(min > 1e-3)
            {
                auto Hmat_sp = Hmat_->matrix_sp();
                auto Vec1_sp = Vec1_.matrix_sp();
                auto Vec2_sp = Vec2_.matrix_sp();
                singlePrecisionT alpha_sp = static_cast<singlePrecisionT>(alpha);
                singlePrecisionT beta_sp = static_cast<singlePrecisionT>(beta);

                CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(cublasH_,
                                                                        CUBLAS_OP_N,
                                                                        CUBLAS_OP_N,
                                                                        Hmat_sp->rows(),
                                                                        block,
                                                                        Hmat_sp->cols(),
                                                                        &alpha_sp,
                                                                        Hmat_sp->data(),
                                                                        Hmat_sp->ld(),
                                                                        Vec1_sp->data() + offset * Vec1_sp->ld() + locked_ * Vec1_sp->ld(),
                                                                        Vec1_sp->ld(),
                                                                        &beta_sp,
                                                                        Vec2_sp->data() + offset * Vec2_sp->ld() + locked_ * Vec2_sp->ld(),
                                                                        Vec2_sp->ld()));
            }
            else
            {
                CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(cublasH_,
                                                                        CUBLAS_OP_N,
                                                                        CUBLAS_OP_N,
                                                                        Hmat_->rows(),
                                                                        block,
                                                                        Hmat_->cols(),
                                                                        &alpha,
                                                                        Hmat_->data(),
                                                                        Hmat_->ld(),
                                                                        Vec1_.data() + offset * Vec1_.ld() + locked_ * Vec1_.ld(),
                                                                        Vec1_.ld(),
                                                                        &beta,
                                                                        Vec2_.data() + offset * Vec2_.ld() + locked_ * Vec2_.ld(),
                                                                        Vec2_.ld()));
            }
            
        }
        else
#endif
*/   
        {
            CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(cublasH_,
                                                                    CUBLAS_OP_N,
                                                                    CUBLAS_OP_N,
                                                                    Hmat_->rows(),
                                                                    block,
                                                                    Hmat_->cols(),
                                                                    &alpha,
                                                                    Hmat_->data(),
                                                                    Hmat_->ld(),
                                                                    Vec1_.data() + offset * Vec1_.ld() + locked_ * Vec1_.ld(),
                                                                    Vec1_.ld(),
                                                                    &beta,
                                                                    Vec2_.data() + offset * Vec2_.ld() + locked_ * Vec2_.ld(),
                                                                    Vec2_.ld()));
        }

        Vec1_.swap(Vec2_);
    }

    void QR(std::size_t fixednev, chase::Base<T> cond) override
    {
        SCOPED_NVTX_RANGE();
        chase::linalg::internal::cuda::t_lacpy('A', 
                                                Vec2_.rows(), 
                                                locked_, 
                                                Vec1_.data(), 
                                                Vec1_.ld(),
                                                Vec2_.data(), 
                                                Vec2_.ld());   

        int disable = config_.DoCholQR() ? 0 : 1;
        char* cholddisable = getenv("CHASE_DISABLE_CHOLQR");
        if (cholddisable) {
            disable = std::atoi(cholddisable);
        }

        Base<T> cond_threshold_upper = (sizeof(Base<T>) == 8) ? 1e8 : 1e4;
        Base<T> cond_threshold_lower = (sizeof(Base<T>) == 8) ? 2e1 : 1e1;

        char* chol_threshold = getenv("CHASE_CHOLQR1_THLD");
        if (chol_threshold)
        {
            cond_threshold_lower = std::atof(chol_threshold);
        }

        //int display_bounds = 0;
        //char* display_bounds_env = getenv("CHASE_DISPLAY_BOUNDS");
        //if (display_bounds_env)
        //{
        //    display_bounds = std::atoi(display_bounds_env);
        //}

        if (disable == 1)
        {
            chase::linalg::internal::cuda::houseHoulderQR(cusolverH_,
                                                        Vec1_,
                                                        d_return_,
                                                        devInfo_,
                                                        d_work_,
                                                        lwork_);
        }
        else
        {
#ifdef CHASE_OUTPUT
        std::cout << std::setprecision(2) << "cond(V): " << cond << std::endl;
#endif
            //if (display_bounds != 0)
            //{
            //  dla_->estimated_cond_evaluator(locked_, cond);
            //}
            int info = 1;

            if (cond > cond_threshold_upper)
            {
                
                info = chase::linalg::internal::cuda::shiftedcholQR2(cublasH_,
                                                              cusolverH_,
                                                              Vec1_,
                                                              d_work_,
                                                              lwork_,
                                                              &A_);
            }
            else if(cond < cond_threshold_lower)
            {
                info = chase::linalg::internal::cuda::cholQR1(cublasH_,
                                                              cusolverH_,
                                                              Vec1_,
                                                              d_work_,
                                                              lwork_,
                                                              &A_); 
            }
            else
            {
                info = chase::linalg::internal::cuda::cholQR2(cublasH_,
                                                              cusolverH_,
                                                              Vec1_,
                                                              d_work_,
                                                              lwork_,
                                                              &A_);
  
            }

            if (info != 0)
            {
#ifdef CHASE_OUTPUT
                std::cout << "CholeskyQR doesn't work, Househoulder QR will be used." << std::endl;
#endif
                chase::linalg::internal::cuda::houseHoulderQR(cusolverH_,
                                                            Vec1_,
                                                            d_return_,
                                                            devInfo_,
                                                            d_work_,
                                                            lwork_);
            }
        }

        chase::linalg::internal::cuda::t_lacpy('A', 
                                                Vec1_.rows(), 
                                                locked_, 
                                                Vec2_.data(), 
                                                Vec2_.ld(),
                                                Vec1_.data(), 
                                                Vec1_.ld());    
    }

    void RR(chase::Base<T>* ritzv, std::size_t block) override
    {
        SCOPED_NVTX_RANGE();
        std::size_t locked = (nev_ + nex_) - block;
        chase::linalg::internal::cuda::rayleighRitz(cublasH_,
                                                    cusolverH_,
                                                    Hmat_,
                                                    Vec1_,
                                                    Vec2_,
                                                    ritzvs_,
                                                    locked,
                                                    block,
                                                    devInfo_,
                                                    d_work_,
                                                    lwork_,
                                                    &A_);
 
        Vec1_.swap(Vec2_);
    }
    
    void Sort(chase::Base<T> * ritzv, chase::Base<T> * residLast, chase::Base<T> * resid) override
    {

    }

    void Resd(chase::Base<T>* ritzv, chase::Base<T>* resd, std::size_t fixednev) override
    {
        SCOPED_NVTX_RANGE();
        std::size_t unconverged = (nev_ + nex_) - fixednev;
        chase::linalg::internal::cuda::residuals(cublasH_,
                                                 Hmat_,
                                                 Vec1_,
                                                 ritzvs_.data(),
                                                 resid_.data(),
                                                 fixednev,
                                                 unconverged,
                                                 &Vec2_);
        CHECK_CUDA_ERROR(cudaMemcpy(resd, 
                                    resid_.data() + fixednev, 
                                    unconverged * sizeof(chase::Base<T>),
                                    cudaMemcpyDeviceToHost));      
    }

    void Swap(std::size_t i, std::size_t j) override
    {
        SCOPED_NVTX_RANGE();
        chase::linalg::internal::cuda::t_lacpy('A',
                                               Vec1_.rows(),
                                               1,
                                               Vec1_.data() + i * Vec1_.ld(),
                                               Vec1_.ld(),
                                               tmp_,
                                               N_);
        chase::linalg::internal::cuda::t_lacpy('A',
                                               Vec1_.rows(),
                                               1,
                                               Vec1_.data() + j * Vec1_.ld(),
                                               Vec1_.ld(),
                                               Vec1_.data() + i * Vec1_.ld(),
                                               Vec1_.ld());
        chase::linalg::internal::cuda::t_lacpy('A',
                                               Vec1_.rows(),
                                               1,
                                               tmp_,
                                               N_,
                                               Vec1_.data() + j * Vec1_.ld(),
                                               Vec1_.ld());                                       
    }

    void Lock(std::size_t new_converged) override
    {
        locked_ += new_converged;
    }
 
    void End() override 
    {
        SCOPED_NVTX_RANGE();
        Vec1_.D2H();
    }
        
private:
    std::size_t N_;              /**< Size of the matrix. */
    T *H_;                       /**< Pointer to the matrix \( H \). */
    T *V1_;                      /**< Pointer to the matrix \( V_1 \). */
    std::size_t ldh_;            /**< Leading dimension of matrix \( H \). */
    std::size_t ldv_;            /**< Leading dimension of matrix \( V_1 \). */
    chase::Base<T> *ritzv_;      /**< Pointer to the Ritz values vector. */
    std::size_t nev_;            /**< Number of eigenvalues to compute. */
    std::size_t nex_;            /**< Number of extra vectors. */
    std::size_t nevex_;          /**< Total number of eigenvalues and extra vectors. */

    T *tmp_;                     /**< Temporary buffer for GPU computations. */
    bool is_sym_;                ///< Flag for matrix symmetry.
    bool is_pseudoHerm_;                ///< Flag for matrix symmetry.

    std::size_t locked_;         /**< Counter for the number of converged eigenvalues. */
    MatrixType *Hmat_;       /**< GPU matrix \( H \). */
    chase::matrix::Matrix<T, chase::platform::GPU> Vec1_;       /**< GPU matrix \( V_1 \). */
    chase::matrix::Matrix<T, chase::platform::GPU> Vec2_;       /**< GPU matrix for additional vectors. */
    chase::matrix::Matrix<T, chase::platform::GPU> A_;          /**< GPU matrix for computations. */
    chase::matrix::Matrix<chase::Base<T>, chase::platform::GPU> ritzvs_; /**< GPU matrix for Ritz values. */
    chase::matrix::Matrix<chase::Base<T>, chase::platform::GPU> resid_;  /**< GPU matrix for residuals. */
    chase::ChaseConfig<T> config_;   /**< Configuration object for the Chase algorithm. */

    cudaStream_t stream_;          /**< CUDA stream for asynchronous operations. */
    cublasHandle_t cublasH_;       /**< CUBLAS handle for GPU-accelerated linear algebra operations. */
    cusolverDnHandle_t cusolverH_; /**< CUSOLVER handle for eigenvalue computations. */

    int* devInfo_;                 /**< Pointer to device information for CUDA operations. */
    T* d_return_;                  /**< Pointer to device buffer for eigenvalues. */
    T* d_work_;                    /**< Pointer to work buffer for matrix operations. */
    int lwork_ = 0;                    /**< Workspace size for matrix operations. */
}; 

}    
}

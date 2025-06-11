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
#include "external/lapackpp/lapackpp.hpp"
#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"
#include "grid/mpiGrid2D.hpp"
#include "linalg/internal/cuda/cuda_kernels.hpp"
#include "linalg/internal/cuda_aware_mpi/cuda_mpi_kernels.hpp"
#ifdef HAS_NCCL
#include "linalg/internal/nccl/nccl_kernels.hpp"
#endif
#ifdef HAS_SCALAPACK
#include "external/scalapackpp/scalapackpp.hpp"
#endif
#include "algorithm/types.hpp"

#include "Impl/config/config.hpp"
#include "Impl/chase_gpu/nvtx.hpp"

#include "../../linalg/internal/typeTraits.hpp"

using namespace chase::linalg;

template <typename Backend>
struct MGPUKernelNamspaceSelector;

// Specialization for MPI backend
template <>
struct MGPUKernelNamspaceSelector<chase::grid::backend::MPI> {
    using type = chase::linalg::internal::cuda_mpi; // Use the struct as a type
    template <typename GridType>
    static auto getColCommunicator(GridType* grid) {
        return grid->get_col_comm(); // MPI communicator
    }    
};

#ifdef HAS_NCCL
// Specialization for NCCL backend
template <>
struct MGPUKernelNamspaceSelector<chase::grid::backend::NCCL> {
    using type = chase::linalg::internal::cuda_nccl; // Use the struct as a type
    template <typename GridType>
    static auto getColCommunicator(GridType* grid) {
        return grid->get_nccl_col_comm(); // NCCL communicator
    }    
};

#endif
namespace chase
{
namespace Impl
{
/**
 * @page pChASEGPU
 * 
 * @section intro_sec Introduction
 * This class implements the GPU-based parallel version of the Chase algorithm using NVIDIA NCCL 
 * (NVIDIA Collective Communication Library) for efficient multi-GPU communication. The class inherits 
 * from `ChaseBase` and is designed for solving large-scale generalized eigenvalue problems on 
 * distributed GPU environments. It operates on matrix and multi-vector data types, supporting 
 * configurations where matrix data and eigenvectors are mapped to a shared MPI grid for distributed 
 * computation.
 * 
 * @section constructor_sec Constructors and Destructor
 * The constructor for `pChASEGPU` initializes essential components, including the Hamiltonian matrix, 
 * input multi-vectors, result multi-vectors, and configurations for NVIDIA libraries such as cuBLAS 
 * and cuSolver. It also sets up memory allocations for GPU-based operations and verifies that 
 * the matrix is square and that the matrix and eigenvectors share the same MPI grid.
 * 
 * @section members_sec Private Members
 * The private members include matrix and multi-vector data, configuration settings, and GPU-specific 
 * resources like cuBLAS and cuSolver handles. Additional private members manage MPI-related properties, 
 * including process rank and grid coordinates, as well as device memory allocations used for intermediate 
 * computations.
 */    

/**
* @brief GPU-based parallel Chase algorithm with NCCL.
* 
* This class provides an implementation of the Chase algorithm for solving generalized eigenvalue 
* problems on GPU-based platforms. It leverages NVIDIA's NCCL for efficient GPU-to-GPU communication 
* and cuSolver and cuBLAS for GPU-based linear algebra operations. Designed for distributed GPU 
* environments, it operates on matrix and multi-vector data types with MPI parallelism.
* 
* @tparam MatrixType The matrix type, such as `chase::distMatrix::BlockBlockMatrix` or 
*                    `chase::distMatrix::BlockCyclicMatrix`.
* @tparam InputMultiVectorType The input multi-vector type, typically used for storing eigenvectors.
*/ 
template <typename MatrixType, typename InputMultiVectorType, typename BackendType = chase::grid::backend::NCCL>
class pChASEGPU : public ChaseBase<typename MatrixType::value_type>
{
    using T = typename MatrixType::value_type;
    using ResultMultiVectorType = typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type;
    //using backend = typename chase::grid::backend::MPI;
    using backend = BackendType;
//    using kernelNamespace = typename MGPUKernelNamspaceSelector<backend>::type;
    using kernelNamespace = typename MGPUKernelNamspaceSelector<backend>::type;
    
public:
    /**
     * @brief Constructs the `pChASEGPU` object.
     * 
     * Initializes the Chase algorithm parameters, matrix data, multi-vector data, and GPU-specific 
     * resources. Sets up memory allocations for intermediate data and configures cuBLAS and cuSolver 
     * for GPU computation.
     * 
     * @param nev Number of eigenvalues to compute.
     * @param nex Number of extra vectors for iterative refinement.
     * @param H Pointer to the Hamiltonian matrix.
     * @param V Pointer to the input multi-vector (typically for eigenvectors).
     * @param ritzv Pointer to the Ritz values, used for storing eigenvalues.
     */
    pChASEGPU(std::size_t nev,
                 std::size_t nex,
                 MatrixType *H,
                 InputMultiVectorType *V,
                 chase::Base<T> *ritzv): nev_(nev), nex_(nex), nevex_(nev + nex), config_(H->g_rows(), nev, nex), N_(H->g_rows())
    {
        SCOPED_NVTX_RANGE();

        if(H->g_rows() != H->g_cols())
        {
            throw std::runtime_error("ChASE requires the matrix solved to be squared");
        }

        if( H->getMpiGrid() != V->getMpiGrid())
        {   
            throw  std::runtime_error("ChASE requires the matrix and eigenvectors mapped to same MPI grid");
        }    

        N_ = H->g_rows();
        Hmat_ = H;
        V1_ = V;
        V2_ = V1_->template clone2<InputMultiVectorType>();
        W1_ = V1_->template clone2<ResultMultiVectorType>();
        W2_ = V1_->template clone2<ResultMultiVectorType>();

        ritzv_ = std::make_unique<chase::distMatrix::RedundantMatrix<chase::Base<T>, chase::platform::GPU>>(nevex_, 1, nevex_, ritzv, Hmat_->getMpiGrid_shared_ptr());
        resid_ = std::make_unique<chase::distMatrix::RedundantMatrix<chase::Base<T>, chase::platform::GPU>>(nevex_, 1, Hmat_->getMpiGrid_shared_ptr());

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

        if constexpr (std::is_same<MatrixType, chase::distMatrix::QuasiHermitianBlockBlockMatrix<T,chase::platform::GPU>>::value ||
		      std::is_same<MatrixType, chase::distMatrix::QuasiHermitianBlockCyclicMatrix<T,chase::platform::GPU>>::value )

	{
	
		is_sym_ = false;
		is_pseudoHerm_ = true;
        
		A_ = std::make_unique<chase::distMatrix::RedundantMatrix<T, chase::platform::GPU>>(2*nevex_, nevex_, Hmat_->getMpiGrid_shared_ptr());
		//A_ = std::make_unique<chase::distMatrix::RedundantMatrix<T, chase::platform::GPU>>(3*nevex_, nevex_, Hmat_->getMpiGrid_shared_ptr());
        	
		CHECK_CUSOLVER_ERROR(chase::linalg::cusolverpp::cusolverDnTheevd_bufferSize(
                                                            cusolverH_, 
                                                            CUSOLVER_EIG_MODE_VECTOR, 
                                                            CUBLAS_FILL_MODE_LOWER,
                                                            nevex_, 
                                                            A_->l_data(), 
                                                            A_->l_ld(), 
                                                            ritzv_->l_data(), 
							    &lwork_heevd));
/*
#ifdef XGEEV_EXISTS
		CHECK_CUSOLVER_ERROR(cusolverDnCreateParams(&params_));

		std::size_t temp_ldwork = 0;
		std::size_t temp_lhwork = 0;

            	CHECK_CUSOLVER_ERROR(
                	chase::linalg::cusolverpp::cusolverDnTgeev_bufferSize(
                    		cusolverH_, params_, CUSOLVER_EIG_MODE_NOVECTOR,
                    		CUSOLVER_EIG_MODE_VECTOR, nevex_, A_->l_data(), A_->l_ld(),
                    		V2_->l_data(), NULL, 1, V1_->l_data(), V1_->l_ld(),
                    		&temp_ldwork, &temp_lhwork));

		lwork_heevd = (int)temp_ldwork;
		lhwork_ = (int)temp_lhwork;

		h_work_ = std::unique_ptr<T[]>(new T[lhwork_]);
#endif
*/
	}
	else
	{
	
		is_sym_ = true;
		is_pseudoHerm_ = false;

		A_ = std::make_unique<chase::distMatrix::RedundantMatrix<T, chase::platform::GPU>>(nevex_, nevex_, Hmat_->getMpiGrid_shared_ptr());

        	CHECK_CUSOLVER_ERROR(chase::linalg::cusolverpp::cusolverDnTheevd_bufferSize(
                                                            cusolverH_, 
                                                            CUSOLVER_EIG_MODE_VECTOR, 
                                                            CUBLAS_FILL_MODE_LOWER,
                                                            nevex_, 
                                                            A_->l_data(), 
                                                            A_->l_ld(), 
                                                            ritzv_->l_data(), 
                                                            &lwork_heevd));
	}
        	
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

        if(nevex_ * (nevex_ + 1) / 2 > lwork_)
        {
        	lwork_ = nevex_ * (nevex_ + 1) / 2;
       	}

        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_work_, sizeof(T) * lwork_));    

        CHECK_CUDA_ERROR(cudaMalloc((void**)&states_,
                             sizeof(curandStatePhilox4_32_10_t) * (256 * 32)));

        std::vector<std::size_t> diag_xoffs, diag_yoffs;

        if constexpr(std::is_same<MatrixType, chase::distMatrix::BlockBlockMatrix<T, chase::platform::GPU>>::value || 
		     std::is_same<MatrixType, chase::distMatrix::QuasiHermitianBlockBlockMatrix<T, chase::platform::GPU>>::value )
        {
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
        }else if constexpr(std::is_same<MatrixType, chase::distMatrix::BlockCyclicMatrix<T, chase::platform::GPU>>::value ||
		     	   std::is_same<MatrixType, chase::distMatrix::QuasiHermitianBlockCyclicMatrix<T, chase::platform::GPU>>::value )
        {
            auto m_contiguous_global_offs = Hmat_->m_contiguous_global_offs();
            auto n_contiguous_global_offs = Hmat_->n_contiguous_global_offs();
            auto m_contiguous_local_offs = Hmat_->m_contiguous_local_offs();
            auto n_contiguous_local_offs = Hmat_->n_contiguous_local_offs();
            auto m_contiguous_lens = Hmat_->m_contiguous_lens();
            auto n_contiguous_lens = Hmat_->n_contiguous_lens();
            auto mblocks = Hmat_->mblocks();
            auto nblocks = Hmat_->nblocks();

            for (std::size_t j = 0; j < nblocks; j++)
            {
                for (std::size_t i = 0; i < mblocks; i++)
                {
                    for (std::size_t q = 0; q < n_contiguous_lens[j]; q++)
                    {
                        for (std::size_t p = 0; p < m_contiguous_lens[i]; p++)
                        {
                            if (q + n_contiguous_global_offs[j] == p + m_contiguous_global_offs[i])
                            {
                                diag_xoffs.push_back(p + m_contiguous_local_offs[i]);
                                diag_yoffs.push_back(q + n_contiguous_local_offs[j]);

                            }
                        }
                    }
                }
            }
        }else
        {
            throw std::runtime_error("Matrix type is not supported");
        }
        
        diag_cnt = diag_xoffs.size();

        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_diag_xoffs, sizeof(std::size_t) * diag_cnt));    
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_diag_yoffs, sizeof(std::size_t) * diag_cnt));    

        CHECK_CUDA_ERROR(cudaMemcpy(d_diag_xoffs, diag_xoffs.data(), sizeof(std::size_t) * diag_cnt , cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_diag_yoffs, diag_yoffs.data(), sizeof(std::size_t) * diag_cnt , cudaMemcpyHostToDevice));
 
        if constexpr (std::is_same<T, std::complex<float>>::value)
        {
            if(!A_->isDoublePrecisionEnabled())
            {
                A_->enableDoublePrecision();
            }

            if(!V1_->isDoublePrecisionEnabled())
            {
                V1_->enableDoublePrecision();
            }            
        }    
    }

    pChASEGPU(const pChASEGPU&) = delete;

    ~pChASEGPU() 
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
    int get_rank() { return my_rank_; }

    void loadProblemFromFile(std::string filename)
    {
       SCOPED_NVTX_RANGE();
       Hmat_->readFromBinaryFile(filename);
    }

    void saveProblemToFile(std::string filename)
    {
        Hmat_->saveToBinaryFile(filename);
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

        is_sym_ = kernelNamespace::checkSymmetryEasy(cublasH_, *Hmat_);  
        return is_sym_;
    }

    bool isSym() override {return is_sym_;} 
    
    bool checkPseudoHermicityEasy() override
    {
        SCOPED_NVTX_RANGE();
	//is_pseudoHerm_= 0;
        return is_pseudoHerm_;
    }
    
    bool isPseudoHerm() override { return is_pseudoHerm_; }

    void symOrHermMatrix(char uplo) override
    {
        SCOPED_NVTX_RANGE();

        kernelNamespace::symOrHermMatrix(uplo, *Hmat_);   
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

        kernelNamespace::lanczos_dispatch(cublasH_,
                                              m, 
                                              *Hmat_, 
                                              *V1_, 
                                              upperb);   
    }

    void Lanczos(std::size_t M, std::size_t numvec, chase::Base<T>* upperb,
                         chase::Base<T>* ritzv, chase::Base<T>* Tau, chase::Base<T>* ritzV) override
    {
        SCOPED_NVTX_RANGE();

        kernelNamespace::lanczos_dispatch(cublasH_,
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
        
        kernelNamespace::shiftDiagonal(*Hmat_, d_diag_xoffs, d_diag_yoffs, diag_cnt, std::real(c));

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
                    kernelNamespace::template MatrixMultiplyMultiVectors<singlePrecisionT>(cublasH_,
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
                    kernelNamespace::template MatrixMultiplyMultiVectors<singlePrecisionT>(cublasH_,
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
                    kernelNamespace::MatrixMultiplyMultiVectors(cublasH_,
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
                    kernelNamespace::MatrixMultiplyMultiVectors(cublasH_,
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
                kernelNamespace::MatrixMultiplyMultiVectors(cublasH_,
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
                kernelNamespace::MatrixMultiplyMultiVectors(cublasH_,
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
        
        if constexpr (std::is_same<MatrixType, chase::distMatrix::QuasiHermitianBlockBlockMatrix<T,chase::platform::GPU>>::value ||
                std::is_same<MatrixType, chase::distMatrix::QuasiHermitianBlockCyclicMatrix<T,chase::platform::GPU>>::value )
        {	
            kernelNamespace::flipLowerHalfMatrixSign(*V1_, 0, locked_);
        }

        if (disable == 1)
        {
#ifdef HAS_SCALAPACK
            kernelNamespace::houseHoulderQR(*V1_);
#else
            throw std::runtime_error("For ChASE-MPI, distributed Householder QR requires ScaLAPACK, which is not detected\n");
#endif
        }else if(nevex_ >= MINIMAL_N_INVOKE_MODIFIED_GRAM_SCHMIDT_QR_GPU_NCCL)
        {
            info = kernelNamespace::modifiedGramSchmidtCholQR(cublasH_,
                                                            cusolverH_,
                                                            V1_->l_rows(), 
                                                            V1_->l_cols(), 
                                                            locked_,
                                                            V1_->l_data(),  
                                                            V1_->l_ld(), 
                                                            MGPUKernelNamspaceSelector<backend>::getColCommunicator(V1_->getMpiGrid()),
                                                            //V1_->getMpiGrid()->get_nccl_col_comm(),
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


                info = kernelNamespace::shiftedcholQR2(cublasH_,
                                                                cusolverH_,
                                                                V1_->g_rows(),
                                                                V1_->l_rows(), 
                                                                V1_->l_cols(), 
                                                                V1_->l_data(),  
                                                                V1_->l_ld(), 
                                                                //V1_->getMpiGrid()->get_nccl_col_comm(),
                                                                MGPUKernelNamspaceSelector<backend>::getColCommunicator(V1_->getMpiGrid()),
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
                    kernelNamespace::houseHoulderQR(*V1_);
#else
                    throw std::runtime_error("For ChASE-MPI, distributed Householder QR requires ScaLAPACK, which is not detected\n");
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
                if constexpr (std::is_same<T, std::complex<float>>::value)
                {
                    V1_->copyTo();

                    auto V1_d = V1_->getDoublePrecisionMatrix();
                    auto A_d = A_->getDoublePrecisionMatrix();
                    std::complex<double> *d_work_d ;
                    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_work_d, sizeof(std::complex<double>) * lwork_));
                    info = kernelNamespace::shiftedcholQR2(cublasH_,
                                                                cusolverH_,
                                                                V1_->g_rows(),
                                                                V1_->l_rows(), 
                                                                V1_->l_cols(), 
                                                                V1_d->l_data(),  
                                                                V1_->l_ld(), 
                                                                //V1_->getMpiGrid()->get_nccl_col_comm(),
                                                                MGPUKernelNamspaceSelector<backend>::getColCommunicator(V1_->getMpiGrid()),
                                                                d_work_d,
                                                                lwork_,
                                                                A_d->l_data()); 
                    V1_->copyback();
                    CHECK_CUDA_ERROR(cudaFree(d_work_d));
                }
                else
                {
                info = kernelNamespace::shiftedcholQR2(cublasH_,
                                                                cusolverH_,
                                                                V1_->g_rows(),
                                                                V1_->l_rows(), 
                                                                V1_->l_cols(), 
                                                                V1_->l_data(),  
                                                                V1_->l_ld(), 
                                                                //V1_->getMpiGrid()->get_nccl_col_comm(),
                                                                MGPUKernelNamspaceSelector<backend>::getColCommunicator(V1_->getMpiGrid()),
                                                                d_work_,
                                                                lwork_,
                                                                A_->l_data()); 
                }                                               
            }
            else if(cond < cond_threshold_lower)
            {
                /*if constexpr (std::is_same<T, std::complex<float>>::value)
                {
                    if(V1_->isDoublePrecisionEnabled())
                    {
                        V1_->disableDoublePrecision();
                    }
                    if(A_->isDoublePrecisionEnabled())
                    {
                        A_->disableDoublePrecision();
                    }
                }
                */

                /*if constexpr (std::is_same<T, std::complex<float>>::value)
                {
                    V1_->enableDoublePrecision();
                    A_->enableDoublePrecision();
                    auto V1_d = V1_->getDoublePrecisionMatrix();
                    auto A_d = A_->getDoublePrecisionMatrix();
                    info = kernelNamespace::cholQR1(cublasH_,
                                                                cusolverH_,
                                                                V1_->l_rows(), 
                                                                V1_->l_cols(), 
                                                                V1_d->l_data(),  
                                                                V1_->l_ld(), 
                                                                //V1_->getMpiGrid()->get_nccl_col_comm(),
                                                                MGPUKernelNamspaceSelector<backend>::getColCommunicator(V1_->getMpiGrid()),
                                                                reinterpret_cast<std::complex<double>*>(d_work_),
                                                                lwork_,
                                                                A_d->l_data()); 
                    V1_->disableDoublePrecision(true);
                    A_->disableDoublePrecision();
                }else*/
                {
                    info = kernelNamespace::cholQR1(cublasH_,
                                                                    cusolverH_,
                                                                    V1_->l_rows(), 
                                                                    V1_->l_cols(), 
                                                                    V1_->l_data(),  
                                                                    V1_->l_ld(), 
                                                                    //V1_->getMpiGrid()->get_nccl_col_comm(),
                                                                    MGPUKernelNamspaceSelector<backend>::getColCommunicator(V1_->getMpiGrid()),
                                                                    d_work_,
                                                                    lwork_,
                                                                    A_->l_data());  
                }

                                                            
            }
            else
            {   
                /*if constexpr (std::is_same<T, std::complex<float>>::value)
                {
                    if(V1_->isDoublePrecisionEnabled())
                    {
                        V1_->disableDoublePrecision();
                    }
                    if(A_->isDoublePrecisionEnabled())
                    {
                        A_->disableDoublePrecision();
                    }
                }*/

                /*if constexpr (std::is_same<T, std::complex<float>>::value)
                {
                    V1_->enableDoublePrecision();
                    A_->enableDoublePrecision();
                    auto V1_d = V1_->getDoublePrecisionMatrix();
                    auto A_d = A_->getDoublePrecisionMatrix();
                    info = kernelNamespace::cholQR2(cublasH_,
                                                                    cusolverH_,
                                                                    V1_->l_rows(), 
                                                                    V1_->l_cols(), 
                                                                    V1_d->l_data(),  
                                                                    V1_->l_ld(), 
                                                                    //V1_->getMpiGrid()->get_nccl_col_comm(),
                                                                    MGPUKernelNamspaceSelector<backend>::getColCommunicator(V1_->getMpiGrid()),
                                                                    reinterpret_cast<std::complex<double>*>(d_work_),
                                                                    lwork_,
                                                                    A_d->l_data()); 
                    V1_->disableDoublePrecision(true);
                    A_->disableDoublePrecision();
                }else*/
                {
                    info = kernelNamespace::cholQR2(cublasH_,
                                                                    cusolverH_,
                                                                    V1_->l_rows(), 
                                                                    V1_->l_cols(), 
                                                                    V1_->l_data(),  
                                                                    V1_->l_ld(), 
                                                                    //V1_->getMpiGrid()->get_nccl_col_comm(),
                                                                    MGPUKernelNamspaceSelector<backend>::getColCommunicator(V1_->getMpiGrid()),
                                                                    d_work_,
                                                                    lwork_,
                                                                    A_->l_data()); 
                }                             
            }

            if (info != 0)
            {
#ifdef HAS_SCALAPACK
#ifdef CHASE_OUTPUT
                if(my_rank_ == 0){
                    std::cout << "CholeskyQR doesn't work, Househoulder QR will be used." << std::endl;
                }
#endif
                kernelNamespace::houseHoulderQR(*V1_);
#else
                throw std::runtime_error("For ChASE-MPI, distributed Householder QR requires ScaLAPACK, which is not detected\n");
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

        if constexpr (std::is_same<MatrixType, chase::distMatrix::QuasiHermitianBlockBlockMatrix<T,chase::platform::GPU>>::value ||
		      std::is_same<MatrixType, chase::distMatrix::QuasiHermitianBlockCyclicMatrix<T,chase::platform::GPU>>::value )

	{
        	kernelNamespace::quasi_hermitian_rayleighRitz_v2(cublasH_,
                                                   cusolverH_,
						   params_,
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
						   //h_work_.get(),
				   		   //lhwork_,
                                                   A_.get());
	}
	else
	{

        	kernelNamespace::rayleighRitz(cublasH_,
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
                                                   A_.get());
	}

        chase::linalg::internal::cuda::t_lacpy('A',
                                         V2_->l_rows(),
                                         block,
                                         V1_->l_data() + locked_ * V1_->l_ld(),
                                         V1_->l_ld(),
                                         V2_->l_data() + locked_ * V2_->l_ld(),
                                         V2_->l_ld());       
    }
    
    void Sort(chase::Base<T> * ritzv, chase::Base<T> * residLast, chase::Base<T> * resid) override
    {

    }

    void Resd(chase::Base<T>* ritzv, chase::Base<T>* resd, std::size_t fixednev) override 
    {
        SCOPED_NVTX_RANGE();
        
        kernelNamespace::residuals(cublasH_,
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
    /**
     * @enum NextOp
     * @brief Represents the next operation to be performed in the computation.
     */
    enum NextOp
    {
        cAb, /**< Represents the operation `c = A * b`. */
        bAc  /**< Represents the operation `b = A * c`. */
    };
    
    NextOp next_; /**< Holds the next operation in the computation sequence. */

    bool is_sym_; /**< Indicates whether the matrix is symmetric. */
    bool is_pseudoHerm_; /**< Indicates whether the matrix is pseudo-hermitian. */
    
    std::size_t nev_; /**< Number of eigenvalues to compute. */
    std::size_t nex_; /**< Number of additional vectors for iterative refinement. */
    std::size_t nevex_; /**< Total number of vectors (nev + nex) used in the algorithm. */
    std::size_t locked_; /**< Count of locked vectors in the eigenvalue problem. */
    
    std::size_t N_; /**< Dimension of the square matrix. */

    int nprocs_; /**< Total number of MPI processes. */
    int my_rank_; /**< Rank of the current MPI process. */
    int *coords_; /**< Pointer to coordinates of the current process in the MPI grid. */
    int *dims_; /**< Pointer to dimensions of the MPI grid. */

    MatrixType *Hmat_; /**< Pointer to the Hamiltonian matrix. */
    InputMultiVectorType *V1_; /**< Pointer to the first input multi-vector for eigenvectors. */
    std::unique_ptr<InputMultiVectorType> V2_; /**< Unique pointer to the second input multi-vector for computations. */
    std::unique_ptr<ResultMultiVectorType> W1_; /**< Unique pointer to the first result multi-vector. */
    std::unique_ptr<ResultMultiVectorType> W2_; /**< Unique pointer to the second result multi-vector. */

    std::unique_ptr<chase::distMatrix::RedundantMatrix<chase::Base<T>, chase::platform::GPU>> ritzv_; /**< Matrix holding Ritz values (eigenvalues). */
    std::unique_ptr<chase::distMatrix::RedundantMatrix<chase::Base<T>, chase::platform::GPU>> resid_; /**< Matrix holding residuals for eigenvalues. */
    std::unique_ptr<chase::distMatrix::RedundantMatrix<T, chase::platform::GPU>> A_; /**< Auxiliary matrix for intermediate calculations. */

    cudaStream_t stream_; /**< CUDA stream for asynchronous GPU operations. */
    cublasHandle_t cublasH_; /**< Handle to the cuBLAS library for GPU linear algebra operations. */
    cusolverDnHandle_t cusolverH_; /**< Handle to the cuSolver library for GPU-based eigenvalue solvers. */
    cusolverDnParams_t params_; /**< CUSOLVER structure with information for Xgeev. */

    curandStatePhilox4_32_10_t* states_ = NULL; /**< Random number generator state for GPU, used for initializations. */

    int* devInfo_; /**< Pointer to device memory for storing operation status (e.g., success or failure) in cuSolver calls. */
    T* d_return_; /**< Pointer to device memory for storing results of GPU operations. */
    T* d_work_; /**< Pointer to workspace on the device for GPU operations. */
    int lwork_ = 0; /**< Size of the workspace on the device, used for GPU operations. */
        
    std::unique_ptr<T[]> h_work_; /**< Pointer to work buffer on host for geev
                                     in the Quasi Hermitian case. */
    int lhwork_ = 0; /**< Workspace size for host geev operations in the Quasi
                        Hermitian case. */

    std::size_t *d_diag_xoffs; /**< Pointer to device memory holding x offsets for diagonal elements in computations. */
    std::size_t *d_diag_yoffs; /**< Pointer to device memory holding y offsets for diagonal elements in computations. */
    std::size_t diag_cnt; /**< Count of diagonal elements used in the algorithm. */

    chase::ChaseConfig<T> config_; /**< Configuration settings for the Chase algorithm, including problem parameters. */
};

}
}

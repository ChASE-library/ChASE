/* -*- Mode: C++; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
// This file is a part of ChASE.
// Copyright (c) 2015-2021, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include <assert.h>
#include <complex>
#include <cuComplex.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "ChASE-MPI/blas_cuda_wrapper.hpp"

#include "ChASE-MPI/blas_templates.hpp"
#include "ChASE-MPI/chase_mpi_properties.hpp"
#include "ChASE-MPI/chase_mpidla_interface.hpp"

void chase_rand_normal(curandState *states, float *v, int n, cudaStream_t stream_ );

void chase_rand_normal(curandState *states, double *v, int n, cudaStream_t stream_ );

void chase_rand_normal(curandState *states, std::complex<float> *v, int n, cudaStream_t stream_ );

void chase_rand_normal(curandState *states, std::complex<double> *v, int n, cudaStream_t stream_ );

void chase_shift_mgpu_matrix(float* A, std::size_t* off_m, std::size_t* off_n,
                             std::size_t offsize, std::size_t ldH, float shift,
                             cudaStream_t stream_);

void chase_shift_mgpu_matrix(double* A, std::size_t* off_m, std::size_t* off_n,
                             std::size_t offsize, std::size_t ldH, double shift,
                             cudaStream_t stream_);

void chase_shift_mgpu_matrix(std::complex<double>* A, std::size_t* off_m,
                             std::size_t* off_n, std::size_t offsize,
                             std::size_t ldH, double shift,
                             cudaStream_t stream_);

void chase_shift_mgpu_matrix(std::complex<float>* A, std::size_t* off_m,
                             std::size_t* off_n, std::size_t offsize,
                             std::size_t ldH, float shift,
                             cudaStream_t stream_);

namespace chase
{
namespace mpi
{
//
//  This Class is meant to be used with MatrixFreeMPI
//
//! A derived class of ChaseMpiDLAInterface which implements the inter-node
//! computation for a pure-CPU MPI-based implementation of ChASE.
template <class T>
class ChaseMpiDLAMultiGPU2 : public ChaseMpiDLAInterface<T>
{
public:
    //! A constructor of ChaseMpiDLABlaslapack.
    //! @param matrix_properties: it is an object of ChaseMpiProperties, which
    //! defines the MPI environment and data distribution scheme in ChASE-MPI.
    ChaseMpiDLAMultiGPU2(ChaseMpiProperties<T>* matrix_properties,
                        ChaseMpiMatrices<T>& matrices)
    {
        // TODO
        // ldc_ = matrix_properties->get_ldc();
        // ldb_ = matrix_properties->get_ldb();
#ifdef USE_NSIGHT
        nvtxRangePushA("ChaseMpiDLAMultiGPU2: Init");
#endif	 
        n_ = matrix_properties->get_n();
        m_ = matrix_properties->get_m();
        N_ = matrix_properties->get_N();
        nev_ = matrix_properties->GetNev();
        nex_ = matrix_properties->GetNex();
        H_ = matrices.get_H();
        ldh_ = matrices.get_ldh();
        if (H_ == nullptr)
        {
            H_ = matrix_properties->get_H();
            ldh_ = matrix_properties->get_ldh();
        }

        B_ = matrices.get_V2();
        C_ = matrices.get_V1();
        C2_ = matrix_properties->get_C2();
        B2_ = matrix_properties->get_B2();
        A_ = matrix_properties->get_A();

        off_ = matrix_properties->get_off();

        matrix_properties->get_offs_lens(r_offs_, r_lens_, r_offs_l_, c_offs_,
                                         c_lens_, c_offs_l_);
        mb_ = matrix_properties->get_mb();
        nb_ = matrix_properties->get_nb();

        mblocks_ = matrix_properties->get_mblocks();
        nblocks_ = matrix_properties->get_nblocks();

        matrix_properties_ = matrix_properties;

        MPI_Comm row_comm = matrix_properties_->get_row_comm();
        MPI_Comm col_comm = matrix_properties_->get_col_comm();

        MPI_Comm_rank(row_comm, &mpi_row_rank);
        MPI_Comm_rank(col_comm, &mpi_col_rank);

        int num_devices;
        mpi_rank_ = matrix_properties_->get_my_rank();

        cuda_exec(cudaGetDeviceCount(&num_devices));

        std::size_t maxBlock = matrix_properties_->get_max_block();

        /* Register H, B, C as pinned-memories on host */
        cuda_exec(cudaHostRegister((void*)H_, m_ * n_ * sizeof(T),
                                   cudaHostRegisterDefault));
        cuda_exec(cudaHostRegister((void*)B_, n_ * maxBlock * sizeof(T),
                                   cudaHostRegisterDefault));
        cuda_exec(cudaHostRegister((void*)C_, m_ * maxBlock * sizeof(T),
                                   cudaHostRegisterDefault));
        cuda_exec(cudaHostRegister((void*)B2_, n_ * maxBlock * sizeof(T),
                                   cudaHostRegisterDefault));
        cuda_exec(cudaHostRegister((void*)C2_, m_ * maxBlock * sizeof(T),
                                   cudaHostRegisterDefault));

        cuda_exec(cudaMalloc((void**)&(d_H_),  m_ * n_ * sizeof(T)));
        cuda_exec(cudaMallocPitch((void**)&(d_C_), &pitchC, (nev_+nex_) * sizeof(T), m_));
        cuda_exec(cudaMallocPitch((void**)&(d_C2_), &pitchC2, (nev_+nex_) * sizeof(T), m_));	
        cuda_exec(cudaMallocPitch((void**)&(d_B_), &pitchB, (nev_+nex_) * sizeof(T), n_));
        cuda_exec(cudaMallocPitch((void**)&(d_B2_), &pitchB2, (nev_+nex_) * sizeof(T), n_));
        cuda_exec(cudaMalloc((void**)&d_resid_, sizeof(Base<T>) * (nev_ + nex_)));
        cuda_exec(cudaMalloc((void**)&d_A_, sizeof(T) * (nev_ + nex_) * (nev_ + nex_)));
        cuda_exec(cudaMalloc((void**)&d_ritz_, sizeof(Base<T>) * (nev_ + nex_)));
        cuda_exec(cudaMalloc((void**)&states_, sizeof(curandState) * (256 * 32)));

	cublasCreate(&cublasH_);
	cusolverDnCreate(&cusolverH_);
        cuda_exec(cudaMalloc((void**)&devInfo_, sizeof(int)));
	int lwork_heevd = 0;
        cusolver_status_ = cusolverDnTheevd_bufferSize(
            cusolverH_, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
            nev_ + nex_, d_A_, nev_ + nex_, d_ritz_, &lwork_heevd);
        assert(cusolver_status_ == CUSOLVER_STATUS_SUCCESS);

        if (lwork_heevd > lwork_)
        {
            lwork_ = lwork_heevd;
        }

        int lwork_potrf = 0;
        cusolver_status_ = cusolverDnTpotrf_bufferSize(
            cusolverH_, CUBLAS_FILL_MODE_UPPER, nev_ + nex_, d_A_, nev_ + nex_,
            &lwork_potrf);
        assert(cusolver_status_ == CUSOLVER_STATUS_SUCCESS);

        if (lwork_potrf > lwork_)
        {
            lwork_ = lwork_potrf;
        }
        cuda_exec(cudaMalloc((void**)&d_work_, sizeof(T) * lwork_));

        //for shifting matrix
        std::vector<std::size_t> off_m, off_n;
        for(std::size_t j = 0; j < nblocks_; j++)
        {
            for(std::size_t i = 0; i < mblocks_; i++)
            {
                for(std::size_t q = 0; q < c_lens_[j]; q++)
                {
                    for(std::size_t p = 0; p < r_lens_[i]; p++)
                    {
                        if(q + c_offs_[j] == p + r_offs_[i])
                        {
                            off_m.push_back(p + r_offs_l_[i]);
                            off_n.push_back(q + c_offs_l_[j]);
                        }
                    }
                }
            }
        }
        diag_off_size_ = off_m.size();
        cudaMalloc((void**)&(d_off_m_), diag_off_size_ * sizeof(std::size_t));
        cudaMalloc((void**)&(d_off_n_), diag_off_size_ * sizeof(std::size_t));
	cudaMemcpy(d_off_m_, off_m.data(), diag_off_size_* sizeof(std::size_t), cudaMemcpyHostToDevice);
	cudaMemcpy(d_off_n_, off_n.data(), diag_off_size_* sizeof(std::size_t), cudaMemcpyHostToDevice);
    	cuda_exec(cudaStreamCreate(&stream1_));
	cuda_exec(cudaStreamCreate(&stream2_));
#ifdef USE_NSIGHT
        nvtxRangePop();
#endif    
    }

    ~ChaseMpiDLAMultiGPU2()
    {
        cudaStreamDestroy(stream1_);
        cudaStreamDestroy(stream2_);
	cublasDestroy(cublasH_); 
        cusolverDnDestroy(cusolverH_);	
        cuda_exec(cudaHostUnregister(H_));
        cuda_exec(cudaHostUnregister(B_));
        cuda_exec(cudaHostUnregister(C_));
	cuda_exec(cudaHostUnregister(C2_));
        cuda_exec(cudaHostUnregister(B2_));
        cuda_exec(cudaFree(d_H_));
        cuda_exec(cudaFree(d_C_));
        cuda_exec(cudaFree(d_B_));
	cuda_exec(cudaFree(d_A_));
	cuda_exec(cudaFree(d_C2_));
	cuda_exec(cudaFree(d_B2_));
	cuda_exec(cudaFree(d_work_));
	cuda_exec(cudaFree(devInfo_));
        cuda_exec(cudaFree(d_off_m_));	
	cuda_exec(cudaFree(d_off_n_));
	cuda_exec(cudaFree(d_ritz_));
	cuda_exec(cudaFree(d_resid_));
    	cuda_exec(cudaFree(states_));
    }
    void initVecs() override
    {
        //cuda_exec(cudaMemcpyAsync(d_H_, H_, m_ * n_ * sizeof(T), cudaMemcpyHostToDevice, stream1_));
        cuda_exec(cudaMemcpy(d_H_, H_, m_ * n_ * sizeof(T), cudaMemcpyHostToDevice));
    }
    void initRndVecs() override {
        /*std::mt19937 gen(1337.0);
        std::normal_distribution<> d;

        for(auto j = 0; j < m_ * (nev_ + nex_); j++){	
            auto rnd = getRandomT<T>([&]() { return d(gen); });
            C_[j] = rnd;
        }
        */
	chase_rand_normal(states_, d_C_, m_ * (nev_ + nex_), (cudaStream_t) 0);   
        cuda_exec(cudaMemcpy(C_, d_C_, m_ * (nev_ + nex_) * sizeof(T), cudaMemcpyDeviceToHost));        	    		    
    }

    /*! - For ChaseMpiDLABlaslapack, `preApplication` is implemented within
       ChaseMpiDLA.
        - **Parallelism on distributed-memory system SUPPORT**
        - For the meaning of this function, please visit ChaseMpiDLAInterface.
    */
    void preApplication(T* V, std::size_t locked, std::size_t block) override
    {
        next_ = NextOp::bAc;
        //cuda_exec(cudaMemcpyAsync(d_C_ + locked * m_, C_ + locked *m_, m_ * block * sizeof(T), cudaMemcpyHostToDevice, stream1_));
        if(locked > 0)
	{	
	    cuda_exec(cudaMemcpy(d_C_ + locked * m_, C_ + locked *m_, m_ * block * sizeof(T), cudaMemcpyHostToDevice));
        }
    }

    /*! - For ChaseMpiDLABlaslapack, `preApplication` is implemented within
       ChaseMpiDLA.
        - **Parallelism on distributed-memory system SUPPORT**
        - For the meaning of this function, please visit ChaseMpiDLAInterface.
    */
    void preApplication(T* V1, T* V2, std::size_t locked,
                        std::size_t block) override
    {
        this->preApplication(V1, locked, block);
    }
    /*!
       - For ChaseMpiDLABlaslapack, the matrix-matrix multiplication of local
       matrices are implemented in with `GEMM` routine provided by `BLAS`.
       - The collective communication based on MPI which **ALLREDUCE** the
       product of local matrices either within the column communicator or row
       communicator, is implemented within ChaseMpiDLA.
       - **Parallelism on distributed-memory system SUPPORT**
       - **Parallelism is SUPPORT within node if multi-threading is actived**
       - For the meaning of this function, please visit ChaseMpiDLAInterface.
   */
    void apply(T alpha, T beta, std::size_t offset, std::size_t block,
               std::size_t locked) override
    {
        //cudaStreamSynchronize(stream1_);	    
	T Zero = T(0.0);
        if (next_ == NextOp::bAc) 
        {
            if (mpi_col_rank != 0) 
            {
                beta = Zero;
            }
            cuda_exec(cudaMemcpy(d_C_ + offset * m_ + locked * m_, C_ + offset * m_ + locked * m_, block * m_ * sizeof(T), cudaMemcpyHostToDevice));
            //cuda_exec(cudaMemcpyAsync(d_C_ + offset * m_ + locked * m_, C_ + offset * m_ + locked * m_, block * m_ * sizeof(T), cudaMemcpyHostToDevice, stream1_ ));
	    cublas_status_ = cublasTgemm(cublasH_, CUBLAS_OP_C, CUBLAS_OP_N,
                  n_, block, m_,
                  &alpha,
                  d_H_, m_,
                  d_C_ + locked * m_ + offset * m_, m_,
                  &beta,
                  d_B_ + locked * n_ + offset * n_, n_
                      );
            assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);
	    //cublas_status_ = cublasGetMatrixAsync(n_, block, sizeof(T), d_B_+ locked * n_ + offset * n_, n_, B_+ locked * n_ + offset * n_, n_, stream1_ );
            cublas_status_ = cublasGetMatrix(n_, block, sizeof(T), d_B_+ locked * n_ + offset * n_, n_, B_+ locked * n_ + offset * n_, n_ );            
	    assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);
            next_ = NextOp::cAb;
        } else {
            if (mpi_row_rank != 0) {
                beta = Zero;
            }
	    cuda_exec(cudaMemcpy(d_B_+ locked * n_ + offset * n_, B_ + locked * n_ + offset * n_, block * n_ * sizeof(T), cudaMemcpyHostToDevice));
            //cuda_exec(cudaMemcpyAsync(d_B_+ locked * n_ + offset * n_, B_ + locked * n_ + offset * n_, block * n_ * sizeof(T), cudaMemcpyHostToDevice, stream1_));
	    cublas_status_ = cublasTgemm(cublasH_, CUBLAS_OP_N, CUBLAS_OP_N,
                  m_, block, n_,
                  &alpha,
                  d_H_, m_,
                  d_B_+ locked * n_ + offset * n_ , n_,
                  &beta,
                  d_C_+ locked * m_ + offset * m_ , m_
                      );
            assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);
            //cublas_status_ = cublasGetMatrixAsync(m_, block, sizeof(T), d_C_+ locked * m_ + offset * m_, m_, C_+ locked * m_ + offset * m_, m_, stream1_ );
            cublas_status_ = cublasGetMatrix(m_, block, sizeof(T), d_C_+ locked * m_ + offset * m_, m_, C_+ locked * m_ + offset * m_, m_);
	    assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);		    
	    next_ = NextOp::bAc;
        }
    }
    /*!
       - For ChaseMpiDLABlaslapack,  `postApplication` is implemented in
       ChaseMpiDLA, with asynchronously brocasting the final product of `HEMM`
       to each MPI rank.
       - **Parallelism on distributed-memory system SUPPORT**
       - For the meaning of this function, please visit ChaseMpiDLAInterface.
    */
    bool postApplication(T* V, std::size_t block, std::size_t locked) override
    {
        cuda_exec(cudaMemcpy(C_ + locked * m_, d_C_ + locked *m_, m_ * block * sizeof(T), cudaMemcpyDeviceToHost));	    
        //cuda_exec(cudaMemcpyAsync(C_ + locked * m_, d_C_ + locked *m_, m_ * block * sizeof(T), cudaMemcpyDeviceToHost, stream1_));
	return false;
    }

    /*!
      - For ChaseMpiDLABlaslapack,  `shiftMatrix` is implemented in ChaseMpiDLA.
      - **Parallelism on distributed-memory system SUPPORT**
      - For the meaning of this function, please visit ChaseMpiDLAInterface.
    */
    void shiftMatrix(T c, bool isunshift = false) override
    {
        chase_shift_mgpu_matrix(d_H_, d_off_m_, d_off_n_, diag_off_size_, m_, std::real(c), (cudaStream_t) 0);
    }

    void asynCxHGatherC(std::size_t locked, std::size_t block, bool isCcopied ) override
    {
        T alpha = T(1.0);
        T beta = T(0.0);
  
        if(!isCcopied){
	    cuda_exec(cudaMemcpy(d_C_ + locked * m_, C_ + locked * m_, block * m_ * sizeof(T), cudaMemcpyHostToDevice));
	}
	//cuda_exec(cudaMemcpyAsync(d_C_ + locked * m_, C_ + locked * m_, block * m_ * sizeof(T), cudaMemcpyHostToDevice, stream1_));
        cublas_status_ = cublasTgemm(cublasH_, CUBLAS_OP_C, CUBLAS_OP_N,
                  n_, block, m_,
                  &alpha,
                  d_H_, m_,
                  d_C_ + locked * m_, m_,
                  &beta,
                  d_B_ + locked * n_, n_
                  );
        assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);
        //cuda_exec(cudaMemcpyAsync(B_ + locked * n_, d_B_ + locked * n_, block * n_ * sizeof(T), cudaMemcpyDeviceToHost, stream1_));
        cuda_exec(cudaMemcpy(B_ + locked * n_, d_B_ + locked * n_, block * n_ * sizeof(T), cudaMemcpyDeviceToHost));        
    }

    /*!
      - For ChaseMpiDLABlaslapack,  `applyVec` is implemented in ChaseMpiDLA.
      - **Parallelism on distributed-memory system SUPPORT**
      - For the meaning of this function, please visit ChaseMpiDLAInterface.
    */
    void applyVec(T* B, T* C) override {}

    void get_off(std::size_t* xoff, std::size_t* yoff, std::size_t* xlen,
                 std::size_t* ylen) const override
    {
        *xoff = 0;
        *yoff = 0;
        *xlen = static_cast<std::size_t>(N_);
        *ylen = static_cast<std::size_t>(N_);
    }

    T* get_H() const override { return matrix_properties_->get_H(); }
    std::size_t get_mblocks() const override
    {
        return matrix_properties_->get_mblocks();
    }
    std::size_t get_nblocks() const override
    {
        return matrix_properties_->get_nblocks();
    }
    std::size_t get_n() const override { return matrix_properties_->get_n(); }
    std::size_t get_m() const override { return matrix_properties_->get_m(); }
    int* get_coord() const override { return matrix_properties_->get_coord(); }
    void get_offs_lens(std::size_t*& r_offs, std::size_t*& r_lens,
                       std::size_t*& r_offs_l, std::size_t*& c_offs,
                       std::size_t*& c_lens,
                       std::size_t*& c_offs_l) const override
    {
        matrix_properties_->get_offs_lens(r_offs, r_lens, r_offs_l, c_offs,
                                          c_lens, c_offs_l);
    }
    int get_nprocs() const override { return matrix_properties_->get_nprocs(); }
    void Start() override {}
    void End() override {}

    /*!
      - For ChaseMpiDLABlaslapack, `lange` is implemented using `LAPACK` routine
      `xLANGE`.
      - **Parallelism is SUPPORT within node if multi-threading is enabled.**
      - For the meaning of this function, please visit ChaseMpiDLAInterface.
    */
    Base<T> lange(char norm, std::size_t m, std::size_t n, T* A,
                  std::size_t lda) override
    {
        return t_lange(norm, m, n, A, lda);
    }

    /*!
      - For ChaseMpiDLABlaslapack, `axpy` is implemented in ChaseMpiDLA.
     - **Parallelism is SUPPORT within node if multi-threading is enabled.**
      - For the meaning of this function, please visit ChaseMpiDLAInterface.
    */
    void axpy(std::size_t N, T* alpha, T* x, std::size_t incx, T* y,
              std::size_t incy) override
    {
        t_axpy(N, alpha, x, incx, y, incy);
    }

    /*!
      - For ChaseMpiDLABlaslapack, `scal` is implemented in ChaseMpiDLA
      - **Parallelism is SUPPORT within node if multi-threading is enabled.**
      - For the meaning of this function, please visit ChaseMpiDLAInterface.
    */
    void scal(std::size_t N, T* a, T* x, std::size_t incx) override
    {
        t_scal(N, a, x, incx);
    }

    /*!
      - For ChaseMpiDLABlaslapack, `nrm2` is implemented using `BLAS` routine
      `xNRM2`.
      - **Parallelism is SUPPORT within node if multi-threading is enabled.**
      - For the meaning of this function, please visit ChaseMpiDLAInterface.
    */
    Base<T> nrm2(std::size_t n, T* x, std::size_t incx) override
    {
        return t_nrm2(n, x, incx);
    }

    /*!
      - For ChaseMpiDLABlaslapack, `dot` is implemented using `BLAS` routine
      `xDOT`.
      - **Parallelism is SUPPORT within node if multi-threading is enabled.**
      - For the meaning of this function, please visit ChaseMpiDLAInterface.
    */
    T dot(std::size_t n, T* x, std::size_t incx, T* y,
          std::size_t incy) override
    {
        return t_dot(n, x, incx, y, incy);
    }
    /*!
     - For ChaseMpiDLABlaslapack, `gemm` is implemented in ChaseMpiDLA.
     - **Parallelism is SUPPORT within node if multi-threading is enabled.**
     - For the meaning of this function, please visit ChaseMpiDLAInterface.
    */
    void gemm(CBLAS_LAYOUT Layout, CBLAS_TRANSPOSE transa,
              CBLAS_TRANSPOSE transb, std::size_t m, std::size_t n,
              std::size_t k, T* alpha, T* a, std::size_t lda, T* b,
              std::size_t ldb, T* beta, T* c, std::size_t ldc) override
    {
        t_gemm(Layout, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c,
               ldc);
    }

    /*!
     - For ChaseMpiDLABlaslapack, `stemr` with scalar being real and double
     precision, is implemented using `LAPACK` routine `DSTEMR`.
     - **Parallelism is SUPPORT within node if multi-threading is enabled.**
     - For the meaning of this function, please visit ChaseMpiDLAInterface.
    */
    std::size_t stemr(int matrix_layout, char jobz, char range, std::size_t n,
                      double* d, double* e, double vl, double vu,
                      std::size_t il, std::size_t iu, int* m, double* w,
                      double* z, std::size_t ldz, std::size_t nzc, int* isuppz,
                      lapack_logical* tryrac) override
    {
        return t_stemr<double>(matrix_layout, jobz, range, n, d, e, vl, vu, il,
                               iu, m, w, z, ldz, nzc, isuppz, tryrac);
    }

    /*!
     - For ChaseMpiDLABlaslapack, `stemr` with scalar being real and single
     precision, is implemented using `LAPACK` routine `SSTEMR`.
     - **Parallelism is SUPPORT within node if multi-threading is enabled.**
     - For the meaning of this function, please visit ChaseMpiDLAInterface.
    */
    std::size_t stemr(int matrix_layout, char jobz, char range, std::size_t n,
                      float* d, float* e, float vl, float vu, std::size_t il,
                      std::size_t iu, int* m, float* w, float* z,
                      std::size_t ldz, std::size_t nzc, int* isuppz,
                      lapack_logical* tryrac) override
    {
        return t_stemr<float>(matrix_layout, jobz, range, n, d, e, vl, vu, il,
                              iu, m, w, z, ldz, nzc, isuppz, tryrac);
    }

    /*!
        - For ChaseMpiDLABlaslapack, `RR` is implemented by `GEMM` routine
       provided by `BLAS` and `(SY)HEEVD` routine provided by `LAPACK`.
          - The 1st operation `A <- W^T * V` is implemented by `GEMM` from
       `BLAS`.
          - The 2nd operation which computes the eigenpairs of `A`, is
       implemented by `(SY)HEEVD` from `LAPACK`.
          - The 3rd operation which computes `W<-V*A` is implemented by `GEMM`
       from `BLAS`.
        - **Parallelism is SUPPORT within node if multi-threading is enabled.**
        - For the meaning of this function, please visit ChaseMpiDLAInterface.
    */
    void RR(std::size_t block, std::size_t locked, Base<T>* ritzv) override {
        T One = T(1.0);
        T Zero = T(0.0);
        cuda_exec(cudaMemcpy(d_B_ + locked * n_, B_ + locked * n_, block * n_ * sizeof(T), cudaMemcpyHostToDevice));
        cuda_exec(cudaMemcpy(d_B2_ + locked * n_, B2_ + locked * n_, block * n_ * sizeof(T), cudaMemcpyHostToDevice));

	cublas_status_ =
            cublasTgemm(cublasH_, CUBLAS_OP_C, CUBLAS_OP_N, block, block, n_,
                        &One, d_B2_ + locked * n_, n_, d_B_ + locked * n_, n_,
                        &Zero, d_A_, nev_ + nex_);
        assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);
        cublas_status_ = cublasGetMatrix(block, block, sizeof(T), d_A_, nev_ + nex_, A_, nev_ + nex_ );
        assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);
    }

    void V2C(T* v1, std::size_t off1, T* v2, std::size_t off2,
             std::size_t block) override
    {
    }

    void C2V(T* v1, std::size_t off1, T* v2, std::size_t off2,
             std::size_t block) override
    {
    }
    void syherk(char uplo, char trans, std::size_t n, std::size_t k, T* alpha,
                T* a, std::size_t lda, T* beta, T* c, std::size_t ldc,
		bool first ) override
    {
        if(first)
	{
	    cuda_exec(cudaMemcpy(d_C_, C_, (nev_ + nex_) * m_ * sizeof(T), cudaMemcpyHostToDevice));	 }
	Base<T> One = Base<T>(1.0);
        Base<T> Zero = Base<T>(0.0);
        cublasOperation_t transa;
	if(sizeof(T) == sizeof(Base<T>)){
	    transa = CUBLAS_OP_T;
	}else{
	    transa = CUBLAS_OP_C;
	}

        cublas_status_ = cublasTsyherk(cublasH_,
        	CUBLAS_FILL_MODE_UPPER, transa, nev_ + nex_, m_, &One, d_C_, m_, &Zero,
        	d_A_, nev_ + nex_); 
	assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);
        
	cublas_status_ = cublasGetMatrix(nev_ + nex_, nev_ + nex_, sizeof(T), d_A_, nev_ + nex_, A_, nev_ + nex_ );
        assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);
    }

    int potrf(char uplo, std::size_t n, T* a, std::size_t lda) override
    {
        cublas_status_ = cublasSetMatrix(nev_ + nex_, nev_ + nex_, sizeof(T), a, lda, d_A_, nev_ + nex_ );
        assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);
#ifdef USE_NSIGHT
        nvtxRangePushA("cusolverDnTpotrf");
#endif
        cusolver_status_ = cusolverDnTpotrf(cusolverH_, CUBLAS_FILL_MODE_UPPER, nev_ + nex_,
        		   d_A_, nev_ + nex_, d_work_, lwork_, devInfo_);
#ifdef USE_NSIGHT
        nvtxRangePop();
#endif
        assert(CUSOLVER_STATUS_SUCCESS == cusolver_status_);

	int info;	
	cuda_exec(cudaMemcpy(&info, devInfo_, 1 * sizeof(int), cudaMemcpyDeviceToHost));

	return info;
    }

    void trsm(char side, char uplo, char trans, char diag, std::size_t m,
              std::size_t n, T* alpha, T* a, std::size_t lda, T* b,
              std::size_t ldb, bool first ) override
    {
	cublas_status_ = cublasTtrsm(cublasH_, 
			    	     CUBLAS_SIDE_RIGHT, 
			    	     CUBLAS_FILL_MODE_UPPER,
                                     CUBLAS_OP_N,
        			     CUBLAS_DIAG_NON_UNIT, 
				     m_, 
				     nev_ + nex_, 
				     alpha, 
				     d_A_, 
				     nev_+nex_, 
				     d_C_, 
				     m_);
	assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);
        
	if(!first)
	{
	    cublas_status_ = cublasGetMatrix(m_, nev_ + nex_, sizeof(T), d_C_, m_, C_, m_ );	
	}
        assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);
    }

    void Resd(Base<T>* ritzv, Base<T>* resid, std::size_t locked,
              std::size_t unconverged) override
    {
#ifdef HAS_OMP
        char* omp_threads;
        omp_threads = getenv("OMP_NUM_THREADS");
        int num_threads = 1;
        if(omp_threads){
            num_threads = std::atoi(omp_threads);
        }
        omp_set_num_threads(1);
#endif
        for (auto i = 0; i < unconverged; i++)
        {
            T alpha = -ritzv[i];
            t_axpy(n_, &alpha, B2_ + locked * n_ + i * n_, 1,
                   B_ + locked * n_ + i * n_, 1);

            Base<T> tmp = t_nrm2(n_, B_ + locked * n_ + i * n_, 1);
            resid[i] = std::pow(tmp, 2);
        }	    
#ifdef HAS_OMP
	omp_set_num_threads(num_threads);
#endif	
    }

    void heevd(int matrix_layout, char jobz, char uplo, std::size_t n, T* a,
               std::size_t lda, Base<T>* w) override
    {
	T One = T(1.0);
        T Zero = T(0.0);
        std::size_t locked = nev_ + nex_ - n;	
        cublasSetMatrix(n, n, sizeof(T), a, lda, d_A_, nev_ + nex_);	 
#ifdef USE_NSIGHT
        nvtxRangePushA("cusolverDnTheevd");
#endif      	
        cusolver_status_ = cusolverDnTheevd(
            cusolverH_, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, n,
            d_A_, nev_ + nex_, d_ritz_, d_work_, lwork_, devInfo_);
        assert(cusolver_status_ == CUSOLVER_STATUS_SUCCESS);
#ifdef USE_NSIGHT
        nvtxRangePop();
#endif	
        cuda_exec(cudaMemcpy(w, d_ritz_, n * sizeof(Base<T>), cudaMemcpyDeviceToHost));
	cublas_status_ = cublasSetMatrix(m_, n, sizeof(T), C2_ + locked * m_, m_, d_C2_ + locked * m_, m_ );
        assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);
	cublas_status_ =
            cublasTgemm(cublasH_, CUBLAS_OP_N, CUBLAS_OP_N, m_,
                        n, n, &One, d_C2_ + locked * m_, m_,
                        d_A_, nev_ + nex_, &Zero,
                        d_C_ + locked * m_, m_);
        assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);

	cublas_status_ = cublasGetMatrix(m_, n, sizeof(T), d_C_ + locked * m_, m_, C_ + locked * m_, m_ );
        assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);	
    }

    void hhQR(std::size_t locked) override {}

    void cholQR(std::size_t locked) override {}

    void Swap(std::size_t i, std::size_t j) override {}

    void getLanczosBuffer(T** V1, T** V2, std::size_t* ld, T** v0, T** v1,
                          T** w) override
    {
    }
    void getLanczosBuffer2(T** v0, T** v1, T** w) override {}

    void LanczosDos(std::size_t idx, std::size_t m, T* ritzVc) override {}

private:
    enum NextOp
    {
        cAb,
        bAc
    };

    NextOp next_;
    std::size_t N_;

    std::size_t n_;
    std::size_t m_;
    std::size_t ldh_;
    T* H_;
    T* B_;
    T* B2_;
    T* C_;
    T* C2_;
    T *A_;

    std::size_t* off_;
    std::size_t* r_offs_;
    std::size_t* r_lens_;
    std::size_t* r_offs_l_;
    std::size_t* c_offs_;
    std::size_t* c_lens_;
    std::size_t* c_offs_l_;
    std::size_t nb_;
    std::size_t mb_;
    std::size_t nblocks_;
    std::size_t mblocks_;
    std::size_t nev_;
    std::size_t nex_;
    int mpi_row_rank;
    int mpi_col_rank;
    
    //for shifting matrix H
    std::size_t * d_off_m_;
    std::size_t * d_off_n_;
    std::size_t diag_off_size_;
    
    int mpi_rank_;
    cublasHandle_t cublasH_;
    cusolverDnHandle_t cusolverH_;
    cublasStatus_t cublas_status_ = CUBLAS_STATUS_SUCCESS;
    cusolverStatus_t cusolver_status_ = CUSOLVER_STATUS_SUCCESS;
    cudaStream_t stream1_, stream2_;
    curandState *states_ = NULL;
    T *d_H_;
    T *d_C_;
    T *d_C2_;
    T *d_B_;
    T *d_B2_;
    Base<T>* d_ritz_ = NULL;
    T* d_A_;
    Base<T>* d_resid_ = NULL;
    T *d_work_ = NULL;
    std::size_t pitchB, pitchB2, pitchC, pitchC2;
    int* devInfo_ = NULL;

    int lwork_ = 0;
    ChaseMpiProperties<T>* matrix_properties_;
};

template <typename T>
struct is_skewed_matrixfree<ChaseMpiDLAMultiGPU2<T>>
{
    static const bool value = true;
};

} // namespace mpi
} // namespace chase

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

#include "ChASE-MPI/blas_cuda_wrapper.hpp"

#include "ChASE-MPI/blas_templates.hpp"
#include "ChASE-MPI/chase_mpi_properties.hpp"
#include "ChASE-MPI/chase_mpidla_interface.hpp"
#include "ChASE-MPI/impl/mgpu_cudaDLA.hpp"

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
class ChaseMpiDLAMultiGPU : public ChaseMpiDLAInterface<T>
{
public:
    //! A constructor of ChaseMpiDLABlaslapack.
    //! @param matrix_properties: it is an object of ChaseMpiProperties, which
    //! defines the MPI environment and data distribution scheme in ChASE-MPI.
    ChaseMpiDLAMultiGPU(ChaseMpiProperties<T>* matrix_properties,
                        ChaseMpiMatrices<T>& matrices)
    {
        // TODO
        // ldc_ = matrix_properties->get_ldc();
        // ldb_ = matrix_properties->get_ldb();
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
	B2_ = matrix_properties->get_B2();
        C_ = matrices.get_V1();
        C2_ = matrix_properties->get_C2();
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

        previous_offset_ = -1;
        /* Register H, B, C and IMT as pinned-memories on host */
        cuda_exec(cudaHostRegister((void*)H_, m_ * n_ * sizeof(T),
                                   cudaHostRegisterDefault));
        cuda_exec(cudaHostRegister((void*)B_, n_ * maxBlock * sizeof(T),
                                   cudaHostRegisterDefault));
        cuda_exec(cudaHostRegister((void*)C_, m_ * maxBlock * sizeof(T),
                                   cudaHostRegisterDefault));
        /// Construct a new object for handling multi-GPU HEMM execution
        mgpuDLA = new mgpu_cudaDLA<T>(matrix_properties, m_, n_, maxBlock);
    }

    ~ChaseMpiDLAMultiGPU()
    {
        cuda_exec(cudaHostUnregister(H_));
        cuda_exec(cudaHostUnregister(B_));
        cuda_exec(cudaHostUnregister(C_));
        delete mgpuDLA;
    }
    void initVecs() override
    {
        mgpuDLA->distribute_H(H_, m_);
        mgpuDLA->synchronizeAll();
    }
    void initRndVecs() override {
        std::mt19937 gen(1337.0);
        std::normal_distribution<> d;

        for(auto j = 0; j < m_ * (nev_ + nex_); j++){
            auto rnd = getRandomT<T>([&]() { return d(gen); });
            C_[j] = rnd;
        }    
    }
    void initRndVecsFromFile(std::string rnd_file) override {}

    /*! - For ChaseMpiDLABlaslapack, `preApplication` is implemented within
       ChaseMpiDLA.
        - **Parallelism on distributed-memory system SUPPORT**
        - For the meaning of this function, please visit ChaseMpiDLAInterface.
    */
    void preApplication(T* V, std::size_t locked, std::size_t block) override
    {
        next_ = NextOp::bAc;
        mgpuDLA->set_operation(next_);
        previous_offset_ = -1;
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

        T Zero = T(0.0);
        T* buf_init;
        T* buf_target;
        std::size_t m, n, k;
        cublasOperation_t transa;
        std::size_t leading_dim;
        std::size_t ldBufInit;
        std::size_t ldBufTarget;

        if (next_ == NextOp::bAc)
        {
            buf_init = C_ + offset * m_ + locked * m_;
            buf_target = B_ + offset * n_ + locked * n_;
            m = n_;
            n = block;
            k = m_;
            ldBufInit = m_;
            ldBufTarget = n_;
            transa = CUBLAS_OP_C;
            next_ = NextOp::cAb;
            if (mpi_col_rank != 0)
            {
                beta = Zero;
            }
        }
        else
        {
            buf_init = B_ + offset * n_ + locked * n_;
            buf_target = C_ + offset * m_ + locked * m_;
            m = m_;
            n = block;
            k = n_;
            ldBufInit = n_;
            ldBufTarget = m_;
            transa = CUBLAS_OP_N;
            next_ = NextOp::bAc;
            if (mpi_row_rank != 0)
            {
                beta = Zero;
            }
        }

        std::size_t W_offset;
        if (previous_offset_ == -1)
        {
            W_offset = 0;
        }
        else
        {
            W_offset = offset - previous_offset_;
        }

        mgpuDLA->distribute_V(buf_init, ldBufInit, block);
        //mgpuDLA->synchronizeAll();
        mgpuDLA->computeHemm(block, W_offset, alpha, beta);
        //mgpuDLA->synchronizeAll();
        mgpuDLA->return_W(buf_target, ldBufTarget, block, W_offset);
        //mgpuDLA->synchronizeAll();

        mgpuDLA->switch_operation();
        previous_offset_ = offset;
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
        mgpuDLA->synchronizeAll();
        previous_offset_ = -1;
        return false;
    }

    /*!
      - For ChaseMpiDLABlaslapack,  `shiftMatrix` is implemented in ChaseMpiDLA.
      - **Parallelism on distributed-memory system SUPPORT**
      - For the meaning of this function, please visit ChaseMpiDLAInterface.
    */
    void shiftMatrix(T c, bool isunshift = false) override
    {
        mgpuDLA->distribute_H(H_, m_);
        mgpuDLA->shiftMatrix(c);
        mgpuDLA->synchronizeAll();
        if (!isunshift)
        {
            previous_offset_ = -1;
        }
    }

    void asynCxHGatherC(std::size_t locked, std::size_t block, bool isCcopied = false) override
    {
        T alpha = T(1.0);
        T beta = T(0.0);
/*
        t_gemm<T>(CblasColMajor, CblasConjTrans, CblasNoTrans, n_,
                  static_cast<std::size_t>(block), m_, &alpha, H_, ldh_,
                  C_ + locked * m_, m_, &beta, B_ + locked * n_, n_);
*/
        T* buf_init = C_ + locked * m_;
        T* buf_target = B_ + locked * n_;
        
        cublasOperation_t transa = CUBLAS_OP_C;
        std::size_t ldBufInit = m_;
        std::size_t ldBufTarget = n_;
	next_ = NextOp::cAb;
	mgpuDLA->distribute_V(buf_init, ldBufInit, block);
	mgpuDLA->computeHemm(block, 0, alpha, beta);
        mgpuDLA->return_W(buf_target, ldBufTarget, block, 0);	
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

    	this->gemm(CblasColMajor, CblasConjTrans, CblasNoTrans, block, block,
                   n_, &One, B2_ + locked * n_, n_, B_ + locked * n_, n_, &Zero,
                   A_, block);    
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
		bool first = true) override
    {
        t_syherk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
    }

    int potrf(char uplo, std::size_t n, T* a, std::size_t lda) override
    {
        return t_potrf(uplo, n, a, lda);
    }

    void trsm(char side, char uplo, char trans, char diag, std::size_t m,
              std::size_t n, T* alpha, T* a, std::size_t lda, T* b,
              std::size_t ldb, bool first = false) override
    {
        t_trsm(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb);
    }

    void Resd(Base<T>* ritzv, Base<T>* resid, std::size_t locked,
              std::size_t unconverged) override
    {
        for (auto i = 0; i < unconverged; i++)
        {
            T alpha = -ritzv[i];
            t_axpy(n_, &alpha, B2_ + locked * n_ + i * n_, 1,
                   B_ + locked * n_ + i * n_, 1);

            Base<T> tmp = t_nrm2(n_, B_ + locked * n_ + i * n_, 1);
            resid[i] = std::pow(tmp, 2);
        }
    }

    void heevd(int matrix_layout, char jobz, char uplo, std::size_t n, T* a,
               std::size_t lda, Base<T>* w) override
    {
	T One = T(1.0);
        T Zero = T(0.0);
        std::size_t locked = nev_ + nex_ - n;	
	    
        t_heevd(matrix_layout, jobz, uplo, n, a, nev_ + nex_, w);
        this->gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m_, n, n,
                   &One, C2_ + locked * m_, m_, A_, nev_ + nex_, &Zero,
                   C_ + locked * m_, m_);
	
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

    int mpi_rank_;
    int previous_offset_;
    mgpu_cudaDLA<T>* mgpuDLA;

    ChaseMpiProperties<T>* matrix_properties_;
};

template <typename T>
struct is_skewed_matrixfree<ChaseMpiDLAMultiGPU<T>>
{
    static const bool value = true;
};

} // namespace mpi
} // namespace chase

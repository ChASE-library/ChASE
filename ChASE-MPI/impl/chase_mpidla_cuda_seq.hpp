/* -*- Mode: C++; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
// This file is a part of ChASE.
// Copyright (c) 2015-2021, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include <cuComplex.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <assert.h>
#include <complex>

#include "ChASE-MPI/blas_cuda_wrapper.hpp"
#include "ChASE-MPI/chase_mpidla_interface.hpp"

void chase_shift_matrix(float* A, int n, float shift, cudaStream_t* stream_);

void chase_shift_matrix(double* A, int n, double shift, cudaStream_t* stream_);

void chase_shift_matrix(std::complex<float>* A, int n, float shift,
                        cudaStream_t* stream_);

void chase_shift_matrix(std::complex<double>* A, int n, double shift,
                        cudaStream_t* stream_);

namespace chase
{
namespace mpi
{

//! A derived class of ChaseMpiDLAInterface which implements ChASE targeting
//! shared-memory architectures, some selected computation tasks are offloaded
//! to one single GPU card.
template <class T>
class ChaseMpiDLACudaSeq : public ChaseMpiDLAInterface<T>
{
public:
    //! A constructor of ChaseMpiDLACudaSeq.
    //! This constructor sets up the CUDA environment, handles, streams and
    //! allocates required memory on device.
    /*! @param matrices: it is an object of ChaseMpiMatrices, which allocates
       the required buffer.
        @param n: size of matrix defining the eigenproblem.
        @param maxBlock: maximum column number of matrix `V`, which equals to
       `nev+nex`.
    */
    ChaseMpiDLACudaSeq(ChaseMpiMatrices<T>& matrices, std::size_t N,
                       std::size_t nev, std::size_t nex)
        : N_(N), copied_(false), nev_(nev), nex_(nex), max_block_(nev + nex),
          V1_(matrices.get_V1()), V2_(matrices.get_V2()), H_(matrices.get_H())
    {
        cuda_exec(cudaSetDevice(0));

        cuda_exec(cudaMalloc((void**)&(d_V1_), N_ * (nev_ + nex_) * sizeof(T)));
        cuda_exec(cudaMalloc((void**)&(d_V2_), N_ * (nev_ + nex_) * sizeof(T)));
        cuda_exec(cudaMalloc((void**)&(d_H_), N_ * N_ * sizeof(T)));
        cuda_exec(cudaMalloc((void**)&d_A_,
                             sizeof(T) * (nev_ + nex_) * (nev_ + nex_)));
        cuda_exec(
            cudaMalloc((void**)&d_ritz_, sizeof(Base<T>) * (nev_ + nex_)));
        cuda_exec(cudaMalloc((void**)&(v0_), N_ * sizeof(T)));
        cuda_exec(cudaMalloc((void**)&(v1_), N_ * sizeof(T)));
        cuda_exec(cudaMalloc((void**)&(w_), N_ * sizeof(T)));

        cublasCreate(&cublasH_);
        cublasCreate(&cublasH2_);

        cusolverDnCreate(&cusolverH_);
        cuda_exec(cudaStreamCreate(&stream_));
        cublasSetStream(cublasH_, stream_);
        cublasSetStream(cublasH2_, stream_);
        cublasSetPointerMode(cublasH2_, CUBLAS_POINTER_MODE_DEVICE);
        cuda_exec(
            cudaMalloc((void**)&d_resids_, sizeof(Base<T>) * (nev_ + nex_)));

        cusolverDnSetStream(cusolverH_, stream_);

        cuda_exec(cudaMalloc((void**)&devInfo_, sizeof(int)));
        cuda_exec(cudaMalloc((void**)&d_return_, sizeof(T) * max_block_));

        int lwork_geqrf = 0;
        int lwork_orgqr = 0;
        cusolver_status_ = cusolverDnTgeqrf_bufferSize(
            cusolverH_, N_, max_block_, d_V1_, N_, &lwork_geqrf);
        assert(cusolver_status_ == CUSOLVER_STATUS_SUCCESS);
        cuda_exec(cudaSetDevice(0));
        cusolver_status_ =
            cusolverDnTgqr_bufferSize(cusolverH_, N_, max_block_, max_block_,
                                      d_V1_, N_, d_return_, &lwork_orgqr);
        assert(cusolver_status_ == CUSOLVER_STATUS_SUCCESS);

        lwork_ = (lwork_geqrf > lwork_orgqr) ? lwork_geqrf : lwork_orgqr;

        int lwork_heevd = 0;
        cusolver_status_ = cusolverDnTheevd_bufferSize(
            cusolverH_, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
            max_block_, d_A_, max_block_, d_ritz_, &lwork_heevd);

        if (lwork_heevd > lwork_)
        {
            lwork_ = lwork_heevd;
        }

        int lwork_potrf = 0;
        cusolver_status_ = cusolverDnTpotrf_bufferSize(
            cusolverH_, CUBLAS_FILL_MODE_UPPER, nev_ + nex_, d_A_, nev_ + nex_,
            &lwork_potrf);

        if (lwork_potrf > lwork_)
        {
            lwork_ = lwork_potrf;
        }
        cuda_exec(cudaMalloc((void**)&d_work_, sizeof(T) * lwork_));
    }

    ~ChaseMpiDLACudaSeq()
    {
        if (d_V1_)
            cudaFree(d_V1_);
        if (d_V2_)
            cudaFree(d_V2_);
        if (d_H_)
            cudaFree(d_H_);
        if (cublasH_)
            cublasDestroy(cublasH_);
        if (cusolverH_)
            cusolverDnDestroy(cusolverH_);
        if (d_work_)
            cudaFree(d_work_);
        if (devInfo_)
            cudaFree(devInfo_);
        if (d_return_)
            cudaFree(d_return_);
        if (d_A_)
            cudaFree(d_A_);
        if (d_ritz_)
            cudaFree(d_ritz_);
        if (v0_)
            cudaFree(v0_);
        if (v1_)
            cudaFree(v1_);
        if (w_)
            cudaFree(w_);
    }
    void initVecs() override
    {
        cuda_exec(cudaMemcpy(d_V1_, V1_, (nev_ + nex_) * N_ * sizeof(T),
                             cudaMemcpyHostToDevice));
        cuda_exec(cudaMemcpy(d_V2_, d_V1_, (nev_ + nex_) * N_ * sizeof(T),
                             cudaMemcpyDeviceToDevice));
        cuda_exec(
            cudaMemcpy(d_H_, H_, N_ * N_ * sizeof(T), cudaMemcpyHostToDevice));
    }
    void initRndVecs() override
    {
        std::mt19937 gen(1337.0);
        std::normal_distribution<> d;
        for (auto j = 0; j < (nev_ + nex_); j++)
        {
            for (auto i = 0; i < N_; i++)
            {
                V1_[i + j * N_] = getRandomT<T>([&]() { return d(gen); });
            }
        }
    }

    // host->device: v1 on host, v2 on device
    void V2C(T* v1, std::size_t off1, T* v2, std::size_t off2,
             std::size_t block) override
    {
        cuda_exec(cudaMemcpy(v2 + off2 * N_, v1 + off1 * N_,
                             block * N_ * sizeof(T), cudaMemcpyHostToDevice));
    }
    // device->host: v1 on device, v2 on host
    void C2V(T* v1, std::size_t off1, T* v2, std::size_t off2,
             std::size_t block) override
    {
        cuda_exec(cudaMemcpy(v2 + off2 * N_, v1 + off1 * N_,
                             block * N_ * sizeof(T), cudaMemcpyDeviceToHost));
    }

    /*! - For ChaseMpiDLACudaSeq, the core of `preApplication` is implemented
       with `cudaMemcpyAsync, which copies `block` vectors from `V` on Host to
       `V1` on GPU device.
        - **Parallelism is NOT SUPPORT**
        - For the meaning of this function, please visit ChaseMpiDLAInterface.
    */
    void preApplication(T* V, std::size_t locked, std::size_t block) override
    {
        locked_ = locked;
        cuda_exec(cudaMemcpy(d_V1_, V + locked_ * N_, block * N_ * sizeof(T),
                             cudaMemcpyDeviceToDevice));
    }

    /*! - For ChaseMpiDLACudaSeq, `apply` is implemented with `cublasXgemm`
       provided by `cuBLAS`.
        - **Parallelism is SUPPORT within one GPU card**
        - For the meaning of this function, please visit ChaseMpiDLAInterface.
    */
    void apply(T alpha, T beta, std::size_t offset, std::size_t block,
               std::size_t locked) override
    {
        cublas_status_ =
            cublasTgemm(cublasH_, CUBLAS_OP_N, CUBLAS_OP_N, N_,
                        static_cast<std::size_t>(block), N_, &alpha, d_H_, N_,
                        d_V1_ + offset * N_ + locked * N_, N_, &beta,
                        d_V2_ + locked * N_ + offset * N_, N_);
        assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);
        std::swap(d_V1_, d_V2_);
    }

    /*! - For ChaseMpiDLACudaSeq, the core of `postApplication` is implemented
       with `cudaMemcpyAsync, which copies `block` vectors from `V1_` on GPU
       device to `V` on Host.
        - **Parallelism is NOT SUPPORT**
        - For the meaning of this function, please visit ChaseMpiDLAInterface.
    */
    bool postApplication(T* V, std::size_t block, std::size_t locked) override
    {
        return false;
    }

    /*! - For ChaseMpiDLACudaSeq, `shiftMatrix` is offloaded to GPU device and
       implemented by `CUDA`.
        - **Parallelism is SUPPORT within one GPU card**
        - For the meaning of this function, please visit ChaseMpiDLAInterface.
    */
    void shiftMatrix(T c, bool isunshift = false) override
    {
        chase_shift_matrix(d_H_, N_, std::real(c), &stream_);
    }

    void asynCxHGatherC(std::size_t locked, std::size_t block,
                        bool isCcopied = false) override
    {
    }

    /*! - For ChaseMpiDLACudaSeq, `applyVec` is implemented with `GEMM` provided
       by `BLAS`.
        - **Parallelism is SUPPORT within node if multi-threading is actived**
        - For the meaning of this function, please visit ChaseMpiDLAInterface.
    */
    void applyVec(T* B, T* C) override
    {
        T One = T(1.0);
        T Zero = T(0.0);

        cublas_status_ = cublasTgemm(cublasH_, CUBLAS_OP_N, CUBLAS_OP_N, N_, 1,
                                     N_, &One, d_H_, N_, B, N_, &Zero, C, N_);
        assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);
    }

    void get_off(std::size_t* xoff, std::size_t* yoff, std::size_t* xlen,
                 std::size_t* ylen) const override
    {
        *xoff = 0;
        *yoff = 0;
        *xlen = N_;
        *ylen = N_;
    }

    T* get_H() const override { return H_; }
    std::size_t get_mblocks() const override { return 1; }
    std::size_t get_nblocks() const override { return 1; }
    std::size_t get_n() const override { return N_; }
    std::size_t get_m() const override { return N_; }
    int* get_coord() const override
    {
        int* coord = new int[2];
        coord[0] = 0;
        coord[1] = 0;
        return coord;
    }
    int get_nprocs() const override { return 1; }
    void get_offs_lens(std::size_t*& r_offs, std::size_t*& r_lens,
                       std::size_t*& r_offs_l, std::size_t*& c_offs,
                       std::size_t*& c_lens,
                       std::size_t*& c_offs_l) const override
    {

        std::size_t r_offs_[1] = {0};
        std::size_t r_lens_[1];
        r_lens_[0] = N_;
        std::size_t r_offs_l_[1] = {0};
        std::size_t c_offs_[1] = {0};
        std::size_t c_lens_[1];
        r_lens_[0] = N_;
        std::size_t c_offs_l_[1] = {0};

        r_offs = r_offs_;
        r_lens = r_lens_;
        r_offs_l = r_offs_l_;
        c_offs = c_offs_;
        c_lens = c_lens_;
        c_offs_l = c_offs_l_;
    }

    void Start() override {}
    void End() override
    {
        cuda_exec(cudaMemcpy(V1_, d_V1_, max_block_ * N_ * sizeof(T),
                             cudaMemcpyDeviceToHost));
    }

    /*!
      - For ChaseMpiDLACudaSeq, `lange` is implemented using `LAPACK` routine
      `xLANGE`.
      - **Parallelism is SUPPORT within node if multi-threading is actived**
      - For the meaning of this function, please visit ChaseMpiDLAInterface.
    */
    Base<T> lange(char norm, std::size_t m, std::size_t n, T* A,
                  std::size_t lda) override
    {
        return t_lange(norm, m, n, A, lda);
    }

    /*!
      - For ChaseMpiDLACudaSeq, `axpy` is implemented using `BLAS` routine
      `xAXPY`.
      - **Parallelism is SUPPORT within node if multi-threading is actived**
      - For the meaning of this function, please visit ChaseMpiDLAInterface.
    */
    void axpy(std::size_t N, T* alpha, T* x, std::size_t incx, T* y,
              std::size_t incy) override
    {
        cublas_status_ = cublasTaxpy(cublasH_, N, alpha, x, incx, y, incy);
        assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);
    }

    /*!
      - For ChaseMpiDLACudaSeq, `scal` is implemented using `BLAS` routine
      `xSCAL`.
      - **Parallelism is SUPPORT within node if multi-threading is actived**
      - For the meaning of this function, please visit ChaseMpiDLAInterface.
    */
    void scal(std::size_t N, T* a, T* x, std::size_t incx) override
    {
        cublas_status_ = cublasTscal(cublasH_, N, a, x, incx);
        assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);
    }

    /*!
      - For ChaseMpiDLACudaSeq, `nrm2` is implemented using `BLAS` routine
      `xNRM2`.
      - **Parallelism is SUPPORT within node if multi-threading is actived**
      - For the meaning of this function, please visit ChaseMpiDLAInterface.
    */
    Base<T> nrm2(std::size_t n, T* x, std::size_t incx) override
    {
        Base<T> nrm;
        cublas_status_ = cublasTnrm2(cublasH_, n, x, incx, &nrm);
        assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);
        return nrm;
    }

    /*!
      - For ChaseMpiDLACudaSeq, `dot` is implemented using `BLAS` routine
      `xDOT`.
      - **Parallelism is SUPPORT within node if multi-threading is actived**
      - For the meaning of this function, please visit ChaseMpiDLAInterface.
    */
    T dot(std::size_t n, T* x, std::size_t incx, T* y,
          std::size_t incy) override
    {
        T d;
        cublas_status_ = cublasTdot(cublasH_, n, x, incx, y, incy, &d);
        assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);
        return d;
    }

    /*!
      - For ChaseMpiDLACudaSeq, `gemm` is implemented using `BLAS` routine
      `xGEMM`.
      - **Parallelism is SUPPORT within node if multi-threading is actived**
      - For the meaning of this function, please visit ChaseMpiDLAInterface.
    */
    void gemm(CBLAS_LAYOUT Layout, CBLAS_TRANSPOSE transa,
              CBLAS_TRANSPOSE transb, std::size_t m, std::size_t n,
              std::size_t k, T* alpha, T* a, std::size_t lda, T* b,
              std::size_t ldb, T* beta, T* c, std::size_t ldc) override
    {
    }

    /*!
     - For ChaseMpiDLACudaSeq, `stemr` with scalar being real and double
     precision, is implemented using `LAPACK` routine `DSTEMR`.
      - **Parallelism is SUPPORT within node if multi-threading is actived**
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
     - For ChaseMpiDLACudaSeq, `stemr` with scalar being real and single
     precision, is implemented using `LAPACK` routine `SSTEMR`.
      - **Parallelism is SUPPORT within node if multi-threading is actived**
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
        - For ChaseMpiDLACudaSeq, `RR` is implemented by `cublasXgemm` routine
       provided by `cuBLAS` and `(SY)HEEVD` routine provided by `LAPACK`.
          - The 1st operation `A <- W^T * V` is implemented by `cublasXgemm`
       from `cuBLAS`.
          - The 2nd operation which computes the eigenpairs of `A`, is
       implemented by `(SY)HEEVD` from `LAPACK`.
          - The 3rd operation which computes `W<-V*A` is implemented by
       `cublasXgemm` from `cuBLAS`.
        - **for (SY)HHEVD, parallelism is SUPPORT within node if multi-threading
       is actived**
        - **for cublasXgemm, parallelism is SUPPORT within one GPU card**
        - For the meaning of this function, please visit ChaseMpiDLAInterface.
    */

    void RR(std::size_t block, std::size_t locked, Base<T>* ritzv) override
    {
        T One = T(1.0);
        T Zero = T(0.0);

        cublas_status_ = cublasTgemm(
            cublasH_, CUBLAS_OP_C, CUBLAS_OP_N, N_, block, N_, &One, d_H_, N_,
            d_V1_ + locked * N_, N_, &Zero, d_V2_ + locked * N_, N_);

        assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);

        cublas_status_ =
            cublasTgemm(cublasH_, CUBLAS_OP_C, CUBLAS_OP_N, block, block, N_,
                        &One, d_V2_ + locked * N_, N_, d_V1_ + locked * N_, N_,
                        &Zero, d_A_, max_block_);
        assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);
        cusolver_status_ = cusolverDnTheevd(
            cusolverH_, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, block,
            d_A_, max_block_, d_ritz_, d_work_, lwork_, devInfo_);
        assert(cusolver_status_ == CUSOLVER_STATUS_SUCCESS);

        cuda_exec(cudaMemcpy(ritzv, d_ritz_, block * sizeof(Base<T>),
                             cudaMemcpyDeviceToHost));

        cublas_status_ =
            cublasTgemm(cublasH_, CUBLAS_OP_N, CUBLAS_OP_N, N_, block, block,
                        &One, d_V1_ + locked * N_, N_, d_A_, max_block_, &Zero,
                        d_V2_ + locked * N_, N_);
        assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);

        std::swap(d_V1_, d_V2_);
    }

    void getLanczosBuffer(T** V1, T** V2, std::size_t* ld, T** v0, T** v1,
                          T** w) override
    {
        *V1 = d_V1_;
        *V2 = d_V2_;
        *ld = N_;
        cudaMemset(v0_, 0, sizeof(T) * N_);
        cudaMemset(v1_, 0, sizeof(T) * N_);
        cudaMemset(w_, 0, sizeof(T) * N_);
        *v0 = v0_;
        *v1 = v1_;
        *w = w_;
    }

    void getLanczosBuffer2(T** v0, T** v1, T** w) override
    {
        std::mt19937 gen(2342.0);
        std::normal_distribution<> normal_distribution;
        T* vtmp = new T[N_];
        for (std::size_t k = 0; k < N_; ++k)
        {
            vtmp[k] = getRandomT<T>([&]() { return normal_distribution(gen); });
        }
        cuda_exec(
            cudaMemcpy(v1_, vtmp, N_ * sizeof(T), cudaMemcpyHostToDevice));
        cudaMemset(v0_, 0, sizeof(T) * N_);
        cudaMemset(w_, 0, sizeof(T) * N_);
        *v0 = v0_;
        *v1 = v1_;
        *w = w_;
        delete[] vtmp;
    }

    void syherk(char uplo, char trans, std::size_t n, std::size_t k, T* alpha,
                T* a, std::size_t lda, T* beta, T* c, std::size_t ldc,
                bool first = true) override
    {
    }

    int potrf(char uplo, std::size_t n, T* a, std::size_t lda) override
    {
        return 0;
    }

    void trsm(char side, char uplo, char trans, char diag, std::size_t m,
              std::size_t n, T* alpha, T* a, std::size_t lda, T* b,
              std::size_t ldb, bool first = false) override
    {
    }

    void heevd(int matrix_layout, char jobz, char uplo, std::size_t n, T* a,
               std::size_t lda, Base<T>* w) override
    {
    }

    void Resd(Base<T>* ritzv, Base<T>* resid, std::size_t locked,
              std::size_t unconverged) override
    {
        T alpha = T(1.0);
        T beta = T(0.0);

        cublas_status_ = cublasTgemm(
            cublasH_, CUBLAS_OP_C, CUBLAS_OP_N, N_, unconverged, N_, &alpha,
            d_H_, N_, d_V1_ + locked * N_, N_, &beta, d_V2_ + locked * N_, N_);
        assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);

        for (std::size_t i = 0; i < unconverged; ++i)
        {
            beta = T(-ritzv[i]);
            cublas_status_ =
                cublasTaxpy(cublasH_, N_, &beta, (d_V1_ + locked * N_) + N_ * i,
                            1, (d_V2_ + locked * N_) + N_ * i, 1);
            assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);
            cublas_status_ =
                cublasTnrm2(cublasH2_, N_, (d_V2_ + locked * N_) + N_ * i, 1,
                            &d_resids_[i]);
            assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);
        }

        cuda_exec(cudaMemcpy(resid, d_resids_, unconverged * sizeof(Base<T>),
                             cudaMemcpyDeviceToHost));
    }

    void hhQR(std::size_t locked) override
    {
        auto nevex = nev_ + nex_;
        cuda_exec(cudaMemcpy(d_V2_, d_V1_, locked * N_ * sizeof(T),
                             cudaMemcpyDeviceToDevice));
        cusolver_status_ =
            cusolverDnTgeqrf(cusolverH_, N_, nevex, d_V1_, N_, d_return_,
                             d_work_, lwork_, devInfo_);
        assert(CUSOLVER_STATUS_SUCCESS == cusolver_status_);
        cusolver_status_ =
            cusolverDnTgqr(cusolverH_, N_, nevex, nevex, d_V1_, N_, d_return_,
                           d_work_, lwork_, devInfo_);
        assert(CUSOLVER_STATUS_SUCCESS == cusolver_status_);
        cuda_exec(cudaMemcpy(d_V1_, d_V2_, locked * N_ * sizeof(T),
                             cudaMemcpyDeviceToDevice));
    }

    void cholQR(std::size_t locked) override { this->hhQR(locked); }

    void Swap(std::size_t i, std::size_t j) override
    {
        cuda_exec(cudaMemcpy(v1_, d_V1_ + N_ * i, N_ * sizeof(T),
                             cudaMemcpyDeviceToDevice));
        cuda_exec(cudaMemcpy(d_V1_ + N_ * i, d_V1_ + N_ * j, N_ * sizeof(T),
                             cudaMemcpyDeviceToDevice));
        cuda_exec(cudaMemcpy(d_V1_ + N_ * j, v1_, N_ * sizeof(T),
                             cudaMemcpyDeviceToDevice));
    }

    void LanczosDos(std::size_t idx, std::size_t m, T* ritzVc) override
    {
        T alpha = T(1.0);
        T beta = T(0.0);
        cublasSetMatrix(m, idx, sizeof(T), ritzVc, m, d_A_, max_block_);
        cublas_status_ =
            cublasTgemm(cublasH_, CUBLAS_OP_N, CUBLAS_OP_N, N_, idx, m, &alpha,
                        d_V1_, N_, d_A_, max_block_, &beta, d_V2_, N_);
        assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);
        cuda_exec(cudaMemcpy(d_V1_, d_V2_, m * N_ * sizeof(T),
                             cudaMemcpyDeviceToDevice));
    }

private:
    std::size_t N_;
    std::size_t locked_;
    std::size_t max_block_;
    std::size_t nev_;
    std::size_t nex_;

    int* devInfo_ = NULL;
    T* d_V_ = NULL;
    T* d_return_ = NULL;
    T* d_work_ = NULL;
    int lwork_ = 0;
    cusolverStatus_t cusolver_status_ = CUSOLVER_STATUS_SUCCESS;
    cublasStatus_t cublas_status_ = CUBLAS_STATUS_SUCCESS;

    T* d_V1_;
    T* d_V2_;
    T* d_H_;
    T* H_;
    T* V1_;
    T* V2_;
    T* v0_;
    T* v1_;
    T* w_;
    Base<T>* d_ritz_ = NULL;
    T* d_A_;
    Base<T>* d_resids_ = NULL;
    cudaStream_t stream_, stream2_;
    cublasHandle_t cublasH_;
    cublasHandle_t cublasH2_;
    cublasHandle_t cublasBatched_[32];
    cudaStream_t streamBatched_[32];
    cusolverDnHandle_t cusolverH_;
    bool copied_;
};

template <typename T>
struct is_skewed_matrixfree<ChaseMpiDLACudaSeq<T>>
{
    static const bool value = false;
};

} // namespace mpi
} // namespace chase

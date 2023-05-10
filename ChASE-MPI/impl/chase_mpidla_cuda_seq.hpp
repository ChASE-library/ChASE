/* -*- Mode: C++; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
// This file is a part of ChASE.
// Copyright (c) 2015-2023, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include <cuComplex.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <assert.h>
#include <complex>

#include "ChASE-MPI/blas_cuda_wrapper.hpp"
#include "ChASE-MPI/chase_mpidla_interface.hpp"
/** @defgroup chase-cuda-utility Interface to the CUDA kernels functions used by
 * ChASE
 *  @brief This module provides the calling for CUDA:
 *     1. generate random number in normal distribution on device
 *     2. shift the diagonal of a matrix on single GPU.
 *     3. shift the diagonal of a global matrix which has already distributed on
 * multiGPUs (both block-block and block-cyclic distributions)
 *  @{
 */

//! shift the diagonal of a `nxn` square matrix `A` in float real data type on a
//! single GPU.
//!
//! @param[in,out] A a pointer to the matrix to be shifted
//! @param[in] n the row and column of matrix `A`
//! @param[in] shift the value for shifting the diagonal of matrix `A`
//! @param[in] stream_ an asynchronous CUDA stream which allows to run this
//! function asynchronously
void chase_shift_matrix(float* A, int n, float shift, cudaStream_t* stream_);
//! shift the diagonal of a `nxn` square matrix `A` in double real data type on
//! a single GPU.
//!
//! @param[in,out] A a pointer to the matrix to be shifted
//! @param[in] n the row and column of matrix `A`
//! @param[in] shift the value for shifting the diagonal of matrix `A`
//! @param[in] stream_ an asynchronous CUDA stream which allows to run this
//! function asynchronously
void chase_shift_matrix(double* A, int n, double shift, cudaStream_t* stream_);
//! shift the diagonal of a `nxn` square matrix `A` in float complex data type
//! on a single GPU.
//!
//! @param[in,out] A a pointer to the matrix to be shifted
//! @param[in] n the row and column of matrix `A`
//! @param[in] shift the value for shifting the diagonal of matrix `A`
//! @param[in] stream_ an asynchronous CUDA stream which allows to run this
//! function asynchronously
void chase_shift_matrix(std::complex<float>* A, int n, float shift,
                        cudaStream_t* stream_);
//! shift the diagonal of a `nxn` square matrix `A` in double complex data type
//! on a single GPU.
//!
//! @param[in,out] A a pointer to the matrix to be shifted
//! @param[in] n the row and column of matrix `A`
//! @param[in] shift the value for shifting the diagonal of matrix `A`
//! @param[in] stream_ an asynchronous CUDA stream which allows to run this
//! function asynchronously
void chase_shift_matrix(std::complex<double>* A, int n, double shift,
                        cudaStream_t* stream_);
/** @} */ // end of chase-cuda-utility
namespace chase
{
namespace mpi
{

//! @brief A derived class of ChaseMpiDLAInterface which implements ChASE
//! targeting shared-memory architectures, some selected computation tasks are
//! offloaded to one single GPU card.
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
          V1_(matrices.get_V1()), V2_(matrices.get_V2()), H_(matrices.get_H()),
          ldh_(matrices.get_ldh())
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
        cuda_exec(
            cudaMalloc((void**)&states_, sizeof(curandStatePhilox4_32_10_t) * (256 * 32)));

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
        if(states_)
            cudaFree(states_);

    }
    void initVecs() override
    {
        cuda_exec(cudaMemcpy(d_V1_, V1_, (nev_ + nex_) * N_ * sizeof(T),
                             cudaMemcpyHostToDevice));
        cuda_exec(cudaMemcpy(d_V2_, d_V1_, (nev_ + nex_) * N_ * sizeof(T),
                             cudaMemcpyDeviceToDevice));
        //cuda_exec(
        //    cudaMemcpy(d_H_, H_, N_ * N_ * sizeof(T), cudaMemcpyHostToDevice));
        cublasSetMatrix(N_, N_, sizeof(T), H_, ldh_, d_H_, N_);    
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

    void preApplication(T* V, std::size_t locked, std::size_t block) override
    {
        locked_ = locked;
        cuda_exec(cudaMemcpy(d_V1_, V + locked_ * N_, block * N_ * sizeof(T),
                             cudaMemcpyDeviceToDevice));
    }

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

    bool postApplication(T* V, std::size_t block, std::size_t locked) override
    {
        return false;
    }

    void shiftMatrix(T c, bool isunshift = false) override
    {
        chase_shift_matrix(d_H_, N_, std::real(c), &stream_);
    }

    void asynCxHGatherC(std::size_t locked, std::size_t block,
                        bool isCcopied = false) override
    {
    }

    void applyVec(T* B, T* C) override
    {
        T One = T(1.0);
        T Zero = T(0.0);

        cublas_status_ = cublasTgemm(cublasH_, CUBLAS_OP_N, CUBLAS_OP_N, N_, 1,
                                     N_, &One, d_H_, N_, B, N_, &Zero, C, N_);
        assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);
    }

    int get_nprocs() const override { return 1; }
    void Start() override {}
    void End() override
    {
        cuda_exec(cudaMemcpy(V1_, d_V1_, max_block_ * N_ * sizeof(T),
                             cudaMemcpyDeviceToHost));
    }

    void axpy(std::size_t N, T* alpha, T* x, std::size_t incx, T* y,
              std::size_t incy) override
    {
        cublas_status_ = cublasTaxpy(cublasH_, N, alpha, x, incx, y, incy);
        assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);
    }

    void scal(std::size_t N, T* a, T* x, std::size_t incx) override
    {
        cublas_status_ = cublasTscal(cublasH_, N, a, x, incx);
        assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);
    }

    Base<T> nrm2(std::size_t n, T* x, std::size_t incx) override
    {
        Base<T> nrm;
        cublas_status_ = cublasTnrm2(cublasH_, n, x, incx, &nrm);
        assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);
        return nrm;
    }

    T dot(std::size_t n, T* x, std::size_t incx, T* y,
          std::size_t incy) override
    {
        T d;
        cublas_status_ = cublasTdot(cublasH_, n, x, incx, y, incy, &d);
        assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);
        return d;
    }

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

    void cholQR(std::size_t locked, Base<T> cond) override
    {
        this->hhQR(locked);
    }

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
    void Lanczos(std::size_t M, int idx, Base<T>* d, Base<T>* e, Base<T> *r_beta) override
    {}

    void B2C(T* B, std::size_t off1, T* C, std::size_t off2, std::size_t block) override
    {}    
private:
    std::size_t N_;      //!< global dimension of the symmetric/Hermtian matrix
    std::size_t locked_; //!< the number of converged eigenpairs
    std::size_t max_block_; //!< `maxBlock_=nev_ + nex_`
    std::size_t nev_;       //!< number of required eigenpairs
    std::size_t nex_;       //!< number of extral searching space

    int* devInfo_ =
        NULL; //!< for the return of information from any cuSOLVER routines
    T* d_return_ = NULL; //!< a pointer to a local buffer of size `nev_+nex_`
    T* d_work_ =
        NULL; //!< a pointer to a local buffer on GPU, which is reserved for the
              //!< extra buffer required for any cuSOLVER routines
    int lwork_ = 0; //!< size of required extra buffer by any cuSOLVER routines
    cusolverStatus_t cusolver_status_ =
        CUSOLVER_STATUS_SUCCESS; //!< `cuSOLVER` status
    cublasStatus_t cublas_status_ = CUBLAS_STATUS_SUCCESS; //!< `cuBLAS` status

    T* d_V1_; //!< a pointer to a local buffer of size `N_*(nev_+nex_)` on GPU,
              //!< which is mapped to `V1_`.
    T* d_V2_; //!< a pointer to a local buffer of size `N_*(nev_+nex_)` on GPU,
              //!< which is mapped to `V2_`.
    T* d_H_;  //!< a pointer to a local buffer of size `N_*N_` on GPU, which is
              //!< mapped to `H_`.
    T* H_;    //!< a pointer to the Symmetric/Hermtian matrix
    std::size_t ldh_; //!< leading dimension of Hermitian matrix    
    T* V1_;   //!< a matrix of size `N_*(nev_+nex_)`
    T* V2_;   //!< a matrix of size `N_*(nev_+nex_)`
    T* v0_;   //!< a vector of size `N_`, which is allocated in this class for
              //!< Lanczos
    T* v1_;   //!< a vector of size `N_`, which is allocated in this class for
              //!< Lanczos
    T* w_;    //!< a vector of size `N_`, which is allocated in this class for
              //!< Lanczos
    Base<T>* d_ritz_ =
        NULL; //!< a pointer to a local buffer of size `nev_+nex_` on GPU for
              //!< storing computed ritz values
    T* d_A_;  //!< a matrix of size `(nev_+nex_)*(nev_+nex_)`
    Base<T>* d_resids_ =
        NULL; //!< a pointer to a local buffer of size `nev_+nex_` on GPU for
              //!< storing computed residuals
    cudaStream_t
        stream_; //!< CUDA stream for asynchronous exectution of kernels
    cudaStream_t
        stream2_; //!< CUDA stream for asynchronous exectution of kernels
    cublasHandle_t cublasH_;       //!< `cuBLAS` handle
    cublasHandle_t cublasH2_;      //!< `cuBLAS` handle
    cusolverDnHandle_t cusolverH_; //!< `cuSOLVER` handle
    bool copied_; //!< a flag indicates if the matrix has already been copied to
                  //!< device
    curandStatePhilox4_32_10_t *states_ = NULL;
};

template <typename T>
struct is_skewed_matrixfree<ChaseMpiDLACudaSeq<T>>
{
    static const bool value = false;
};

} // namespace mpi
} // namespace chase

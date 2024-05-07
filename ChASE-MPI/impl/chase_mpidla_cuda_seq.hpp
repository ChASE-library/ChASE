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
void residual_gpu(int m, int n, std::complex<double>* dA, int lda,
                  std::complex<double>* dB, int ldb, double* d_ritzv,
                  double* d_resids, bool is_sqrt, cudaStream_t stream_);
void residual_gpu(int m, int n, std::complex<float>* dA, int lda,
                  std::complex<float>* dB, int ldb, float* d_ritzv,
                  float* d_resids, bool is_sqrt, cudaStream_t stream_);
void residual_gpu(int m, int n, double* dA, int lda, double* dB, int ldb,
                  double* d_ritzv, double* d_resids, bool is_sqrt,
                  cudaStream_t stream_);
void residual_gpu(int m, int n, float* dA, int lda, float* dB, int ldb,
                  float* d_ritzv, float* d_resids, bool is_sqrt,
                  cudaStream_t stream_);
//! generate `n` random float numbers in normal distribution on each GPU device.
//!
//! @param[in] seed the seed of random number generator
//! @param[in] states the states of the sequence of random number generator
//! @param[in,out] v a pointer to the device memory to store the random
//! generated numbers
//! @param[in] stream_ an asynchronous CUDA stream which allows to run this
//! function asynchronously
void chase_rand_normal(unsigned long long seed, curandStatePhilox4_32_10_t* states, float* v,
                       int n, cudaStream_t stream_);
//! generate `n` random double numbers in normal distribution on each GPU
//! device.
//!
//! @param[in] seed the seed of random number generator
//! @param[in] states the states of the sequence of random number generator
//! @param[in,out] v a pointer to the device memory to store the random
//! generated numbers
//! @param[in] stream_ an asynchronous CUDA stream which allows to run this
//! function asynchronously
void chase_rand_normal(unsigned long long seed, curandStatePhilox4_32_10_t* states, double* v,
                       int n, cudaStream_t stream_);
//! generate `n` random complex float numbers in normal distribution on each GPU
//! device. The real part and the imaginary part of each individual random
//! number are the same.
//!
//! @param[in] seed the seed of random number generator
//! @param[in] states the states of the sequence of random number generator
//! @param[in,out] v a pointer to the device memory to store the random
//! generated numbers
//! @param[in] stream_ an asynchronous CUDA stream which allows to run this
//! function asynchronously
void chase_rand_normal(unsigned long long seed, curandStatePhilox4_32_10_t* states,
                       std::complex<float>* v, int n, cudaStream_t stream_);
//! generate `n` random complex double numbers in normal distribution on each
//! GPU device. The real part and the imaginary part of each individual random
//! number are the same.
//!
//! @param[in] seed the seed of random number generator
//! @param[in] states the states of the sequence of random number generator
//! @param[in,out] v a pointer to the device memory to store the random
//! generated numbers
//! @param[in] stream_ an asynchronous CUDA stream which allows to run this
//! function asynchronously
void chase_rand_normal(unsigned long long seed, curandStatePhilox4_32_10_t* states,
                       std::complex<double>* v, int n, cudaStream_t stream_);


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


void absTrace_gpu(float* d_matrix, float* d_trace, int n, int ld, cudaStream_t stream_);
void absTrace_gpu(double* d_matrix, double* d_trace, int n, int ld, cudaStream_t stream_);
void absTrace_gpu(std::complex<float>* d_matrix, float* d_trace, int n, int ld, cudaStream_t stream_);
void absTrace_gpu(std::complex<double>* d_matrix, double* d_trace, int n, int ld, cudaStream_t stream_);

// currently, only full copy is support
void t_lacpy_gpu(char uplo, int m, int n, float* dA, int ldda, float* dB,
                 int lddb, cudaStream_t stream_);
void t_lacpy_gpu(char uplo, int m, int n, double* dA, int ldda, double* dB,
                 int lddb, cudaStream_t stream_);
void t_lacpy_gpu(char uplo, int m, int n, std::complex<double>* ddA, int ldda,
                 std::complex<double>* ddB, int lddb, cudaStream_t stream_);
void t_lacpy_gpu(char uplo, int m, int n, std::complex<float>* ddA, int ldda,
                 std::complex<float>* ddB, int lddb, cudaStream_t stream_);

void find_max_and_add_abs_scalar_gpu(float *array, int size, float *scalar, float *max, cudaStream_t stream_);
void find_max_and_add_abs_scalar_gpu(double *array, int size, double *scalar, double *max, cudaStream_t stream_);
void find_max_and_add_abs_scalar_gpu(float *array, int size, std::complex<float> *scalar, float *max, cudaStream_t stream_);
void find_max_and_add_abs_scalar_gpu(double *array, int size, std::complex<double> *scalar, double *max, cudaStream_t stream_);
void computeNegative_gpu(float *d, float *neg, cudaStream_t stream_);
void computeNegative_gpu(double *d, double *neg, cudaStream_t stream_);
void computeNegative_gpu(std::complex<float> *d, std::complex<float> *neg, cudaStream_t stream_);
void computeNegative_gpu(std::complex<double> *d, std::complex<double> *neg, cudaStream_t stream_);
void InverseAndConvert_gpu(float *d, float *inv, cudaStream_t stream_);
void InverseAndConvert_gpu(double *d, double *inv, cudaStream_t stream_);
void InverseAndConvert_gpu(float *d, std::complex<float> *inv, cudaStream_t stream_);
void InverseAndConvert_gpu(double *d, std::complex<double> *inv, cudaStream_t stream_);
void convert2Real_gpu(float *a, float *b,  cudaStream_t stream_);
void convert2Real_gpu(double *a, double *b,  cudaStream_t stream_);
void convert2Real_gpu(std::complex<float> *a, float *b,  cudaStream_t stream_);
void convert2Real_gpu(std::complex<double> *a, double *b,  cudaStream_t stream_);
void convert2Complex_gpu(float *a, float *b,  cudaStream_t stream_);
void convert2Complex_gpu(double *a, double *b,  cudaStream_t stream_);
void convert2Complex_gpu(float *a, std::complex<float> *b,  cudaStream_t stream_);
void convert2Complex_gpu(double *a, std::complex<double> *b,  cudaStream_t stream_);
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
    ChaseMpiDLACudaSeq(T *H, std::size_t ldh, T *V1, Base<T> *ritzv, std::size_t N,
                       std::size_t nev, std::size_t nex)
        : N_(N), copied_(false), nev_(nev), nex_(nex), max_block_(nev + nex),
          matrices_(1, N_, nev_ + nex_, H, ldh, V1, ritzv)
    {
        H_  = matrices_.H().host();
        ldh_ = matrices_.get_ldh();
        d_H_  = matrices_.H().device();
        V1_ = matrices_.C().host();
        d_V1_ = matrices_.C().device();
        d_V2_ = matrices_.B().device();
	d_A_  = matrices_.A().device();
	d_ritz_ = matrices_.Ritzv().device();
	d_resids_ = matrices_.Resid().device();

        cuda_exec(cudaSetDevice(0));
        cuda_exec(cudaMalloc((void**)&(d_v1_), N_ * sizeof(T)));
        cuda_exec(cudaMalloc((void**)&(d_w_), N_ * sizeof(T)));
	
        cudaMallocHost(&v0_, N_ * sizeof(T));
        cudaMallocHost(&v1_, N_ * sizeof(T));
        cudaMallocHost(&w_, N_ * sizeof(T));

        cublasCreate(&cublasH_);
        cusolverDnCreate(&cusolverH_);
        cuda_exec(cudaStreamCreate(&stream_));
        cublasSetStream(cublasH_, stream_);
        cusolverDnSetStream(cusolverH_, stream_);

        cublasCreate(&cublasH2_);
        cublasSetPointerMode(cublasH2_, CUBLAS_POINTER_MODE_DEVICE);
        cublasSetStream(cublasH2_, stream_);

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

#if defined(USE_BLANCZOS)
        block_size = 8;
        int M = 25;
        alpha = Matrix<T>(2, block_size, block_size);
        beta = Matrix<T>(2, block_size, block_size);
        submatrix = Matrix<T>(2, M, M);
        v0 = Matrix<T>(2, N_, block_size);
        v1 = Matrix<T>(2, N_, block_size);
        w = Matrix<T>(2, N_, block_size);
        A = Matrix<T>(2, block_size, 2 * block_size);
        ritzv_ = Matrix<Base<T>>(2, M+1, 1);         

#else
        cuda_exec(cudaMalloc((void**)&(d_v0), 1 * N_ * sizeof(T))); 
        cuda_exec(cudaMalloc((void**)&(d_v1), 1 * N_ * sizeof(T))); 
        cuda_exec(cudaMalloc((void**)&(d_w), 1 * N_ * sizeof(T))); 
#endif


    }

    ~ChaseMpiDLACudaSeq()
    {
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
        if (d_v1_)
            cudaFree(d_v1_);
        if (d_w_)
            cudaFree(d_w_);
        if(states_)
            cudaFree(states_);
	cudaFreeHost(v0_);
	cudaFreeHost(v1_);
	cudaFreeHost(w_);
        if (d_v1)
            cudaFree(d_v1);    
        if (d_v0)
            cudaFree(d_v0); 
        if (d_v1)
            cudaFree(d_w);             
    }
    void initVecs() override
    {
	cuda_exec(cudaMemcpy(d_V2_, d_V1_, (nev_ + nex_) * N_ * sizeof(T),
                             cudaMemcpyDeviceToDevice));
        cublasSetMatrix(N_, N_, sizeof(T), H_, ldh_, d_H_, N_);    
    }
    void initRndVecs() override
    {
	unsigned long long seed = 24141;
        chase_rand_normal(seed, states_, d_V1_, N_ * (nev_ + nex_),
                          (cudaStream_t)0);
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

    void shiftMatrix(T c, bool isunshift = false) override
    {
        chase_shift_matrix(d_H_, N_, std::real(c), &stream_);
    }

    void asynCxHGatherC(std::size_t locked, std::size_t block,
                        bool isCcopied = false) override
    {
    }

    void applyVec(T* B, T* C, std::size_t n) override
    {
        T One = T(1.0);
        T Zero = T(0.0);

        if(n == 1){
            cublas_status_ = cublasTgemv(cublasH_, CUBLAS_OP_N, N_, N_, &One, d_H_, 
                                            N_, B, 1, &Zero, C, 1);                   
            assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);

        }else
        {
            cublas_status_ = cublasTgemm(cublasH_, CUBLAS_OP_N, CUBLAS_OP_N, N_, n,
                                        N_, &One, d_H_, N_, B, N_, &Zero, C, N_);
        
            assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);
        }

    }

    bool checkSymmetryEasy() override 
    {
        std::vector<T> v(N_);
        std::vector<T> u(N_);
        std::vector<T> uT(N_);

        std::mt19937 gen(1337.0);
        std::normal_distribution<> d;
        for (auto i = 0; i < N_; i++)
        {
            v[i] = getRandomT<T>([&]() { return d(gen); });
        }
        
        T One = T(1.0);
        T Zero = T(0.0);

        t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, //
               N_, 1, N_,                                 //
               &One,                                      //
               H_, ldh_,                                  //
               v.data(), N_,                             //
               &Zero,                                     //
               u.data(), N_);

        t_gemm(CblasColMajor, CblasConjTrans, CblasNoTrans, //
               N_, 1, N_,                                 //
               &One,                                      //
               H_, ldh_,                                  //
               v.data(), N_,                             //
               &Zero,                                     //
               uT.data(), N_);

        bool is_sym = true;
        for(auto i = 0; i < N_; i++)
        {
            if(!(u[i] == uT[i]))
            {
                is_sym = false;
                return is_sym;
            }
        }

        return is_sym;
    }

    void symOrHermMatrix(char uplo) override 
    {
        if(uplo == 'U')
        {
            for(auto j = 0; j < N_; j++)
            {
                for(auto i = 0; i < j; i++)
                {
                    H_[j + i * ldh_]= conjugate(H_[i + j * ldh_]);
                }
            }
        }else
        {
            for(auto i = 0; i < N_; i++)
            {
                for(auto j = 0; j < i; j++)
                {
                    H_[j + i * ldh_]= conjugate(H_[i + j * ldh_]);
                }
            }
        }
    }

    int get_nprocs() const override { return 1; }
    void Start() override {}
    void End() override
    {
        cuda_exec(cudaMemcpy(V1_, d_V1_, max_block_ * N_ * sizeof(T),
                             cudaMemcpyDeviceToHost));
    }
    Base<T> *get_Resids() override{
	return matrices_.Resid().host();   
    }

    Base<T> *get_Ritzv() override{
        return matrices_.Ritzv().host();
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

    int potrf(char uplo, std::size_t n, T* a, std::size_t lda, bool isinfo = true) override  
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

        residual_gpu(N_, unconverged, d_V2_ + locked * N_, N_,
                     d_V1_ + locked * N_, N_, d_ritz_,
                     d_resids_, true, (cudaStream_t)0);

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

    }

    int cholQR1(std::size_t locked) override
    {
        T one = T(1.0);
        T zero = T(0.0);

        Base<T> One = Base<T>(1.0);
        Base<T> Zero = Base<T>(0.0);
        cublasOperation_t transa;
        if (sizeof(T) == sizeof(Base<T>))
        {
            transa = CUBLAS_OP_T;
        }
        else
        {
            transa = CUBLAS_OP_C;
        }

        int info = 1;

        cuda_exec(cudaMemcpy(d_V2_, d_V1_, locked * N_ * sizeof(T),
                             cudaMemcpyDeviceToDevice));
        assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);

        cublas_status_ = cublasTsyherk(cublasH_, CUBLAS_FILL_MODE_UPPER, transa,
                                       nev_ + nex_, N_, &One, d_V1_, N_,
                                       &Zero, d_A_, nev_ + nex_);
        assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);

        cusolver_status_ = cusolverDnTpotrf(
            cusolverH_, CUBLAS_FILL_MODE_UPPER, nev_ + nex_, d_A_,
            nev_ + nex_, d_work_, lwork_, devInfo_);

        cuda_exec(cudaMemcpy(&info, devInfo_, 1 * sizeof(int),
                                cudaMemcpyDeviceToHost));
        if(info != 0)
        {
            return info;
        }else
        {
            cublas_status_ =
                cublasTtrsm(cublasH_, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
                            CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, N_, nev_ + nex_,
                            &one, d_A_, nev_ + nex_, d_V1_, N_);
#ifdef CHASE_OUTPUT
            std::cout << std::setprecision(2) << "choldegree: 1" << std::endl;
#endif                    
            return info;  
        }         
    }

    int cholQR2(std::size_t locked) override
    {
        T one = T(1.0);
        T zero = T(0.0);

        Base<T> One = Base<T>(1.0);
        Base<T> Zero = Base<T>(0.0);
        cublasOperation_t transa;
        if (sizeof(T) == sizeof(Base<T>))
        {
            transa = CUBLAS_OP_T;
        }
        else
        {
            transa = CUBLAS_OP_C;
        }

        int info = 1;

        cuda_exec(cudaMemcpy(d_V2_, d_V1_, locked * N_ * sizeof(T),
                             cudaMemcpyDeviceToDevice));

        cublas_status_ = cublasTsyherk(cublasH_, CUBLAS_FILL_MODE_UPPER, transa,
                                       nev_ + nex_, N_, &One, d_V1_, N_,
                                       &Zero, d_A_, nev_ + nex_);
        assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);

        cusolver_status_ = cusolverDnTpotrf(
            cusolverH_, CUBLAS_FILL_MODE_UPPER, nev_ + nex_, d_A_,
            nev_ + nex_, d_work_, lwork_, devInfo_);

        cuda_exec(cudaMemcpy(&info, devInfo_, 1 * sizeof(int),
                                cudaMemcpyDeviceToHost));
        
        if(info != 0)
        {
            return info;
        }else
        {
            cublas_status_ =
                cublasTtrsm(cublasH_, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
                            CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, N_, nev_ + nex_,
                            &one, d_A_, nev_ + nex_, d_V1_, N_);

            cublas_status_ = cublasTsyherk(cublasH_, CUBLAS_FILL_MODE_UPPER, transa,
                                        nev_ + nex_, N_, &One, d_V1_, N_,
                                        &Zero, d_A_, nev_ + nex_);
            assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);

            cusolver_status_ = cusolverDnTpotrf(
                cusolverH_, CUBLAS_FILL_MODE_UPPER, nev_ + nex_, d_A_,
                nev_ + nex_, d_work_, lwork_, devInfo_);

            cublas_status_ =
                cublasTtrsm(cublasH_, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
                            CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, N_, nev_ + nex_,
                            &one, d_A_, nev_ + nex_, d_V1_, N_);                
#ifdef CHASE_OUTPUT
            std::cout << std::setprecision(2) << "choldegree: 2" << std::endl;
#endif                    
            return info;  
        } 
    }   

    int shiftedcholQR2(std::size_t locked) override
    {
        T one = T(1.0);
        T zero = T(0.0);

        Base<T> One = Base<T>(1.0);
        Base<T> Zero = Base<T>(0.0);
        Base<T> shift;
        cublasOperation_t transa;
        if (sizeof(T) == sizeof(Base<T>))
        {
            transa = CUBLAS_OP_T;
        }
        else
        {
            transa = CUBLAS_OP_C;
        }

        int info = 1;

        cuda_exec(cudaMemcpy(d_V2_, d_V1_, locked * N_ * sizeof(T),
                             cudaMemcpyDeviceToDevice));

        cublas_status_ = cublasTsyherk(cublasH_, CUBLAS_FILL_MODE_UPPER, transa,
                                       nev_ + nex_, N_, &One, d_V1_, N_,
                                       &Zero, d_A_, nev_ + nex_);
        assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);

        Base<T> nrmf = 0.0;
        this->computeDiagonalAbsSum(d_A_, &nrmf, nev_ + nex_, nev_ + nex_);

        shift = std::sqrt(N_) * nrmf * std::numeric_limits<Base<T>>::epsilon();
        this->shiftMatrixForQR(d_A_, nev_ + nex_, (T)shift);      

        cusolver_status_ = cusolverDnTpotrf(
            cusolverH_, CUBLAS_FILL_MODE_UPPER, nev_ + nex_, d_A_,
            nev_ + nex_, d_work_, lwork_, devInfo_);

        cuda_exec(cudaMemcpy(&info, devInfo_, 1 * sizeof(int),
                                cudaMemcpyDeviceToHost));
        
        if(info != 0)
        {
            return info;
        }

        cublas_status_ =
        cublasTtrsm(cublasH_, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
                    CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, N_, nev_ + nex_,
                    &one, d_A_, nev_ + nex_, d_V1_, N_);

        cublas_status_ = cublasTsyherk(cublasH_, CUBLAS_FILL_MODE_UPPER, transa,
                                    nev_ + nex_, N_, &One, d_V1_, N_,
                                    &Zero, d_A_, nev_ + nex_);
        assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);

        cusolver_status_ = cusolverDnTpotrf(
            cusolverH_, CUBLAS_FILL_MODE_UPPER, nev_ + nex_, d_A_,
            nev_ + nex_, d_work_, lwork_, devInfo_);

        cuda_exec(cudaMemcpy(&info, devInfo_, 1 * sizeof(int),
                                cudaMemcpyDeviceToHost));
        
        if(info != 0)
        {
            return info;
        }
        
        cublas_status_ =
            cublasTtrsm(cublasH_, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
                        CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, N_, nev_ + nex_,
                        &one, d_A_, nev_ + nex_, d_V1_, N_);                                            

        cublas_status_ = cublasTsyherk(cublasH_, CUBLAS_FILL_MODE_UPPER, transa,
                                    nev_ + nex_, N_, &One, d_V1_, N_,
                                    &Zero, d_A_, nev_ + nex_);
        assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);

        cusolver_status_ = cusolverDnTpotrf(
            cusolverH_, CUBLAS_FILL_MODE_UPPER, nev_ + nex_, d_A_,
            nev_ + nex_, d_work_, lwork_, devInfo_);

        cublas_status_ =
            cublasTtrsm(cublasH_, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
                        CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, N_, nev_ + nex_,
                        &one, d_A_, nev_ + nex_, d_V1_, N_);    
#ifdef CHASE_OUTPUT
        std::cout << std::setprecision(2) << "choldegree: 2, shift = " << shift << std::endl;
#endif 
        return info;
    }

    void estimated_cond_evaluator(std::size_t locked, Base<T> cond)
    {
        auto nevex = nev_ + nex_;
        std::vector<Base<T>> S(nevex - locked);
        std::vector<Base<T>> norms(nevex - locked);
        std::vector<T> V2(N_ * (nevex));

        cuda_exec(cudaMemcpy(V2.data(), d_V1_, N_ * nevex * sizeof(T),
                                cudaMemcpyDeviceToHost));
        T* U;
        std::size_t ld = 1;
        T* Vt;
        t_gesvd('N', 'N', N_, nevex - locked, V2.data() + N_ * locked,
                N_, S.data(), U, ld, Vt, ld);
        
        for (auto i = 0; i < nevex - locked; i++)
        {
            norms[i] = std::sqrt(t_sqrt_norm(S[i]));
        }
        
        std::sort(norms.begin(), norms.end());

        std::cout << "estimate: " << cond << ", rcond: "
                    << norms[nev_ + nex_ - locked - 1] / norms[0]
                    << ", ratio: "
                    << cond * norms[0] / norms[nev_ + nex_ - locked - 1]
                    << std::endl;

    }

    void lockVectorCopyAndOrthoConcatswap(std::size_t locked, bool isHHqr)
    {
        cuda_exec(cudaMemcpy(d_V1_, d_V2_, locked * N_ * sizeof(T),
                             cudaMemcpyDeviceToDevice));        
    } 

    void Swap(std::size_t i, std::size_t j) override
    {
        cuda_exec(cudaMemcpy(d_v1_, d_V1_ + N_ * i, N_ * sizeof(T),
                             cudaMemcpyDeviceToDevice));
        cuda_exec(cudaMemcpy(d_V1_ + N_ * i, d_V1_ + N_ * j, N_ * sizeof(T),
                             cudaMemcpyDeviceToDevice));
        cuda_exec(cudaMemcpy(d_V1_ + N_ * j, d_v1_, N_ * sizeof(T),
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

    void Lanczos(std::size_t M, Base<T> *r_beta) override 
    {
#ifdef USE_BLANCZOS  
        std::cout << "calling Block Lanczos..." << std::endl;

        std::size_t block_size = 8;
        std::size_t div = std::floor(static_cast<double>(M) / block_size);
        M = 25;
        //M = (div * block_size + 1 == M) ? M : block_size * (div + 1) + 1;
        //std::vector<Base<T>> ritzv_(M);

        T One = T(1.0);
        T Zero = T(0.0);
        T NegOne = T(-1.0);   

        Base<T> one = Base<T>(1.0);
        Base<T> zero = Base<T>(0.0);   

        cuda_exec(cudaMemcpy(v1.device(), d_V1_, N_ * block_size * sizeof(T), cudaMemcpyDeviceToDevice));
        //CholQR
        cublasOperation_t transa;
        if (sizeof(T) == sizeof(Base<T>))
        {
            transa = CUBLAS_OP_T;
        }
        else
        {
            transa = CUBLAS_OP_C;
        }

        cusolver_status_ =
            cusolverDnTgeqrf(cusolverH_, N_, block_size, v1.device(), N_, d_return_,
                             d_work_, lwork_, devInfo_);
        assert(CUSOLVER_STATUS_SUCCESS == cusolver_status_);
        cusolver_status_ =
            cusolverDnTgqr(cusolverH_, N_, block_size, block_size, v1.device(), N_, d_return_,
                           d_work_, lwork_, devInfo_);
        assert(CUSOLVER_STATUS_SUCCESS == cusolver_status_);
                
/*
        int info = -1;
        cublas_status_ = cublasTsyherk(cublasH_, CUBLAS_FILL_MODE_UPPER, transa,
                                block_size, N_, &one, v1.device(), N_,
                                &zero, A.device(), block_size);
        assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);

        cusolver_status_ = cusolverDnTpotrf(
            cusolverH_, CUBLAS_FILL_MODE_UPPER, block_size, A.device(),
            block_size, d_work_, lwork_, devInfo_);

        cuda_exec(cudaMemcpy(&info, devInfo_, 1 * sizeof(int),
                                cudaMemcpyDeviceToHost));

        if(info != 0)
        {
            std::vector<T> v_tmp(N_ * block_size);
            std::vector<T> tau(block_size);
            cuda_exec(cudaMemcpy(v_tmp.data(), v1.device(), block_size * N_ * sizeof(T), cudaMemcpyDeviceToHost));
            t_geqrf(LAPACK_COL_MAJOR, N_, block_size, v_tmp.data(), N_, tau.data());
            t_gqr(LAPACK_COL_MAJOR, N_, block_size, block_size, v_tmp.data(), N_, tau.data());
            cuda_exec(cudaMemcpy(v1.device(), v_tmp.data(), block_size * N_ * sizeof(T), cudaMemcpyHostToDevice));            
        }else
        {
            cublas_status_ =
                cublasTtrsm(cublasH_, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
                            CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, N_, block_size,
                            &One, A.device(), block_size, v1.device(), N_);   
            assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);                              
        }   
*/
        for(auto k = 0; k < M - 1; k = k + block_size)
        {   
            auto nb = std::min(block_size, M - k);
            this->applyVec(v1.device(), w.device(), nb);
            //alpha = v1^T * w
            cublas_status_ = cublasTgemm(
                cublasH_, CUBLAS_OP_C, CUBLAS_OP_N, nb, nb, N_, &One,
                v1.device(), N_, w.device(), N_, &Zero, alpha.device(), block_size);
            assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);

            //w = - v * alpha + w
            cublas_status_ = cublasTgemm(
                cublasH_, CUBLAS_OP_N, CUBLAS_OP_N, N_, nb, nb, &NegOne,
                v1.device(), N_, alpha.device(), block_size, &One, w.device(), N_);
            assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);

            t_lacpy_gpu('A', nb, nb, alpha.device(), block_size, submatrix.device() + k + k * M, M, NULL);

            //w = - v0 * beta + w
            if(k > 0)
            {
                cublas_status_ = cublasTgemm(
                    cublasH_, CUBLAS_OP_N, CUBLAS_OP_N, N_, nb, nb, &NegOne,
                    v0.device(), N_, beta.device(), block_size, &One, w.device(), N_);
                assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);            
            }

            cusolver_status_ =
                cusolverDnTgeqrf(cusolverH_, N_, nb, w.device(), N_, d_return_,
                                d_work_, lwork_, devInfo_);
            assert(CUSOLVER_STATUS_SUCCESS == cusolver_status_);

            t_lacpy_gpu('U', nb, nb, w.device(), N_, beta.device(), block_size, NULL);

            cusolver_status_ =
                cusolverDnTgqr(cusolverH_, N_, nb, nb, w.device(), N_, d_return_,
                            d_work_, lwork_, devInfo_);
            assert(CUSOLVER_STATUS_SUCCESS == cusolver_status_);  
            //CholeskyQR2
/*            
            cublas_status_ = cublasTsyherk(cublasH_, CUBLAS_FILL_MODE_UPPER, transa,
                                    nb, N_, &one, w.device(), N_,
                                    &zero, A.device(), block_size);
            assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);  

            cusolver_status_ = cusolverDnTpotrf(
                cusolverH_, CUBLAS_FILL_MODE_UPPER, nb, A.device(),
                block_size, d_work_, lwork_, devInfo_);
            
            cuda_exec(cudaMemcpy(&info, devInfo_, 1 * sizeof(int),
                                    cudaMemcpyDeviceToHost));  

            if(info == 0)
            {
                cublas_status_ =
                    cublasTtrsm(cublasH_, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
                                CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, N_, nb,
                                &One, A.device(), block_size, w.device(), N_);   
                assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);  

                cublas_status_ = cublasTsyherk(cublasH_, CUBLAS_FILL_MODE_UPPER, transa,
                                        nb, N_, &one, w.device(), N_,
                                        &zero, A.device() + block_size * block_size, block_size);
                assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);     

                cusolver_status_ = cusolverDnTpotrf(
                    cusolverH_, CUBLAS_FILL_MODE_UPPER, nb, A.device() + block_size * block_size,
                    block_size, d_work_, lwork_, devInfo_);     

                cublas_status_ =
                    cublasTtrsm(cublasH_, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
                                CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, N_, nb,
                                &One, A.device() + block_size * block_size, block_size, w.device(), N_);   
                assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);                      

                cublas_status_ = cublasTgemm(
                    cublasH_, CUBLAS_OP_N, CUBLAS_OP_N, nb, nb, nb, &One,
                    A.device() + block_size * block_size, block_size, A.device(), block_size, &Zero, beta.device(), block_size);
                assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);  
                     
            }else
            {
                std::vector<T> w_tmp(N_ * nb);
                std::vector<T> tau(nb);
                cuda_exec(cudaMemcpy(w_tmp.data(), w.device(), nb * N_ * sizeof(T), cudaMemcpyDeviceToHost));
                t_geqrf(LAPACK_COL_MAJOR, N_, nb, w_tmp.data(), N_, tau.data());
                cublas_status_ = cublasSetMatrix(nb, nb, sizeof(T), w_tmp.data(), N_, beta.device(), block_size);
                t_gqr(LAPACK_COL_MAJOR, N_, nb, nb, w_tmp.data(), N_, tau.data());
                cuda_exec(cudaMemcpy(w.device(), w_tmp.data(), nb * N_ * sizeof(T), cudaMemcpyHostToDevice));                  
            }
*/

            if(k > 0)
            {
                t_lacpy_gpu('U', nb, nb, beta.device(), block_size, submatrix.device() + k * M + (k - 1), M, NULL);
            }            

            v1.swap(v0);
            v1.swap(w);   
        }

        T *d_alpha, *d_scalar;
        Base<T> *d_real_alpha;
        cuda_exec(cudaMalloc((void**)&d_real_alpha, sizeof(Base<T>)));
        cuda_exec(cudaMalloc((void**)&d_alpha, sizeof(T)));
        cuda_exec(cudaMalloc((void**)&d_scalar, sizeof(T)));

        this->applyVec(v1.device(), w.device(), 1);
        cublas_status_ = cublasTdot(cublasH2_, N_, v1.device(), 1, w.device(), 1, d_alpha);
        assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);
        computeNegative_gpu(d_alpha, d_scalar, stream_);
        cublas_status_ = cublasTaxpy(cublasH2_, N_, d_scalar, v1.device(), 1, w.device(), 1);
        assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);        
        cudaMemcpy(submatrix.device()+ M * M - 1, d_alpha, sizeof(T), cudaMemcpyDeviceToDevice);

        computeNegative_gpu(&(beta.device()[0]), d_scalar, stream_); 
        cublas_status_ = cublasTaxpy(cublasH2_, N_, d_scalar, v0.device(), 1, w.device(), 1);
        assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);

        cublas_status_ = cublasTnrm2(cublasH2_, N_, w.device(), 1, d_real_alpha);
        assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);
        convert2Complex_gpu(d_real_alpha, d_scalar, stream_);
            
        cusolver_status_ = cusolverDnTheevd(
            cusolverH_, CUSOLVER_EIG_MODE_NOVECTOR, CUBLAS_FILL_MODE_UPPER, M,
            submatrix.device(), M, ritzv_.device(), d_work_, lwork_, devInfo_);
        assert(cusolver_status_ == CUSOLVER_STATUS_SUCCESS);
        //ritzv allocated as M+1, the first M contains ritzv, and the last one is reserved
        //for the output of function below
        //find_max_and_add_abs_scalar_gpu(ritzv_.device(), (int)M, &beta.device()[0], &(ritzv_.device()[M]), NULL);
        find_max_and_add_abs_scalar_gpu(ritzv_.device(), (int)M, d_scalar, &(ritzv_.device()[M]), NULL);
        cudaMemcpy(r_beta, &(ritzv_.device()[M]), 1 * sizeof(Base<T>), cudaMemcpyDeviceToHost);

#else
        std::cout << "calling Lanczos..." << std::endl;
/*
        Matrix<Base<T>> ritzv_(2, M + 1, 1);
        Matrix<Base<T>> subMatrix(2, M, M); 
        Base<T> *real_alpha, *real_beta;
        T *alpha, *beta;
        T One = T(1.0);
        T Zero = T(0.0);
        cuda_exec(cudaMalloc((void**)&real_alpha, sizeof(Base<T>)));
        cuda_exec(cudaMalloc((void**)&real_beta, sizeof(Base<T>)));
        cuda_exec(cudaMalloc((void**)&alpha, sizeof(T)));
        cuda_exec(cudaMalloc((void**)&beta, sizeof(T)));

        cuda_exec(cudaMemcpy(d_v1, d_V1_, 
                    1 * N_ * sizeof(T), cudaMemcpyDeviceToDevice)); 

        cublas_status_ = cublasTnrm2(cublasH2_, N_, d_V2_, 1, real_alpha);
        assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);
        
        InverseAndConvert_gpu(real_alpha, alpha, stream_);
        cublas_status_ = cublasTscal(cublasH2_, N_, alpha, d_v1, 1);
        assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);

        for (std::size_t k = 0; k < M; k = k + 1)
        {
            cublas_status_ = cublasTgemv(cublasH_, CUBLAS_OP_N, N_, N_, &One, d_H_, 
                                N_, d_v1, 1, &Zero, d_w, 1);                   
            assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);

            cublas_status_ = cublasTdot(cublasH2_, N_, d_v1, 1, d_w, 1, alpha);
            assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);
            convert2Real_gpu(alpha, real_alpha, stream_);
            computeNegative_gpu(alpha, beta, stream_);
            cublas_status_ = cublasTaxpy(cublasH2_, N_, beta, d_v1, 1, d_w, 1);
            assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);
            cudaMemcpy(subMatrix.device()+ k + k * M, real_alpha, sizeof(Base<T>), cudaMemcpyDeviceToDevice);
            
            if(k > 0)
            {
                convert2Complex_gpu(real_beta, beta, stream_);
                computeNegative_gpu(beta, beta, stream_);
                cublas_status_ = cublasTaxpy(cublasH2_, N_, beta, d_v0, 1, d_w, 1);
                assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);
            }

            cublas_status_ = cublasTnrm2(cublasH2_, N_, d_w, 1, real_beta);
            assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);
            InverseAndConvert_gpu(real_beta, beta, stream_);

            if (k == M - 1)
                break;
           
            cublas_status_ = cublasTscal(cublasH2_, N_, beta, d_w, 1);
            assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);
            if(k > 0){
                cudaMemcpy(&(subMatrix.device()[k - 1 + k * M]), real_beta, 1 * sizeof(Base<T>), cudaMemcpyDeviceToDevice);
            }
            std::swap(d_v1,d_v0);
            std::swap(d_v1, d_w);  
        } 

        int lwork_heevd = 0;
        cusolver_status_ = cusolverDnTheevd_bufferSize(
            cusolverH_, CUSOLVER_EIG_MODE_NOVECTOR, CUBLAS_FILL_MODE_UPPER,
            M, subMatrix.device(), M, ritzv_.device(), &lwork_heevd);

        Matrix<Base<T>> dwork__(2, lwork_heevd, 1);

        cusolver_status_ = cusolverDnTheevd(
            cusolverH_, CUSOLVER_EIG_MODE_NOVECTOR, CUBLAS_FILL_MODE_UPPER, M,
            subMatrix.device(), M, ritzv_.device(), dwork__.device(), lwork_heevd, devInfo_);
        assert(cusolver_status_ == CUSOLVER_STATUS_SUCCESS);
        //ritzv allocated as M+1, the first M contains ritzv, and the last one is reserved
        //for the output of function below
        find_max_and_add_abs_scalar_gpu(ritzv_.device(), (int)M, real_beta, &(ritzv_.device()[M]), stream_);
        cudaMemcpy(r_beta, &(ritzv_.device()[M]), 1 * sizeof(Base<T>), cudaMemcpyDeviceToHost);

*/
        std::vector<Base<T>> ritzv_(M);

        Base<T> real_beta;
        std::vector<Base<T>> d(M);
        std::vector<Base<T>> e(M);

        T alpha = T(1.0);
        T beta = T(0.0);

        cuda_exec(cudaMemcpy(d_v1, d_V1_, 
                    1 * N_ * sizeof(T), cudaMemcpyDeviceToDevice));  

        Base<T> real_alpha = this->nrm2(N_,d_V2_, 1);
        alpha = T(1 / real_alpha);
        this->scal(N_, &alpha, d_v1, 1);    

        for (std::size_t k = 0; k < M; k = k + 1)
        {
            this->applyVec(d_v1, d_w, 1);

            alpha = this->dot(N_, d_v1, 1, d_w, 1);
            alpha = -alpha;
            this->axpy(N_, &alpha, d_v1, 1, d_w, 1);
            alpha = -alpha;
            d[k] = std::real(alpha);                
        
            if(k > 0){
                beta = T(-real_beta);
                this->axpy(N_, &beta, d_v0, 1, d_w, 1);
                beta = -beta; 
            }

            real_beta = this->nrm2(N_, d_w, 1);
            beta = T(1.0 / real_beta); 
        
            if (k == M - 1)
                break;
           
            this->scal(N_, &beta, d_w, 1);
            e[k] = real_beta;
            
            std::swap(d_v1,d_v0);
            std::swap(d_v1, d_w);            
        }            
        int notneeded_m;
        std::size_t vl, vu;
        Base<T> ul, ll;
        int tryrac = 0;
        std::vector<int> isuppz(2 * M);

        t_stemr<Base<T>>(LAPACK_COL_MAJOR, 'N', 'A', M, d.data(), e.data(), ul, ll, vl, vu,
                         &notneeded_m, ritzv_.data(), NULL, M, M, isuppz.data(), &tryrac);

        *r_beta = std::max(std::abs(ritzv_[0]), std::abs(ritzv_[M - 1])) +
                  std::abs(real_beta);
                  
#endif                   
                    
    }
    void mLanczos(std::size_t M, int numvec, Base<T>* d, Base<T>* e, Base<T> *r_beta) override
    {
        std::vector<T> alpha(numvec, T(1.0));
        std::vector<T> beta(numvec, T(0.0));     
        std::vector<Base<T>> real_alpha(numvec);
        std::vector<Base<T>> real_beta(numvec);  

        T *d_v0, *d_v1, *d_w;
        cuda_exec(cudaMalloc((void**)&(d_v0), numvec * N_ * sizeof(T))); 
        cuda_exec(cudaMalloc((void**)&(d_v1), numvec * N_ * sizeof(T))); 
        cuda_exec(cudaMalloc((void**)&(d_w), numvec * N_ * sizeof(T))); 

        cuda_exec(cudaMemset(d_v0, 0, numvec * N_ * sizeof(T)));

        cuda_exec(cudaMemcpy(d_v1, d_V2_, 
                        numvec * N_ * sizeof(T), cudaMemcpyDeviceToDevice));  
        
        for(auto i = 0; i < numvec; i++)
        {             
            real_alpha[i] = this->nrm2(N_,d_V2_ + i * N_, 1);
            alpha[i] = T(1 / real_alpha[i]);
            this->scal(N_, &alpha[i], d_v1 + i * N_, 1);
        }

        for (std::size_t k = 0; k < M; k = k + 1)
        {
            cuda_exec(cudaMemcpy(d_V1_, d_v1, 
                        numvec * N_ * sizeof(T), cudaMemcpyDeviceToDevice));  

            this->applyVec(d_v1, d_w, numvec);

            for(auto i = 0; i < numvec; i++)
            {
                alpha[i] = this->dot(N_, d_v1 + i * N_, 1, d_w + i * N_, 1);
                alpha[i] = -alpha[i];
                this->axpy(N_, &alpha[i], d_v1 + i * N_, 1, d_w + i * N_, 1);
                alpha[i] = -alpha[i];
                d[k + M * i] = std::real(alpha[i]);                
            }
       
            for(auto i = 0; i < numvec; i++)
            {
                beta[i] = T(-real_beta[i]);
                this->axpy(N_, &beta[i], d_v0 + i * N_, 1, d_w + i * N_, 1);
                beta[i] = -beta[i]; 
                real_beta[i] = this->nrm2(N_, d_w + i * N_, 1);
                beta[i] = T(1.0 / real_beta[i]); 
            }

            if (k == M - 1)
                break;

            for(auto i = 0; i < numvec; i++)
            {             
                this->scal(N_, &beta[i], d_w + i * N_, 1);
            }

            for(auto i = 0; i < numvec; i++)
            {
                e[k + M * i] = real_beta[i];
            }
            
            std::swap(d_v1,d_v0);
            std::swap(d_v1, d_w);            
        }

        for(auto i = 0; i < numvec; i++)
        {
            r_beta[i] = real_beta[i];
        }

        if (d_v0)
            cudaFree(d_v0);
        if (d_v1)
            cudaFree(d_v1);
        if (d_w)
            cudaFree(d_w);    
              
    }
    
    void B2C(T* B, std::size_t off1, T* C, std::size_t off2, std::size_t block) override
    {}

    void lacpy(char uplo, std::size_t m, std::size_t n,
             T* a, std::size_t lda, T* b, std::size_t ldb) override
    {}

    void shiftMatrixForQR(T *A, std::size_t n, T shift) override
    {
        chase_shift_matrix(A, n, std::real(shift), &stream_);
    }

    void computeDiagonalAbsSum(T *A, Base<T> *sum, std::size_t n, std::size_t ld)
    {
        Base<T> *d_sum;
        cudaMalloc((void**)&d_sum, sizeof(Base<T>));
        absTrace_gpu(A, d_sum, n, ld, (cudaStream_t)0);
        cudaMemcpy(sum, d_sum, sizeof(Base<T>), cudaMemcpyDeviceToHost);
    }

    ChaseMpiMatrices<T> *getChaseMatrices() override
    {
        return &matrices_;    
    }
    
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
    T* d_v1_;
    T* d_w_;

    Base<T>* d_ritz_ =
        NULL; //!< a pointer to a local buffer of size `nev_+nex_` on GPU for
              //!< storing computed ritz values
    T* d_A_;  //!< a matrix of size `(nev_+nex_)*(nev_+nex_)`
    Base<T>* d_resids_ =
        NULL; //!< a pointer to a local buffer of size `nev_+nex_` on GPU for
              //!< storing computed residuals
    cudaStream_t
        stream_; //!< CUDA stream for asynchronous exectution of kernels
    cublasHandle_t cublasH_;       //!< `cuBLAS` handle
    cublasHandle_t cublasH2_;       //!< `cuBLAS` handle

    cusolverDnHandle_t cusolverH_; //!< `cuSOLVER` handle
    bool copied_; //!< a flag indicates if the matrix has already been copied to
                  //!< device
    curandStatePhilox4_32_10_t *states_ = NULL;

    ChaseMpiMatrices<T> matrices_;

#if defined(USE_BLANCZOS)
    int block_size;    
    Matrix<T> alpha;
    Matrix<T> beta;
    Matrix<T> submatrix;
    Matrix<T> v0;
    Matrix<T> v1;
    Matrix<T> w;
    Matrix<T> A;
    Matrix<Base<T>> ritzv_;    
#endif
    T *d_v0, *d_v1, *d_w;
};

template <typename T>
struct is_skewed_matrixfree<ChaseMpiDLACudaSeq<T>>
{
    static const bool value = false;
};

} // namespace mpi
} // namespace chase



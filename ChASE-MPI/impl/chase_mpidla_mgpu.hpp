/* -*- Mode: C++; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
// This file is a part of ChASE.
// Copyright (c) 2015-2023, Simulation and Data Laboratory Quantum Materials,
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

/** @defgroup chase-cuda-utility Interface to the CUDA kernels functions used by
 * ChASE
 *  @brief This module provides the calling for CUDA:
 *     1. generate random number in normal distribution on device
 *     2. shift the diagonal of a matrix on single GPU.
 *     3. shift the diagonal of a global matrix which has already distributed on
 * multiGPUs (both block-block and block-cyclic distributions)
 *  @{
 */

void residual_gpu(int m, int n, std::complex<double> *dA, int lda, std::complex<double> *dB,
                         int ldb, double *d_ritzv, double *d_resids, bool is_sqrt, cudaStream_t stream_);
void residual_gpu(int m, int n, std::complex<float> *dA, int lda, std::complex<float> *dB,
                         int ldb, float *d_ritzv, float *d_resids, bool is_sqrt, cudaStream_t stream_);
void residual_gpu(int m, int n, double *dA, int lda,  double *dB,
                         int ldb, double *d_ritzv, double *d_resids, bool is_sqrt, cudaStream_t stream_);
void residual_gpu(int m, int n, float *dA, int lda, float *dB,
                         int ldb, float *d_ritzv, float *d_resids, bool is_sqrt, cudaStream_t stream_);

//currently, only full copy is support
void t_lacpy_gpu(char uplo, int m, int n, float *dA, int ldda, float *dB, int lddb, cudaStream_t stream_ );
void t_lacpy_gpu(char uplo, int m, int n, double *dA, int ldda, double *dB, int lddb, cudaStream_t stream_ );
void t_lacpy_gpu(char uplo, int m, int n, std::complex<double> *ddA, int ldda, std::complex<double> *ddB, int lddb, cudaStream_t stream_ );
void t_lacpy_gpu(char uplo, int m, int n, std::complex<float> *ddA, int ldda, std::complex<float> *ddB, int lddb, cudaStream_t stream_ );
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
//! shift the diagonal of a `nxn` square matrix `A` in float real data type.
//! which has been distributed on multi-GPUs in either block-block of
//! block-cyclic faison. Each GPU may contains different number of diagonal part
//! of the global matrix, espeically for the block-cyclic distribution On each
//! GPU, it may contains multiple pieces of diagonal to be shifted, and each
//! piece may be of different size. Hence each piece is determined by `off_m`,
//! `off_n`, `offsize` within the local matrix on each GPU device.
//!
//! @param[in,out] A a pointer to the local piece of matrix to be shifted on
//! each GPU
//! @param[in] off_m the offset of the row of the first element each piece of
//! diagonal to be shifted within the local matrix `A`
//! @param[in] off_n the offset of the column of the first element each piece of
//! diagonal to be shifted within the local matrix `A`
//! @param[in] offsize number of elements to be shifted on each piece
//! @param[in,out] ldh the leading dimension of local matrix to be shifted
//! @param[in] shift the value for shifting the diagonal
//! @param[in] stream_ an asynchronous CUDA stream which allows to run this
//! function asynchronously
void chase_shift_mgpu_matrix(float* A, std::size_t* off_m, std::size_t* off_n,
                             std::size_t offsize, std::size_t ldH, float shift,
                             cudaStream_t stream_);
//! shift the diagonal of a `nxn` square matrix `A` in double real data type.
//! which has been distributed on multi-GPUs in either block-block of
//! block-cyclic faison. Each GPU may contains different number of diagonal part
//! of the global matrix, espeically for the block-cyclic distribution On each
//! GPU, it may contains multiple pieces of diagonal to be shifted, and each
//! piece may be of different size. Hence each piece is determined by `off_m`,
//! `off_n`, `offsize` within the local matrix on each GPU device.
//!
//! @param[in,out] A a pointer to the local piece of matrix to be shifted on
//! each GPU
//! @param[in] off_m the offset of the row of the first element each piece of
//! diagonal to be shifted within the local matrix `A`
//! @param[in] off_n the offset of the column of the first element each piece of
//! diagonal to be shifted within the local matrix `A`
//! @param[in] offsize number of elements to be shifted on each piece
//! @param[in,out] ldh the leading dimension of local matrix to be shifted
//! @param[in] shift the value for shifting the diagonal
//! @param[in] stream_ an asynchronous CUDA stream which allows to run this
//! function asynchronously
void chase_shift_mgpu_matrix(double* A, std::size_t* off_m, std::size_t* off_n,
                             std::size_t offsize, std::size_t ldH, double shift,
                             cudaStream_t stream_);
//! shift the diagonal of a `nxn` square matrix `A` in double complex data type.
//! which has been distributed on multi-GPUs in either block-block of
//! block-cyclic faison. Each GPU may contains different number of diagonal part
//! of the global matrix, espeically for the block-cyclic distribution On each
//! GPU, it may contains multiple pieces of diagonal to be shifted, and each
//! piece may be of different size. Hence each piece is determined by `off_m`,
//! `off_n`, `offsize` within the local matrix on each GPU device.
//!
//! @param[in,out] A a pointer to the local piece of matrix to be shifted on
//! each GPU
//! @param[in] off_m the offset of the row of the first element each piece of
//! diagonal to be shifted within the local matrix `A`
//! @param[in] off_n the offset of the column of the first element each piece of
//! diagonal to be shifted within the local matrix `A`
//! @param[in] offsize number of elements to be shifted on each piece
//! @param[in,out] ldh the leading dimension of local matrix to be shifted
//! @param[in] shift the value for shifting the diagonal
//! @param[in] stream_ an asynchronous CUDA stream which allows to run this
//! function asynchronously
void chase_shift_mgpu_matrix(std::complex<double>* A, std::size_t* off_m,
                             std::size_t* off_n, std::size_t offsize,
                             std::size_t ldH, double shift,
                             cudaStream_t stream_);
//! shift the diagonal of a `nxn` square matrix `A` in float complex data type.
//! which has been distributed on multi-GPUs in either block-block of
//! block-cyclic faison. Each GPU may contains different number of diagonal part
//! of the global matrix, espeically for the block-cyclic distribution On each
//! GPU, it may contains multiple pieces of diagonal to be shifted, and each
//! piece may be of different size. Hence each piece is determined by `off_m`,
//! `off_n`, `offsize` within the local matrix on each GPU device.
//!
//! @param[in,out] A a pointer to the local piece of matrix to be shifted on
//! each GPU
//! @param[in] off_m the offset of the row of the first element each piece of
//! diagonal to be shifted within the local matrix `A`
//! @param[in] off_n the offset of the column of the first element each piece of
//! diagonal to be shifted within the local matrix `A`
//! @param[in] offsize number of elements to be shifted on each piece
//! @param[in,out] ldh the leading dimension of local matrix to be shifted
//! @param[in] shift the value for shifting the diagonal
//! @param[in] stream_ an asynchronous CUDA stream which allows to run this
//! function asynchronously
void chase_shift_mgpu_matrix(std::complex<float>* A, std::size_t* off_m,
                             std::size_t* off_n, std::size_t offsize,
                             std::size_t ldH, float shift,
                             cudaStream_t stream_);

/** @} */ // end of chase-cuda-utility

namespace chase
{
namespace mpi
{
//
//  This Class is meant to be used with MatrixFreeMPI
//
//! @brief A derived class of ChaseMpiDLAInterface which implements the
//! inter-node computation for a multi-GPUs MPI-based implementation of ChASE.
template <class T>
class ChaseMpiDLAMultiGPU : public ChaseMpiDLAInterface<T>
{
public:
    //! A constructor of ChaseMpiDLABlaslapack.
    //! @param matrix_properties: it is an object of ChaseMpiProperties, which
    //! defines the MPI environment and data distribution scheme in ChASE-MPI.
    //! @param matrices: it is an instance of ChaseMpiMatrices, which
    //!  allocates the required buffers in ChASE-MPI.
    ChaseMpiDLAMultiGPU(ChaseMpiProperties<T>* matrix_properties,
                        ChaseMpiMatrices<T>& matrices)
    {
#ifdef USE_NSIGHT
        nvtxRangePushA("ChaseMpiDLAMultiGPU: Init");
#endif
        n_ = matrix_properties->get_n();
        m_ = matrix_properties->get_m();
        N_ = matrix_properties->get_N();
        nev_ = matrix_properties->GetNev();
        nex_ = matrix_properties->GetNex();
        H_ = matrices.get_H();
        ldh_ = matrices.get_ldh();
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
        cuda_exec(cudaHostRegister((void*)H_, ldh_ * n_ * sizeof(T),
                                   cudaHostRegisterDefault));
        cuda_exec(cudaHostRegister((void*)B_, n_ * maxBlock * sizeof(T),
                                   cudaHostRegisterDefault));
        cuda_exec(cudaHostRegister((void*)C_, m_ * maxBlock * sizeof(T),
                                   cudaHostRegisterDefault));
        cuda_exec(cudaHostRegister((void*)B2_, n_ * maxBlock * sizeof(T),
                                   cudaHostRegisterDefault));
        cuda_exec(cudaHostRegister((void*)C2_, m_ * maxBlock * sizeof(T),
                                   cudaHostRegisterDefault));

        cuda_exec(cudaMalloc((void**)&(d_H_), m_ * n_ * sizeof(T)));
        cuda_exec(cudaMallocPitch((void**)&(d_C_), &pitchC,
                                  (nev_ + nex_) * sizeof(T), m_));
        cuda_exec(cudaMallocPitch((void**)&(d_C2_), &pitchC2,
                                  (nev_ + nex_) * sizeof(T), m_));
        cuda_exec(cudaMallocPitch((void**)&(d_B_), &pitchB,
                                  (nev_ + nex_) * sizeof(T), n_));
        cuda_exec(cudaMallocPitch((void**)&(d_B2_), &pitchB2,
                                  (nev_ + nex_) * sizeof(T), n_));
        cuda_exec(cudaMalloc((void**)&d_A_,
                             sizeof(T) * (nev_ + nex_) * (nev_ + nex_)));
        cuda_exec(
            cudaMalloc((void**)&d_ritz_, sizeof(Base<T>) * (nev_ + nex_)));
        cuda_exec(
            cudaMalloc((void**)&d_resids_, sizeof(Base<T>) * (nev_ + nex_)));
	cuda_exec(
            cudaMalloc((void**)&states_, sizeof(curandStatePhilox4_32_10_t) * (256 * 32)));
        cuda_exec(cudaMalloc((void**)&d_v_, sizeof(T) * m_));
        cuda_exec(cudaMalloc((void**)&d_w_, sizeof(T) * n_));

        cublasCreate(&cublasH_);
	cublasCreate(&cublasH2_);
        cusolverDnCreate(&cusolverH_);
	cublasSetPointerMode(cublasH2_, CUBLAS_POINTER_MODE_DEVICE);
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

        // for shifting matrix
        std::vector<std::size_t> off_m, off_n;
        for (std::size_t j = 0; j < nblocks_; j++)
        {
            for (std::size_t i = 0; i < mblocks_; i++)
            {
                for (std::size_t q = 0; q < c_lens_[j]; q++)
                {
                    for (std::size_t p = 0; p < r_lens_[i]; p++)
                    {
                        if (q + c_offs_[j] == p + r_offs_[i])
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
        cudaMemcpy(d_off_m_, off_m.data(), diag_off_size_ * sizeof(std::size_t),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_off_n_, off_n.data(), diag_off_size_ * sizeof(std::size_t),
                   cudaMemcpyHostToDevice);
        cuda_exec(cudaStreamCreate(&stream1_));
        cuda_exec(cudaStreamCreate(&stream2_));
#if defined(CUDA_AWARE)
	cudaMalloc((void**)&(vv_), m_ * sizeof(T));
#else
	vv_ = new T[m_];
#endif	
#ifdef USE_NSIGHT
        nvtxRangePop();
#endif
    }

    //! Destructor.
    ~ChaseMpiDLAMultiGPU()
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
        cuda_exec(cudaFree(states_));
#if defined(CUDA_AWARE)
        cuda_exec(cudaFree(vv_));	
#else
	delete[] vv_;
#endif	
    }
    //! - This function set initially the operation for apply() used in
    //! ChaseMpi::Lanczos()
    //! - This function also copies the local block of Symmetric/Hermtian matrix
    //! to GPU
    void initVecs() override
    {
        cublas_status_ = cublasSetMatrix(m_, n_, sizeof(T), H_, ldh_, d_H_, m_);
        assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);
        next_ = NextOp::bAc;
    }
    //! This function generates the random values for each MPI proc using device
    //! API of cuRAND
    //!     - each MPI proc with a same MPI rank among different column
    //!     communicator
    //!       same a same seed of RNG
    void initRndVecs() override
    {
        MPI_Comm col_comm = matrix_properties_->get_col_comm();
        int mpi_col_rank;
        MPI_Comm_rank(col_comm, &mpi_col_rank);
        unsigned long long seed = 1337 + mpi_col_rank;

        chase_rand_normal(seed, states_, d_C_, m_ * (nev_ + nex_),
                          (cudaStream_t)0);
        cuda_exec(cudaMemcpy(C_, d_C_, m_ * (nev_ + nex_) * sizeof(T),
                             cudaMemcpyDeviceToHost));
    }

    //! - This function set initially the operation for apply() in filter
    //! - it copies also `C_` to device buffer `d_C`
    void preApplication(T* V, std::size_t locked, std::size_t block) override
    {
        next_ = NextOp::bAc;
        if (locked > 0)
        {
            cuda_exec(cudaMemcpy(d_C_ + locked * m_, C_ + locked * m_,
                                 m_ * block * sizeof(T),
                                 cudaMemcpyHostToDevice));
        }
    }

    //! - This function performs the local computation of `GEMM` for
    //! ChaseMpiDLA::apply()
    //! - It is implemented based on `cuBLAS`'s `cublasXgemm`.
    void apply(T alpha, T beta, std::size_t offset, std::size_t block,
               std::size_t locked) override
    {
        // cudaStreamSynchronize(stream1_);
        T Zero = T(0.0);
        if (next_ == NextOp::bAc)
        {
            if (mpi_col_rank != 0)
            {
                beta = Zero;
            }
#if !defined(CUDA_AWARE)	    
            cuda_exec(cudaMemcpy(d_C_ + offset * m_ + locked * m_,
                                 C_ + offset * m_ + locked * m_,
                                 block * m_ * sizeof(T),
                                 cudaMemcpyHostToDevice));
#endif
	    cublas_status_ =
                cublasTgemm(cublasH_, CUBLAS_OP_C, CUBLAS_OP_N, n_, block, m_,
                            &alpha, d_H_, m_, d_C_ + locked * m_ + offset * m_,
                            m_, &beta, d_B_ + locked * n_ + offset * n_, n_);
            assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);
            cublas_status_ = cublasGetMatrix(
                n_, block, sizeof(T), d_B_ + locked * n_ + offset * n_, n_,
                B_ + locked * n_ + offset * n_, n_);
            assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);
            next_ = NextOp::cAb;
        }
        else
        {
            if (mpi_row_rank != 0)
            {
                beta = Zero;
            }
#if !defined(CUDA_AWARE)	    
            cuda_exec(cudaMemcpy(d_B_ + locked * n_ + offset * n_,
                                 B_ + locked * n_ + offset * n_,
                                 block * n_ * sizeof(T),
                                 cudaMemcpyHostToDevice));
#endif
	    cublas_status_ =
                cublasTgemm(cublasH_, CUBLAS_OP_N, CUBLAS_OP_N, m_, block, n_,
                            &alpha, d_H_, m_, d_B_ + locked * n_ + offset * n_,
                            n_, &beta, d_C_ + locked * m_ + offset * m_, m_);
            assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);

            cublas_status_ = cublasGetMatrix(
                m_, block, sizeof(T), d_C_ + locked * m_ + offset * m_, m_,
                C_ + locked * m_ + offset * m_, m_);
            assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);
            next_ = NextOp::bAc;
        }
    }
    //! - This function copies a number of column of `d_C_` to `C_`.
    //! - This memory copying is required for ChaseMpi::Lanczos in which `axpy`,
    //! etc
    //! - are performed on CPUs.
    bool postApplication(T* V, std::size_t block, std::size_t locked) override
    {
        cuda_exec(cudaMemcpy(C_ + locked * m_, d_C_ + locked * m_,
                             m_ * block * sizeof(T), cudaMemcpyDeviceToHost));
        return false;
    }

    //! This function performs the shift of diagonal of a global matrix
    //! - This global is already distributed on GPUs, so the shifting operation
    //! takes place on the local
    //!   block of global matrix on each GPU.
    //! - This function is naturally in parallel among all MPI procs and also
    //! with each GPU.
    void shiftMatrix(T c, bool isunshift = false) override
    {
        chase_shift_mgpu_matrix(d_H_, d_off_m_, d_off_n_, diag_off_size_, m_,
                                std::real(c), (cudaStream_t)0);
    }
    //! - This function performs the local computation of `GEMM` for
    //! ChaseMpiDLA::asynCxHGatherC()
    //! - It is implemented based on `cuBLAS`'s `cublasXgemm`.
    void asynCxHGatherC(std::size_t locked, std::size_t block,
                        bool isCcopied) override
    {
        T alpha = T(1.0);
        T beta = T(0.0);
#if !defined(CUDA_AWARE)
        if (!isCcopied)
        {
            cuda_exec(cudaMemcpy(d_C_ + locked * m_, C_ + locked * m_,
                                 block * m_ * sizeof(T),
                                 cudaMemcpyHostToDevice));
        }
#endif
        cublas_status_ = cublasTgemm(
            cublasH_, CUBLAS_OP_C, CUBLAS_OP_N, n_, block, m_, &alpha, d_H_, m_,
            d_C_ + locked * m_, m_, &beta, d_B_ + locked * n_, n_);
        assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);
#if !defined(CUDA_AWARE)
	cuda_exec(cudaMemcpy(B_ + locked * n_, d_B_ + locked * n_,
                             block * n_ * sizeof(T), cudaMemcpyDeviceToHost));
#endif
#if defined(CUDA_AWARE)
        cuda_exec(cudaMemcpy(B2_ + locked * n_, d_B2_ + locked * n_,
                             block * n_ * sizeof(T), cudaMemcpyDeviceToHost));	
#endif	
    }

    //! - All required operations for this function has been done in for
    //! ChaseMpiDLA::applyVec().
    //! - This function contains nothing in this class.
    void applyVec(T* v, T* w) override 
    {
        T alpha = T(1.0);
        T beta = T(0.0);
	    
        cuda_exec(cudaMemcpy(d_v_, v, m_ * sizeof(T), cudaMemcpyHostToDevice));
	std::size_t k = 1;
	cublas_status_ = cublasTgemm(
            cublasH_, CUBLAS_OP_C, CUBLAS_OP_N, n_, k, m_, &alpha, d_H_, m_,
            d_v_, m_, &beta, d_w_, n_);
        assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);

        cuda_exec(cudaMemcpy(w, d_w_, n_ * sizeof(T), cudaMemcpyDeviceToHost));

    }
    int get_nprocs() const override { return matrix_properties_->get_nprocs(); }
    void Start() override {}
    void End() override {}

    //! It is an interface to BLAS `?axpy`.
    void axpy(std::size_t N, T* alpha, T* x, std::size_t incx, T* y,
              std::size_t incy) override
    {
        t_axpy(N, alpha, x, incx, y, incy);
    }

    //! It is an interface to BLAS `?scal`.
    void scal(std::size_t N, T* a, T* x, std::size_t incx) override
    {
        t_scal(N, a, x, incx);
    }

    //! It is an interface to BLAS `?nrm2`.
    Base<T> nrm2(std::size_t n, T* x, std::size_t incx) override
    {
#if defined(CUDA_AWARE)	   
        Base<T> nrm;
        cublas_status_ = cublasTnrm2(cublasH_, n, x, incx, &nrm);
        assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);
        return nrm;	    
#else	    
        return t_nrm2(n, x, incx);
#endif
    }

    //! It is an interface to BLAS `?dot`.
    T dot(std::size_t n, T* x, std::size_t incx, T* y,
          std::size_t incy) override
    {
        return t_dot(n, x, incx, y, incy);
    }

    //! - This function performs the local computation of `GEMM` for
    //! ChaseMpiDLA::RR()
    //! - It is implemented based on `cuBLAS`'s `cublasXgemm`.
    void RR(std::size_t block, std::size_t locked, Base<T>* ritzv) override
    {
        T One = T(1.0);
        T Zero = T(0.0);
#if !defined(CUDA_AWARE)    
    	cuda_exec(cudaMemcpy(d_B_ + locked * n_, B_ + locked * n_,
                             block * n_ * sizeof(T), cudaMemcpyHostToDevice));

	cuda_exec(cudaMemcpy(d_B2_ + locked * n_, B2_ + locked * n_,
                             block * n_ * sizeof(T), cudaMemcpyHostToDevice));
#endif
        cublas_status_ =
            cublasTgemm(cublasH_, CUBLAS_OP_C, CUBLAS_OP_N, block, block, n_,
                        &One, d_B2_ + locked * n_, n_, d_B_ + locked * n_, n_,
                        &Zero, d_A_, nev_ + nex_);
        assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);
#if !defined(CUDA_AWARE)
	cublas_status_ = cublasGetMatrix(block, block, sizeof(T), d_A_,
                                         nev_ + nex_, A_, nev_ + nex_);
        assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);
#endif    
    }
    //! It is an interface to cuBLAS `cublasXsy(he)rk`.
    void syherk(char uplo, char trans, std::size_t n, std::size_t k, T* alpha,
                T* a, std::size_t lda, T* beta, T* c, std::size_t ldc,
                bool first) override
    {
	if (first)
        {
            cuda_exec(cudaMemcpy(d_C_, C_, (nev_ + nex_) * m_ * sizeof(T),
                                 cudaMemcpyHostToDevice));
        }
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

        cublas_status_ =
            cublasTsyherk(cublasH_, CUBLAS_FILL_MODE_UPPER, transa, nev_ + nex_,
                          m_, &One, d_C_, m_, &Zero, d_A_, nev_ + nex_);
        assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);
#if !defined(CUDA_AWARE)
        cublas_status_ = cublasGetMatrix(nev_ + nex_, nev_ + nex_, sizeof(T),
                                         d_A_, nev_ + nex_, A_, nev_ + nex_);
        assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);
#endif    
    }
    //! It is an interface to cuSOLVER `cusolverXpotrf`.
    int potrf(char uplo, std::size_t n, T* a, std::size_t lda) override
    {
#if !defined(CUDA_AWARE)	    
        cublas_status_ = cublasSetMatrix(nev_ + nex_, nev_ + nex_, sizeof(T), a,
                                         lda, d_A_, nev_ + nex_);
        assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);
#endif
#ifdef USE_NSIGHT
        nvtxRangePushA("cusolverDnTpotrf");
#endif
        cusolver_status_ =
            cusolverDnTpotrf(cusolverH_, CUBLAS_FILL_MODE_UPPER, nev_ + nex_,
                             d_A_, nev_ + nex_, d_work_, lwork_, devInfo_);
#ifdef USE_NSIGHT
        nvtxRangePop();
#endif
        assert(CUSOLVER_STATUS_SUCCESS == cusolver_status_);

        int info;
        cuda_exec(cudaMemcpy(&info, devInfo_, 1 * sizeof(int),
                             cudaMemcpyDeviceToHost));

        return info;
    }

    //! It is an interface to cuBLAS `cublasXtrsm`.
    void trsm(char side, char uplo, char trans, char diag, std::size_t m,
              std::size_t n, T* alpha, T* a, std::size_t lda, T* b,
              std::size_t ldb, bool first) override
    {
        cublas_status_ =
            cublasTtrsm(cublasH_, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
                        CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m_, nev_ + nex_,
                        alpha, d_A_, nev_ + nex_, d_C_, m_);
        assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);
#if !defined(CUDA_AWARE)
        if (!first)
        {
            cublas_status_ =
                cublasGetMatrix(m_, nev_ + nex_, sizeof(T), d_C_, m_, C_, m_);
        }
        assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);
#endif    
    }
    //! - This function performs the local computation of residuals for
    //! ChaseMpiDLA::Resd()
    //! - It is implemented based on `BLAS`'s `?axpy` and `?nrm2`.
    //! - This function computes only the residuals of local part of vectors on
    //! each MPI proc.
    //! - The final results are obtained in ChaseMpiDLA::Resd() with an
    //! MPI_Allreduce operation
    //!      within the row communicator.
    void Resd(Base<T>* ritzv, Base<T>* resid, std::size_t locked,
              std::size_t unconverged) override
    {
#if defined(CUDA_AWARE)	    
	residual_gpu(n_, unconverged, d_B_ + locked * n_, n_, d_B2_ + locked * n_,
                     n_, d_ritz_, d_resids_, false, (cudaStream_t)0);

        cuda_exec(cudaMemcpy(resid, d_resids_, unconverged * sizeof(Base<T>),
                             cudaMemcpyDeviceToHost));
	
#else
    	for (auto i = 0; i < unconverged; i++)
        {
            T alpha = -ritzv[i];
            t_axpy(n_, &alpha, B2_ + locked * n_ + i * n_, 1,
                   B_ + locked * n_ + i * n_, 1);

            resid[i] = t_norm_p2(n_, B_ + locked * n_ + i * n_);

        }
#endif	
    }
    //! - This function performs the local computation for ChaseMpiDLA::heevd()
    //! - It is implemented based on `cuBLAS`'s `xgemm` and cuSOLVER's
    //! `cusolverXsy(he)evd`.
    void heevd(int matrix_layout, char jobz, char uplo, std::size_t n, T* a,
               std::size_t lda, Base<T>* w) override
    {
        T One = T(1.0);
        T Zero = T(0.0);
        std::size_t locked = nev_ + nex_ - n;
#if !defined(CUDA_AWARE)
	cublasSetMatrix(n, n, sizeof(T), a, lda, d_A_, nev_ + nex_);
#endif
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
        cuda_exec(cudaMemcpy(w, d_ritz_, n * sizeof(Base<T>),
                             cudaMemcpyDeviceToHost));
#if !defined(CUDA_AWARE)
	cublas_status_ = cublasSetMatrix(m_, n, sizeof(T), C2_ + locked * m_,
                                         m_, d_C2_ + locked * m_, m_);
        assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);
#endif
	cublas_status_ =
            cublasTgemm(cublasH_, CUBLAS_OP_N, CUBLAS_OP_N, m_, n, n, &One,
                        d_C2_ + locked * m_, m_, d_A_, nev_ + nex_, &Zero,
                        d_C_ + locked * m_, m_);
        assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);
#if !defined(CUDA_AWARE)
        cublas_status_ = cublasGetMatrix(m_, n, sizeof(T), d_C_ + locked * m_,
                                         m_, C_ + locked * m_, m_);
        assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);
#endif    
    }
    //! - All required operations for this function has been done in for
    //! ChaseMpiDLA::hhQR().
    //! - This function contains nothing in this class.
    void hhQR(std::size_t locked) override {}
    //! - All required operations for this function has been done in for
    //! ChaseMpiDLA::cholQR().
    //! - This function contains nothing in this class.
    void cholQR(std::size_t locked, Base<T> cond) override {}
    //! - All required operations for this function has been done in for
    //! ChaseMpiDLA::Swap().
    //! - This function contains nothing in this class.
    void Swap(std::size_t i, std::size_t j) override {}
    //! - All required operations for this function has been done in for
    //! ChaseMpiDLA::LanczosDos().
    //! - This function contains nothing in this class.
    void LanczosDos(std::size_t idx, std::size_t m, T* ritzVc) override {
#if defined(CUDA_AWARE)
        cuda_exec(cudaMemcpy(d_C_, C_, m_ * (nev_ + nex_) * sizeof(T), cudaMemcpyHostToDevice));    
#endif    
    }
    void Lanczos(std::size_t M, int idx, Base<T>* d, Base<T>* e, Base<T> *r_beta) override
    {}

    void B2C(T* B, std::size_t off1, T* C, std::size_t off2, std::size_t block) override
    {} 

    void getMpiWorkSpace(T **C, T **B, T **A, T **C2, T **B2, T **vv) override
    {
#if defined(CUDA_AWARE)
        *C = d_C_;
        *B = d_B_;
	*A = d_A_;
	*C2 = d_C2_;
	*B2 = d_B2_;
	*vv = vv_;
#else	    
        *C = C_;
        *B = B_;  
        *A = A_;
	*C2 = C2_;
	*B2 = B2_;	
	*vv = vv_;
#endif
    }
    void getMpiCollectiveBackend(int *allreduce_backend, int *bcast_backend) override
    {
#if defined(CUDA_AWARE)	    
    #if defined(HAS_NCCL)	    
        *allreduce_backend = NCCL_BACKEND;
        *bcast_backend = MPI_BACKEND;   
    #else
        *allreduce_backend = MPI_BACKEND;
        *bcast_backend = MPI_BACKEND;	
    #endif
#else	
        *allreduce_backend = MPI_BACKEND;
        *bcast_backend = MPI_BACKEND;
#endif
	
    }

    bool isCudaAware()
    {
#if defined(CUDA_AWARE)
	return true;
#else
	return false;
#endif    
    }

    void lacpy(char uplo, std::size_t m, std::size_t n,
             T* a, std::size_t lda, T* b, std::size_t ldb) override
    {
#if defined(CUDA_AWARE)
        t_lacpy_gpu(uplo, m, n, a, lda, b, ldb, NULL);
#else
        t_lacpy(uplo, m, n, a, lda, b, ldb);

#endif    
    }

    void shiftMatrixForQR(T *A, std::size_t n, T shift) override
    {
#if defined(CUDA_AWARE)	    
    	chase_shift_matrix(A, n, std::real(shift), &stream1_);
#else
        for (auto i = 0; i < n; i++)
        {
            A[i * n + i] += (T)shift;
        }
#endif	
    }

    void retrieveC(T **C, std::size_t locked, std::size_t block, bool copy) override
    {
#if defined(CUDA_AWARE)
	if(copy)    
	{
	    cuda_exec(cudaMemcpy(C_ + locked * m_, d_C_ + locked * m_, block * m_ * sizeof(T),
                             cudaMemcpyDeviceToHost));        	    
	}
#endif
	*C = C_;	
    }

    void retrieveB(T **B, std::size_t locked, std::size_t block, bool copy) override
    {
#if defined(CUDA_AWARE)
	if(copy)
	{
            cuda_exec(cudaMemcpy(B_ + locked * n_, d_B_ + locked * n_, block * n_ *sizeof(T),
                             cudaMemcpyDeviceToHost));
	}
#endif
        *B = B_;    
    }

    void putC(T *C, std::size_t locked, std::size_t block) override
    {
#if defined(CUDA_AWARE)
        cuda_exec(cudaMemcpy(d_C_ + locked * m_, C + locked * m_, block * m_ * sizeof(T),
                             cudaMemcpyHostToDevice));
#endif	    
    }

private:
    enum NextOp
    {
        cAb,
        bAc
    };

    NextOp next_; //!< it is to manage the switch of operation from `V2=H*V1` to
                  //!< `V1=H'*V2` in filter
    std::size_t N_; //!< global dimension of the symmetric/Hermtian matrix

    std::size_t n_; //!< number of columns of local matrix of the
                    //!< symmetric/Hermtian matrix
    std::size_t
        m_; //!< number of rows of local matrix of the symmetric/Hermtian matrix
    std::size_t ldh_; //!< leading dimension of local matrix on each MPI proc
    T* H_;            //!< a pointer to the local matrix on each MPI proc
    T* B_;  //!< a matrix of size `n_*(nev_+nex_)`, which is allocated in
            //!< ChaseMpiMatrices
    T* B2_; //!< a matrix of size `n_*(nev_+nex_)`, which is allocated in
            //!< ChaseMpiProperties
    T* C_;  //!< a matrix of size `m_*(nev_+nex_)`, which is allocated in
            //!< ChaseMpiMatrices
    T* C2_; //!< a matrix of size `m_*(nev_+nex_)`, which is allocated in
            //!< ChaseMpiProperties
    T* A_;  //!< a matrix of size `(nev_+nex_)*(nev_+nex_)`, which is allocated
            //!< in ChaseMpiProperties

    std::size_t* off_;      //!< identical to ChaseMpiProperties::off_
    std::size_t* r_offs_;   //!< identical to ChaseMpiProperties::r_offs_
    std::size_t* r_lens_;   //!< identical to ChaseMpiProperties::r_lens_
    std::size_t* r_offs_l_; //!< identical to ChaseMpiProperties::r_offs_l_
    std::size_t* c_offs_;   //!< identical to ChaseMpiProperties::c_offs_
    std::size_t* c_lens_;   //!< identical to ChaseMpiProperties::c_lens_
    std::size_t* c_offs_l_; //!< identical to ChaseMpiProperties::c_offs_l_
    std::size_t nb_;        //!< identical to ChaseMpiProperties::nb_
    std::size_t mb_;        //!< identical to ChaseMpiProperties::mb_
    std::size_t nblocks_;   //!< identical to ChaseMpiProperties::nblocks_
    std::size_t mblocks_;   //!< identical to ChaseMpiProperties::mblocks_
    std::size_t nev_;       //!< number of required eigenpairs
    std::size_t nex_;       //!< number of extral searching space
    int mpi_row_rank;       //!< rank within each row communicator
    int mpi_col_rank;       //!< rank within each column communicator

    // for shifting matrix H
    std::size_t*
        d_off_m_; //!< the offset of the row of the first element each piece of
                  //!< diagonal to be shifted within the local matrix `A`
    std::size_t*
        d_off_n_; //!< the offset of the column of the first element each piece
                  //!< of diagonal to be shifted within the local matrix `A`
    std::size_t
        diag_off_size_; //!< number of elements to be shifted on each piece

    int mpi_rank_; //!< the MPI rank within the working MPI communicator
    cublasHandle_t cublasH_;       //!< `cuBLAS` handle
    cublasHandle_t cublasH2_;       //!< `cuBLAS` handle    
    cusolverDnHandle_t cusolverH_; //!< `cuSOLVER` handle
    cublasStatus_t cublas_status_ = CUBLAS_STATUS_SUCCESS; //!< `cuBLAS` status
    cusolverStatus_t cusolver_status_ =
        CUSOLVER_STATUS_SUCCESS; //!< `cuSOLVER` status
    cudaStream_t
        stream1_; //!< CUDA stream for asynchronous exectution of kernels
    cudaStream_t
        stream2_; //!< CUDA stream for asynchronous exectution of kernels
    //curandState* states_ = NULL; //!< a pointer of `curandState` for the cuRAND
    curandStatePhilox4_32_10_t *states_ = NULL;
    T* d_H_;  //!< a pointer to a local buffer of size `m_*n_` on GPU, which is
              //!< mapped to `H_`.
    T* d_C_;  //!< a pointer to a local buffer of size `m_*(nev_+nex_)` on GPU,
              //!< which is mapped to `C_`.
    T* d_C2_; //!< a pointer to a local buffer of size `m_*(nev_+nex_)` on GPU,
              //!< which is mapped to `C2_`.
    T* d_B_;  //!< a pointer to a local buffer of size `n_*(nev_+nex_)` on GPU,
              //!< which is mapped to `B_`.
    T* d_B2_; //!< a pointer to a local buffer of size `n_*(nev_+nex_)` on GPU,
              //!< which is mapped to `B2_`.
    Base<T>* d_ritz_ =
        NULL; //!< a pointer to a local buffer of size `nev_+nex_` on GPU for
              //!< storing computed ritz values
    T* d_A_;  //!< a pointer to a local buffer of size `(nev_+nex_)*(nev_+nex_)`
              //!< on GPU, which is mapped to `A_`.
    T* d_work_ =
        NULL; //!< a pointer to a local buffer on GPU, which is reserved for the
              //!< extra buffer required for any cuSOLVER routines
    T *d_v_;
    T *d_w_;

    Base<T> *d_resids_ = NULL;
    T *vv_;
    std::size_t pitchB;  //!< pitch for `B_` and `d_B_`
    std::size_t pitchB2; //!< pitch for `B2_` and `d_B2_`
    std::size_t pitchC;  //!< pitch for `C_` and `d_C_`
    std::size_t pitchC2; //!< pitch for `C_` and `d_C2_`
    int* devInfo_ =
        NULL; //!< for the return of information from any cuSOLVER routines

    int lwork_ = 0; //!< size of required extra buffer by any cuSOLVER routines
    ChaseMpiProperties<T>*
        matrix_properties_; //!< an object of class ChaseMpiProperties
};

template <typename T>
struct is_skewed_matrixfree<ChaseMpiDLAMultiGPU<T>>
{
    static const bool value = true;
};

} // namespace mpi
} // namespace chase

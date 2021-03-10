/* -*- Mode: C++; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
// This file is a part of ChASE.
// Copyright (c) 2015-2018, Simulation Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany
// and
// Copyright (c) 2016-2018, Aachen Institute for Advanced Study in Computational
//   Engineering Science, RWTH Aachen University, Germany All rights reserved.
// License is 3-clause BSD:
// https://github.com/SimLabQuantumMaterials/ChASE/

#pragma once

#include <assert.h>
#include <complex>

#include "ChASE-MPI/blas_cuda_wrapper.hpp"
#include "ChASE-MPI/chase_mpidla_interface.hpp"

void chase_zshift_matrix(std::complex<double>* A, int n, double shift,
                         cudaStream_t* stream_);

namespace chase {
namespace mpi {

template <class T>
class ChaseMpiDLACudaSeq : public ChaseMpiDLAInterface<T> {
 public:
  ChaseMpiDLACudaSeq(ChaseMpiMatrices<T>& matrices, std::size_t n,
                      std::size_t maxBlock)
      : n_(n), copied_(false), max_block_(maxBlock) {
    cuda_exec(cudaMalloc(&(V1_), n_ * maxBlock * sizeof(T)));
    cuda_exec(cudaMalloc(&(V2_), n_ * maxBlock * sizeof(T)));
    cuda_exec(cudaMalloc(&(H_), n_ * n_ * sizeof(T)));

    OrigH_ = matrices.get_H();

    std::size_t pitch_host = n_ * sizeof(T);
    std::size_t pitch_device = n_ * sizeof(T);

    cuda_exec(cudaSetDevice(0));
    cublasCreate(&handle_);
    cuda_exec(cudaStreamCreate(&stream_));
    cublasSetStream(handle_, stream_);

    cuda_exec(cudaSetDevice(0));
    cusolverDnCreate(&cusolverH_);
    cuda_exec(cudaSetDevice(0));
    cuda_exec(cudaMalloc ((void**)&devInfo_, sizeof(int)));
    cuda_exec(cudaMalloc ((void**)&d_V_  , sizeof(T)*n_*max_block_));
    cuda_exec(cudaMalloc ((void**)&d_return_  , sizeof(T)*max_block_));

    int lwork_geqrf = 0;
    int lwork_orgqr = 0;
    cuda_exec(cudaSetDevice(0));	
    cusolver_status_ = cusolverDnTgeqrf_bufferSize(
            cusolverH_[0],
            N_,
            nev_ + nex_,
            d_V_,
            N_,
            &lwork_geqrf);
    assert (cusolver_status_ == CUSOLVER_STATUS_SUCCESS);
    cuda_exec(cudaSetDevice(0));
    cusolver_status_ = cusolverDnTgqr_bufferSize(
            cusolverH_[0],
            N_,
            nev_ + nex_,
            nev_ + nex_,
            d_V_,
            N_,
            d_return_,
            &lwork_orgqr);
    assert (cusolver_status_ == CUSOLVER_STATUS_SUCCESS);

    lwork_ = (lwork_geqrf > lwork_orgqr)? lwork_geqrf : lwork_orgqr;
    
    cuda_exec(cudaSetDevice(0));
    cuda_exec(cudaMalloc((void**)&d_work_, sizeof(T)*lwork_));
    

  }

  ~ChaseMpiDLACudaSeq() {
    cudaFree(V1_);
    cudaFree(V2_);
    cudaFree(H_);
    cudaStreamDestroy(stream_);
    cublasDestroy(handle_);
    cusolverDnDestroy(cusolverH_);
    if (devInfo_) cudaFree(devInfo_);
    if (d_V_) cudaFree(d_V_);
    if (d_return_) cudaFree(d_return_);
    if (d_work_) cudaFree(d_work_);	

  }

  void preApplication(T* V, std::size_t locked, std::size_t block) {
    locked_ = locked;
    cuda_exec(cudaMemcpyAsync(V1_, V + locked * n_, block * n_ * sizeof(T),
                              cudaMemcpyHostToDevice, stream_));
  }

  void preApplication(T* V1, T* V2, std::size_t locked, std::size_t block) {
    cuda_exec(cudaMemcpyAsync(V2_, V2 + locked * n_, block * n_ * sizeof(T),
                              cudaMemcpyHostToDevice, stream_));

    this->preApplication(V1, locked, block);
  }

  void apply(T alpha, T beta, std::size_t offset, std::size_t block) {
    cublasTgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N,  //
                n_, block, n_,                      //
                &alpha,                             //
                H_, n_,                             //
                V1_ + offset * n_, n_,              //
                &beta,                              //
                V2_ + offset * n_, n_);             //
    std::swap(V1_, V2_);
  }

  bool postApplication(T* V, std::size_t block) {
    cuda_exec(cudaMemcpyAsync(V + locked_ * n_, V1_, block * n_ * sizeof(T),
                              cudaMemcpyDeviceToHost, stream_));
    cudaStreamSynchronize(stream_);
    return false;
  }

  void shiftMatrix(T c, bool isunshift = false) {
    // for (std::size_t i = 0; i < n_; ++i) {
    //   OrigH_[i + i * n_] += c;
    // }

    if (!copied_) {
      cuda_exec(cudaMemcpyAsync(H_, OrigH_, n_ * n_ * sizeof(T),
                                cudaMemcpyHostToDevice, stream_));
      copied_ = true;
    }

    chase_zshift_matrix(H_, n_, std::real(c), &stream_);
  }

  void applyVec(T* B, T* C) {
    T alpha = T(1.0);
    T beta = T(0.0);

    t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n_, 1, n_, &alpha, OrigH_,
           n_, B, n_, &beta, C, n_);

    // this->preApplication(B, 0, 1);
    // this->apply(alpha, beta, 0, 1);
    // this->postApplication(C, 1);
  }

  void get_off(std::size_t* xoff, std::size_t* yoff, std::size_t* xlen,
               std::size_t* ylen) const override {
    *xoff = 0;
    *yoff = 0;
    *xlen = n_;
    *ylen = n_;
  }

  T* get_H() const override { return OrigH_; }
  std::size_t get_mblocks() const override {return 1;}
  std::size_t get_nblocks() const override {return 1;}
  std::size_t get_n() const override {return n_;}
  std::size_t get_m() const override {return n_;}
  int *get_coord() const override {
          int coord[2] = {0, 0};
          return coord;
  }
  void get_offs_lens(std::size_t* &r_offs, std::size_t* &r_lens, std::size_t* &r_offs_l,
                  std::size_t* &c_offs, std::size_t* &c_lens, std::size_t* &c_offs_l) const override{

          std::size_t r_offs_[1] = {0};
          std::size_t r_lens_[1]; r_lens_[0] = n_;
          std::size_t r_offs_l_[1] = {0};
          std::size_t c_offs_[1] = {0};
          std::size_t c_lens_[1]; r_lens_[0] = n_;
          std::size_t c_offs_l_[1] = {0};

          r_offs = r_offs_;
          r_lens = r_lens_;
          r_offs_l = r_offs_l_;
          c_offs = c_offs_;
          c_lens = c_lens_;
          c_offs_l = c_offs_l_;
  }

  void Start() override { copied_ = false; }

  Base<T> lange(char norm, std::size_t m, std::size_t n, T* A, std::size_t lda){
      return t_lange(norm, m, n, A, lda);
  }

  void axpy(std::size_t N, T * alpha, T * x, std::size_t incx, T *y, std::size_t incy){
      t_axpy(N, alpha, x, incx, y, incy);
  }

  void scal(std::size_t N, T *a, T *x, std::size_t incx){
      t_scal(N, a, x, incx);
  }

  Base<T> nrm2(std::size_t n, T *x, std::size_t incx){
      return t_nrm2(n, x, incx);
  }

  T dot(std::size_t n, T* x, std::size_t incx, T* y, std::size_t incy){
      return t_dot(n, x, incx, y, incy);
  }

  void gemm_small(CBLAS_LAYOUT Layout, CBLAS_TRANSPOSE transa,
                         CBLAS_TRANSPOSE transb, std::size_t m,
                         std::size_t n, std::size_t k, T* alpha,
                         T* a, std::size_t lda, T* b,
                         std::size_t ldb, T* beta, T* c, std::size_t ldc)
  {
      t_gemm(Layout, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  }

  void gemm_large(CBLAS_LAYOUT Layout, CBLAS_TRANSPOSE transa,
                         CBLAS_TRANSPOSE transb, std::size_t m,
                         std::size_t n, std::size_t k, T* alpha,
                         T* a, std::size_t lda, T* b,
                         std::size_t ldb, T* beta, T* c, std::size_t ldc)
  {
      t_gemm(Layout, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  }

  std::size_t stemr(int matrix_layout, char jobz, char range, std::size_t n,
                    double* d, double* e, double vl, double vu, std::size_t il, std::size_t iu,
                    int* m, double* w, double* z, std::size_t ldz, std::size_t nzc,
                    int* isuppz, lapack_logical* tryrac){
      return t_stemr<double>(matrix_layout, jobz, range, n, d, e, vl, vu, il, iu, m, w, z, ldz, nzc, isuppz, tryrac);
  }

  std::size_t stemr(int matrix_layout, char jobz, char range, std::size_t n,
                    float* d, float* e, float vl, float vu, std::size_t il, std::size_t iu,
                    int* m, float* w, float* z, std::size_t ldz, std::size_t nzc,
                    int* isuppz, lapack_logical* tryrac){
      return t_stemr<float>(matrix_layout, jobz, range, n, d, e, vl, vu, il, iu, m, w, z, ldz, nzc, isuppz, tryrac);
  }

  void gegqr(std::size_t N, std::size_t nevex, T * approxV, std::size_t LDA){
    	cudaSetDevice(0);
	cuda_exec(cudaMemcpy(d_V_, approxV, sizeof(T)*N*nevex, cudaMemcpyHostToDevice));
        cudaSetDevice(shmrank_*num_devices_per_rank_);
	cusolver_status_ = cusolverDnTgeqrf(
            cusolverH_[0],
            N,
            nevex,
            d_V_,
            LDA,
            d_return_,
            d_work_,
            lwork_,
            devInfo_);
        assert(CUSOLVER_STATUS_SUCCESS == cusolver_status_);
        cudaSetDevice(0);
	cusolver_status_ = cusolverDnTgqr(
            cusolverH_[0],
            N,
            nevex,
            nevex,
            d_V_,
            LDA,
            d_return_,
            d_work_,
            lwork_,
            devInfo_);
        assert(CUSOLVER_STATUS_SUCCESS == cusolver_status_);
        cudaSetDevice(shmrank_*num_devices_per_rank_);
	cuda_exec(cudaMemcpy(approxV, d_V_, sizeof(T)*N*nevex, cudaMemcpyDeviceToHost));
  }

  void RR_kernel(std::size_t N, std::size_t block, T *approxV, std::size_t locked, T *workspace, T One, T Zero, Base<T> *ritzv){
        T *A = new T[block * block];

        T *d_A_ = NULL;
        T *d_W_ = NULL;

        cudaSetDevice(shmrank_*num_devices_per_rank_);	
        cuda_exec(cudaMalloc ((void**)&d_W_, sizeof(T) * N * block));
        cuda_exec(cudaMalloc ((void**)&d_A_, sizeof(T) * block * block));
        cudaSetDevice(shmrank_*num_devices_per_rank_);
	cuda_exec(cudaMemcpy(d_V_, approxV + locked * N, sizeof(T)* N * block, cudaMemcpyHostToDevice));
        cuda_exec(cudaMemcpy(d_W_, workspace + locked * N, sizeof(T)* N * block, cudaMemcpyHostToDevice));
        cudaSetDevice(shmrank_*num_devices_per_rank_);
	cublas_status_ = cublasTgemm(
	  handle_[0],
	  CUBLAS_OP_C,
	  CUBLAS_OP_N,
	  block,
	  block,
	  N,
	  &One,
          d_V_, N,
          d_W_, N,
          &Zero,
          d_A_, 
	  block);
        assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);


        cuda_exec(cudaMemcpy(A, d_A_, sizeof(T)* block * block, cudaMemcpyDeviceToHost));	
        t_heevd(LAPACK_COL_MAJOR, 'V', 'L', block, A, block, ritzv);
        cuda_exec(cudaMemcpy(d_A_, A, sizeof(T)* block * block, cudaMemcpyHostToDevice));

	cudaSetDevice(shmrank_*num_devices_per_rank_);
      	cublas_status_ = cublasTgemm(
            handle_[0],
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            N,
            block,
            block,
            &One,
            d_V_, N,
            d_A_, block,
            &Zero,
            d_W_,
            N);
      	assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);
        cudaSetDevice(shmrank_*num_devices_per_rank_);
  	cuda_exec(cudaMemcpy(approxV + locked * N, d_V_, sizeof(T)* N * block, cudaMemcpyDeviceToHost));
      	cuda_exec(cudaMemcpy(workspace + locked * N, d_W_, sizeof(T)* N * block, cudaMemcpyDeviceToHost));

      	if (d_A_) cudaFree(d_A_);
      	if (d_W_) cudaFree(d_W_);
	
  }

 private:
  std::size_t n_;
  std::size_t locked_;
  std::size_t max_block_;

  int *devInfo_ = NULL;
  T *d_V_ = NULL;	
  T *d_return_ = NULL;
  T *d_work_ = NULL;
  int lwork_ = 0;
  
  cusolverStatus_t cusolver_status_ = CUSOLVER_STATUS_SUCCESS;
  cublasStatus_t cublas_status_ = CUBLAS_STATUS_SUCCESS;

  T* V1_;
  T* V2_;
  T* H_;
  T* OrigH_;
  cudaStream_t stream_;
  cublasHandle_t handle_;
  cusolverDnHandle_t cusolverH_;
  bool copied_;

};

template <typename T>
struct is_skewed_matrixfree<ChaseMpiDLACudaSeq<T>> {
  static const bool value = false;
};

}  // namespace mpi
}  // namespace chase

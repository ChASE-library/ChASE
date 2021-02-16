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
#include "ChASE-MPI/chase_mpihemm_interface.hpp"

void chase_zshift_matrix(std::complex<double>* A, int n, double shift,
                         cudaStream_t* stream_);

namespace chase {
namespace mpi {

template <class T>
class ChaseMpiHemmCudaSeq : public ChaseMpiHemmInterface<T> {
 public:
  ChaseMpiHemmCudaSeq(ChaseMpiMatrices<T>& matrices, std::size_t n,
                      std::size_t maxBlock)
      : n_(n), copied_(false) {
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
  }

  ~ChaseMpiHemmCudaSeq() {
    cudaFree(V1_);
    cudaFree(V2_);
    cudaFree(H_);
    cudaStreamDestroy(stream_);
    cublasDestroy(handle_);
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

 private:
  std::size_t n_;
  std::size_t locked_;
  T* V1_;
  T* V2_;
  T* H_;
  T* OrigH_;
  cudaStream_t stream_;
  cublasHandle_t handle_;
  bool copied_;
};

template <typename T>
struct is_skewed_matrixfree<ChaseMpiHemmCudaSeq<T>> {
  static const bool value = false;
};

}  // namespace mpi
}  // namespace chase

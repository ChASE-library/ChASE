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

#include "blas_cuda_wrapper.hpp"
#include "chase_mpihemm_interface.hpp"

namespace chase {
namespace mpi {

template <class T>
class ChaseMpiHemmCudaSeq : public ChaseMpiHemmInterface<T> {
 public:
  ChaseMpiHemmCudaSeq(ChaseMpiMatrices<T>& matrices, std::size_t n,
                      std::size_t maxBlock)
      : n_(n) {
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

    cuda_exec(
        cudaMemcpy(H_, OrigH_, n_ * n_ * sizeof(T), cudaMemcpyHostToDevice));
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
    cuda_exec(cudaMemcpy(V1_, V + locked * n_, block * n_ * sizeof(T),
                         cudaMemcpyHostToDevice));
  }

  void preApplication(T* V1, T* V2, std::size_t locked, std::size_t block) {
    cuda_exec(cudaMemcpy(V2_, V2 + locked * n_, block * n_ * sizeof(T),
                         cudaMemcpyHostToDevice));

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
    cuda_exec(cudaMemcpy(V + locked_ * n_, V1_, block * n_ * sizeof(T),
                         cudaMemcpyDeviceToHost));

    return false;
  }

  void shiftMatrix(T c) {
    for (std::size_t i = 0; i < n_; ++i) {
      OrigH_[i + i * n_] += c;
    }

    cuda_exec(
        cudaMemcpy(H_, OrigH_, n_ * n_ * sizeof(T), cudaMemcpyHostToDevice));
  }

  void applyVec(T* B, T* C) {
    T alpha = T(1.0);
    T beta = T(0.0);

    t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n_, 1, n_, &alpha, OrigH_,
           n_, B, n_, &beta, C, n_);
  }

  void get_off(std::size_t* xoff, std::size_t* yoff, std::size_t* xlen,
               std::size_t* ylen) const override {
    *xoff = 0;
    *yoff = 0;
    *xlen = n_;
    *ylen = n_;
  }

  T* get_H() const override { return OrigH_; }

 private:
  std::size_t n_;
  std::size_t locked_;
  T* V1_;
  T* V2_;
  T* H_;
  T* OrigH_;
  cudaStream_t stream_;
  cublasHandle_t handle_;
};

template <typename T>
struct is_skewed_matrixfree<ChaseMpiHemmCudaSeq<T>> {
  static const bool value = false;
};

}  // namespace mpi
}  // namespace chase

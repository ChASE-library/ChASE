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
#include <cuComplex.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <complex>

#include "blas_cuda_wrapper.hpp"
#include "blas_templates.hpp"
#include "chase_mpihemm_interface.hpp"

namespace chase {
namespace mpi {

template <class T>
class ChaseMpiHemmCuda : public ChaseMpiHemmInterface<T> {
 public:
  ChaseMpiHemmCuda(ChaseMpiProperties<T>* matrix_properties) {
    n_ = matrix_properties->get_n();
    m_ = matrix_properties->get_m();
    N_ = matrix_properties->get_N();

    orig_H_ = matrix_properties->get_H();
    orig_B_ = matrix_properties->get_B();
    orig_C_ = matrix_properties->get_C();
    orig_IMT_ = matrix_properties->get_IMT();

    matrix_properties_ = matrix_properties;

    auto maxBlock = matrix_properties_->get_max_block();
    cuda_exec(cudaMalloc(&(B_), std::max(n_, m_) * maxBlock * sizeof(T)));
    cuda_exec(cudaMalloc(&(IMT_), std::max(n_, m_) * maxBlock * sizeof(T)));
    cuda_exec(cudaMalloc(&(H_), m_ * n_ * sizeof(T)));

    cuda_exec(cudaSetDevice(0));
    cublasCreate(&handle_);
    cuda_exec(cudaStreamCreate(&stream_));
    cublasSetStream(handle_, stream_);
  }

  ~ChaseMpiHemmCuda() {
    cudaFree(B_);
    cudaFree(IMT_);
    cudaFree(H_);
    cudaStreamDestroy(stream_);
    cublasDestroy(handle_);
  }

  void preApplication(T* V, std::size_t locked, std::size_t block) {
    next_ = NextOp::bAc;
  }

  void preApplication(T* V, T* V2, std::size_t locked, std::size_t block) {
    cuda_exec(cudaMemcpy(B_, orig_B_, block * n_ * sizeof(T),
                         cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();
    this->preApplication(V, locked, block);
  }

  void apply(T alpha, T beta, std::size_t offset, std::size_t block) {
    T* buf_init;
    T* buf_target;
    std::size_t m, n, k;
    cublasOperation_t transa;
    std::size_t leading_dim;

    if (next_ == NextOp::bAc) {
      buf_init = orig_C_ + offset * m_;
      buf_target = orig_IMT_ + offset * n_;
      m = n_;
      n = block;
      k = m_;
      transa = CUBLAS_OP_C;
      next_ = NextOp::cAb;
    } else {
      buf_init = orig_B_ + offset * n_;
      buf_target = orig_IMT_ + offset * m_;
      m = m_;
      n = block;
      k = n_;
      transa = CUBLAS_OP_N;
      next_ = NextOp::bAc;
    }

    cuda_exec(cudaMemcpyAsync(B_, buf_init, block * k * sizeof(T),
                              cudaMemcpyHostToDevice, stream_));

    cublasTgemm(handle_, transa, CUBLAS_OP_N, m, n, k, &alpha, H_, m_, B_, k,
                &beta, IMT_, m);

    cuda_exec(cudaMemcpyAsync(buf_target, IMT_, m * block * sizeof(T),
                              cudaMemcpyDeviceToHost, stream_));

    cudaStreamSynchronize(stream_);
  }

  bool postApplication(T* V, std::size_t block) { return false; }

  void shiftMatrix(T c) {
    // The MPI part already shifts H_, so we just copy to the device
    cuda_exec(
        cudaMemcpy(H_, orig_H_, n_ * m_ * sizeof(T), cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();
  }

  void applyVec(T* B, T* C) {
    T alpha = T(1.0);
    T beta = T(0.0);

    t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n_, 1, n_, &alpha,
           orig_H_, n_, B, n_, &beta, C, n_);
  }

  void get_off(std::size_t* xoff, std::size_t* yoff, std::size_t* xlen,
               std::size_t* ylen) const {
    *xoff = 0;
    *yoff = 0;
    *xlen = m_;
    *ylen = n_;
  }

  T* get_H() const override { return matrix_properties_->get_H(); }

 private:
  enum NextOp { cAb, bAc };

  std::size_t n_;
  std::size_t m_;
  std::size_t N_;

  NextOp next_;

  T* B_;
  T* IMT_;
  T* H_;

  T* orig_B_;
  T* orig_C_;
  T* orig_IMT_;
  T* orig_H_;

  cudaStream_t stream_;
  cublasHandle_t handle_;
  ChaseMpiProperties<T>* matrix_properties_;
};

template <typename T>
struct is_skewed_matrixfree<ChaseMpiHemmCuda<T>> {
  static const bool value = true;
};

}  // namespace mpi
}  // namespace chase

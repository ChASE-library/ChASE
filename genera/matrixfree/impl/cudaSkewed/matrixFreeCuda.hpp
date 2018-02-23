#pragma once

#include <assert.h>
#include <cuComplex.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <complex>

#include "genera/matrixfree/blas_templates.h"
#include "genera/matrixfree/cuda_wrapper.h"
#include "genera/matrixfree/matrixfree_interface.h"

namespace chase {
namespace matrixfree {

template <class T>
class MatrixFreeCudaSkewed : public MatrixFreeInterface<T> {
 public:
  MatrixFreeCudaSkewed(
      std::shared_ptr<SkewedMatrixProperties<T>> matrix_properties) {
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
    // cuda_exec(cudaMalloc(&(C_), m_ * maxBlock * sizeof(T)));
    cuda_exec(cudaMalloc(&(IMT_), std::max(n_, m_) * maxBlock * sizeof(T)));
    cuda_exec(cudaMalloc(&(H_), m_ * n_ * sizeof(T)));

    cuda_exec(cudaSetDevice(0));
    cublasCreate(&handle_);
    cuda_exec(cudaStreamCreate(&stream_));
    cublasSetStream(handle_, stream_);
  }

  ~MatrixFreeCudaSkewed() {}

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
    /*
    if (next_ == NextOp::bAc) {

        cuda_exec(
            cudaMemcpy(
                C_ + offset * m_,
                orig_C_ + offset * m_,
                block * m_ * sizeof(T),
                cudaMemcpyHostToDevice));

        cublasTgemm(
            handle_, CUBLAS_OP_C, CUBLAS_OP_N,
            n_, block, m_,
            &alpha,
            H_, m_,
            C_ + offset * m_, m_,
            &beta,
            IMT_ + offset * n_, n_);

        cudaStreamSynchronize(stream_);

        cuda_exec(
            cudaMemcpy(
                orig_IMT_ + offset * n_,
                IMT_ + offset * n_,
                n_ * block * sizeof(T),
                cudaMemcpyDeviceToHost));

        next_ = NextOp::cAb;
    } else { // cAb
        cuda_exec(
            cudaMemcpy(
                B_ + offset * n_,
                orig_B_ + offset * n_,
                block * n_ * sizeof(T),
                cudaMemcpyHostToDevice));

        cublasTgemm(
            handle_, CUBLAS_OP_N, CUBLAS_OP_N,
            m_, block, n_,
            &alpha,
            H_, m_,
            B_ + offset * n_, n_,
            &beta,
            IMT_ + offset * m_, m_);

        cudaStreamSynchronize(stream_);

        cuda_exec(
            cudaMemcpy(
                orig_IMT_ + offset * m_,
                IMT_ + offset * m_,
                m_ * block * sizeof(T),
                cudaMemcpyDeviceToHost));

        next_ = NextOp::bAc;
    }
    */
  }

  bool postApplication(T* V, std::size_t block) {
    // std::size_t pitch_host = n_ * sizeof(T);
    // std::size_t pitch_device = n_ * sizeof(T);

    // cuda_exec(
    //     cudaMemcpy2D(V, pitch_host,
    //         V1_, pitch_device,
    //         n_ * sizeof(T), block,
    //         cudaMemcpyDeviceToHost));
    // std::memcpy(V, V1_, n_ * block * sizeof(T));

    return false;
  }

  void shiftMatrix(T c) {
    // shiftMatrixGPU(H_, n_, n_, c, 0, stream_);
    // for (std::size_t i = 0; i < n_; ++i) {
    //     OrigH_[i + i * n_] += c;
    // }

    cuda_exec(
        cudaMemcpy(H_, orig_H_, n_ * m_ * sizeof(T), cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();
  }

  void applyVec(T* B, T* C) {
    T alpha = T(1.0);
    T beta = T(0.0);

    // std::cout << "not implemented\n";
    t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n_, 1, n_, &alpha,
           orig_H_, n_, B, n_, &beta, C, n_);
  }

  void get_off(CHASE_INT* xoff, CHASE_INT* yoff, CHASE_INT* xlen,
               CHASE_INT* ylen) const {
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
  T* C_;
  T* IMT_;
  T* H_;

  T* orig_B_;
  T* orig_C_;
  T* orig_IMT_;
  T* orig_H_;

  cudaStream_t stream_;
  cublasHandle_t handle_;
  std::shared_ptr<SkewedMatrixProperties<T>> matrix_properties_;
};
}  // namespace matrixfree
}  // namespace chase

#pragma once

#include <assert.h>
#include <complex>

#include "../matrixFreeInterface/cuda_wrapper.hpp"
#include "../matrixFreeInterface/matrixFreeInterface.hpp"
#include "../matrixFreeInterface/template_wrapper.hpp"

using U = std::complex<double>;

// the idea is to do the gemm in double precision (on the CPU),
// and then compare to CPU and GPU single precision

template <class T>
class MatrixFreeDebug : public MatrixFreeInterface<T> {
 public:
  MatrixFreeDebug(T* H, std::size_t n, std::size_t maxBlock) : n_(n) {
    cuda_exec(cudaMalloc(&(V1_), n_ * maxBlock * sizeof(T)));
    cuda_exec(cudaMalloc(&(V2_), n_ * maxBlock * sizeof(T)));
    cuda_exec(cudaMalloc(&(H_), n_ * n_ * sizeof(T)));

    OrigH_ = H;

    cuda_exec(cudaSetDevice(0));
    cublasCreate(&handle_);
    cuda_exec(cudaStreamCreate(&stream_));
    cublasSetStream(handle_, stream_);

    cuda_exec(
        cudaMemcpy(H_, OrigH_, n_ * n_ * sizeof(T), cudaMemcpyHostToDevice));

    // single precision for GEMM
    V1_CPU_ = new T[n_ * maxBlock];
    V2_CPU_ = new T[n_ * maxBlock];

    // we alloc the double precision variants
    OrigH_D_ = new U[n_ * n_];
    V1_D_ = new U[n_ * maxBlock];
    V2_D_ = new U[n_ * maxBlock];

    // std::size_t pitch_host = n_ * sizeof(T);
    // std::size_t pitch_device = n_ * sizeof(T);
    // cuda_exec(cudaMemcpy2D(H_, pitch_device, OrigH_, pitch_host, n_ *
    // sizeof(T), n_, cudaMemcpyHostToDevice));
  }

  ~MatrixFreeDebug() {}

  void preApplication(T* V, std::size_t locked, std::size_t block) {
    locked_ = locked;
    // std::size_t pitch_host = n_ * sizeof(T);
    // std::size_t pitch_device = n_ * sizeof(T);
    // cuda_exec(cudaMemcpy2D(V1_, pitch_device, V + locked * n_, pitch_host,
    //                        n_ * sizeof(T), block, cudaMemcpyHostToDevice));

    cuda_exec(cudaMemcpy(V1_, V + locked * n_, block * n_ * sizeof(T),
                         cudaMemcpyHostToDevice));

    // copy V1 in single and double precision
    std::copy(V + locked * n_, V + locked * n_ + n_ * block, V1_D_);
    std::copy(V + locked * n_, V + locked * n_ + n_ * block, V1_CPU_);

    Base<U> norm;
    U avg;

    norm = 0;
    for (std::size_t it = 0; it < n_ * block; ++it) {
      norm += std::abs(V[+locked * n_ + it]) * std::abs(V[+locked * n_ + it]);
    }
    norm = std::sqrt(norm);
    std::cout << "norm of init EV: " << norm << "\n";
  }

  void preApplication(T* V1, T* V2, std::size_t locked, std::size_t block) {
    // std::size_t pitch_host = n_ * sizeof(T);
    // std::size_t pitch_device = n_ * sizeof(T);
    // cuda_exec(cudaMemcpy2D(V2_, pitch_device, V2 + locked * n_, pitch_host,
    //                        n_ * sizeof(T), block, cudaMemcpyHostToDevice));

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

    // populate H in double precision
    std::copy(OrigH_, OrigH_ + n_ * n_, OrigH_D_);
    U alphaU = alpha;
    U betaU = beta;

    // do gemm in double precision
    t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans,  //
           n_, block, n_,                              //
           &alphaU,                                    // V2 <-
           OrigH_D_, n_,                               //      alpha * H*V1
           V1_D_ + offset * n_, n_,                    //      + beta * V2
           &betaU,                                     //
           V2_D_ + offset * n_, n_);

    // do gemm in single precision
    t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans,  //
           n_, block, n_,                              //
           &alpha,                                     // V2 <-
           OrigH_, n_,                                 //      alpha * H*V1
           V1_CPU_ + offset * n_, n_,                  //      + beta * V2
           &beta,                                      //
           V2_CPU_ + offset * n_, n_);

    Base<U> norm;
    U avg;
    Base<U> max;

    std::cout << "block: " << block << "\n";

    norm = 0;
    for (std::size_t it = 0; it < n_ * block; ++it) {
      norm += std::abs(V1_D_[it]) * std::abs(V1_D_[it]);
    }
    norm = std::sqrt(norm);
    std::cout << "norm of EV[0]: " << norm << "\n";

    norm = 0;
    for (std::size_t it = 0; it < n_ * block; ++it) {
      norm += std::abs(V2_D_[it]) * std::abs(V2_D_[it]);
    }
    norm = std::sqrt(norm);
    std::cout << "norm of EV: " << norm << "\n";

    // compare double prec w CPU
    norm = 0;
    avg = 0;
    max = 0;
    for (std::size_t it = 0; it < n_ * block; ++it) {
      max = std::max(max, std::abs(V2_D_[it] - V2_CPU_[it]));
      avg += V2_D_[it] - V2_CPU_[it];
      norm +=
          std::abs(V2_D_[it] - V2_CPU_[it]) * std::abs(V2_D_[it] - V2_CPU_[it]);
    }

    avg = avg / (n_ * block);
    norm = std::sqrt(norm);
    std::cout << "CPU::avg " << avg << "\t CPU::norm " << norm << "  CPU::max "
              << max << "\n";

    // load gpu version into cpu buffer and compare
    // compare double prec w GPU
    norm = 0;
    avg = 0;
    max = 0;
    cuda_exec(cudaMemcpy(V2_CPU_, V2_, block * n_ * sizeof(T),
                         cudaMemcpyDeviceToHost));
    for (std::size_t it = 0; it < n_ * block; ++it) {
      max = std::max(max, std::abs(V2_D_[it] - V2_CPU_[it]));
      avg += V2_D_[it] - V2_CPU_[it];
      norm +=
          std::abs(V2_D_[it] - V2_CPU_[it]) * std::abs(V2_D_[it] - V2_CPU_[it]);
    }

    avg = avg / (n_ * block);
    norm = std::sqrt(norm);
    std::cout << "GPU::avg " << avg << "\t GPU::norm " << norm << "  CPU::max "
              << max << " \n";

    std::cout << "********************\n";

    std::swap(V1_, V2_);
    std::swap(V1_D_, V2_D_);
    std::swap(V1_CPU_, V2_CPU_);
  }

  bool postApplication(T* V, std::size_t block) {
    // std::size_t pitch_host = n_ * sizeof(T);
    // std::size_t pitch_device = n_ * sizeof(T);
    // cuda_exec(cudaMemcpy2D(V + locked_ * n_, pitch_host, V1_, pitch_device,
    //                        n_ * sizeof(T), block, cudaMemcpyDeviceToHost));

    cuda_exec(cudaMemcpy(V + locked_ * n_, V1_, block * n_ * sizeof(T),
                         cudaMemcpyDeviceToHost));

    return false;
  }

  void shiftMatrix(T c) {
    for (std::size_t i = 0; i < n_; ++i) {
      OrigH_[i + i * n_] += c;
    }

    // std::size_t pitch_host = n_ * sizeof(T);
    // std::size_t pitch_device = n_ * sizeof(T);
    // cuda_exec(cudaMemcpy2D(H_, pitch_device, OrigH_, pitch_host, n_ *
    // sizeof(T), n_, cudaMemcpyHostToDevice));

    cuda_exec(
        cudaMemcpy(H_, OrigH_, n_ * n_ * sizeof(T), cudaMemcpyHostToDevice));
  }

  void applyVec(T* B, T* C) {
    T alpha = T(1.0);
    T beta = T(0.0);

    t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans,  //
           n_, 1, n_,                                  //
           &alpha,                                     //
           OrigH_, n_,                                 //
           B, n_,                                      //
           &beta,                                      //
           C, n_);                                     //
  }

  void get_off(CHASE_INT* xoff, CHASE_INT* yoff, CHASE_INT* xlen,
               CHASE_INT* ylen) const override {
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

  U* V1_D_;
  U* V2_D_;
  U* H_D_;
  U* OrigH_D_;

  T* V1_CPU_;
  T* V2_CPU_;
  T* H_CPU_;

  cudaStream_t stream_;
  cublasHandle_t handle_;
};

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
#include <cuda_profiler_api.h>

#include "blas_cuda_wrapper.hpp"
#include "blas_templates.hpp"
#include "chase_mpihemm_interface.hpp"
#include "mgpu_cudaHemm.hpp"

void chase_zshift_mpi_matrix(std::complex<double>* A, std::size_t* off,
                             std::size_t n, std::size_t m, double shift,
                             cudaStream_t* stream_);

void chase_zshift_matrix(std::complex<double>* A, int n, double shift,
                         cudaStream_t* stream_);
namespace chase {
namespace mpi {

template <class T>
class ChaseMpiHemmMultiGPU : public ChaseMpiHemmInterface<T> {
 public:
  ChaseMpiHemmMultiGPU(ChaseMpiProperties<T>* matrix_properties) {
    n_ = matrix_properties->get_n();
    m_ = matrix_properties->get_m();
    N_ = matrix_properties->get_N();

    orig_H_ = matrix_properties->get_H();
    orig_B_ = matrix_properties->get_B();
    orig_C_ = matrix_properties->get_C();
    orig_IMT_ = matrix_properties->get_IMT();

    off_ = matrix_properties->get_off();
    //copied_ = false;

    matrix_properties_ = matrix_properties;

	// Remove allocation of memories. Will be done in mgpu_gpu class (constructor)

    int num_of_devices;
    int mpi_rank = matrix_properties_->get_my_rank();
    cuda_exec(cudaGetDeviceCount(&num_of_devices));

    //int device_id = mpi_rank % num_of_devices;
    //cuda_exec(cudaSetDevice(device_id));

    std::size_t maxBlock = matrix_properties_->get_max_block();

    std::cout << "[CHASE_MGPU] MPI rank " << mpi_rank << " running on "<< num_of_devices << std::endl;
	std::cout << "[CHASE_MGPU] MPI rank " << mpi_rank << "local dimensions: " << std::endl << "\t m = " <<  m_ << std::endl << "\t n = " << n_ << std::endl << "\t block = " << maxBlock << std::endl;

	/* Register H, B, C and IMT as pinned-memories on host */
	cuda_exec(cudaHostRegister((void*)orig_H_, m_*n_*sizeof(T), cudaHostRegisterDefault));
	cuda_exec(cudaHostRegister((void*)orig_B_, n_*maxBlock*sizeof(T), cudaHostRegisterDefault));
	cuda_exec(cudaHostRegister((void*)orig_IMT_, std::max(n_,m_)*maxBlock*sizeof(T), cudaHostRegisterDefault));
	cuda_exec(cudaHostRegister((void*)orig_C_, m_*maxBlock*sizeof(T), cudaHostRegisterDefault));
		
    //cuda_exec(cudaMalloc(&(B_), std::max(n_, m_) * maxBlock * sizeof(T)));
    //cuda_exec(cudaMalloc(&(IMT_), std::max(n_, m_) * maxBlock * sizeof(T)));
    //cuda_exec(cudaMalloc(&(H_), m_ * n_ * sizeof(T)));

    /// Create CUBLAS context
    //cublasCreate(&handle_);

	/// Create and set cuda stream
    //cuda_exec(cudaStreamCreate(&stream_));
    //cublasSetStream(handle_, stream_);

	/// Construct a new object for handling multi-GPU HEMM execution
	//mgpuHemm = new mgpu_cudaHemm<T>(handle_, stream_, m_, n_, maxBlock);
	cudaProfilerStart();
	mgpuHemm = new mgpu_cudaHemm<T>(m_, n_, maxBlock);
	cudaProfilerStop();
  }

  ~ChaseMpiHemmMultiGPU() {
    //cudaFree(B_);
    //cudaFree(IMT_);
    //cudaFree(H_);
    cuda_exec(cudaHostUnregister(orig_H_));
    cuda_exec(cudaHostUnregister(orig_B_));
    cuda_exec(cudaHostUnregister(orig_C_));
    cuda_exec(cudaHostUnregister(orig_IMT_));
    delete mgpuHemm;
    //cudaStreamDestroy(stream_);
    //cublasDestroy(handle_);
  }

  void preApplication(T* V, std::size_t locked, std::size_t block) {
    next_ = NextOp::bAc;
  }

  void preApplication(T* V, T* V2, std::size_t locked, std::size_t block) {
    // cuda_exec(cudaMemcpy(B_, orig_B_, block * n_ * sizeof(T),
    //                      cudaMemcpyHostToDevice));
    // cudaDeviceSynchronize();
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

    /*cuda_exec(cudaMemcpyAsync(B_, buf_init, block * k * sizeof(T),
                              cudaMemcpyHostToDevice, stream_));

    cublasTgemm(handle_, transa, CUBLAS_OP_N, m, n, k, &alpha, H_, m_, B_, k,
                &beta, IMT_, m);

    cuda_exec(cudaMemcpyAsync(buf_target, IMT_, m * block * sizeof(T),
                              cudaMemcpyDeviceToHost, stream_));
	*/
	cudaProfilerStart();
	/// Transfer block-vector to GPUs
    mgpuHemm->distribute_V(buf_init, k, block);

	/// Compute Hemm
	//mgpuHemm->computeHemm(buf_init, buf_target, m, n, k, block, alpha, beta, transa);
	//mgpuHemm->computeHemm(m, n, k, alpha, beta, transa);
	mgpuHemm->computeHemm(block, alpha, beta);

	mgpuHemm->synchronizeAll();

	/// Return computed block-vector to CPU
	mgpuHemm->return_W(buf_target, m, block);

	mgpuHemm->switch_operation();
	cudaProfilerStop();
  }

  bool postApplication(T* V, std::size_t block) {
    //cudaStreamSynchronize(stream_);
    /*  */
	mgpuHemm->synchronizeAll();

    return false;
  }

  void shiftMatrix(T c, bool isunshift = false) {
    /*if (!copied_) {
      cuda_exec(cudaMemcpyAsync(H_, orig_H_, m_ * n_ * sizeof(T),
                                cudaMemcpyHostToDevice, stream_));
      copied_ = true;
    }

    // cudaDeviceSynchronize();
    cuda_exec(
        cudaMemcpy(H_, orig_H_, n_ * m_ * sizeof(T), cudaMemcpyHostToDevice));
	*/
	cudaProfilerStart();
	mgpuHemm->distribute_H(orig_H_, m_);
	cudaProfilerStop();

    // chase_zshift_mpi_matrix(H_, off_, n_, m_, std::real(c), &stream_);
    // chase_zshift_matrix(H_, n_, std::real(c), &stream_);
  }

  void applyVec(T* B, T* C) {
    T alpha = T(1.0);
    T beta = T(0.0);

    // this->preApplication(B, 0, 1);
    // this->apply(alpha, beta, 0, 1);
    // this->postApplication(C, 1);

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
  void Start() override { copied_ = false; }

 private:
  enum NextOp { cAb, bAc };

  std::size_t n_;
  std::size_t m_;
  std::size_t N_;

  NextOp next_;

  mgpu_cudaHemm<T> *mgpuHemm;

  //T* B_;
  //T* IMT_;
  //T* H_;

  T* orig_B_;
  T* orig_C_;
  T* orig_IMT_;
  T* orig_H_;

  std::size_t* off_;

  bool copied_;

  //cudaStream_t *stream_;
  //cublasHandle_t *handle_;

  /// Matrix properties
  ChaseMpiProperties<T>* matrix_properties_;
};

template <typename T>
struct is_skewed_matrixfree<ChaseMpiHemmMultiGPU<T>> {
  static const bool value = true;
};

}  // namespace mpi
}  // namespace chase

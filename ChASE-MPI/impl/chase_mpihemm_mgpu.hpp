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

#include <chrono>

#include "blas_cuda_wrapper.hpp"
#include "blas_templates.hpp"
#include "chase_mpihemm_interface.hpp"
#include "mgpu_cudaHemm.hpp"

void chase_zshift_mpi_matrix(std::complex<double>* A, std::size_t* off,
                             std::size_t n, std::size_t m, double shift,
                             cudaStream_t* stream_);

void chase_zshift_matrix(std::complex<double>* A, int n, double shift,
                         cudaStream_t* stream_);

using namespace std::chrono;

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

    matrix_properties_ = matrix_properties;

	// Remove allocation of memories. Will be done in mgpu_gpu class (constructor)

    int num_devices;
    mpi_rank = matrix_properties_->get_my_rank();
    cuda_exec(cudaGetDeviceCount(&num_devices));

    std::size_t maxBlock = matrix_properties_->get_max_block();

#ifdef MGPU_TIMER    
	std::cout << "[MGPU_HEMM] MPI rank " << mpi_rank << " found " << num_devices << " GPU devices" << std::endl;
	std::cout << "[MGPU_HEMM] MPI rank " << mpi_rank << " operating on: m = " <<  m_ << ", n = " << n_ << ", block = " << maxBlock << std::endl;
#endif
	/* Register H, B, C and IMT as pinned-memories on host */
	cuda_exec(cudaHostRegister((void*)orig_H_, m_*n_*sizeof(T), cudaHostRegisterDefault));
	cuda_exec(cudaHostRegister((void*)orig_B_, n_*maxBlock*sizeof(T), cudaHostRegisterDefault));
	cuda_exec(cudaHostRegister((void*)orig_IMT_, std::max(n_,m_)*maxBlock*sizeof(T), cudaHostRegisterDefault));
	cuda_exec(cudaHostRegister((void*)orig_C_, m_*maxBlock*sizeof(T), cudaHostRegisterDefault));

	/// Construct a new object for handling multi-GPU HEMM execution
	mgpuHemm = new mgpu_cudaHemm<T>(m_, n_, maxBlock);

	time_copy_H = std::chrono::milliseconds::zero(); 
	time_copy_W = std::chrono::milliseconds::zero();
	time_copy_V = std::chrono::milliseconds::zero();
	time_gemm = std::chrono::milliseconds::zero();
	time_apply_vec = std::chrono::milliseconds::zero();
  }

  ~ChaseMpiHemmMultiGPU() {
    cuda_exec(cudaHostUnregister(orig_H_));
    cuda_exec(cudaHostUnregister(orig_B_));
    cuda_exec(cudaHostUnregister(orig_C_));
    cuda_exec(cudaHostUnregister(orig_IMT_));
    delete mgpuHemm;

#ifdef MGPU_TIMER
	std::cout << "[MGPU_HEMM] Multi-GPU HEMM timings (per component): " << std::endl;
	std::cout << "[MGPU_HEMM] Copy H   = " << time_copy_H.count()/1000 << " sec" << std::endl;
	std::cout << "[MGPU_HEMM] Copy V   = " << time_copy_V.count()/1000 << " sec" << std::endl;
	std::cout << "[MGPU_HEMM] Return W = " << time_copy_W.count()/1000 << " sec"   << std::endl;
	std::cout << "[MGPU_HEMM] Hemm     = " << time_gemm.count()/1000 << " sec"  << std::endl;
	std::cout << "[MGPU_HEMM] AppyVec  = " << time_apply_vec.count()/1000 << " sec"  << std::endl;
	std::cout << std::endl;
  }

  void preApplication(T* V, std::size_t locked, std::size_t block) {
    next_ = NextOp::bAc;
	mgpuHemm->set_operation(next_);
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
	std::size_t ldBufInit;
	std::size_t ldBufTarget;

    if (next_ == NextOp::bAc) {
      buf_init = orig_C_ + offset * m_;
      buf_target = orig_IMT_ + offset * n_;
      m = n_;
      n = block;
      k = m_;
      ldBufInit = m_;
	  ldBufTarget = n_; 
      transa = CUBLAS_OP_C;
      next_ = NextOp::cAb;
    } else {
      buf_init = orig_B_ + offset * n_;
      buf_target = orig_IMT_ + offset * m_;
      m = m_;
      n = block;
      k = n_;
	  ldBufInit = n_;
      ldBufTarget = m_;
      transa = CUBLAS_OP_N;
      next_ = NextOp::bAc;
    }

	/// Transfer block-vector to GPUs
	auto start = high_resolution_clock::now();
    mgpuHemm->distribute_V(buf_init, ldBufInit, block);
	mgpuHemm->synchronizeAll();
	auto stop = high_resolution_clock::now();
	time_copy_V += stop - start;

	/// Compute Hemm
	start = high_resolution_clock::now();
	mgpuHemm->computeHemm(block, alpha, beta);
	mgpuHemm->synchronizeAll();
	stop = high_resolution_clock::now();
	time_gemm += stop - start;

	/// Return computed block-vector to CPU
	start = high_resolution_clock::now();
	mgpuHemm->return_W(buf_target, ldBufTarget, block);
	mgpuHemm->synchronizeAll();
	stop = high_resolution_clock::now();
	time_copy_W += stop - start;

	mgpuHemm->switch_operation();
	//cudaProfilerStop();
  }

  bool postApplication(T* V, std::size_t block) {
    //cudaStreamSynchronize(stream_);
	mgpuHemm->synchronizeAll();

    return false;
  }

  void shiftMatrix(T c, bool isunshift = false) {

	auto start = high_resolution_clock::now();
	mgpuHemm->distribute_H(orig_H_, m_);
	mgpuHemm->synchronizeAll();

    // chase_zshift_mpi_matrix(H_, off_, n_, m_, std::real(c), &stream_);
    // chase_zshift_matrix(H_, n_, std::real(c), &stream_);
	auto stop = high_resolution_clock::now();
	time_copy_H += stop - start;
  }

  void applyVec(T* B, T* C) {
    T alpha = T(1.0);
    T beta = T(0.0);

    // this->preApplication(B, 0, 1);
    // this->apply(alpha, beta, 0, 1);
    // this->postApplication(C, 1);
	auto start = high_resolution_clock::now();

    t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n_, 1, n_, &alpha,
           orig_H_, n_, B, n_, &beta, C, n_);

	auto stop = high_resolution_clock::now();
	time_apply_vec += stop - start;
  }

  void get_off(std::size_t* xoff, std::size_t* yoff, std::size_t* xlen,
               std::size_t* ylen) const {
    *xoff = 0;
    *yoff = 0;
    *xlen = m_;
    *ylen = n_;
  }

  T* get_H() const override { return matrix_properties_->get_H(); }
  std::size_t get_mblocks() const override {return matrix_properties_->get_mblocks();}
  std::size_t get_nblocks() const override {return matrix_properties_->get_nblocks();}
  std::size_t get_n() const override {return matrix_properties_->get_n();}
  std::size_t get_m() const override {return matrix_properties_->get_m();}
  int *get_coord() const override {return matrix_properties_->get_coord();}
  void get_offs_lens(std::size_t* &r_offs, std::size_t* &r_lens, std::size_t* &r_offs_l,
                  std::size_t* &c_offs, std::size_t* &c_lens, std::size_t* &c_offs_l) const override{
     matrix_properties_->get_offs_lens(r_offs, r_lens, r_offs_l, c_offs, c_lens, c_offs_l);
  }

  void Start() override { copied_ = false; }

 private:
  enum NextOp { cAb, bAc };

  std::size_t n_;
  std::size_t m_;
  std::size_t N_;

  NextOp next_;

  mgpu_cudaHemm<T> *mgpuHemm;

  T* orig_B_;
  T* orig_C_;
  T* orig_IMT_;
  T* orig_H_;

  std::size_t* off_;

  int mpi_rank;

  bool copied_;

  /// Matrix properties
  ChaseMpiProperties<T>* matrix_properties_;

  /// Timing variables
  std::chrono::duration<double, std::milli> time_copy_H;
  std::chrono::duration<double, std::milli> time_copy_W;
  std::chrono::duration<double, std::milli> time_copy_V;
  std::chrono::duration<double, std::milli> time_gemm;
  std::chrono::duration<double, std::milli> time_apply_vec;
};

template <typename T>
struct is_skewed_matrixfree<ChaseMpiHemmMultiGPU<T>> {
  static const bool value = true;
};

}  // namespace mpi
}  // namespace chase

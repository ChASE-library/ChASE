/* -*- Mode: C++; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
// This file is a part of ChASE.
// Copyright (c) 2015-2021, Simulation and Data Laboratory Quantum Materials, 
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

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
#include "chase_mpidla_interface.hpp"
#include "mgpu_cudaDLA.hpp"

void chase_shift_mgpu_matrix(float* A, std::size_t* off_m, std::size_t* off_n,
                            std::size_t offsize, std::size_t ldH, float shift,
                             cudaStream_t stream_);

void chase_shift_mgpu_matrix(double* A, std::size_t* off_m, std::size_t* off_n,
                            std::size_t offsize, std::size_t ldH, double shift,
                             cudaStream_t stream_);

void chase_shift_mgpu_matrix(std::complex<double>* A, std::size_t* off_m, std::size_t* off_n,
                             std::size_t offsize, std::size_t ldH, double shift,
                             cudaStream_t stream_);

void chase_shift_mgpu_matrix(std::complex<float>* A, std::size_t* off_m, std::size_t* off_n,
                             std::size_t offsize, std::size_t ldH, float shift,
                             cudaStream_t stream_);


using namespace std::chrono;

namespace chase {
namespace mpi {

//! A derived class of ChaseMpiDLAInterface which implements the inter-node computation for a MPI-based implementation with GPU supports of ChASE.
/*! This implementation supports to use all the GPU cards available on each node. The multi-GPU within node can be managed either with 1 MPI rank or each GPU be bounded to one MPI rank. 
*/
template <class T>
class ChaseMpiDLAMultiGPU : public ChaseMpiDLAInterface<T> {
 public:
  //! A constructor of ChaseMpiDLAMultiGPU.
  //! This construct set up the CUDA environment and register device memory which will be used by ChASE.
  //! @param matrix_properties: it is an object of ChaseMpiProperties, which defines the MPI environment and data distribution scheme in ChASE-MPI.  
  ChaseMpiDLAMultiGPU(ChaseMpiProperties<T>* matrix_properties) {
    n_ = matrix_properties->get_n();
    m_ = matrix_properties->get_m();
    N_ = matrix_properties->get_N();

    orig_H_ = matrix_properties->get_H();
    orig_B_ = matrix_properties->get_B();
    orig_C_ = matrix_properties->get_C();

    off_ = matrix_properties->get_off();

    matrix_properties_ = matrix_properties;

	// Remove allocation of memories. Will be done in mgpu_gpu class (constructor)

    int num_devices;
    mpi_rank = matrix_properties_->get_my_rank();
	
	MPI_Comm row_comm = matrix_properties_->get_row_comm();
	MPI_Comm col_comm = matrix_properties_->get_col_comm();

	MPI_Comm_rank(row_comm, &mpi_row_rank);
	MPI_Comm_rank(col_comm, &mpi_col_rank);

    cuda_exec(cudaGetDeviceCount(&num_devices));

    std::size_t maxBlock = matrix_properties_->get_max_block();

	previous_offset_ = -1;

#ifdef MGPU_TIMER    
	std::cout << "[MGPU_HEMM] MPI rank " << mpi_rank << " found " << num_devices << " GPU devices" << std::endl;
	std::cout << "[MGPU_HEMM] MPI rank " << mpi_rank << " operating on: m = " <<  m_ << ", n = " << n_ << ", block = " << maxBlock << std::endl;
#endif
	/* Register H, B, C and IMT as pinned-memories on host */
	cuda_exec(cudaHostRegister((void*)orig_H_, m_*n_*sizeof(T), cudaHostRegisterDefault));
	cuda_exec(cudaHostRegister((void*)orig_B_, n_*maxBlock*sizeof(T), cudaHostRegisterDefault));
	cuda_exec(cudaHostRegister((void*)orig_C_, m_*maxBlock*sizeof(T), cudaHostRegisterDefault));

	/// Construct a new object for handling multi-GPU HEMM execution
	mgpuDLA = new mgpu_cudaDLA<T>(matrix_properties, m_, n_, maxBlock);

	time_copy_H = std::chrono::milliseconds::zero(); 
	time_copy_W = std::chrono::milliseconds::zero();
	time_copy_V = std::chrono::milliseconds::zero();
	time_gemm = std::chrono::milliseconds::zero();
	time_apply_vec = std::chrono::milliseconds::zero();

  }

  ~ChaseMpiDLAMultiGPU() {
    cuda_exec(cudaHostUnregister(orig_H_));
    cuda_exec(cudaHostUnregister(orig_B_));
    cuda_exec(cudaHostUnregister(orig_C_));
    delete mgpuDLA;

#ifdef MGPU_TIMER
	std::cout << "[MGPU_HEMM] Multi-GPU HEMM timings (per component): " << std::endl;
	std::cout << "[MGPU_HEMM] Copy H   = " << time_copy_H.count()/1000 << " sec" << std::endl;
	std::cout << "[MGPU_HEMM] Copy V   = " << time_copy_V.count()/1000 << " sec" << std::endl;
	std::cout << "[MGPU_HEMM] Return W = " << time_copy_W.count()/1000 << " sec"   << std::endl;
	std::cout << "[MGPU_HEMM] Hemm     = " << time_gemm.count()/1000 << " sec"  << std::endl;
	std::cout << "[MGPU_HEMM] AppyVec  = " << time_apply_vec.count()/1000 << " sec"  << std::endl;
	std::cout << std::endl;
#endif
  }

  /*! - For ChaseMpiDLAMultiGPU, `preApplication` is implemented only with the operation of switching operation flags.
      - For the meaning of this function, please visit ChaseMpiDLAInterface.
  */
  void preApplication(T* V, std::size_t locked, std::size_t block)  override {
    next_ = NextOp::bAc;
	mgpuDLA->set_operation(next_);
	previous_offset_ = -1;
  }

  /*! - For ChaseMpiDLAMultiGPU, `preApplication` is implemented only with the operation of switching operation flags.
      - For the meaning of this function, please visit ChaseMpiDLAInterface.
  */
  void preApplication(T* V, T* V2, std::size_t locked, std::size_t block)  override {
    // cuda_exec(cudaMemcpy(B_, orig_B_, block * n_ * sizeof(T),
    //                      cudaMemcpyHostToDevice));
    // cudaDeviceSynchronize();
    this->preApplication(V, locked, block);
  }

   /*!
      - For ChaseMpiDLAMultiGPU, the matrix-matrix multiplication of local matrices are offloaded to multi-GPUs available on the node. The collection of local product among multi-GPUs is done by MPI communication scheme. The computation on each GPU card is implemented with `cublasXgemm` provided by `cuBLAS`.
      - The multi-GPU HEMM within each node is implemented in class mgpu_cudaDLA.
      - The collective communication based on MPI which **ALLREDUCE** the product of local matrices either within the column communicator or row communicator, is implemented within ChaseMpiDLA.
      - **Parallelism on distributed-memory system SUPPORT**
      - **Parallelism within each node among multi-GPUs SUPPORT** 
      - **Parallelism within each GPU SUPPORT**             
      - For the meaning of this function, please visit ChaseMpiDLAInterface.
  */
  void apply(T alpha, T beta, std::size_t offset, std::size_t block) override  {

	T Zero = T(0.0);
    T* buf_init;
    T* buf_target;
    std::size_t m, n, k;
    cublasOperation_t transa;
    std::size_t leading_dim;
	std::size_t ldBufInit;
	std::size_t ldBufTarget;

    if (next_ == NextOp::bAc) {
      buf_init = orig_C_ + offset * m_;
      buf_target = orig_B_ + offset * n_;
      m = n_;
      n = block;
      k = m_;
      ldBufInit = m_;
	  ldBufTarget = n_; 
      transa = CUBLAS_OP_C;
      next_ = NextOp::cAb;
	  if (mpi_col_rank != 0) {
			beta = Zero;
	  }
    } else {
      buf_init = orig_B_ + offset * n_;
      buf_target = orig_C_ + offset * m_;
      m = m_;
      n = block;
      k = n_;
	  ldBufInit = n_;
      ldBufTarget = m_;
      transa = CUBLAS_OP_N;
      next_ = NextOp::bAc;
	  if (mpi_row_rank != 0) {
			beta = Zero;
	  }
    }

	/* Set local offset for W. The column-matrix from the previuos step 
 	 * (already distributed among GPUs) had different (equal or smaller) 
 	 * offset. We have to find the differene between the offset used in
 	 * the previous call and the current one
 	 */
	std::size_t W_offset;
	if(previous_offset_ == -1) {
		W_offset = 0;
	} else {
		W_offset = offset - previous_offset_;
	}

	/// Transfer block-vector to GPUs
	// TODO: Do not distribute buf_init if beta == 0 -> spare one copying
    auto start = high_resolution_clock::now();
    mgpuDLA->distribute_V(buf_init, ldBufInit, block);
    mgpuDLA->synchronizeAll();
    auto stop = high_resolution_clock::now();
    time_copy_V += stop - start;

	/// Compute Hemm
	start = high_resolution_clock::now();
	mgpuDLA->computeHemm(block, W_offset, alpha, beta);
	mgpuDLA->synchronizeAll();
	stop = high_resolution_clock::now();
	time_gemm += stop - start;

	/// Return computed block-vector to CPU
	start = high_resolution_clock::now();
	mgpuDLA->return_W(buf_target, ldBufTarget, block, W_offset);
	mgpuDLA->synchronizeAll();
	stop = high_resolution_clock::now();
	time_copy_W += stop - start;

	mgpuDLA->switch_operation();

	previous_offset_ = offset;
  }

  /*!
     - For ChaseMpiDLAMultiGPU,  `postApplication` is implemented for the synchronization among the GPUs bounded to a same MPI rank.
     - For the meaning of this function, please visit ChaseMpiDLAInterface.  
  */
  bool postApplication(T* V, std::size_t block)  override {
    //cudaStreamSynchronize(stream_);
	mgpuDLA->synchronizeAll();

	previous_offset_ = -1;

    return false;
  }

  /*!
    - For ChaseMpiDLAMultiGPU, `shiftMatrix` is implemented in ChaseMpiDLA which is executed on CPUs.
    - **Parallelism on distributed-memory system SUPPORT**
    - The distribution of matrix `H` owned by each MPI rank to the GPUs bounded to it, it is called in this function. The implementation is within class mgpu_cudaDLA.
    - For the meaning of this function, please visit ChaseMpiDLAInterface.    
  */
  void shiftMatrix(T c, bool isunshift = false)  override {

	auto start = high_resolution_clock::now();
	mgpuDLA->distribute_H(orig_H_, m_);
        mgpuDLA->shiftMatrix(c);
	mgpuDLA->synchronizeAll();

    // chase_zshift_mpi_matrix(H_, off_, n_, m_, std::real(c), &stream_);
    // chase_zshift_matrix(H_, n_, std::real(c), &stream_);
	auto stop = high_resolution_clock::now();
	time_copy_H += stop - start;
  }

  /*!
    - For ChaseMpiDLAMultiGPU,  `applyVec` is implemented in ChaseMpiDLA.
    - **Parallelism on distributed-memory system SUPPORT**
    - **Parallelism within each node among multi-GPUs SUPPORT** 
    - **Parallelism within each GPU SUPPORT**     
    - For the meaning of this function, please visit ChaseMpiDLAInterface.    
  */
  void applyVec(T* B, T* C)  override {
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
               std::size_t* ylen) const  override  {
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

  /*!
    - For ChaseMpiDLAMultiGPU, `lange` is implemented using `LAPACK` routine `xLANGE`.
    - **Parallelism is SUPPORT within node if multi-threading is enabled**
    - For the meaning of this function, please visit ChaseMpiDLAInterface.
  */
  Base<T> lange(char norm, std::size_t m, std::size_t n, T* A, std::size_t lda) override {
      return t_lange(norm, m, n, A, lda);
  }

  /*!
    - For ChaseMpiDLAMultiGPU, `gegqr` is implemented by calling `gegqr` function of class mgpu_cudaDLA, whose implementation is based on `cuSOLVER` routines `cusolverDnXgeqrf` and `cusolverDnXumgqr`.
    - **Parallelism is SUPPORT within one GPU card**
    - For the meaning of this function, please visit ChaseMpiDLAInterface.
  */
  void gegqr(std::size_t N, std::size_t nevex, T * approxV, std::size_t LDA) override {
      mgpuDLA->gegqr(N, nevex, approxV, LDA);
  }

  /*!
    - For ChaseMpiDLAMultiGPU, `axpy` is implemented in ChaseMpiDLA.
   - **Parallelism is SUPPORT within node if multi-threading is enabled**    
    - For the meaning of this function, please visit ChaseMpiDLAInterface.
  */
  void axpy(std::size_t N, T * alpha, T * x, std::size_t incx, T *y, std::size_t incy) override { }

  /*!
    - For ChaseMpiDLAMultiGPU, `scal` is implemented in ChaseMpiDLA
    - **Parallelism is SUPPORT within node if multi-threading is enabled**   
    - For the meaning of this function, please visit ChaseMpiDLAInterface.
  */  
  void scal(std::size_t N, T *a, T *x, std::size_t incx) override { }

  /*!
    - For ChaseMpiDLAMultiGPU, `nrm2` is implemented using `BLAS` routine `xNRM2`.
    - **Parallelism is SUPPORT within node if multi-threading is enabled**    
    - For the meaning of this function, please visit ChaseMpiDLAInterface.
  */
  Base<T> nrm2(std::size_t n, T *x, std::size_t incx) override {
      return t_nrm2(n, x, incx);
  }

 /*!
    - For ChaseMpiDLAMultiGPU, `dot` is implemented using `BLAS` routine `xDOT`.
    - **Parallelism is SUPPORT within node if multi-threading is enabled**       
    - For the meaning of this function, please visit ChaseMpiDLAInterface.
  */
  T dot(std::size_t n, T* x, std::size_t incx, T* y, std::size_t incy) override {
      return t_dot(n, x, incx, y, incy);
  }

  /*!
   - For ChaseMpiDLAMultiGPU, `gemm_small` is implemented in ChaseMpiDLA.
   - **Parallelism is SUPPORT within node if multi-threading is enabled**    
   - For the meaning of this function, please visit ChaseMpiDLAInterface.
  */
  void gemm_small(CBLAS_LAYOUT Layout, CBLAS_TRANSPOSE transa,
                         CBLAS_TRANSPOSE transb, std::size_t m,
                         std::size_t n, std::size_t k, T* alpha,
                         T* a, std::size_t lda, T* b,
                         std::size_t ldb, T* beta, T* c, std::size_t ldc) override 
  {}

  /*!
   - For ChaseMpiDLAMultiGPU, `gemm_large` is implemented in ChaseMpiDLA.
   - **Parallelism is SUPPORT within node if multi-threading is enabled**    
   - For the meaning of this function, please visit ChaseMpiDLAInterface.
  */
  void gemm_large(CBLAS_LAYOUT Layout, CBLAS_TRANSPOSE transa,
                         CBLAS_TRANSPOSE transb, std::size_t m,
                         std::size_t n, std::size_t k, T* alpha,
                         T* a, std::size_t lda, T* b,
                         std::size_t ldb, T* beta, T* c, std::size_t ldc) override 
  {}

  /*!
   - For ChaseMpiDLAMultiGPU, `stemr` with scalar being real and double precision, is implemented using `LAPACK` routine `DSTEMR`.
   - **Parallelism is SUPPORT within node if multi-threading is enabled**    
   - For the meaning of this function, please visit ChaseMpiDLAInterface.
  */
  std::size_t stemr(int matrix_layout, char jobz, char range, std::size_t n,
                    double* d, double* e, double vl, double vu, std::size_t il, std::size_t iu,
                    int* m, double* w, double* z, std::size_t ldz, std::size_t nzc,
                    int* isuppz, lapack_logical* tryrac) override {
      return t_stemr<double>(matrix_layout, jobz, range, n, d, e, vl, vu, il, iu, m, w, z, ldz, nzc, isuppz, tryrac);
  }

  /*!
   - For ChaseMpiDLAMultiGPU, `stemr` with scalar being real and single precision, is implemented using `LAPACK` routine `SSTEMR`.
   - **Parallelism is SUPPORT within node if multi-threading is enabled**    
   - For the meaning of this function, please visit ChaseMpiDLAInterface.
  */
  std::size_t stemr(int matrix_layout, char jobz, char range, std::size_t n,
                    float* d, float* e, float vl, float vu, std::size_t il, std::size_t iu,
                    int* m, float* w, float* z, std::size_t ldz, std::size_t nzc,
                    int* isuppz, lapack_logical* tryrac) override {
      return t_stemr<float>(matrix_layout, jobz, range, n, d, e, vl, vu, il, iu, m, w, z, ldz, nzc, isuppz, tryrac);
  }

  /*!
    - For ChaseMpiDLAMultiGPU, `RR_kernel` is implemented by calling the `RR_kernel` function of class mgpu_cudaDLA, whose implementation is based on `cuBLAS` routine `cublasXgemm` and `LAPACK` routine `(SY)HEEVD`.
        - The 1st operation `A <- W^T * V` is implemented by `cublasXgemm` from `cuBLAS`.
        - The 2nd operation which computes the eigenpairs of `A`, is implemented by `(SY)HEEVD` from `LAPACK`.
        - The 3rd operation which computes `W<-V*A` is implemented by `cublasXgemm` from `cuBLAS`.
    - **for (SY)HHEVD, parallelism is SUPPORT within node if multi-threading is actived**
    - **for cublasXgemm, parallelism is SUPPORT within one GPU card**
    - For the meaning of this function, please visit ChaseMpiDLAInterface.
  */  
  void RR_kernel(std::size_t N, std::size_t block, T *approxV, std::size_t locked, T *workspace, T One, T Zero, Base<T> *ritzv) override {
      mgpuDLA->RR_kernel(N, block, approxV, locked, workspace, One, Zero, ritzv);
  }

 private:
  enum NextOp { cAb, bAc };

  std::size_t n_;
  std::size_t m_;
  std::size_t N_;

  NextOp next_;

  mgpu_cudaDLA<T> *mgpuDLA;

  T* orig_B_;
  T* orig_C_;
  T* orig_H_;

  std::size_t* off_;

  int mpi_rank;
  int mpi_row_rank;
  int mpi_col_rank;

  bool copied_;

  int previous_offset_;

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
struct is_skewed_matrixfree<ChaseMpiDLAMultiGPU<T>> {
  static const bool value = true;
};

}  // namespace mpi
}  // namespace chase

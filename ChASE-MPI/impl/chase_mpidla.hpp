/* -*- Mode: C++; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
// This file is a part of ChASE.
// Copyright (c) 2015-2021, Simulation and Data Laboratory Quantum Materials, 
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include <mpi.h>
#include <iterator>
#include <numeric>

#include "ChASE-MPI/chase_mpi_properties.hpp"
#include "ChASE-MPI/chase_mpidla_interface.hpp"

namespace chase {
namespace mpi {

//! A derived class of ChaseMpiDLAInterface which implements mostly the MPI collective communications part of ChASE-MPI targeting the distributed-memory systens with or w/o GPUs.
/*! The computation in node are mostly implemented in ChaseMpiDLABlaslapack and ChaseMpiDLAMultiGPU. 
    It supports both `Block Distribution` and `Block-Cyclic Distribution` schemes.
*/ 
template <class T>
class ChaseMpiDLA : public ChaseMpiDLAInterface<T> {
 public:
  //! A constructor of ChaseMpiDLA.
  /*!
    @param matrix_properties: it is an object of ChaseMpiProperties, which defines the MPI environment and data distribution scheme in ChASE-MPI.
    @param dla: it is an object of ChaseMpiDLAInterface, which defines the implementation of in-node computation for ChASE-MPI. Currently, it can be one of ChaseMpiDLABlaslapack and ChaseMpiDLAMultiGPU. 
  */
  ChaseMpiDLA(ChaseMpiProperties<T>* matrix_properties,
               ChaseMpiDLAInterface<T>* dla)
      : dla_(dla) {
    ldc_ = matrix_properties->get_ldc();
    ldb_ = matrix_properties->get_ldb();

    N_ = matrix_properties->get_N();
    n_ = matrix_properties->get_n();
    m_ = matrix_properties->get_m();

    B_ = matrix_properties->get_B();
    C_ = matrix_properties->get_C();

    std::size_t max_block_ = matrix_properties->get_max_block();

    matrix_properties_ = matrix_properties;

    row_comm_ = matrix_properties->get_row_comm();
    col_comm_ = matrix_properties->get_col_comm();

    dims_ = matrix_properties->get_dims();
    coord_ = matrix_properties->get_coord();
    off_ = matrix_properties->get_off();

    data_layout = matrix_properties->get_dataLayout();

    matrix_properties->get_offs_lens(r_offs_, r_lens_, r_offs_l_, c_offs_, c_lens_, c_offs_l_);
    mb_ = matrix_properties->get_mb();
    nb_ = matrix_properties->get_nb();    

    mblocks_ = matrix_properties->get_mblocks();
    nblocks_ = matrix_properties->get_nblocks();

    int sign = 0;
    if(data_layout.compare("Block-Cyclic") == 0){
	sign = 1;
    }

    Buff_ = new T[sign * N_ *  max_block_];

  }

  ~ChaseMpiDLA() {
    delete[] Buff_;
  }

  /*! - For ChaseMpiDLA, `preApplication` is implemented within `std::memcpy`.
      - **Parallelism on distributed-memory system SUPPORT**
      - For the meaning of this function, please visit ChaseMpiDLAInterface.
  */
  void preApplication(T* V, std::size_t locked, std::size_t block) override {
    next_ = NextOp::bAc;
    locked_ = locked;

    for (auto j = 0; j < block; j++){
   	for(auto i = 0; i < mblocks_; i++){
	    std::memcpy(C_ + j * m_ + r_offs_l_[i], V + j * N_ + locked * N_ + r_offs_[i], r_lens_[i] * sizeof(T));
	} 
    }

    dla_->preApplication(V, locked, block);
  }

  /*! - For ChaseMpiDLA, `preApplication` is implemented within `std::memcpy`.
      - **Parallelism on distributed-memory system SUPPORT**
      - For the meaning of this function, please visit ChaseMpiDLAInterface.
  */
  void preApplication(T* V1, T* V2, std::size_t locked, std::size_t block) override {
    for (auto j = 0; j < block; j++) {
	for(auto i = 0; i < nblocks_; i++){
            std::memcpy(B_ + j * n_ + c_offs_l_[i], V2 + j * N_ + locked * N_ + c_offs_[i], c_lens_[i] * sizeof(T));	    
	}	
    }

    dla_->preApplication(V1, V2, locked, block);

    this->preApplication(V1, locked, block);
  }

   /*!
      - In ChaseMpiDLA, collective communication of `HEMM` operation based on MPI which **ALLREDUCE** the product of local matrices either within the column communicator or row communicator.
      - In ChaseMpiDLA, `scal` and `axpy` are implemented with the one provided by `BLAS`.
      - **Parallelism on distributed-memory system SUPPORT**
      - **Parallelism within node for ChaseMpiDLABlaslapack if multi-threading is enabled**          
      - **Parallelism within node among multi-GPUs for ChaseMpiDLAMultiGPU**
      - **Parallelism within each GPU for ChaseMpiDLAMultiGPU**             
      - For the meaning of this function, please visit ChaseMpiDLAInterface.
  */
  void apply(T alpha, T beta, std::size_t offset, std::size_t block) override {
    T One = T(1.0);
    T Zero = T(0.0);

    std::size_t dim;
    if (next_ == NextOp::bAc) {

      dim = n_ * block;

      dla_->apply(alpha, beta, offset, block);

      MPI_Allreduce(MPI_IN_PLACE, B_ + offset * n_, dim, getMPI_Type<T>(),
                    MPI_SUM, col_comm_);

      next_ = NextOp::cAb;
    } else {  // cAb

      dim = m_ * block;

      dla_->apply(alpha, beta, offset, block);

      MPI_Allreduce(MPI_IN_PLACE, C_ + offset * m_, dim, getMPI_Type<T>(),
                    MPI_SUM, row_comm_);

      next_ = NextOp::bAc;
    }
  }

  /*!
     - For ChaseMpiDLA,  `postApplication` operation brocasts asynchronously the final product of `HEMM` to each MPI rank. 
     - **Parallelism on distributed-memory system SUPPORT**
     - For the meaning of this function, please visit ChaseMpiDLAInterface.  
  */
  bool postApplication(T* V, std::size_t block) override {
    dla_->postApplication(V, block);

    std::size_t N = N_;
    std::size_t dimsIdx;
    std::size_t subsize;
    std::size_t blocksize;
    std::size_t nbblocks;

    T* buff;
    MPI_Comm comm;
    std::size_t *offs, *lens, *offs_l;  

    T* targetBuf = V + locked_ * N;

    if (next_ == NextOp::bAc) {
      subsize = m_;
      blocksize = mb_;
      buff = C_;
      comm = col_comm_;
      dimsIdx = 0;
      offs = r_offs_;
      lens = r_lens_;
      offs_l = r_offs_l_;
      nbblocks = mblocks_;
    } else {
      subsize = n_;
      blocksize = nb_;
      buff = B_;
      comm = row_comm_;
      dimsIdx = 1;
      offs = c_offs_;
      lens = c_lens_;
      offs_l = c_offs_l_; 
      nbblocks = nblocks_;      
    }

    int gsize, rank;
    MPI_Comm_size(comm, &gsize);
    MPI_Comm_rank(comm, &rank);

    auto& blockcounts = matrix_properties_->get_blockcounts()[dimsIdx];
    auto& blocklens = matrix_properties_->get_blocklens()[dimsIdx];
    auto& blockdispls = matrix_properties_->get_blockdispls()[dimsIdx];
    auto& sendlens = matrix_properties_->get_sendlens()[dimsIdx];
    auto& g_offset = matrix_properties_->get_g_offsets()[dimsIdx];

    std::vector<std::vector<int>> block_cyclic_displs;
    block_cyclic_displs.resize(gsize);

    int displs_cnt = 0;
    for (auto j = 0; j < gsize; ++j) {
	    block_cyclic_displs[j].resize(blockcounts[j]);
    	for (auto i = 0; i < blockcounts[j]; ++i){
	        block_cyclic_displs[j][i] = displs_cnt;
            displs_cnt += blocklens[j][i];
	    }
    }

    std::vector<MPI_Request> reqs(gsize);
    std::vector<MPI_Datatype> newType(gsize);

    for (auto j = 0; j < gsize; ++j) {
      int array_of_sizes[2] = {static_cast<int>(N_), 1};
      int array_of_subsizes[2] = {static_cast<int>(sendlens[j]), 1};
      int array_of_starts[2] = {block_cyclic_displs[j][0], 0};

      MPI_Type_create_subarray(2, array_of_sizes, array_of_subsizes,
                               array_of_starts, MPI_ORDER_FORTRAN,
                               getMPI_Type<T>(), &(newType[j]));

      MPI_Type_commit(&(newType[j]));
    }

    if(data_layout.compare("Block-Cyclic") == 0){
      for(auto i = 0; i < gsize; i++){
        if (rank == i) {
          MPI_Ibcast(buff, sendlens[i] * block, getMPI_Type<T>(), i, comm, &reqs[i]);
        } else {
	  MPI_Ibcast(Buff_, block, newType[i], i, comm, &reqs[i]);
	}
      }
    }else{
      for(auto i = 0; i < gsize; i++){
        if (rank == i) {
          MPI_Ibcast(buff, sendlens[i] * block, getMPI_Type<T>(), i, comm, &reqs[i]);
        } else {
          MPI_Ibcast(targetBuf, block, newType[i], i, comm, &reqs[i]);
        }
      }    
    }

    int i = rank;

    if(data_layout.compare("Block-Cyclic") == 0){
    	for (auto j = 0; j < block; ++j) {
      	    std::memcpy(Buff_ + j * N_ + block_cyclic_displs[i][0], buff + sendlens[i] * j, sendlens[i] * sizeof(T));
    	}
    }else{
        for (auto j = 0; j < block; ++j) {
            std::memcpy(targetBuf + j * N_ + block_cyclic_displs[i][0], buff + sendlens[i] * j, sendlens[i] * sizeof(T));
        }    
    }

    MPI_Waitall(gsize, reqs.data(), MPI_STATUSES_IGNORE);

    if(data_layout.compare("Block-Cyclic") == 0){

    	for(auto j = 0; j < gsize; j++){
	      for (auto i = 0; i < blockcounts[j]; ++i){
	        t_lacpy('A', blocklens[j][i], block, Buff_ + block_cyclic_displs[j][i], 
			             N_, targetBuf + blockdispls[j][i] , N_);
	     }
    	}

    }

    for (auto j = 0; j < gsize; j++) {
        MPI_Type_free(&newType[j]);
    }

    return true;
  }

  /*!
    - For ChaseMpiDLA,  `shiftMatrix` is implemented in nested loop for both `Block Distribution` and `Block-Cyclic Distribution`.
    - **Parallelism on distributed-memory system SUPPORT**
    - For the meaning of this function, please visit ChaseMpiDLAInterface.    
  */
  void shiftMatrix(T c, bool isunshift = false) override {
    dla_->shiftMatrix(c, isunshift);
  }

  /*!
    - For ChaseMpiDLA,  `applyVec` is implemented by `preApplication`, `apply` and `postApplication` implemented in this class.
    - **Parallelism on distributed-memory system SUPPORT**
    - **Parallelism within node for ChaseMpiDLABlaslapack if multi-threading is enabled**          
    - **Parallelism within node among multi-GPUs for ChaseMpiDLAMultiGPU**
    - **Parallelism within each GPU for ChaseMpiDLAMultiGPU**    
    - For the meaning of this function, please visit ChaseMpiDLAInterface.    
  */
  void applyVec(T* B, T* C) override {
    // TODO

    T One = T(1.0);
    T Zero = T(0.0);

    this->preApplication(B, 0, 1);
    this->apply(One, Zero, 0, 1);
    this->postApplication(C, 1);

    // gemm_->applyVec(B, C);
  }

  void get_off(std::size_t* xoff, std::size_t* yoff, std::size_t* xlen,
               std::size_t* ylen) const override {
    *xoff = off_[0];
    *yoff = off_[1];
    *xlen = m_;
    *ylen = n_;
  }

  T* get_H() const override { return matrix_properties_->get_H(); }
  std::size_t get_mblocks() const override {return mblocks_;}
  std::size_t get_nblocks() const override {return nblocks_;}
  std::size_t get_n() const override {return n_;}
  std::size_t get_m() const override {return m_;}  
  int *get_coord() const override {return matrix_properties_->get_coord();}
  void get_offs_lens(std::size_t* &r_offs, std::size_t* &r_lens, std::size_t* &r_offs_l,
                  std::size_t* &c_offs, std::size_t* &c_lens, std::size_t* &c_offs_l) const override{
     matrix_properties_->get_offs_lens(r_offs, r_lens, r_offs_l, c_offs, c_lens, c_offs_l); 
  }
  int get_nprocs() const override {return matrix_properties_->get_nprocs();}
  void Start() override { dla_->Start(); }

  /*!
    - For ChaseMpiDLA, `lange` is implemented by calling the one in ChaseMpiDLABlaslapack and ChaseMpiDLAMultiGPU.
    - This implementation is the same for both with or w/o GPUs.
    - **Parallelism is SUPPORT within node if multi-threading is actived**
    - For the meaning of this function, please visit ChaseMpiDLAInterface.
  */
  Base<T> lange(char norm, std::size_t m, std::size_t n, T* A, std::size_t lda) override {
      return dla_->lange(norm, m, n, A, lda);
  }

  /*!
    - For ChaseMpiDLA, `gegqr` is implemented by calling the one in ChaseMpiDLABlaslapack and ChaseMpiDLAMultiGPU.
    - This implementation is the same for both with or w/o GPUs.
    - **Parallelism is SUPPORT within node if multi-threading is actived**
    - For the meaning of this function, please visit ChaseMpiDLAInterface.
  */
  void gegqr(std::size_t N, std::size_t nevex, T * approxV, std::size_t LDA) override {
      dla_->gegqr(N, nevex, approxV, LDA);
  }

  /*!
    - For ChaseMpiDLA, `axpy` is implemented by calling the one in ChaseMpiDLABlaslapack and ChaseMpiDLAMultiGPU.
    - This implementation is the same for both with or w/o GPUs.
    - **Parallelism is SUPPORT within node if multi-threading is actived**
    - For the meaning of this function, please visit ChaseMpiDLAInterface.
  */
  void axpy(std::size_t N, T * alpha, T * x, std::size_t incx, T *y, std::size_t incy) override {
      t_axpy(N, alpha, x, incx, y, incy);
      dla_->axpy(N, alpha, x, incx, y, incy);
  }

  /*!
    - For ChaseMpiDLA, `scal` is implemented by calling the one in ChaseMpiDLABlaslapack and ChaseMpiDLAMultiGPU.
    - This implementation is the same for both with or w/o GPUs.
    - **Parallelism is SUPPORT within node if multi-threading is actived**
    - For the meaning of this function, please visit ChaseMpiDLAInterface.
  */
  void scal(std::size_t N, T *a, T *x, std::size_t incx) override {
      t_scal(N, a, x, incx);
      dla_->scal(N, a, x, incx);
  }

  /*!
    - For ChaseMpiDLA, `nrm2` is implemented by calling the one in ChaseMpiDLABlaslapack and ChaseMpiDLAMultiGPU.
    - This implementation is the same for both with or w/o GPUs.
    - **Parallelism is SUPPORT within node if multi-threading is actived**
    - For the meaning of this function, please visit ChaseMpiDLAInterface.
  */
  Base<T> nrm2(std::size_t n, T *x, std::size_t incx) override {
      return dla_->nrm2(n, x, incx);
  }

  /*!
    - For ChaseMpiDLA, `dot` is implemented by calling the one in ChaseMpiDLABlaslapack and ChaseMpiDLAMultiGPU.
    - This implementation is the same for both with or w/o GPUs.
    - **Parallelism is SUPPORT within node if multi-threading is actived**
    - For the meaning of this function, please visit ChaseMpiDLAInterface.
  */
  T dot(std::size_t n, T* x, std::size_t incx, T* y, std::size_t incy) override {
      return dla_->dot(n, x, incx, y, incy);
  }
 
   /*!
   - For ChaseMpiDLA, `gemm_small` is implemented with `xGEMM` provided by `BLAS`.
   - This implementation is the same for both with or w/o GPUs.
   - **Parallelism is SUPPORT within node if multi-threading is actived**    
   - For the meaning of this function, please visit ChaseMpiDLAInterface.
  */
  void gemm_small(CBLAS_LAYOUT Layout, CBLAS_TRANSPOSE transa,
                         CBLAS_TRANSPOSE transb, std::size_t m,
                         std::size_t n, std::size_t k, T* alpha,
                         T* a, std::size_t lda, T* b,
                         std::size_t ldb, T* beta, T* c, std::size_t ldc) override
  {
      t_gemm(Layout, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
      dla_->gemm_small(Layout, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  }

   /*!
   - For ChaseMpiDLA, `gemm_large` is implemented with `xGEMM` provided by `BLAS`.
   - This implementation is the same for both with or w/o GPUs.   
   - **Parallelism is SUPPORT within node if multi-threading is actived**    
   - For the meaning of this function, please visit ChaseMpiDLAInterface.
  */
  void gemm_large(CBLAS_LAYOUT Layout, CBLAS_TRANSPOSE transa,
                         CBLAS_TRANSPOSE transb, std::size_t m,
                         std::size_t n, std::size_t k, T* alpha,
                         T* a, std::size_t lda, T* b,
                         std::size_t ldb, T* beta, T* c, std::size_t ldc) override
  {
      t_gemm(Layout, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
      dla_->gemm_large(Layout, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  }

  /*!
    - For ChaseMpiDLA, `stemr` with real and double precision scalar, is implemented by calling the one in ChaseMpiDLABlaslapack and ChaseMpiDLAMultiGPU.
    - This implementation is the same for both with or w/o GPUs.
    - **Parallelism is SUPPORT within node if multi-threading is actived**
    - For the meaning of this function, please visit ChaseMpiDLAInterface.
  */
  std::size_t stemr(int matrix_layout, char jobz, char range, std::size_t n,
                    double* d, double* e, double vl, double vu, std::size_t il, std::size_t iu,
                    int* m, double* w, double* z, std::size_t ldz, std::size_t nzc,
                    int* isuppz, lapack_logical* tryrac) override {
      return dla_->stemr(matrix_layout, jobz, range, n, d, e, vl, vu, il, iu, m, w, z, ldz, nzc, isuppz, tryrac);
  }

  /*!
    - For ChaseMpiDLA, `stemr` with real and single precision scalar, is implemented by calling the one in ChaseMpiDLABlaslapack and ChaseMpiDLAMultiGPU.
    - This implementation is the same for both with or w/o GPUs.
    - **Parallelism is SUPPORT within node if multi-threading is actived**
    - For the meaning of this function, please visit ChaseMpiDLAInterface.
  */
  std::size_t stemr(int matrix_layout, char jobz, char range, std::size_t n,
                    float* d, float* e, float vl, float vu, std::size_t il, std::size_t iu,
                    int* m, float* w, float* z, std::size_t ldz, std::size_t nzc,
                    int* isuppz, lapack_logical* tryrac) override {
      return dla_->stemr(matrix_layout, jobz, range, n, d, e, vl, vu, il, iu, m, w, z, ldz, nzc, isuppz, tryrac);
  }

  /*!
    - For ChaseMpiDLA, `RR_kernel` with real and double precision scalar, is implemented by calling the one in ChaseMpiDLABlaslapack and ChaseMpiDLAMultiGPU.
    - This implementation is the same for both with or w/o GPUs.
    - **Parallelism is SUPPORT within node if multi-threading is actived**
    - For the meaning of this function, please visit ChaseMpiDLAInterface.
  */
  void RR_kernel(std::size_t N, std::size_t block, T *approxV, std::size_t locked, T *workspace, T One, T Zero, Base<T> *ritzv) override {
      dla_->RR_kernel(N, block, approxV, locked, workspace, One, Zero, ritzv);	
  }
  
 private:
  enum NextOp { cAb, bAc };

  std::size_t locked_;
  std::size_t ldc_;
  std::size_t ldb_;

  std::size_t n_;
  std::size_t m_;
  std::size_t N_;

  T* B_;
  T* C_;
  T *Buff_;

  NextOp next_;
  MPI_Comm row_comm_, col_comm_;
  int* dims_;
  int* coord_;
  std::size_t* off_;

  std::size_t *r_offs_;
  std::size_t *r_lens_;
  std::size_t *r_offs_l_;
  std::size_t *c_offs_;
  std::size_t *c_lens_;
  std::size_t *c_offs_l_;
  std::size_t nb_;
  std::size_t mb_;
  std::size_t nblocks_;
  std::size_t mblocks_;

  std::string data_layout;
  std::unique_ptr<ChaseMpiDLAInterface<T>> dla_;
  ChaseMpiProperties<T>* matrix_properties_;
};
}  // namespace mpi
}  // namespace chase

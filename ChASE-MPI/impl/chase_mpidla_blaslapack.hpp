/* -*- Mode: C++; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
// This file is a part of ChASE.
// Copyright (c) 2015-2021, Simulation and Data Laboratory Quantum Materials, 
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "ChASE-MPI/blas_templates.hpp"
#include "ChASE-MPI/chase_mpi_properties.hpp"
#include "ChASE-MPI/chase_mpidla_interface.hpp"

namespace chase {
namespace mpi {
//
//  This Class is meant to be used with MatrixFreeMPI
//
//! A derived class of ChaseMpiDLAInterface which implements the inter-node computation for a pure-CPU MPI-based implementation of ChASE. 
template <class T>
class ChaseMpiDLABlaslapack : public ChaseMpiDLAInterface<T> {
 public:
  //! A constructor of ChaseMpiDLABlaslapack.
  //! @param matrix_properties: it is an object of ChaseMpiProperties, which defines the MPI environment and data distribution scheme in ChASE-MPI.
  ChaseMpiDLABlaslapack(ChaseMpiProperties<T>* matrix_properties, ChaseMpiMatrices<T>& matrices) {
    // TODO
    // ldc_ = matrix_properties->get_ldc();
    // ldb_ = matrix_properties->get_ldb();

    n_ = matrix_properties->get_n();
    m_ = matrix_properties->get_m();
    N_ = matrix_properties->get_N();
    nev_ = matrix_properties->GetNev();
    nex_ = matrix_properties->GetNex();
    H_ = matrices.get_H();
    ldh_ = matrices.get_ldh();
    if(H_ == nullptr){
      H_ = matrix_properties->get_H();
      ldh_ = matrix_properties->get_ldh();
    }

    B_ = matrices.get_V2();
    C_ = matrices.get_V1();
    C2_ = matrix_properties->get_C2();

    off_ = matrix_properties->get_off();

    matrix_properties->get_offs_lens(r_offs_, r_lens_, r_offs_l_, c_offs_, c_lens_, c_offs_l_);
    mb_ = matrix_properties->get_mb();
    nb_ = matrix_properties->get_nb();

    mblocks_ = matrix_properties->get_mblocks();
    nblocks_ = matrix_properties->get_nblocks();

    matrix_properties_ = matrix_properties;

	MPI_Comm row_comm = matrix_properties_->get_row_comm();
	MPI_Comm col_comm = matrix_properties_->get_col_comm();

	MPI_Comm_rank(row_comm, &mpi_row_rank);
	MPI_Comm_rank(col_comm, &mpi_col_rank);

  }

  ~ChaseMpiDLABlaslapack() {}
  void initVecs() override{}  
  void initRndVecs() override {}

  /*! - For ChaseMpiDLABlaslapack, `preApplication` is implemented within ChaseMpiDLA.
      - **Parallelism on distributed-memory system SUPPORT**
      - For the meaning of this function, please visit ChaseMpiDLAInterface.
  */
  void preApplication(T* V, std::size_t locked, std::size_t block) override {
    next_ = NextOp::bAc;
    // std::memcpy(C_, V + locked_ * N_, N_ * block * sizeof(T));
  }

  /*! - For ChaseMpiDLABlaslapack, `preApplication` is implemented within ChaseMpiDLA.
      - **Parallelism on distributed-memory system SUPPORT**
      - For the meaning of this function, please visit ChaseMpiDLAInterface.
  */
  void preApplication(T* V1, T* V2, std::size_t locked, std::size_t block) override {
    // std::memcpy(B_, V2 + locked * N_, N_ * block * sizeof(T));
    this->preApplication(V1, locked, block);
  }

   /*!
      - For ChaseMpiDLABlaslapack, the matrix-matrix multiplication of local matrices are implemented in with `GEMM` routine provided by `BLAS`.
      - The collective communication based on MPI which **ALLREDUCE** the product of local matrices either within the column communicator or row communicator, is implemented within ChaseMpiDLA.
      - **Parallelism on distributed-memory system SUPPORT**
      - **Parallelism is SUPPORT within node if multi-threading is actived**        
      - For the meaning of this function, please visit ChaseMpiDLAInterface.
  */
  void apply(T alpha, T beta, std::size_t offset, std::size_t block,  std::size_t locked) override {

	  T Zero = T(0.0);

    if (next_ == NextOp::bAc) {

      if (mpi_col_rank != 0) {
         beta = Zero;
      }
      t_gemm<T>(CblasColMajor, CblasConjTrans, CblasNoTrans, n_,
                static_cast<std::size_t>(block), m_, &alpha, H_, ldh_,
                C_ + offset * m_ + locked * m_, m_, &beta, B_ + locked * n_ + offset * n_, n_);
      next_ = NextOp::cAb;
    } else {
 
     if (mpi_row_rank != 0) {
         beta = Zero;
      }
      t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m_,
             static_cast<std::size_t>(block), n_, &alpha, H_, ldh_,
             B_ + offset * n_ + locked * n_, n_, &beta, C_ + offset * m_ + locked * m_, m_);
      next_ = NextOp::bAc;
    }
  }

  // deg is always even so we know that we return C?
  /*!
     - For ChaseMpiDLABlaslapack,  `postApplication` is implemented in ChaseMpiDLA, with asynchronously brocasting the final product of `HEMM` to each MPI rank. 
     - **Parallelism on distributed-memory system SUPPORT**
     - For the meaning of this function, please visit ChaseMpiDLAInterface.  
  */
  bool postApplication(T* V, std::size_t block, std::size_t locked) override {
    return false;
  }

  /*!
    - For ChaseMpiDLABlaslapack,  `shiftMatrix` is implemented in ChaseMpiDLA.
    - **Parallelism on distributed-memory system SUPPORT**
    - For the meaning of this function, please visit ChaseMpiDLAInterface.    
  */
  void shiftMatrix(T c, bool isunshift = false) override {

    for(std::size_t j = 0; j < nblocks_; j++){
        for(std::size_t i = 0; i < mblocks_; i++){
            for(std::size_t q = 0; q < c_lens_[j]; q++){
                for(std::size_t p = 0; p < r_lens_[i]; p++){
                    if(q + c_offs_[j] == p + r_offs_[i]){
                        H_[(q + c_offs_l_[j]) * ldh_ + p + r_offs_l_[i]] += c;
                    }
                }
            }
        }
    }

  }

  void asynCxHGatherC(std::size_t locked, std::size_t block) override {
    T alpha = T(1.0);
    T beta = T(0.0); 

    t_gemm<T>(CblasColMajor, CblasConjTrans, CblasNoTrans, n_,
                static_cast<std::size_t>(block), m_, &alpha, H_, ldh_,
                C_ + locked * m_, m_, &beta, B_ + locked * n_, n_);
  }


  /*!
    - For ChaseMpiDLABlaslapack,  `applyVec` is implemented in ChaseMpiDLA.
    - **Parallelism on distributed-memory system SUPPORT**
    - For the meaning of this function, please visit ChaseMpiDLAInterface.    
  */
  void applyVec(T* B, T* C) override {
    T alpha = T(1.0);
    T beta = T(0.0);

    t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans,  //
           N_, 1, N_,                                  //
           &alpha,                                     //
           H_, N_,                                     //
           B, N_,                                      //
           &beta,                                      //
           C, N_);                                     //
  }

  void get_off(std::size_t* xoff, std::size_t* yoff, std::size_t* xlen,
               std::size_t* ylen) const override {
    *xoff = 0;
    *yoff = 0;
    *xlen = static_cast<std::size_t>(N_);
    *ylen = static_cast<std::size_t>(N_);
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
  int get_nprocs() const override {return matrix_properties_->get_nprocs();}
  void Start() override {}

  /*!
    - For ChaseMpiDLABlaslapack, `lange` is implemented using `LAPACK` routine `xLANGE`.
    - **Parallelism is SUPPORT within node if multi-threading is enabled.**
    - For the meaning of this function, please visit ChaseMpiDLAInterface.
  */
  Base<T> lange(char norm, std::size_t m, std::size_t n, T* A, std::size_t lda) override {
      return t_lange(norm, m, n, A, lda);
  }

  /*!
    - For ChaseMpiDLABlaslapack, `axpy` is implemented in ChaseMpiDLA.
   - **Parallelism is SUPPORT within node if multi-threading is enabled.**    
    - For the meaning of this function, please visit ChaseMpiDLAInterface.
  */
  void axpy(std::size_t N, T * alpha, T * x, std::size_t incx, T *y, std::size_t incy) override { 
      t_axpy(N, alpha, x, incx, y, incy);
  }

  /*!
    - For ChaseMpiDLABlaslapack, `scal` is implemented in ChaseMpiDLA
    - **Parallelism is SUPPORT within node if multi-threading is enabled.**   
    - For the meaning of this function, please visit ChaseMpiDLAInterface.
  */
  void scal(std::size_t N, T *a, T *x, std::size_t incx) override { 
    t_scal(N, a, x, incx);
}

  /*!
    - For ChaseMpiDLABlaslapack, `nrm2` is implemented using `BLAS` routine `xNRM2`.
    - **Parallelism is SUPPORT within node if multi-threading is enabled.**    
    - For the meaning of this function, please visit ChaseMpiDLAInterface.
  */
  Base<T> nrm2(std::size_t n, T *x, std::size_t incx) override {
      return t_nrm2(n, x, incx);
  }

  /*!
    - For ChaseMpiDLABlaslapack, `dot` is implemented using `BLAS` routine `xDOT`.
    - **Parallelism is SUPPORT within node if multi-threading is enabled.**       
    - For the meaning of this function, please visit ChaseMpiDLAInterface.
  */
  T dot(std::size_t n, T* x, std::size_t incx, T* y, std::size_t incy) override {
      return t_dot(n, x, incx, y, incy);
  }
  /*!
   - For ChaseMpiDLABlaslapack, `gemm` is implemented in ChaseMpiDLA.
   - **Parallelism is SUPPORT within node if multi-threading is enabled.**    
   - For the meaning of this function, please visit ChaseMpiDLAInterface.
  */
  void gemm(CBLAS_LAYOUT Layout, CBLAS_TRANSPOSE transa,
                         CBLAS_TRANSPOSE transb, std::size_t m,
                         std::size_t n, std::size_t k, T* alpha,
                         T* a, std::size_t lda, T* b,
                         std::size_t ldb, T* beta, T* c, std::size_t ldc) override
  {
      t_gemm(Layout, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);

  }

  /*!
   - For ChaseMpiDLABlaslapack, `stemr` with scalar being real and double precision, is implemented using `LAPACK` routine `DSTEMR`.
   - **Parallelism is SUPPORT within node if multi-threading is enabled.**    
   - For the meaning of this function, please visit ChaseMpiDLAInterface.
  */
  std::size_t stemr(int matrix_layout, char jobz, char range, std::size_t n,
                    double* d, double* e, double vl, double vu, std::size_t il, std::size_t iu,
                    int* m, double* w, double* z, std::size_t ldz, std::size_t nzc,
                    int* isuppz, lapack_logical* tryrac) override {
      return t_stemr<double>(matrix_layout, jobz, range, n, d, e, vl, vu, il, iu, m, w, z, ldz, nzc, isuppz, tryrac);
  }

  /*!
   - For ChaseMpiDLABlaslapack, `stemr` with scalar being real and single precision, is implemented using `LAPACK` routine `SSTEMR`.
   - **Parallelism is SUPPORT within node if multi-threading is enabled.**    
   - For the meaning of this function, please visit ChaseMpiDLAInterface.
  */
  std::size_t stemr(int matrix_layout, char jobz, char range, std::size_t n,
                    float* d, float* e, float vl, float vu, std::size_t il, std::size_t iu,
                    int* m, float* w, float* z, std::size_t ldz, std::size_t nzc,
                    int* isuppz, lapack_logical* tryrac) override {
      return t_stemr<float>(matrix_layout, jobz, range, n, d, e, vl, vu, il, iu, m, w, z, ldz, nzc, isuppz, tryrac);
  }

  /*!
      - For ChaseMpiDLABlaslapack, `RR` is implemented by `GEMM` routine provided by `BLAS` and `(SY)HEEVD` routine provided by `LAPACK`.
        - The 1st operation `A <- W^T * V` is implemented by `GEMM` from `BLAS`.
        - The 2nd operation which computes the eigenpairs of `A`, is implemented by `(SY)HEEVD` from `LAPACK`.
        - The 3rd operation which computes `W<-V*A` is implemented by `GEMM` from `BLAS`.
      - **Parallelism is SUPPORT within node if multi-threading is enabled.**    
      - For the meaning of this function, please visit ChaseMpiDLAInterface.
  */  
  void RR(std::size_t block, std::size_t locked, Base<T> *ritzv) override {}

  void V2C(T *v1, std::size_t off1, T *v2, std::size_t off2, std::size_t block) override {}

  void C2V(T *v1, std::size_t off1, T *v2, std::size_t off2, std::size_t block) override {
  }
  void syherk(char uplo, char trans, std::size_t n, std::size_t k, T* alpha, T* a, std::size_t lda, T* beta, T* c, std::size_t ldc)  override  {
      t_syherk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
  }

  int potrf(char uplo, std::size_t n, T* a, std::size_t lda) override{
      return t_potrf(uplo, n, a, lda);
  }

  void trsm(char side, char uplo, char trans, char diag,
                      std::size_t m, std::size_t n, T* alpha,
                      T* a, std::size_t lda, T* b, std::size_t ldb) override{
      t_trsm(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb);
  }

  void Resd(Base<T> *ritzv, Base<T> *resid, std::size_t locked, std::size_t unconverged) override{

  }

  void heevd(int matrix_layout, char jobz, char uplo, std::size_t n,
                    T* a, std::size_t lda, Base<T>* w) override {

      t_heevd(matrix_layout, jobz,uplo, n, a, lda, w);
  }



  void hhQR(std::size_t locked) override {
  }
  
  void cholQR(std::size_t locked) override{
  }

  void Swap(std::size_t i, std::size_t j)override{}

  void getLanczosBuffer(T **V1, T **V2, std::size_t *ld) override{}

 private:
  enum NextOp { cAb, bAc };

  NextOp next_;
  std::size_t N_;

  std::size_t n_;
  std::size_t m_;
  std::size_t ldh_;
  T* H_;
  T* B_;
  T* C_;
  T* C2_;

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
  std::size_t nev_;
  std::size_t nex_;
  int mpi_row_rank;
  int mpi_col_rank;

  ChaseMpiProperties<T>* matrix_properties_;
};

template <typename T>
struct is_skewed_matrixfree<ChaseMpiDLABlaslapack<T>> {
  static const bool value = true;
};

}  // namespace mpi
}  // namespace chase

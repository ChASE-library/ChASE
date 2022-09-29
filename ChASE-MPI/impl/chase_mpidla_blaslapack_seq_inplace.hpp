/* -*- Mode: C++; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
// This file is a part of ChASE.
// Copyright (c) 2015-2021, Simulation and Data Laboratory Quantum Materials, 
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include <cstring>
#include <memory>

#include "ChASE-MPI/chase_mpidla_interface.hpp"

namespace chase {
namespace mpi {

//! A derived class of ChaseMpiDLAInterface which implements ChASE targeting shared-memory architectures with only CPUs available. 
/*! It implements in a inplace mode, in which the buffer of `V1` and `V2` are swapped and reused, which reduces the required memory to be allocted.
*/
template <class T>
class ChaseMpiDLABlaslapackSeqInplace : public ChaseMpiDLAInterface<T> {
 public:
 //! A constructor of ChaseMpiDLABlaslapackSeqInplace.
  /*! @param matrices: it is an object of ChaseMpiMatrices, which allocates the required buffer.
      @param n: size of matrix defining the eigenproblem.
      @param maxBlock: maximum column number of matrix `V`, which equals to `nev+nex`.
  */  
  ChaseMpiDLABlaslapackSeqInplace(ChaseMpiMatrices<T>& matrices, std::size_t n,
                             std::size_t maxBlock)
      : N_(n),
        V1_(matrices.get_V1()),
        V2_(matrices.get_V2()),
        H_(matrices.get_H()) {}

  ~ChaseMpiDLABlaslapackSeqInplace() {}

  /*! - For ChaseMpiDLABlaslapackSeqInplace, the core of `preApplication` is implemented with `std::swap`, which swaps the buffer of `V1` and `V2`.
      - **Parallelism is NOT SUPPORT**
      - For the meaning of this function, please visit ChaseMpiDLAInterface.
  */
  void preApplication(T* V, std::size_t locked, std::size_t block) override {
    locked_ = locked;

    if (V != V1_) std::swap(V1_, V2_);
    assert(V == V1_);
  }

  /*! - For ChaseMpiDLABlaslapackSeqInplace, the core of `preApplication` is implemented with `std::swap`, which swaps the buffer of `V1` and `V2`.
      - **Parallelism is NOT SUPPORT**
      - For the meaning of this function, please visit ChaseMpiDLAInterface.
  */
  void preApplication(T* V1, T* V2, std::size_t locked, std::size_t block) override {
    this->preApplication(V1, locked, block);
  }

  /*! - For ChaseMpiDLABlaslapackSeqInplace, `apply` is implemented with `GEMM` provided by `BLAS`.
      - **Parallelism is SUPPORT within node if multi-threading is actived**
      - For the meaning of this function, please visit ChaseMpiDLAInterface.
  */
  void apply(T alpha, T beta, std::size_t offset, std::size_t block,  std::size_t locked) override {

    assert(V2_ != V1_);
    t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans,  //
           N_, block, N_,                              //
           &alpha,                                     // V2_ <-
           H_, N_,                                     //   H * V1_
           V1_ + (locked_ + offset) * N_, N_,          //   + V2_
           &beta,                                      //
           V2_ + (locked_ + offset) * N_, N_);         //

    std::swap(V1_, V2_);
  }

  /*! - For ChaseMpiDLABlaslapackSeqInplace, `postApplication` doesn't require explicit implementation here.
      - For the meaning of this function, please visit ChaseMpiDLAInterface.
  */
  bool postApplication(T* V, std::size_t block, std::size_t locked) override {
    // this is somewhat a hack, but causes the approxV in the next
    // preApplication to be the same pointer content as V1_
    // std::swap(V1_, V2_);

    assert(V == V1_);

    return false;
  }

  /*! - For ChaseMpiDLABlaslapackSeqInplace, `shiftMatrix` is implemented by a loop of length `N_`.
      - **Parallelism is NOT SUPPORT**
      - For the meaning of this function, please visit ChaseMpiDLAInterface.
  */
  void shiftMatrix(T const c,bool isunshift = false) override {
    for (std::size_t i = 0; i < N_; ++i) {
      H_[i + i * N_] += c;
    }
  }

  void asynCxHGatherC(T *V, std::size_t locked, std::size_t block) override {}

  /*! - For ChaseMpiDLABlaslapackSeqInplace, `applyVec` is implemented with `GEMM` provided by `BLAS`.
      - **Parallelism is SUPPORT within node if multi-threading is actived**
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

  T* get_H() const override { return H_; }

  void get_off(std::size_t* xoff, std::size_t* yoff, std::size_t* xlen,
               std::size_t* ylen) const override {
    *xoff = 0;
    *yoff = 0;
    *xlen = N_;
    *ylen = N_;
  }

  std::size_t get_mblocks() const override {return 1;}
  std::size_t get_nblocks() const override {return 1;}
  std::size_t get_n() const override {return N_;}
  std::size_t get_m() const override {return N_;}
  int *get_coord() const override {
          int *coord = new int[2];
          coord[0] = 0;
          coord[1] = 0;

          return coord;
  }
  void get_offs_lens(std::size_t* &r_offs, std::size_t* &r_lens, std::size_t* &r_offs_l,
                  std::size_t* &c_offs, std::size_t* &c_lens, std::size_t* &c_offs_l) const override{

          std::size_t r_offs_[1] = {0};
          std::size_t r_lens_[1]; r_lens_[0] = N_;
          std::size_t r_offs_l_[1] = {0};
          std::size_t c_offs_[1] = {0};
          std::size_t c_lens_[1]; r_lens_[0] = N_;
          std::size_t c_offs_l_[1] = {0};

          r_offs = r_offs_;
          r_lens = r_lens_;
          r_offs_l = r_offs_l_;
          c_offs = c_offs_;
          c_lens = c_lens_;
          c_offs_l = c_offs_l_;
  }
  int get_nprocs() const override {return 1;}
  void Start() override {}

  /*!
    - For ChaseMpiDLABlaslapackSeqInplace, `lange` is implemented using `LAPACK` routine `xLANGE`.
    - **Parallelism is SUPPORT within node if multi-threading is actived**
    - For the meaning of this function, please visit ChaseMpiDLAInterface.
  */
  Base<T> lange(char norm, std::size_t m, std::size_t n, T* A, std::size_t lda) override {
      return t_lange(norm, m, n, A, lda);
  }  

  /*!
    - For ChaseMpiDLABlaslapackSeqInplace, `gegqr` is implemented using `LAPACK` routine `xGEQRF` and `xUMGQR`.
    - **Parallelism is SUPPORT within node if multi-threading is actived**    
    - For the meaning of this function, please visit ChaseMpiDLAInterface.
  */
  void gegqr(std::size_t N, std::size_t nevex, T * approxV, std::size_t LDA) override {
      
      this->postApplication(approxV, nevex - locked_);

      auto tau = std::unique_ptr<T[]> {
          new T[ nevex ]
      };
      t_geqrf(LAPACK_COL_MAJOR, N, nevex, approxV, LDA, tau.get());
      t_gqr(LAPACK_COL_MAJOR, N, nevex, nevex, approxV, LDA, tau.get());
  }

  /*!
    - For ChaseMpiDLABlaslapackSeqInplace, `axpy` is implemented using `BLAS` routine `xAXPY`.
   - **Parallelism is SUPPORT within node if multi-threading is actived**    
    - For the meaning of this function, please visit ChaseMpiDLAInterface.
  */
  void axpy(std::size_t N, T * alpha, T * x, std::size_t incx, T *y, std::size_t incy) override {
      t_axpy(N, alpha, x, incx, y, incy);
  }

  /*!
    - For ChaseMpiDLABlaslapackSeqInplace, `scal` is implemented using `BLAS` routine `xSCAL`.
    - **Parallelism is SUPPORT within node if multi-threading is actived**   
    - For the meaning of this function, please visit ChaseMpiDLAInterface.
  */
  void scal(std::size_t N, T *a, T *x, std::size_t incx) override {
      t_scal(N, a, x, incx);
  }

  /*!
    - For ChaseMpiDLABlaslapackSeqInplace, `nrm2` is implemented using `BLAS` routine `xNRM2`.
    - **Parallelism is SUPPORT within node if multi-threading is actived**    
    - For the meaning of this function, please visit ChaseMpiDLAInterface.
  */
  Base<T> nrm2(std::size_t n, T *x, std::size_t incx) override {
      return t_nrm2(n, x, incx);
  }

  /*!
    - For ChaseMpiDLABlaslapackSeqInplace, `dot` is implemented using `BLAS` routine `xDOT`.
    - **Parallelism is SUPPORT within node if multi-threading is actived**       
    - For the meaning of this function, please visit ChaseMpiDLAInterface.
  */
  T dot(std::size_t n, T* x, std::size_t incx, T* y, std::size_t incy) override {
      return t_dot(n, x, incx, y, incy);
  }

  /*!
    - For ChaseMpiDLABlaslapackSeqInplace, `gemm_small` is implemented using `BLAS` routine `xGEMM`.
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
  }

  /*!
    - For ChaseMpiDLABlaslapackSeqInplace, `gemm_small` is implemented using `BLAS` routine `xGEMM`.
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
  }

  /*!
   - For ChaseMpiDLABlaslapackSeqInplace, `stemr` with scalar being real and double precision, is implemented using `LAPACK` routine `DSTEMR`.
   - **Parallelism is SUPPORT within node if multi-threading is actived**    
   - For the meaning of this function, please visit ChaseMpiDLAInterface.
  */
  std::size_t stemr(int matrix_layout, char jobz, char range, std::size_t n,
                    double* d, double* e, double vl, double vu, std::size_t il, std::size_t iu,
                    int* m, double* w, double* z, std::size_t ldz, std::size_t nzc,
                    int* isuppz, lapack_logical* tryrac) override {
      return t_stemr<double>(matrix_layout, jobz, range, n, d, e, vl, vu, il, iu, m, w, z, ldz, nzc, isuppz, tryrac);
  }

  /*!
   - For ChaseMpiDLABlaslapackSeqInplace, `stemr` with scalar being real and single precision, is implemented using `LAPACK` routine `SSTEMR`.
   - **Parallelism is SUPPORT within node if multi-threading is actived**    
   - For the meaning of this function, please visit ChaseMpiDLAInterface.
  */
  std::size_t stemr(int matrix_layout, char jobz, char range, std::size_t n,
                    float* d, float* e, float vl, float vu, std::size_t il, std::size_t iu,
                    int* m, float* w, float* z, std::size_t ldz, std::size_t nzc,
                    int* isuppz, lapack_logical* tryrac) override {
      return t_stemr<float>(matrix_layout, jobz, range, n, d, e, vl, vu, il, iu, m, w, z, ldz, nzc, isuppz, tryrac);
  }

  /*!
      - For ChaseMpiDLABlaslapackSeqInplace, `RR_kernel` is implemented by `GEMM` routine provided by `BLAS` and `(SY)HEEVD` routine provided by `LAPACK`.
        - The 1st operation `A <- W^T * V` is implemented by `GEMM` from `BLAS`.
        - The 2nd operation which computes the eigenpairs of `A`, is implemented by `(SY)HEEVD` from `LAPACK`.
        - The 3rd operation which computes `W<-V*A` is implemented by `GEMM` from `BLAS`.
      - **Parallelism is SUPPORT within node if multi-threading is actived**    
      - For the meaning of this function, please visit ChaseMpiDLAInterface.
  */  
  void RR_kernel(std::size_t N, std::size_t block, T *approxV, std::size_t locked, T *workspace, T One, T Zero, Base<T> *ritzv) override {
      T *A = new T[block * block];

      // A <- W' * V
      t_gemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
             block, block, N,
             &One,
             approxV + locked * N, N,
             workspace + locked * N, N,
             &Zero,
             A, block
      );

      t_heevd(LAPACK_COL_MAJOR, 'V', 'L', block, A, block, ritzv);

      t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
           N, block, block,
           &One,
           approxV + locked * N, N,
           A, block,
           &Zero,
           workspace + locked * N, N
      );

      delete[] A;
  }

  void LanczosDos(std::size_t N_, std::size_t idx, std::size_t m, T *workspace_, std::size_t ldw, T *ritzVc, std::size_t ldr, T* approxV_, std::size_t ldv) override{
    T alpha = T(1.0);
    T beta = T(0.0);

    t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
           N_, idx, m,
           &alpha,
           workspace_, ldw,
           ritzVc, ldr,
           &beta,
           approxV_, ldv
    );
  }

  void syherk(char uplo, char trans, std::size_t n, std::size_t k, T* alpha, T* a, std::size_t lda, T* beta, T* c, std::size_t ldc)  override  {
  }

  int potrf(char uplo, std::size_t n, T* a, std::size_t lda) override{
            return 0;
  }

  void trsm(char side, char uplo, char trans, char diag,
                      std::size_t m, std::size_t n, T* alpha,
                      T* a, std::size_t lda, T* b, std::size_t ldb) override{
  }

  void heevd(int matrix_layout, char jobz, char uplo, std::size_t n,
                    T* a, std::size_t lda, Base<T>* w) override {
  }

  void heevd2(std::size_t m_, std::size_t block, T* A, std::size_t lda, T *approxV, std::size_t ldv, T* workspace, std::size_t ldw, std::size_t offset, Base<T>* ritzv) override {
  }


  void Resd(T *approxV_, T* workspace_, Base<T> *ritzv, Base<T> *resid, std::size_t locked, std::size_t unconverged) override{

  }

  void hhQR(std::size_t m_, std::size_t nevex, T *approxV, std::size_t ldv) override{

  }

  void hhQR_dist(std::size_t m_, std::size_t nevex,std::size_t locked, T *approxV, std::size_t ldv) override {
 
  }
 
  void cholQR1(std::size_t m_, std::size_t nevex, T *approxV, std::size_t ldv) override{
  }
  void cholQR1_dist(std::size_t N, std::size_t nevex, std::size_t locked, T *approxV, std::size_t ldv) override{
  }
  void Lock(T * workspace_, std::size_t new_converged) override{}

  void Swap(std::size_t i, std::size_t j)override{}

 private:
  std::size_t N_;
  std::size_t locked_;

  T* H_;
  T* V1_;
  T* V2_;
};

template <typename T>
struct is_skewed_matrixfree<ChaseMpiDLABlaslapackSeqInplace<T>> {
  static const bool value = false;
};

}  // namespace mpi
}  // namespace chase

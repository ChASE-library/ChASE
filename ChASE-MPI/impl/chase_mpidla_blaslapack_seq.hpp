/* -*- Mode: C++; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
// This file is a part of ChASE.
// Copyright (c) 2015-2021, Simulation and Data Laboratory Quantum Materials, 
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include <cstring>
#include <memory>

#include "ChASE-MPI/blas_templates.hpp"
#include "ChASE-MPI/chase_mpi_matrices.hpp"

namespace chase {
namespace mpi {

// A very simple implementation of MatrixFreeInterface
// We duplicate the two vector sets from ChASE_Blas and copy
// into the duplicates before each GEMM call.

//! A derived class of ChaseMpiDLAInterface which implements ChASE targeting shared-memory architectures with only CPUs available. 
template <class T>
class ChaseMpiDLABlaslapackSeq : public ChaseMpiDLAInterface<T> {
 public:
  //! A constructor of ChaseMpiDLABlaslapackSeq.
  /*! @param matrices: it is an object of ChaseMpiMatrices, which allocates the required buffer.
      @param n: size of matrix defining the eigenproblem.
      @param maxBlock: maximum column number of matrix `V`, which equals to `nev+nex`.
  */
  explicit ChaseMpiDLABlaslapackSeq(ChaseMpiMatrices<T>& matrices, std::size_t n,
                               std::size_t nex, std::size_t nev)
      : N_(n),
        nev_(nev),
        nex_(nex),
        maxBlock_(nex+nex),
        H_(matrices.get_H()),
        V1_(new T[N_ * maxBlock_]), 
        V2_(new T[N_ * maxBlock_]), 
        A_(new T[maxBlock_ * maxBlock_]){

        V12_ = matrices.get_V1();
        V22_ = matrices.get_V2();
        }

  ChaseMpiDLABlaslapackSeq() = delete;
  ChaseMpiDLABlaslapackSeq(ChaseMpiDLABlaslapackSeq const& rhs) = delete;

  ~ChaseMpiDLABlaslapackSeq() {}
  void initVecs(T *V) override{
    t_lacpy('A', N_, nev_+nex_, V12_, N_, get_V1() , N_);
  }  
  void initRndVecs(T *V) override {
     std::mt19937 gen(1337.0);
     std::normal_distribution<> d;
     for(auto j = 0; j < (nev_ + nex_); j++){
        for(auto i = 0; i < N_; i++){
          V[i + j * N_] = getRandomT<T>([&]() { return d(gen);});    
        }           
     }
  }

  void C2V(T *v1, T *v2, std::size_t block) override {
    std::memcpy(v2, v1, N_ * block *sizeof(T));    
  }

  /*! - For ChaseMpiDLABlaslapackSeq, the core of `preApplication` is implemented with `std::memcpy`, which copies `block` vectors from `V` to `V1`.
      - **Parallelism is NOT SUPPORT**
      - For the meaning of this function, please visit ChaseMpiDLAInterface.
  */
  void preApplication(T* V, std::size_t const locked,
                      std::size_t const block) override {
    locked_ = locked;
    std::memcpy(get_V1(), V + locked * N_, N_ * block * sizeof(T));
  }

  /*! - For ChaseMpiDLABlaslapackSeq, the core of `preApplication` is implemented with `std::memcpy`, which copies `block` vectors from `V2` to `V2_`.
      - **Parallelism is NOT SUPPORT**
      - For the meaning of this function, please visit ChaseMpiDLAInterface.
  */
  void preApplication(T* V1, T* V2, std::size_t const locked,
                      std::size_t const block) override {
    std::memcpy(get_V2(), V2 + locked * N_, N_ * block * sizeof(T));
    this->preApplication(V1, locked, block);
  }

  /*! - For ChaseMpiDLABlaslapackSeq, `apply` is implemented with `GEMM` provided by `BLAS`.
      - **Parallelism is SUPPORT within node if multi-threading is actived**
      - For the meaning of this function, please visit ChaseMpiDLAInterface.
  */
  void apply(T alpha, T beta, std::size_t offset, std::size_t block,  std::size_t locked) override {

    t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans,  //
           N_, block, N_,                              //
           &alpha,                                     // V2 <-
           H_, N_,                                     //      alpha * H*V1
           get_V1() +locked * N_ + offset * N_, N_,                 //      + beta * V2
           &beta,                                      //
           get_V2() +locked * N_ + offset * N_, N_);

    std::swap(V1_, V2_);

  }
  /*! - For ChaseMpiDLABlaslapackSeq, the core of `postApplication` is implemented with `std::memcpy`, which copies `block` vectors from `V1` to `V`.
      - **Parallelism is NOT SUPPORT**
      - For the meaning of this function, please visit ChaseMpiDLAInterface.
  */
  bool postApplication(T* V, std::size_t const block,  std::size_t locked) override {
    std::memcpy(V + locked_ * N_, get_V1(), N_ * block * sizeof(T));
    return false;
  }

  /*! - For ChaseMpiDLABlaslapackSeq, `shiftMatrix` is implemented by a loop of length `N_`.
      - **Parallelism is NOT SUPPORT**
      - For the meaning of this function, please visit ChaseMpiDLAInterface.
  */
  void shiftMatrix(T const c, bool isunshift = false) override {
    for (std::size_t i = 0; i < N_; ++i) {
      H_[i + i * N_] += c;
    }
  }

  void asynCxHGatherC(T *V, std::size_t locked, std::size_t block) override {
    T const alpha = T(1.0);
    T const beta = T(0.0);

    t_gemm(CblasColMajor, CblasConjTrans, CblasNoTrans,  
           N_, block, N_,                              
           &alpha,                                     
           H_, N_,                                     
           get_V1() + locked * N_, N_,                 
           &beta,                                      
           get_V2() + locked * N_, N_);

    std::memcpy(V22_ + locked * N_, V12_ + locked * N_, N_ * block * sizeof(T) );    
  }

  /*! - For ChaseMpiDLABlaslapackSeq, `applyVec` is implemented with `GEMM` provided by `BLAS`.
      - **Parallelism is SUPPORT within node if multi-threading is actived**
      - For the meaning of this function, please visit ChaseMpiDLAInterface.
  */
  void applyVec(T* B, T* C) override {
    T const One = T(1.0);
    T const Zero = T(0.0);

    this->preApplication(B, 0, 1);
    t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans,  //
           N_, 1, N_,                              //
           &One,                                     // V2 <-
           H_, N_,                                     //      alpha * H*V1
           get_V1(), N_,                 //      + beta * V2
           &Zero,                                      //
           get_V2(), N_);
    std::memcpy(C, get_V2(), N_ * sizeof(T));    
  }

  void get_off(std::size_t* xoff, std::size_t* yoff, std::size_t* xlen,
               std::size_t* ylen) const override {
    *xoff = 0;
    *yoff = 0;
    *xlen = N_;
    *ylen = N_;
  }

  T* get_H() const  override { return H_; }

  T* get_V1() const { return V1_.get(); }
  T* get_V2() const { return V2_.get(); }
  std::size_t get_mblocks() const override {return 1;}
  std::size_t get_nblocks() const override {return 1;}
  std::size_t get_n() const override {return N_;}
  std::size_t get_m() const override {return N_;}
  int *get_coord() const override {
	  int *coord = new int[2];
	  coord[0] = coord[1] = 0;
	  return coord;
  }
  int get_nprocs() const override {return 1;}  
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

  void Start() override {}

  /*!
    - For ChaseMpiDLABlaslapackSeq, `lange` is implemented using `LAPACK` routine `xLANGE`.
    - **Parallelism is SUPPORT within node if multi-threading is actived**
    - For the meaning of this function, please visit ChaseMpiDLAInterface.
  */
  Base<T> lange(char norm, std::size_t m, std::size_t n, T* A, std::size_t lda) override {
      return t_lange(norm, m, n, A, lda);
  }


  /*!
    - For ChaseMpiDLABlaslapackSeq, `axpy` is implemented using `BLAS` routine `xAXPY`.
   - **Parallelism is SUPPORT within node if multi-threading is actived**    
    - For the meaning of this function, please visit ChaseMpiDLAInterface.
  */
  void axpy(std::size_t N, T * alpha, T * x, std::size_t incx, T *y, std::size_t incy) override {
      t_axpy(N, alpha, x, incx, y, incy);
  }

  /*!
    - For ChaseMpiDLABlaslapackSeq, `scal` is implemented using `BLAS` routine `xSCAL`.
    - **Parallelism is SUPPORT within node if multi-threading is actived**   
    - For the meaning of this function, please visit ChaseMpiDLAInterface.
  */
  void scal(std::size_t N, T *a, T *x, std::size_t incx) override {
      t_scal(N, a, x, incx);
  }

  /*!
    - For ChaseMpiDLABlaslapackSeq, `nrm2` is implemented using `BLAS` routine `xNRM2`.
    - **Parallelism is SUPPORT within node if multi-threading is actived**    
    - For the meaning of this function, please visit ChaseMpiDLAInterface.
  */
  Base<T> nrm2(std::size_t n, T *x, std::size_t incx) override {
      return t_nrm2(n, x, incx);
  }

  /*!
    - For ChaseMpiDLABlaslapackSeq, `dot` is implemented using `BLAS` routine `xDOT`.
    - **Parallelism is SUPPORT within node if multi-threading is actived**       
    - For the meaning of this function, please visit ChaseMpiDLAInterface.
  */
  T dot(std::size_t n, T* x, std::size_t incx, T* y, std::size_t incy) override {
      return t_dot(n, x, incx, y, incy);
  }

  /*!
    - For ChaseMpiDLABlaslapackSeq, `gemm_small` is implemented using `BLAS` routine `xGEMM`.
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
    - For ChaseMpiDLABlaslapackSeq, `gemm_small` is implemented using `BLAS` routine `xGEMM`.
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
   - For ChaseMpiDLABlaslapackSeq, `stemr` with scalar being real and double precision, is implemented using `LAPACK` routine `DSTEMR`.
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
   - For ChaseMpiDLABlaslapackSeq, `stemr` with scalar being real and single precision, is implemented using `LAPACK` routine `SSTEMR`.
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
      - For ChaseMpiDLABlaslapackSeq, `RR_kernel` is implemented by `GEMM` routine provided by `BLAS` and `(SY)HEEVD` routine provided by `LAPACK`.
        - The 1st operation `A <- W^T * V` is implemented by `GEMM` from `BLAS`.
        - The 2nd operation which computes the eigenpairs of `A`, is implemented by `(SY)HEEVD` from `LAPACK`.
        - The 3rd operation which computes `W<-V*A` is implemented by `GEMM` from `BLAS`.
      - **Parallelism is SUPPORT within node if multi-threading is actived**    
      - For the meaning of this function, please visit ChaseMpiDLAInterface.
  */  

  void RR_kernel(std::size_t N, std::size_t block, T *approxV, std::size_t locked, T *workspace, T One, T Zero, Base<T> *ritzv) override {
    T alpha = T(1.0);
    T beta = T(0.0); 

    this->asynCxHGatherC(approxV, locked, block);

    auto A = std::unique_ptr<T[]> {
      new T[ block * block ]
    };


    // A <- W' * V
    t_gemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
             block, block, N,
             &One,
             V22_ + locked * N, N,
             get_V2() + locked * N, N,
             &Zero,
             A.get(), block
    );

    t_heevd(LAPACK_COL_MAJOR, 'V', 'L', block, A.get(), block, ritzv);

    t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
           N, block, block,
           &One,
           V12_ + locked * N, N,
           A.get(), block,
           &Zero,
           get_V1() + locked * N, N
    );

    std::memcpy(V12_+locked*N_, get_V1()+locked*N_, N_ * block * sizeof(T));
   
  }


  void syherk(char uplo, char trans, std::size_t n, std::size_t k, T* alpha, T* a, std::size_t lda, T* beta, T* c, std::size_t ldc)  override {
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

  void Resd(T *approxV_, T* workspace_, Base<T> *ritzv, Base<T> *resid, std::size_t locked, std::size_t unconverged) override{
    T alpha = T(1.0);
    T beta = T(0.0); 

    this->asynCxHGatherC(approxV_, locked, unconverged);
    for (std::size_t i = 0; i < unconverged; ++i) {
      beta = T(-ritzv[i]);
      axpy(                                      
          N_,                                      
          &beta,                                   
          (V22_ + locked * N_) + N_ * i, 1,   
          (get_V2() + locked * N_) + N_ * i, 1  
      );

      resid[i] = nrm2(N_, (get_V2() + locked * N_) + N_ * i, 1);
    }
  }

  void hhQR(std::size_t m_, std::size_t nevex, std::size_t locked,T *approxV, std::size_t ldv) override {
    std::unique_ptr<T []> tau(new T[nevex]);
    t_geqrf(LAPACK_COL_MAJOR, N_, nevex, get_V1(), N_, tau.get());
    t_gqr(LAPACK_COL_MAJOR, N_, nevex, nevex,  get_V1(), N_, tau.get());
    std::memcpy(get_V1(), V12_, locked * N_ * sizeof(T));
    std::memcpy(V12_+locked * N_, get_V1() + locked * N_, (nevex - locked) * N_ * sizeof(T));  
  }
  

  void cholQR(std::size_t N, std::size_t nevex, std::size_t locked, T *approxV, std::size_t ldv) override{
    auto A_ = std::unique_ptr<T[]> {
      new T[ nevex * nevex ]
    };
    T one = T(1.0);
    T zero = T(0.0);
    int info = -1;

    t_syherk('U', 'C', nevex, N_, &one, get_V1(), N_, &zero, A_.get(), nevex);
    info = t_potrf('U', nevex, A_.get(), nevex);
    
    if(info == 0){
      t_trsm('R', 'U', 'N', 'N', N_, nevex, &one, A_.get(), nevex, get_V1(), N_);

      int choldeg = 2;
      char *choldegenv;
      choldegenv = getenv("CHASE_CHOLQR_DEGREE");
      if(choldegenv){
        choldeg = std::atoi(choldegenv);
      }
#ifdef CHASE_OUTPUT   
      std::cout << "choldegee: " << choldeg << std::endl;
#endif
      for(auto i = 0; i < choldeg - 1; i++){
        t_syherk('U', 'C', nevex, N_, &one, get_V1(), N_, &zero, A_.get(), nevex);
        t_potrf('U', nevex, A_.get(), nevex);
        t_trsm('R', 'U', 'N', 'N', N_, nevex, &one, A_.get(), nevex, get_V1(), N_);
      }
      std::memcpy(get_V1(), V12_, locked * N_ * sizeof(T));
      std::memcpy(V12_+locked * N_, get_V1() + locked * N_, (nevex - locked) * N_ * sizeof(T));  
    }else{
#ifdef CHASE_OUTPUT         
      std::cout << "cholQR failed because of ill-conditioned vector, use Householder QR instead" << std::endl;
#endif      
      this->hhQR(N, nevex, locked, approxV, ldv);
    }

  }
  void Lock(T * workspace_, std::size_t new_converged) override{}

  void Swap(std::size_t i, std::size_t j)override{
    T *tmp = new T[N_];

    memcpy(tmp, get_V1() + N_ * i, N_ * sizeof(T));
    memcpy(get_V1() + N_ * i, get_V1() + N_ * j, N_ * sizeof(T));
    memcpy(get_V1() + N_ * j, tmp, N_ * sizeof(T));

    memcpy(tmp, V12_ + N_ * i, N_ * sizeof(T));
    memcpy(V12_ + N_ * i, V12_ + N_ * j, N_ * sizeof(T));
    memcpy(V12_ + N_ * j, tmp, N_ * sizeof(T));

    delete[] tmp;

  }

  void lanczos(std::size_t mIters, int idx, Base<T> *d, Base<T> *e,  Base<T> *rbeta,  T *V_, T *workspace_)override{
    std::size_t m = mIters;
    std::size_t n = N_;

    T *v0_ = new T[n]();
    T *w_ = new T[n]();

    T *v0 = v0_;
    T *w = w_;

    T alpha = T(1.0);
    T beta = T(0.0);
    T One = T(1.0);
    T Zero = T(0.0);
    
    // V is filled with randomness
    T *v1_ = new T[n]();
    T *v1 = v1_;

    if(idx >= 0){
      this->C2V(V12_ + idx * n, v1, 1);
    }else{
      std::mt19937 gen(2342.0);
      std::normal_distribution<> normal_distribution;

      for (std::size_t k = 0; k < N_; ++k){
        v1[k] = getRandomT<T>([&]() { return normal_distribution(gen); });
      }
    }

    // ENSURE that v1 has one norm
    Base<T> real_alpha = t_nrm2(n, v1, 1);
    alpha = T(1 / real_alpha);
    t_scal(n, &alpha, v1, 1);

    Base<T> real_beta = 0.0;

    for (std::size_t k = 0; k < m; k = k + 1) {
      std::memcpy(V1_.get() + k * n, v1, n * sizeof(T));
 
      this->applyVec(v1, w);

      alpha = t_dot(n, v1, 1, w, 1);
      alpha = -alpha;
      t_axpy(n, &alpha, v1, 1, w, 1);
      alpha = -alpha;

      d[k] = std::real(alpha);


      if (k == m - 1) break;

      beta = T(-real_beta);
      t_axpy(n, &beta, v0, 1, w, 1);
      beta = -beta;

      real_beta = t_nrm2(n, w, 1); 

      beta = T(1.0 / real_beta);

      t_scal(n, &beta, w, 1);

      e[k] = real_beta;

      std::swap(v1, v0);
      std::swap(v1, w); 
                                
    }

    *rbeta = real_beta;

    delete[] w_;
    delete[] v0_;    
  }


  void LanczosDos(std::size_t N_, std::size_t idx, std::size_t m, T *workspace_, std::size_t ldw, T *ritzVc, std::size_t ldr, T* approxV_) override{
    T alpha = T(1.0);
    T beta = T(0.0);

    t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
           N_, idx, m,
           &alpha,
           get_V1(), N_,
           ritzVc, ldr,
           &beta,
           V12_, N_
    );

    std::memcpy(get_V1(),V12_, m * N_ *sizeof(T));
  }

 private:
  std::size_t N_;
  std::size_t locked_;
  std::size_t maxBlock_;
  std::size_t nex_;
  std::size_t nev_;
  T* H_;
  std::unique_ptr<T> V1_; //C
  std::unique_ptr<T> V2_; //B
  std::unique_ptr<T> A_; //for CholeskyQR
  T *V12_; //C2
  T *V22_; //C1
};

template <typename T>
struct is_skewed_matrixfree<ChaseMpiDLABlaslapackSeq<T>> {
  static const bool value = false;
};

}  // namespace mpi
}  // namespace chase

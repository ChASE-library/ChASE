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
    	v0_.resize(N_);
   	v1_.resize(N_);
    	w_.resize(N_);
        }

  ChaseMpiDLABlaslapackSeq() = delete;
  ChaseMpiDLABlaslapackSeq(ChaseMpiDLABlaslapackSeq const& rhs) = delete;

  ~ChaseMpiDLABlaslapackSeq() {}
  void initVecs() override{
    next_ = NextOp::bAc;
    t_lacpy('A', N_, nev_+nex_, V12_, N_, get_V1() , N_);
  }  
  void initRndVecs() override {
     std::mt19937 gen(1337.0);
     std::normal_distribution<> d;
     for(auto j = 0; j < (nev_ + nex_); j++){
        for(auto i = 0; i < N_; i++){
          V12_[i + j * N_] = getRandomT<T>([&]() { return d(gen);});    
        }           
     }
  }
  void initRndVecsFromFile(std::string rnd_file) override {
      std::ostringstream problem(std::ostringstream::ate);
      problem << rnd_file;
      std::ifstream infile(problem.str().c_str(), std::ios::binary);

      infile.read(reinterpret_cast<char*>(V12_), N_ * (nev_+nex_) * sizeof(T));
  }

  //v1->v2
  void V2C(T *v1, std::size_t off1, T *v2, std::size_t off2, std::size_t block) override {
    std::memcpy(v2 + off2 * N_, v1 + off1 * N_, N_ * block *sizeof(T));        
  }

  //v1->v2;
  void C2V(T *v1, std::size_t off1, T *v2, std::size_t off2, std::size_t block) override {
    std::memcpy(v2 + off2 * N_, v1 + off1 * N_, N_ * block *sizeof(T));    
  }

  /*! - For ChaseMpiDLABlaslapackSeq, the core of `preApplication` is implemented with `std::memcpy`, which copies `block` vectors from `V` to `V1`.
      - **Parallelism is NOT SUPPORT**
      - For the meaning of this function, please visit ChaseMpiDLAInterface.
  */
  void preApplication(T* V, std::size_t const locked,
                      std::size_t const block) override {
    next_ = NextOp::bAc;
    locked_ = locked;
    std::memcpy(V12_, V + locked * N_, N_ * block * sizeof(T));
  }

  /*! - For ChaseMpiDLABlaslapackSeq, the core of `preApplication` is implemented with `std::memcpy`, which copies `block` vectors from `V2` to `V2_`.
      - **Parallelism is NOT SUPPORT**
      - For the meaning of this function, please visit ChaseMpiDLAInterface.
  */
  void preApplication(T* V1, T* V2, std::size_t const locked,
                      std::size_t const block) override {
    std::memcpy(V22_, V2 + locked * N_, N_ * block * sizeof(T));
    this->preApplication(V1, locked, block);
  }

  /*! - For ChaseMpiDLABlaslapackSeq, `apply` is implemented with `GEMM` provided by `BLAS`.
      - **Parallelism is SUPPORT within node if multi-threading is actived**
      - For the meaning of this function, please visit ChaseMpiDLAInterface.
  */
  void apply(T alpha, T beta, std::size_t offset, std::size_t block,  std::size_t locked) override {
 
    if (next_ == NextOp::bAc) {
      t_gemm(CblasColMajor, CblasConjTrans, CblasNoTrans, N_,
                static_cast<std::size_t>(block), N_, &alpha, H_, N_,
                V12_ + offset * N_ + locked * N_, N_, &beta, V22_ + locked * N_ + offset * N_, N_);

      next_ = NextOp::cAb;
    }else{
      t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N_,
             static_cast<std::size_t>(block), N_, &alpha, H_, N_,
             V22_ + offset * N_ + locked * N_, N_, &beta, V12_ + offset * N_ + locked * N_, N_);

      next_ = NextOp::bAc;
    }
  }
  /*! - For ChaseMpiDLABlaslapackSeq, the core of `postApplication` is implemented with `std::memcpy`, which copies `block` vectors from `V1` to `V`.
      - **Parallelism is NOT SUPPORT**
      - For the meaning of this function, please visit ChaseMpiDLAInterface.
  */
  bool postApplication(T* V, std::size_t const block,  std::size_t locked) override {
    if (next_ == NextOp::bAc) {
      std::memcpy(V + locked_ * N_, V12_, N_ * block * sizeof(T));
    }else{
      std::memcpy(V + locked_ * N_, V22_, N_ * block * sizeof(T));      
    }
    return true;
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

  void asynCxHGatherC(std::size_t locked, std::size_t block) override {
    T const alpha = T(1.0);
    T const beta = T(0.0);

    t_gemm(CblasColMajor, CblasConjTrans, CblasNoTrans,  
           N_, block, N_,                              
           &alpha,                                     
           H_, N_,                                     
           V12_ + locked * N_, N_,                 
           &beta,                                      
           V22_ + locked * N_, N_);

    std::memcpy(get_V2() + locked * N_, get_V1() + locked * N_, N_ * block * sizeof(T) );    
  }

  /*! - For ChaseMpiDLABlaslapackSeq, `applyVec` is implemented with `GEMM` provided by `BLAS`.
      - **Parallelism is SUPPORT within node if multi-threading is actived**
      - For the meaning of this function, please visit ChaseMpiDLAInterface.
  */
  void applyVec(T* B, T* C) override {
    T One = T(1.0);
    T Zero = T(0.0);

    this->preApplication(B, 0, 1);
    this->apply(One, Zero, 0, 1, 0);
    this->postApplication(C, 1, 0); 
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
    - For ChaseMpiDLABlaslapackSeq, `gemm` is implemented using `BLAS` routine `xGEMM`.
    - **Parallelism is SUPPORT within node if multi-threading is actived**    
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
      - For ChaseMpiDLABlaslapackSeq, `RR` is implemented by `GEMM` routine provided by `BLAS` and `(SY)HEEVD` routine provided by `LAPACK`.
        - The 1st operation `A <- W^T * V` is implemented by `GEMM` from `BLAS`.
        - The 2nd operation which computes the eigenpairs of `A`, is implemented by `(SY)HEEVD` from `LAPACK`.
        - The 3rd operation which computes `W<-V*A` is implemented by `GEMM` from `BLAS`.
      - **Parallelism is SUPPORT within node if multi-threading is actived**    
      - For the meaning of this function, please visit ChaseMpiDLAInterface.
  */  

  void RR(std::size_t block, std::size_t locked, Base<T> *ritzv) override {
      T One = T(1.0);
      T Zero = T(0.0);

    this->asynCxHGatherC(locked, block);

    auto A = std::unique_ptr<T[]> {
      new T[ block * block ]
    };

    // A <- W' * V
    t_gemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
             block, block, N_,
             &One,
             get_V2() + locked * N_, N_,
             V22_ + locked * N_, N_,
             &Zero,
             A.get(), block
    );

    t_heevd(LAPACK_COL_MAJOR, 'V', 'L', block, A.get(), block, ritzv);

    t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
           N_, block, block,
           &One,
           get_V1() + locked * N_, N_,
           A.get(), block,
           &Zero,
           V12_ + locked * N_, N_
    );

    std::memcpy(get_V1() +locked*N_, V12_ +locked*N_, N_ * block * sizeof(T));
   
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

  void Resd(Base<T> *ritzv, Base<T> *resid, std::size_t locked, std::size_t unconverged) override{
    T alpha = T(1.0);
    T beta = T(0.0); 

    this->asynCxHGatherC(locked, unconverged);

    for (std::size_t i = 0; i < unconverged; ++i) {
      beta = T(-ritzv[i]);
      axpy(                                      
          N_,                                      
          &beta,                                   
          (get_V2() + locked * N_) + N_ * i, 1,   
          (V22_ + locked * N_) + N_ * i, 1  
      );

      resid[i] = nrm2(N_, (V22_ + locked * N_) + N_ * i, 1);
    }
  }

  void hhQR(std::size_t locked) override {
    auto nevex = nev_ + nex_;

    std::unique_ptr<T []> tau(new T[nevex]);

    t_geqrf(LAPACK_COL_MAJOR, N_, nevex, V12_, N_, tau.get());
    t_gqr(LAPACK_COL_MAJOR, N_, nevex, nevex,  V12_, N_, tau.get());

    std::memcpy(V12_, get_V1(), locked * N_ * sizeof(T));
    std::memcpy(get_V1()+locked * N_, V12_ + locked * N_, (nevex - locked) * N_ * sizeof(T));  
  }
  

  void cholQR(std::size_t locked) override{
    auto nevex = nev_ + nex_;

    auto A_ = std::unique_ptr<T[]> {
      new T[ nevex * nevex ]
    };
    T one = T(1.0);
    T zero = T(0.0);
    int info = -1;

    t_syherk('U', 'C', nevex, N_, &one, V12_, N_, &zero, A_.get(), nevex);
    info = t_potrf('U', nevex, A_.get(), nevex);
    
    if(info == 0){
      t_trsm('R', 'U', 'N', 'N', N_, nevex, &one, A_.get(), nevex, V12_, N_);

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
        t_syherk('U', 'C', nevex, N_, &one, V12_, N_, &zero, A_.get(), nevex);
        t_potrf('U', nevex, A_.get(), nevex);
        t_trsm('R', 'U', 'N', 'N', N_, nevex, &one, A_.get(), nevex, V12_, N_);
      }
      std::memcpy(V12_, get_V1(), locked * N_ * sizeof(T));
      std::memcpy(get_V1()+locked * N_, V12_ + locked * N_, (nevex - locked) * N_ * sizeof(T));
    }else{
#ifdef CHASE_OUTPUT         
      std::cout << "cholQR failed because of ill-conditioned vector, use Householder QR instead" << std::endl;
#endif      
      this->hhQR(locked);
    }

  }

  void Swap(std::size_t i, std::size_t j)override{
    T *tmp = new T[N_];

    memcpy(tmp, V12_ + N_ * i, N_ * sizeof(T));
    memcpy(V12_ + N_ * i, V12_ + N_ * j, N_ * sizeof(T));
    memcpy(V12_ + N_ * j, tmp, N_ * sizeof(T));

    memcpy(tmp, get_V1() + N_ * i, N_ * sizeof(T));
    memcpy(get_V1() + N_ * i, get_V1() + N_ * j, N_ * sizeof(T));
    memcpy(get_V1() + N_ * j, tmp, N_ * sizeof(T));

    delete[] tmp;

  }

  void getLanczosBuffer(T **V1, T **V2, std::size_t *ld, T **v0, T **v1, T **w) override{
    *V1 = V12_;
    *V2 = get_V1();
    *ld = N_;

    std::fill(v1_.begin(), v1_.end(), T(0));
    std::fill(v0_.begin(), v0_.end(), T(0));
    std::fill(w_.begin(), w_.end(), T(0));

    *v0 = v0_.data();
    *v1 = v1_.data();
    *w = w_.data();
  }

 private:
  enum NextOp { cAb, bAc };
  NextOp next_;

  std::size_t N_;
  std::size_t locked_;
  std::size_t maxBlock_;
  std::size_t nex_;
  std::size_t nev_;
  T* H_;
  std::unique_ptr<T> V1_; //C
  std::unique_ptr<T> V2_; //B
  std::unique_ptr<T> A_; //for CholeskyQR
  std::vector<T> v0_;
  std::vector<T> v1_;
  std::vector<T> w_;
  T *V12_; //C1
  T *V22_; //B1
};

template <typename T>
struct is_skewed_matrixfree<ChaseMpiDLABlaslapackSeq<T>> {
  static const bool value = false;
};

}  // namespace mpi
}  // namespace chase

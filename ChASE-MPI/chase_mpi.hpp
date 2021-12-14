/* -*- Mode: C++; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
// This file is a part of ChASE.
// Copyright (c) 2015-2021, Simulation and Data Laboratory Quantum Materials, 
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include <cstring>  //memcpy
#include <iostream>
#include <memory>
#include <random>
#include "mpi.h"

#include "algorithm/chase.hpp"

#include "blas_templates.hpp"
#include "chase_mpi_matrices.hpp"

#include "./impl/chase_mpidla.hpp"

//#include <omp.h>

namespace chase {
namespace mpi {

//! A derived class of Chase to implement ChASE based on MPI and Dense Linear Algebra (**DLA**) routines.
  /*!
    This is a calls derived from the Chase class which plays the
      role of interface for the kernels used by the library. 
      - All members of the Chase class are virtual functions. These
      functions are re-implemented in the ChaseMpi
      class.
      - All the members functions of ChaseMpi, which are the implementation of the virtual functions in class Chase, 
        are implemented using the *DLA* routines provided by the class ChaseMpiDLAInterface.
      -  The DLA functions in ChaseMpiDLAInterface are also vritual functions, which are differently implemented targeting different
      computing architectures (sequential/parallel, CPU/GPU, shared-memory/distributed-memory, etc). In the class ChaseMpi, the calling of DLA functions are indeed
      calling their implementations from different derived classes. Thus this ChaseMpi 
      class is able to have customized implementation for various architectures.
      - For the implementation of the class ChaseMpi targeting distributed-memory platforms based
      on MPI, the setup of MPI environment and communication scheme,
      and the distribution of data (matrix, vectors) across MPI nodes are following the ChaseMpiProperties class,
      the distribution of matrix can be either **Block** or **Block-Cyclic** scheme.
      @tparam MF: A class derived from ChaseMpiDLAInterface, which indicates the selected implementation of DLA that to be used by ChaseMpi Object.
      @tparam T: the scalar type used for the application. ChASE is templated
      for real and complex numbers with both Single Precision and Double Precision,
      thus `T` can be one of `float`, `double`, `std::complex<float>` and 
      `std::complex<double>`.

  */
template <template <typename> class MF, class T>
class ChaseMpi : public chase::Chase<T> {
 public:
  // case 1:
  // todo? take all arguments of matrices and entirely wrap it?
  //! A constructor of the ChaseMpi class which gives an implenentation of ChASE for shared-memory architecture, without MPI.
  /*!
     The private members of this classes are initialized by the parameters of this constructor.
     - For the
     variable `N_`, it is initialized by the first parameter of this constructor `N`. 
     - The variables `nev_` and
     and `nex_` are initialized by the parameters of this constructor `nev` and `nex`, respectively.
     - The variable `rank_` is set to be 0 since non MPI is supported. The variable `locked_` is initially set to be 0.
     - The variable `config_` is setup by the constructor of ChaseConfig which takes the parameters `N`, `nev` and `nex`.
     - The variable `matrices_` is setup directly by the constructor of ChaseMpiMatrices which takes the paramter `H`, `N`,
     `nev`, `nex`, `V1`, `ritzv`, `V2` and `resid`. 
     - The variable `dla_` is initialized by `MF` which is a derived class
     of ChaseMpiDLAInterface. In ChASE, the candidates of `MF` for non-MPI case are the classes `ChaseMpiDLABlaslapackSeq`,
     `ChaseMpiDLABlaslapackSeqInplace` and `ChaseMpiDLACudaSeq`.

     @param N: size of the square matrix defining the eigenproblem.
     @param nev: Number of desired extremal eigenvalues.
     @param nex: Number of eigenvalues augmenting the search space. Usually a relatively small fraction of `nev`.
     @param V1: a pointer to a rectangular matrix of size `N * (nev+nex)`.
     @param ritzv: a pointer to an array to store the computed Ritz values.
     @param H: a pointer to the memory which stores local part of matrix on each MPI rank.
     @param V2: a pointer to anther rectangular matrix of size `N * (nev+nex)`.
     @param resid: a pointer to an array to store the residual of computed eigenpairs.
  */
  ChaseMpi(std::size_t N, std::size_t nev, std::size_t nex, T *V1 = nullptr,
           Base<T> *ritzv = nullptr, T *H = nullptr, T *V2 = nullptr,
           Base<T> *resid = nullptr)
      : N_(N),
        nev_(nev),
        nex_(nex),
        rank_(0),
        locked_(0),
        config_(N, nev, nex),
        matrices_(N_, nev_ + nex_, V1, ritzv, H, V2, resid),
        dla_(new MF<T>(matrices_, N_, nev_ + nex_)) {

    ritzv_ = matrices_.get_Ritzv();
    resid_ = matrices_.get_Resid();

    V_ = matrices_.get_V1();
    W_ = matrices_.get_V2();

    approxV_ = matrices_.get_V1();
    workspace_ = matrices_.get_V2();
    H_ = dla_->get_H();
  }

  // case 2: MPI
  //! A constructor of the ChaseMpi class which gives an implenentation of ChASE for distributed-memory architecture, with the support of MPI.
  /*!
     The private members of this classes are initialized by the parameters of this constructor. 
     - For the
     variable `N_`, `nev_` and `nex_` are initialized by the first parameter of this constructor `properties_`.
     - The variable `rank_` is initialized by `MPI_Comm_rank`. The variable `locked_` is initially set to be 0.
     - The variable `config_` is setup by the constructor of ChaseConfig which takes the parameters `N`, `nev` and `nex`.
     - The variable `matrices_` is constructed by the `create_matrices` function defined in ChaseMpiProperties.
     - The variable `dla_` for the distributed-memory is initialized by the constructor of ChaseMpiDLA which takes
     both `properties_` and `MF`. In MPI case, the implementation of are split into two classses: 
        - the class ChaseMpiDLA implements mainly the MPI collective communication part of ChASE
     with MPI support. 
        - The local computation tasks within each MPI is implemented by another two classes derived also from the class ChaseMpiDLAInterface:
     `ChaseMpiDLABlaslapack` for pure-CPUs version and `chaseMpiDLAMultiGPU` for multi-GPUs version.
        - Thus, for this constructor, a combination of ChaseMpiDLA and one of ChaseMpiDLABlaslapack and chaseMpiDLAMultiGPU is required.
     
     @param properties: an object of ChaseMpiProperties which setups the MPI environment and data distribution scheme for ChaseMpi targeting distributed-memory systems.
     @param V1: a pointer to a rectangular matrix of size `N * (nev+nex)`.
     @param ritzv: a pointer to an array to store the computed Ritz values.
     @param H: a pointer to the memory which stores local part of matrix on each MPI rank.
     @param V2: a pointer to anther rectangular matrix of size `N * (nev+nex)`.
     @param resid: a pointer to an array to store the residual of computed eigenpairs.
  */    
  ChaseMpi(ChaseMpiProperties<T> *properties, T *V1 = nullptr,
           Base<T> *ritzv = nullptr, T *V2 = nullptr, Base<T> *resid = nullptr)
      : N_(properties->get_N()),
        nev_(properties->GetNev()),
        nex_(properties->GetNex()),
        locked_(0),
        config_(N_, nev_, nex_),
        properties_(properties),
        matrices_(std::move(
            properties_.get()->create_matrices(V1, ritzv, V2, resid))),
        dla_(new ChaseMpiDLA<T>(properties_.get(),
                                  new MF<T>(properties_.get()))) {
    int init;
    MPI_Initialized(&init);
    if (init) MPI_Comm_rank(MPI_COMM_WORLD, &rank_);

    V_ = matrices_.get_V1();
    W_ = matrices_.get_V2();
    ritzv_ = matrices_.get_Ritzv();
    resid_ = matrices_.get_Resid();

    approxV_ = V_;
    workspace_ = W_;

    static_assert(is_skewed_matrixfree<MF<T>>::value,
                  "MatrixFreeChASE Must be skewed");
  }

  //! It prevents the copy operation of the constructor of ChaseMpi.
  ChaseMpi(const ChaseMpi &) = delete;

  ~ChaseMpi() {}

  //! This member function implements the virtual one declared in Chase class.
  //! \return `config_`: an object of ChaseConfig class which defines the eigensolver.
  ChaseConfig<T> &GetConfig()  override { return config_; }

  //! This member function implements the virtual one declared in Chase class.
  //! Returns the rank of matrix A which is distributed within 2D MPI grid.
  //! \return `N_`: the rank of matrix `A`.
  std::size_t GetN() const override { return N_; }

  //! This member function implements the virtual one declared in Chase class.
  //! Returns Number of desired extremal eigenpairs, which was set by users.
  //! \return `nev_`: Number of desired extremal eigenpairs.
  std::size_t GetNev() override { return nev_; }

  //! This member function implements the virtual one declared in Chase class.
  //! Returns the Increment of the search subspace so that its total size, which was set by users.
  //! \return `nex_`: Increment of the search subspace.
  std::size_t GetNex() override { return nex_; }

  //! This member function implements the virtual one declared in Chase class.
  //! Returns the array which stores the computed Ritz values.
  //! \return `ritzv_`: an array stores the computed Ritz values.
  Base<T> *GetRitzv() override { return ritzv_; }

  void Start()  override {
    locked_ = 0;
    dla_->Start();
  }
  void End() override {}

  //! \return `approxV_`: A pointer to the memory allocated to store a rectangular matrix `approxV_`, which will be right-multiplied to `A` during the process of ChASE. The eigenvectors obtained will also stored in `approxV_`.
  T *GetVectorsPtr() { return approxV_; }

  //! A pointer to the memory allocated to store a rectangular matrix `workspace_`, whose conjugate transpose will be left-multiplied to `A` during the process of ChASE.
  T *GetWorkspacePtr() { return workspace_; }

  //! This member function implements the virtual one declared in Chase class.
  //! This member function shifts the diagonal of matrix `A` with a shift value `c`.
  //! It is implemented by calling the different implementation in the derived class of `ChaseMpiDLAInterface`.
  //! For the construction of ChaseMpi with MPI, this function is naturally in parallel across the MPI nodes, since the shift operation only takes places on selected local matrices which stores in a distributed manner. 
  //! @param c: the shift value on the diagonal of matrix `A`.
  void Shift(T c, bool isunshift = false) override {
    if (!isunshift) {
      dla_->preApplication(approxV_, locked_, nev_ + nex_ - locked_);
    }

    dla_->shiftMatrix(c, isunshift);
  };

  // todo this is wrong we want the END of V
  void Cpy(std::size_t new_converged){
      //    memcpy( workspace+locked*N, approxV+locked*N,
      //    N*(new_converged)*sizeof(T) );
      // memcpy(approxV + locked * N, workspace + locked * N,
      //     N * (new_converged) * sizeof(T));
  };

  //! This member function implements the virtual one declared in Chase class.
  //! This member function computes \f$approxV = alpha * A*approxV + beta * approxV\f$.
  //! This operation on `approxV` starts with an offset of column, this value of offset is `offset`.
  /*!
      @param block: the number of vectors in `approxV` for this `HEMM` operation.
      @param alpha: a scalar of type `T` which scales `A*approxV`.
      @param beta: a scalar of type `T` which scales `approxV`.
      @param offset: the offset of column number which the `HEMM` starts from.
  */
  void HEMM(std::size_t block, T alpha, T beta, std::size_t offset) override {
    dla_->apply(alpha, beta, offset, block);
    std::swap(approxV_, workspace_);
  };

  void Hv(T alpha);

  //! This member function implements the virtual one declared in Chase class.
  //! This member function performs a QR factorization with an explicit construction of the unitary matrix `Q`.
  //! After the explicit construction of Q, its first `fixednev` number of vectors are overwritten by the converged eigenvectors stored in `workspace_`.
  //! @param fixednev: total number of converged eigenpairs before this time QR factorization.  
  void QR(std::size_t fixednev) override {
    //dla_->postApplication(approxV_, nev_ + nex_ - locked_);

    std::size_t nevex = nev_ + nex_;
    // we don't need this, as we copy to workspace when locking
    // std::memcpy(workspace_, approxV_, N_ * fixednev * sizeof(T));

    dla_->gegqr(N_, nevex, approxV_, N_);

    std::memcpy(approxV_, workspace_, N_ * fixednev * sizeof(T));
  };

  //! This member function implements the virtual one declared in Chase class.
  /*! This function performs the Rayleigh-Ritz projection, which projects the eigenproblem
      to be a small one, then solves the small problem and reconstructs the eigenvectors.
      @param ritzv: a pointer to the array which stores the computed eigenvalues.
      @param block: the number of non-converged eigenpairs, which determines the size of small eigenproblem.
  */
  void RR(Base<T> *ritzv, std::size_t block) override {

    T One = T(1.0);
    T Zero = T(0.0);

    dla_->preApplication(approxV_, locked_, block);
    dla_->apply(One, Zero, 0, block);
    dla_->postApplication(workspace_, block);

    // W <- H*V
    // t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans,  //
    //        N_, block, N_,                              //
    //        &One,                                       //
    //        H_, N_,                                     //
    //        approxV_ + locked_ * N_, N_,                //
    //        &Zero,                                      //
    //        workspace_ + locked_ * N_, N_);

    dla_->RR_kernel(N_, block, approxV_, locked_, workspace_, One, Zero, ritzv);

    std::swap(approxV_, workspace_);

    // we can swap, since the locked part were copied over as part of the QR

  };

  //! This member function implements the virtual one declared in Chase class.
  //! This member function computes the residuals of unconverged eigenpairs.
  /*!
       @param ritzv: an array stores the eigenvalues.
       @param resid: a pointer to an array which stores the residual for each eigenpairs.
       @param fixednev: number of converged eigenpairs. Thus the number of non-converged one which perform this operation of computing residual is `nev_ + nex_ + fixednev_`. 
  */
  void Resd(Base<T> *ritzv, Base<T> *resid, std::size_t fixednev) override {
    T alpha = T(1.0);
    T beta = T(0.0);
    std::size_t unconverged = (nev_ + nex_) - fixednev;

    dla_->preApplication(approxV_, locked_, unconverged);
    dla_->apply(alpha, beta, 0, unconverged);
    dla_->postApplication(workspace_, unconverged);

    // t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans,  //
    //        N_, unconverged, N_,                        //
    //        &alpha,                                     //
    //        H_, N_,                                     //
    //        approxV_ + locked_ * N_, N_,                //
    //        &beta,                                      //
    //        workspace_ + locked_ * N_, N_);

    dla_ -> Resd(approxV_, workspace_,ritzv, resid, locked_, unconverged);
  };

  //! This member function implements the virtual one declared in Chase class.
  //! It swaps the two matrices of vectors used in the Chebyschev filter
  //! @param i&j: the column indexing `i` and `j` are swapped in both rectangular matrices `approxV_` and `workspace_`.
  void Swap(std::size_t i, std::size_t j) override {
    T *ztmp = new T[N_];
    memcpy(ztmp, approxV_ + N_ * i, N_ * sizeof(T));
    memcpy(approxV_ + N_ * i, approxV_ + N_ * j, N_ * sizeof(T));
    memcpy(approxV_ + N_ * j, ztmp, N_ * sizeof(T));

    memcpy(ztmp, workspace_ + N_ * i, N_ * sizeof(T));
    memcpy(workspace_ + N_ * i, workspace_ + N_ * j, N_ * sizeof(T));
    memcpy(workspace_ + N_ * j, ztmp, N_ * sizeof(T));
    delete[] ztmp;
  };

  //! This member function implements the virtual one declared in Chase class.
  //! It estimates the upper bound of user-interested spectrum by Lanczos eigensolver
  //! @param m: the iterative steps for Lanczos eigensolver.
  //! @param upperb: a pointer to the upper bound estimated by Lanczos eigensolver.
  void Lanczos(std::size_t m, Base<T> *upperb) override {
    // todo
    std::size_t n = N_;
    T *v1 = workspace_;
    // std::random_device rd;
    std::mt19937 gen(2342.0);
    std::normal_distribution<> normal_distribution;

    for (std::size_t k = 0; k < N_; ++k)
      v1[k] = getRandomT<T>([&]() { return normal_distribution(gen); });

    // assert( m >= 1 );
    Base<T> *d = new Base<T>[m]();
    Base<T> *e = new Base<T>[m]();

    // SO C++03 5.3.4[expr.new]/15
    T *v0_ = new T[n]();
    T *w_ = new T[n]();

    T *v0 = v0_;
    T *w = w_;

    T alpha = T(1.0);
    T beta = T(0.0);
    T One = T(1.0);
    T Zero = T(0.0);
    //  T *v1 = V;
    // ENSURE that v1 has one norm
    Base<T> real_alpha = dla_->nrm2(n, v1, 1);
    alpha = T(1 / real_alpha);
    dla_->scal(n, &alpha, v1, 1);
    Base<T> real_beta = 0;
    real_beta = 0;

    for (std::size_t k = 0; k < m; ++k) {
      // t_gemv(CblasColMajor, CblasNoTrans, N_, N_, &One, H_, N_, v1, 1, &Zero,
      // w, 1);
      dla_->applyVec(v1, w);
      alpha = dla_->dot(n, v1, 1, w, 1);

      alpha = -alpha;
      dla_->axpy(n, &alpha, v1, 1, w, 1);
      alpha = -alpha;

      d[k] = std::real(alpha);
      if (k == m - 1) break;

      beta = T(-real_beta);
      dla_->axpy(n, &beta, v0, 1, w, 1);
      beta = -beta;

      real_beta = dla_->nrm2(n, w, 1);
      beta = T(1.0 / real_beta);

      dla_->scal(n, &beta, w, 1);

      e[k] = real_beta;

      std::swap(v1, v0);
      std::swap(v1, w);
    }

    delete[] w_;
    delete[] v0_;

    int notneeded_m;
    std::size_t vl, vu;
    Base<T> ul, ll;
    int tryrac = 0;
    int *isuppz = new int[2 * m];
    Base<T> *ritzv = new Base<T>[m];

    dla_->stemr(LAPACK_COL_MAJOR, 'N', 'A', m, d, e, ul, ll, vl, vu,
                     &notneeded_m, ritzv, NULL, m, m, isuppz, &tryrac);

    *upperb = std::max(std::abs(ritzv[0]), std::abs(ritzv[m - 1])) +
              std::abs(real_beta);

    delete[] ritzv;
    delete[] isuppz;
    delete[] d;
    delete[] e;
  };

  // we need to be careful how we deal with memory here
  // we will operate within Workspace
  //! This member function implements the virtual one declared in Chase class.
  //! It estimates the upper bound of user-interested spectrum by Lanczos eigensolver
  void Lanczos(std::size_t M, std::size_t idx, Base<T> *upperb, Base<T> *ritzv,
               Base<T> *Tau, Base<T> *ritzV) override {
    // todo
    std::size_t m = M;
    std::size_t n = N_;

    // assert( m >= 1 );

    // The first m*N part is reserved for the lanczos vectors
    Base<T> *d = new Base<T>[m]();
    Base<T> *e = new Base<T>[m]();

    // SO C++03 5.3.4[expr.new]/15
    T *v0_ = new T[n]();
    T *w_ = new T[n]();

    T *v0 = v0_;
    T *w = w_;

    T alpha = T(1.0);
    T beta = T(0.0);
    T One = T(1.0);
    T Zero = T(0.0);

    // V is filled with randomness
    T *v1 = workspace_;
    for (std::size_t k = 0; k < N_; ++k) v1[k] = V_[k + idx * N_];

    // ENSURE that v1 has one norm
    Base<T> real_alpha = dla_->nrm2(n, v1, 1);
    alpha = T(1 / real_alpha);
    dla_->scal(n, &alpha, v1, 1);

    Base<T> real_beta = 0.0;

    for (std::size_t k = 0; k < m; ++k) {
      if (workspace_ + k * n != v1)
        memcpy(workspace_ + k * n, v1, n * sizeof(T));

      // t_gemv(CblasColMajor, CblasNoTrans, n, n, &One, H_, n, v1, 1, &Zero, w,
      // 1);
      dla_->applyVec(v1, w);

      // std::cout << "lanczos Av\n";
      // for (std::size_t ll = 0; ll < 2; ++ll)
      //   std::cout << w[ll] << "\n";

      alpha = dla_->dot(n, v1, 1, w, 1);

      alpha = -alpha;
      dla_->axpy(n, &alpha, v1, 1, w, 1);
      alpha = -alpha;

      d[k] = std::real(alpha);
      if (k == m - 1) break;

      beta = T(-real_beta);
      dla_->axpy(n, &beta, v0, 1, w, 1);
      beta = -beta;

      real_beta = dla_->nrm2(n, w, 1);
      beta = T(1.0 / real_beta);

      dla_->scal(n, &beta, w, 1);

      e[k] = real_beta;

      std::swap(v1, v0);
      std::swap(v1, w);
    }

    delete[] w_;
    delete[] v0_;

    int notneeded_m;
    std::size_t vl, vu;
    Base<T> ul, ll;
    int tryrac = 0;
    int *isuppz = new int[2 * m];
    dla_->stemr(LAPACK_COL_MAJOR, 'V', 'A', m, d, e, ul, ll, vl, vu, &notneeded_m,
            ritzv, ritzV, m, m, isuppz, &tryrac);
    *upperb = std::max(std::abs(ritzv[0]), std::abs(ritzv[m - 1])) +
              std::abs(real_beta);

    for (std::size_t k = 1; k < m; ++k) {
      Tau[k] = std::abs(ritzV[k * m]) * std::abs(ritzV[k * m]);
      // std::cout << Tau[k] << "\n";
    }

    delete[] isuppz;
    delete[] d;
    delete[] e;
  };

  //! This member function implements the virtual one declared in Chase class.
  //! It locks the `new_converged` eigenvectors, which makes `locked_ += new_converged`.
  //! @param new_converged: number of newly converged eigenpairs in the present iterative step.
  void Lock(std::size_t new_converged) override {
    std::memcpy(workspace_ + locked_ * N_, approxV_ + locked_ * N_,
                N_ * (new_converged) * sizeof(T));
    locked_ += new_converged;
  };
  /*
  double compare(T *V_) {
    double norm = 0;
    for (std::size_t i = 0; i < (nev_ + nex_) * N; ++i)
      norm += std::abs(V_[i] - approxV[i]) * std::abs(V_[i] - approxV[i]);
    std::cout << "norm: " << norm << "\n";

    norm = 0;
    for (std::size_t i = 0; i < (locked_)*N; ++i)
      norm += std::abs(V_[i] - approxV[i]) * std::abs(V_[i] - approxV[i]);
    std::cout << "norm: " << norm << "\n";
  }
  */
  //! This member function implements the virtual one declared in Chase class.
  //! It estimates the spectral distribution of eigenvalues.
  void LanczosDos(std::size_t idx, std::size_t m, T *ritzVc) override {
    T alpha = T(1.0);
    T beta = T(0.0);

    dla_->gemm_small(CblasColMajor, CblasNoTrans, CblasNoTrans,  //
           N_, idx, m,                                 //
           &alpha,                                     //
           workspace_, N_,                             //
           ritzVc, m,                                  //
           &beta,                                      //
           approxV_, N_                                //
    );
  }

  //! This function return the residual of computed eigenpairs.
  /*! 
    For the computed eigenvectors stored in `approxV` with `nev` number of eigenvectors,
    this function return the largest absolute value of any elemental of a `N * nev_` matrix:
    \f$A * approxV - Ritz*approxV\f$, in which Ritz is a `nev_ * nev_` matrix whose diagonal stores
    the elemental of `ritzv_` in order, and whose other elements are `0`.
  */
  Base<T> Residual() {
    for (std::size_t j = 0; j < N_ * (nev_ + nex_); ++j) {
      workspace_[j] = approxV_[j];
    }

    //    memcpy(W, V, sizeof(MKL_Complex16)*N*nev);
    T one(1.0);
    T zero(0.0);
    T eigval;
    int iOne = 1;
    for (int ttz = 0; ttz < nev_; ttz++) {
      eigval = -1.0 * ritzv_[ttz];
      dla_->scal(N_, &eigval, workspace_ + ttz * N_, 1);
    }

    dla_->preApplication(approxV_, workspace_, 0, nev_);
    dla_->apply(one, one, 0, nev_);
    dla_->postApplication(workspace_, nev_);

    // t_hemm(CblasColMajor, CblasLeft, CblasLower,  //
    //        N_, nev_,                              //
    //        &one,                                  //
    //        H_, N_,                                //
    //        V_, N_,                                //
    //        &one, W_, N_);

    Base<T> norm = dla_->lange('M', N_, nev_, workspace_, N_);
    return norm;
  }

  //! This function checks the orthogonality of computed eigenvectors.
  /*! 
    For the computed eigenvectors stored in `approxV` with `nev` number of eigenvectors,
    this function return the largest absolute value of any elemental of a `nev_ * nev_` matrix:
    \f$approxV^H * approxV - I\f$.
  */
  Base<T> Orthogonality() {
    T one(1.0);
    T zero(0.0);
    // Check eigenvector orthogonality
    auto unity = std::unique_ptr<T[]>(new T[nev_ * nev_]);
    T neg_one(-1.0);
    for (int ttz = 0; ttz < nev_; ttz++) {
      for (int tty = 0; tty < nev_; tty++) {
        if (ttz == tty)
          unity[nev_ * ttz + tty] = 1.0;
        else
          unity[nev_ * ttz + tty] = 0.0;
      }
    }

    dla_->gemm_large(CblasColMajor, CblasConjTrans, CblasNoTrans, nev_, nev_, N_, &one,
           &*approxV_, N_, &*approxV_, N_, &neg_one, &unity[0], nev_);
    Base<T> norm = dla_->lange('M', nev_, nev_, &unity[0], nev_);
    return norm;
  }

#ifdef CHASE_OUTPUT
  void Output(std::string str) override {
    if (rank_ == 0) std::cout << str;
  }
#endif

  //! \return `H_`: A pointer to the memory allocated to store (local part if applicable) of matrix `A`. 
  T *GetMatrixPtr() { return dla_->get_H(); }

  //! Returns the number of submatrices along the row direction in the local matrix on each MPI node.
  //! This member function is usefull for the construction of ChaseMpiProperties with `Block-Cyclic Distribution`.
  std::size_t get_mblocks() {return dla_->get_mblocks();}

 //! Returns the number of submatrices along the column direction in the local matrix on each MPI node.
  //! This member function is usefull for the construction of ChaseMpiProperties with `Block-Cyclic Distribution`.  
  std::size_t get_nblocks() {return dla_->get_nblocks();}

  //! Returns Row number of the local matrix. 
  std::size_t get_m() {return dla_->get_m();}

  //! Returns Column number of the local matrix. 
  std::size_t get_n() {return dla_->get_n();}

  //! Returns the coordinates of each MPI rank in the cartesian communicator grid.
  int *get_coord() {return dla_->get_coord();}

  //! This member function only matters for the Block-Cyclic Distribution.
  //! Returns the pointers to `r_offs_`, `r_lens_`, `r_offs_l_`, `c_offs_`, `c_lens_` and `c_offs_l_` in single member function.
  //! For more details of this member function, please visit `get_offs_lens` function within ChaseMpiProperties.
  void get_offs_lens(std::size_t* &r_offs, std::size_t* &r_lens, std::size_t* &r_offs_l, 
		  std::size_t* &c_offs, std::size_t* &c_lens, std::size_t* &c_offs_l){
      dla_->get_offs_lens(r_offs, r_lens, r_offs_l, c_offs, c_lens, c_offs_l);
  }

  //! Returns the offset of row and column of the local matrix on each MPI node regarding the global index of matrix `A`.
  //! For more details of this member function, please visit `get_off` function within ChaseMpiProperties.
  void GetOff(std::size_t *xoff, std::size_t *yoff, std::size_t *xlen,
              std::size_t *ylen) {
    dla_->get_off(xoff, yoff, xlen, ylen);
  }

  //! This member function implements the virtual one declared in Chase class.
  //! \return `resid_`: a  pointer to the memory allocated to store the residual of each computed eigenpair.  
  Base<T> *GetResid()  override { return resid_; }

 private:
  //!Global size of the matrix A defining the eigenproblem.
  /*!
    - For the constructor of class ChaseMpi without MPI, 
    this variable is initialized by the constructor using the value of the first of its input parameters `N`. 
    - For the constructor of class ChaseMpi with MPI, this variable is initialized by `properties_`, a pointer to an object of ChaseMpiProperties.   
    This variable is private, it can be access by the member function GetN().
  */
  const std::size_t N_;

  //!Number of desired extremal eigenpairs.
  /*!
    - For the constructor of class ChaseMpi without MPI, 
    this variable is initialized by the constructor using the value of the second of its input parameters `nev`. 
    - For the constructor of class ChaseMpi with MPI, this variable is initialized by `properties`, a pointer to an object of ChaseMpiProperties.   
    This variable is private, it can be access by the member function GetNev().
  */
  const std::size_t nev_;

  //!Increment of the search subspace so that its total size is `nev+nex`.
  /*!
    - For the constructor of class ChaseMpi without MPI, 
    this variable is initialized by the constructor using the value of the third of its input parameters `nex`. 
    - For the constructor of class ChaseMpi with MPI, this variable is initialized by `properties`, a pointer to an object of ChaseMpiProperties.   
    This variable is private, it can be access by the member function GetNex().
  */  
  const std::size_t nex_;

  //! The rank of each MPI node with the working MPI communicator
  /*!
    - For the constructor of class ChaseMpi without MPI, this variable is `0`.
    - For the constructor of class ChaseMpi with MPI, this variable is gotten through MPI function `MPI_Comm_rank`.
  */
  int rank_;

  //! The number of eigenvectors to be locked, which indicates the number of eigenpairs converged into acceptable tolerance.
  /*!
    - For the constructor of class ChaseMpi without MPI, 
    this variable is initialized `0`.
    - For the constructor of class ChaseMpi with MPI, this variable is initialized also `0`.
    - During the process of ChASE solving eigenproblem, the value of this variable will increase with more and more eigenpairs computed.
  */ 
  std::size_t locked_;

  //! A pointer to the memory allocated to store the (local part) matrix of A.
  /*!
    - For the constructor of class ChaseMpi without MPI, 
    the memory is allocated directly by ChaseMpiMatrices of size `N_ * N_`.
    - For the constructor of class ChaseMpi with MPI, this variable is a pointer to the local matrix of `A` on each MPI node.
    It is initalized within the construction of ChaseMpiProperties.
    - This variable is private, and it can be assessed through the member function GetMatrixPtr().
  */ 
  T *H_;

  //! A pointer to the memory allocated to store a rectangular matrix `V`, which will be right-multiplied to `A` during the process of ChASE. The eigenvectors obtained will also stored in `V_`.
  /*!
    - For the constructor of class ChaseMpi without MPI, 
    the memory is allocated directly by ChaseMpiMatrices of size `N_ * (nex_ + nex_)`.
    - For the constructor of class ChaseMpi with MPI, this variable is a pointer to a rectangular matrix of `V` of size `N_ * (nex_ + nex_)`, 
    which are identical on each MPI node. It is initalized within the construction of ChaseMpiMatrices.
  */   
  T *V_;

  //! A pointer to the memory allocated to store a rectangular matrix `W`, whose conjugate transpose will be left-multiplied to `A` during the process of ChASE.
  /*!
    - For the constructor of class ChaseMpi without MPI, 
    the memory is allocated directly by ChaseMpiMatrices of size `N_ * (nex_ + nex_)`.
    - For the constructor of class ChaseMpi with MPI, this variable is a pointer to a rectangular matrix of `W` of size `N_ * (nex_ + nex_)`, 
    which are identical on each MPI node. It is initalized within the construction of ChaseMpiMatrices.
  */   
  T *W_;

  //! This pointer is pointing to the same memory of `V_`.
  //! This variable is private, and it can be assessed through the member function GetVectorsPtr().
  T *approxV_;

  //! This pointer is pointing to the same memory of `W_`.
  //! This variable is private, and it can be assessed through the member function GetWorkspacePtr().
  T *workspace_;

  //! A pointer to the memory allocated to store the computed eigenvalues. The values inside are always real since the matrix of eigenproblem is Hermitian (symmetric).
  /*!
    - For the constructor of class ChaseMpi without MPI, 
    the memory is allocated directly by ChaseMpiMatrices of size `nex_ + nex_`.
    - For the constructor of class ChaseMpi with MPI, this variable is a pointer to a vector of size `nex_ + nex_`, 
    which are identical on each MPI node. It is initalized within the construction of ChaseMpiMatrices.
    - The final converged eigenvalues will be stored in it.
    - This variable is private, and it can be assessed through the member function GetRitzv().
  */ 
  Base<T> *ritzv_;

  //! A pointer to the memory allocated to store the residual of each computed eigenpair. The values inside are always real.
  /*!
    - For the constructor of class ChaseMpi without MPI, 
    the memory is allocated directly by ChaseMpiMatrices of size `nex_ + nex_`.
    - For the constructor of class ChaseMpi with MPI, this variable is a pointer to a vector of size `nex_ + nex_`, 
    which are identical on each MPI node. It is initalized within the construction of ChaseMpiMatrices.
    - The residuals of final converged eigenvalues will be stored in it.
    - This variable is private, and it can be assessed through the member function GetResid().
  */   
  Base<T> *resid_;

  //! A smart pointer to an object of ChaseMpiProperties class which is used by ChaseMpi class.
  //! This variable only matters for the constructor of class ChaseMpi with MPI.
  std::unique_ptr<ChaseMpiProperties<T>> properties_;

  //! An object of ChaseMpiMatrices to setup the matrices and vectors used by ChaseMpi class.
  ChaseMpiMatrices<T> matrices_;

  //! A smart pointer to an object of ChaseMpiDLAInterface class which is used by ChaseMpi class.
  /*!
    - For the constructor of class ChaseMpi without MPI, 
   this variable is initialized directly by the **template parameter** `MF`.
    - For the constructor of class ChaseMpi with MPI, this variable is implemented by considering `properties_`  and `MF`.
    It is initalized within the construction of ChaseMpiMatrices.
  */
  std::unique_ptr<ChaseMpiDLAInterface<T>> dla_;

  //! An object of ChaseConfig class which setup all the parameters of ChASE, these parameters are either provided by users, or using the default values.
  //! This variable is initialized by the constructor of ChaseConfig class by using `N`, `nev` and `nex`, which are the parameters of the constructors of ChaseMpi.
  ChaseConfig<T> config_;
};

// TODO
/*
void check_params(std::size_t N, std::size_t nev, std::size_t nex,
                  const double tol, std::size_t deg )
{
  bool abort_flag = false;
  if(tol < 1e-14)
    std::clog << "WARNING: Tolerance too small, may take a while." << std::endl;
  if(deg < 8 || deg > ChASE_Config::chase_max_deg)
    std::clog << "WARNING: Degree should be between 8 and " <<
ChASE_Config::chase_max_deg << "."
              << " (current: " << deg << ")" << std::endl;
  if((double)nex/nev < 0.15 || (double)nex/nev > 0.75)
    {
      std::clog << "WARNING: NEX should be between 0.15*NEV and 0.75*NEV."
                << " (current: " << (double)nex/nev << ")" << std::endl;
      //abort_flag = true;
    }
  if(nev+nex > N)
    {
      std::cerr << "ERROR: NEV+NEX has to be smaller than N." << std::endl;
      abort_flag = true;
    }

  if(abort_flag)
    {
      std::cerr << "Stopping execution." << std::endl;
      exit(-1);
    }
 }
*/
}  // namespace mpi
}  // namespace chase

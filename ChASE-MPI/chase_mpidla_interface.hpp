/* -*- Mode: C++; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
// This file is a part of ChASE.
// Copyright (c) 2015-2023, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include <cstdlib>
#include <tuple>

#include "algorithm/types.hpp"
#include "chase_mpi_matrices.hpp"
#include "chase_mpi_properties.hpp"

namespace chase
{
namespace mpi
{

// Type Trait
template <typename T>
struct is_skewed_matrixfree
{
    static const bool value = false;
};

//! @brief A class to set up an interface to all the Dense Linear Algebra
//! (`DLA`) operations required by ChASE.
/*!
    In the class ChaseMpiDLAInterface, the `DLA` functions are only setup as a
   series of `virtual` functions without direct implementation. The
   implementation of these `DLA` will be laterly implemented by a set of derived
   classes targeting different computing architectures. Currently, in `ChASE`,
   we provide multiple derived classes
   - chase::mpi::ChaseMpiDLABlaslapackSeq: implementing ChASE targeting
   shared-memory architectures with only CPUs available.
   - chase::mpi::ChaseMpiDLABlaslapackSeqInplace: implementing ChASE targeting
   shared-memory architectures with only CPUs available, with a inplace mode,
   in which the buffer of rectangular matrices are swapped and reused. This
   reduces the required memory to be allocted.
   - chase::mpi::ChaseMpiDLACudaSeq: implementing ChASE targeting shared-memory
   architectures, most computation tasks are offloaded to one single GPU card.
   - chase::mpi::ChaseMpiDLA: implementing mostly the MPI collective
   communications part of distributed-memory ChASE targeting the systems with or
   w/o GPUs.
   - chase::mpi::ChaseMpiDLABlaslapack: implementing the inter-node computation
   for a pure-CPU MPI-based implementation of ChASE.
   - chase::mpi::ChaseMpiDLAMultiGPU: implementing the inter-node computation
   for a multi-GPU MPI-based implementation of ChASE.


    @tparam T: the scalar type used for the application. ChASE is templated
    for real and complex numbers with both Single Precision and Double
   Precision, thus `T` can be one of `float`, `double`, `std::complex<float>`
   and `std::complex<double>`.
 */
template <class T>
class ChaseMpiDLAInterface
{
public:
    typedef T value_type;

    virtual ~ChaseMpiDLAInterface(){};

    /*!
      This function shifts the diagonal of global matrix with a constant value
      `c`.
      @param c: shift value
    */
    virtual void shiftMatrix(T c, bool isunshift = false) = 0;
    /*!
      This function is for some pre-application steps for the distributed HEMM
      in ChASE. These steps may vary in different implementations targetting
      different architectures. These steps can be backup of some buffers,
      copy data from CPU to GPU, etc.
      @param V1: a pointer to a matrix
      @param locked: an integer indicating the number of locked (converged)
      eigenvectors
      @param block: an integer indicating the number of non-locked
      (non-converged) eigenvectors
    */
    virtual void preApplication(T* V1, std::size_t locked,
                                std::size_t block) = 0;

    //! Initialise the vectors in ChASE. When solving a sequence of problems,
    //! the solving first problem requires setting the initial vectors with
    //! random numbers in normal distribution.
    virtual void initVecs() = 0;
    //! Fill the initial vectors with random numbers in normal distribution if
    //! necessary.
    virtual void initRndVecs() = 0;
    //! Performs \f$V_2<- \alpha V1H + \beta V_2\f$ and `swap`\f$(V_1,V_2)\f$.
    /*!
      The first `offset` vectors of V1 and V2 are not part of the `HEMM`.
      The number of vectors performed in `V1` and `V2` is `block`
      In `MATLAB` notation, this operation performs:

      `V2[:,start:end]<-alpha*V1[:,start:end]*H+beta*V2[:,start:end]`,

      in which
      `start=locked+offset` and `end=locked+offset+block`.

      @param alpha: a scalar times on `V1*H` in `HEMM` operation.
      @param beta: a scalar times on `V2` in `HEMM` operation.
      @param offset: an offset of number vectors which the `HEMM` starting from.
      @param block: number of non-converged eigenvectors, it indicates the
      number of vectors in `V1` and `V2` to perform `HEMM`.
      @param locked: number of converged eigenvectors.
    */
    virtual void apply(T alpha, T beta, std::size_t offset, std::size_t block,
                       std::size_t locked) = 0;

    //! Performs \f$V_2<- V1H + V_2\f$
    /*!
      The number of vectors performed in `V1` and `V2` is `block`
      In `MATLAB` notation, this operation performs:

      `V2[:,start:end]<-alpha*V1[:,start:end]*H+beta*V2[:,start:end]`,

      in which
      `start=locked` and `end=locked+block`.

      @param locked: number of converged eigenvectors.
      @param block: number of non-converged eigenvectors, it indicates the
      number of vectors in `V1` and `V2` to perform `HEMM`.
      @param isCcopied: a flag indicates is a required buffer `C` has already
      been copied to GPU device. It matters only for ChaseMpiDLAMultiGPU.
    */
    virtual void asynCxHGatherC(std::size_t locked, std::size_t block,
                                bool isCcopied = false) = 0;

    //! Swap the columns indexing `i` and `j` in a rectangular matrix
    //! The operated matrices maybe different in different implementations
    /*!
     *  @param i: index of one column to be swapped
     *  @param j: index of another column to be swapped
     *
     */
    virtual void Swap(std::size_t i, std::size_t j) = 0;

    //! Performs a Generalized Matrix Vector Multiplication (`GEMV`) with
    //! `alpha=1.0` and `beta=0.0`.
    /*!
       The operation is `C=H*B`.
       @param B: the vector to be multiplied on `H`.
       @param C: the vector to store the product of `H` and `B`.
    */
    virtual void applyVec(T* B, T* C, std::size_t n) = 0;
    // Returns ptr to H, which may be used to populate H.
    //! Return the total number of MPI procs within the working MPI
    //! communicator.
    virtual int get_nprocs() const = 0;
    virtual Base<T>* get_Resids() = 0;
    virtual Base<T>* get_Ritzv() = 0;

    //! Starting point of solving an eigenproblem
    virtual void Start() = 0;
    //! Ending point of solving an eigenproblem
    virtual void End() = 0;

    //! A `BLAS-like` function which performs a constant times a vector plus a
    //! vector.
    /*!
        @param[in] N: number of elements in input vector(s).
        @param[in] alpha: a scalar times on `x` in `AXPY` operation.
        @param[in] x: an array of type `T`, dimension `( 1 + ( N - 1 )*abs( incx
       )`.
        @param[in] incx:  storage spacing between elements of `x`.
        @param[in/out] y: an array of type `T`, dimension `( 1 + ( N - 1 )*abs(
       incy )`.
        @param[in] incy:  storage spacing between elements of `y`.
    */
    virtual void axpy(std::size_t N, T* alpha, T* x, std::size_t incx, T* y,
                      std::size_t incy) = 0;
    //! A `BLAS-like` function which scales a vector by a constant.
    /*!
        @param[in] N: number of elements in input vector(s).
        @param[in] a: a scalar of type `T` times on vector `x`.
        @param[in/out] x: an array of type `T`, dimension `( 1 + ( N - 1 )*abs(
       incx )`.
        @param[in] incx:  storage spacing between elements of `x`.
    */
    virtual void scal(std::size_t N, T* a, T* x, std::size_t incx) = 0;

    //! A `BLAS-like` function which returns the euclidean norm of a vector.
    /*!
        @param[in] N: number of elements in input vector(s).
        @param[in] x: an array of type `T`, dimension `( 1 + ( N - 1 )*abs( incx
       )`.
        @param[in] incx:  storage spacing between elements of `x`.
        \return the euclidean norm of vector `x`.
    */
    virtual Base<T> nrm2(std::size_t n, T* x, std::size_t incx) = 0;

    //! A `BLAS-like` function which forms the dot product of two vectors.
    /*!
        @param[in] N: number of elements in input vector(s).
        @param[in] x: an array of type `T`, dimension `( 1 + ( N - 1 )*abs( incx
       )`.
        @param[in] incx:  storage spacing between elements of `x`.
        @param[in] y: an array of type `T`, dimension `( 1 + ( N - 1 )*abs( incy
       )`.
        @param[in] incy:  storage spacing between elements of `y`.
        \return the dot product of vectors `x` and `y`.
    */
    virtual T dot(std::size_t n, T* x, std::size_t incx, T* y,
                  std::size_t incy) = 0;

    //! This function implements the kernel of **Rayleigh-Ritz** (short as `RR`)
    //! step of ChASE.
    /*!
      @param[in] block: the number of vectors are used for these operations
      within `approxV^T` and `workspace`.
      @param[in] locked: number of converged eigenvectors
      @param[out] ritzv: a real vector which stores the computed eigenvalues.
    */
    virtual void RR(std::size_t block, std::size_t locked, Base<T>* ritzv) = 0;

    //! A `LAPACK-like` function which forms one of the symetric/hermitian rank
    //! k operations
    //! - \f$c := alpha*a*a**H + beta*c,\f$
    //! - \f$c := alpha*a**H*a + beta*c,\f$
    /*!
     *  where  alpha and beta  are  real scalars,  c is an  n by n
     symetric/hermitian matrix and  a  is an  n by k  matrix in the first case
     and a  k by n matrix in the second case.

        The parameters of this function is the same as <a
     href="https://netlib.org/lapack/
        explore-html/dc/d17/group__complex16__blas__level3_ga71e68893445a523b923411ebf4c22582.
        html">zherk()</a>, <a
     href="https://netlib.org/lapack/explore-html/db/def/group__
        complex__blas__level3_gade9f14cf41f0cefea7918d716f3e1c20.html">cherk()</a>,
        <a
     href="https://netlib.org/lapack/explore-html/d1/d54/group__double__blas_
        _level3_gae0ba56279ae3fa27c75fefbc4cc73ddf.html">dsyrk()</a> and <a
     href="https://netlib
        .org/lapack/explore-html/db/dc9/group__single__blas__level3_gae953a93420ca237670f
        5c67bbde9d9ff.html">ssyrk()</a>,
    */
    virtual void syherk(char uplo, char trans, std::size_t n, std::size_t k,
                        T* alpha, T* a, std::size_t lda, T* beta, T* c,
                        std::size_t ldc, bool first = true) = 0;

    //! A `LAPACK-like` function which computes the Cholesky factorization of a
    //! symmetric/Hermitian positive definite matrix `a`.
    /*!
        The parameters of this function is the same as <a
       href="https://netlib.org/lapack/explore-html/d1
        /d7a/group__double_p_ocomputational_ga2f55f604a6003d03b5cd4a0adcfb74d6.html">dpotrf()</a>,
        <a
       href="https://netlib.org/lapack/explore-html/d8/db2/group__real_p_ocomputational_gaaf
        31db7ab15b4f4ba527a3d31a15a58e.html">spotrf()</a>,
        <a
       href="https://netlib.org/lapack/explore-html/d6/df6/group__complex_p_ocomputational
        _ga4e85f48dbd837ccbbf76aa077f33de19.html">cpotrf()</a>,
        <a
       href="https://netlib.org/lapack/explore-html/d3/d8d/group__complex16_p_ocomputational_
        ga93e22b682170873efb50df5a79c5e4eb.html">zpotrf()</a>
    */
    virtual int potrf(char uplo, std::size_t n, T* a, std::size_t lda, bool isInfo = true) = 0;

    //! A `LAPACK-like` function which solves one of the matrix equations
    /*!
        \f$op( A )*X = alpha*B\f$ or \f$X*op( A ) = alpha*B\f$,
         where alpha is a scalar, X and B are m by n matrices, A is a unit, or
       non-unit,  upper or lower triangular matrix  and  op( A )  is $A$ itself,
       the transpose or conjugate transpose of itself. The parameters of this
       function is the same as <a href="https://netlib.org/lapack/explore-h
        tml/d1/d54/group__double__blas__level3_ga6a0a7704f4a747562c1bd9487e89795c.html">dtrsm()</a>,
        <a
       href="https://netlib.org/lapack/explore-html/db/dc9/group__single__blas__level3_ga9893c
        ceb3ffc7ce400eee405970191b3.html">strsm()</a>,
        <a
       href="https://netlib.org/lapack/explore-html/db/def/group__complex__blas__level3_
        gaf33844c7fd27e5434496d2ce0c1fc9d4.html">ctrsm()</a>,
        <a
       href="https://netlib.org/lapack/explore-html/dc/d17/group__complex16__blas__level3
        _gac571a0a6d43e969990456d0676edb786.html">ztrsm()</a>
    */
    virtual void trsm(char side, char uplo, char trans, char diag,
                      std::size_t m, std::size_t n, T* alpha, T* a,
                      std::size_t lda, T* b, std::size_t ldb,
                      bool first = false) = 0;

    //! A `LAPACK-like` function which computes all eigenvalues and, optionally,
    //! all eigenvectors of a complex Hermitian/real symmetric matrix using
    //! divide and conquer algorithm.
    /*！
        The parameters of this function is the same as <a
       href="https://netlib.org/lapack/explore-html/d3/d88/gr
        oup__real_s_yeigen_ga6b4d01c8952350ea557b90302ef9de4d.html">ssyevd()</a>,
       <a href="https://netlib.org/lapack/exp
        lore-html/d2/d8a/group__double_s_yeigen_ga77dfa610458b6c9bd7db52533bfd53a1.html">dsyevd()</a>,
       <a href="https://
        netlib.org/lapack/explore-html/d9/de3/group__complex_h_eeigen_ga6084b0819f9642f0db26257e8a3ebd42.html">cheevd()</a>
        and <a
       href="https://netlib.org/lapack/explore-html/df/d9a/group__complex16_h_eeigen_ga9b3e110476166e66f
        2f62fa1fba6344a.html">zheevd()</a>.
    */
    virtual void heevd(int matrix_layout, char jobz, char uplo, std::size_t n,
                       T* a, std::size_t lda, Base<T>* w) = 0;

    //! Compute the residuals of unconverged ritz pairs, the locked ones is
    //! skipped.
    //! @param ritzv the ritz values
    //! @param resid the computed residuals
    //! @param locked the number of converged ritz values
    //! @param unconverged the number of unconverged ritz values
    //! (`=nev_+nex-locked`)
    virtual void Resd(Base<T>* ritzv, Base<T>* resid, std::size_t locked,
                      std::size_t unconverged) = 0;

    //! Househoulder QR factorization on the rectangular matrix `V1`.
    //! It can be geqrf from
    //!     - `LAPACK` ,
    //!     - `ScaLAPACK`,
    //!     - `cuSolver`,
    //!
    //! which depends on
    //! the implementation and targeting architectures.
    //!  @param locked: number of converged eigenvectors.
    virtual void hhQR(std::size_t locked) = 0;
    //! Cholesky QR factorization on the rectangular matrix `V1`.
    //!  @param locked: number of converged eigenvectors.
    virtual void cholQR(std::size_t locked, Base<T> cond) = 0;
    //! Lanczos DOS to estimate the \mu_{nev+nex} for ChASE
    virtual void LanczosDos(std::size_t idx, std::size_t m, T* ritzVc) = 0;

    virtual void Lanczos(std::size_t M, Base<T>* d, Base<T>* e,
                         Base<T>* r_beta) = 0;
    virtual void mLanczos(std::size_t M, int numvec, Base<T>* d, Base<T>* e,
                         Base<T>* r_beta) = 0;

    virtual void B2C(T* B, std::size_t off1, T* C, std::size_t off2,
                     std::size_t block) = 0;

    virtual void lacpy(char uplo, std::size_t m, std::size_t n, T* a,
                       std::size_t lda, T* b, std::size_t ldb) = 0;
    virtual void shiftMatrixForQR(T* A, std::size_t n, T shift) = 0;
    virtual ChaseMpiMatrices<T>* getChaseMatrices() = 0;
};
} // namespace mpi
} // namespace chase

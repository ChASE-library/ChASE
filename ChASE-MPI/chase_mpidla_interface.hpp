/* -*- Mode: C++; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
// This file is a part of ChASE.
// Copyright (c) 2015-2021, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include <cstdlib>
#include <tuple>

#include "algorithm/types.hpp"
#include "chase_mpi_properties.hpp"
//
//  MatrixFreeInterface provides the Matrix-Matrix multiplication for use in
//  ChASE. The core functionality is apply(), which performs a HEMM.

// ASSUMPTION: MatrixFreeInterface assumes that apply() is called an even number
//             of times. This is always the case when used with ChASE_Algorithm.

// ASSUMPTION: shiftMatrix must be called at least once before apply() is called
//             (Some implementations may use shift() to transfer the matrix to
//              the GPU)

// The manner is which the Matrix H is loaded into the class is defined by the
// subclass. Further, the size of the vectors V1 and V2 must be known to any
// subclass.

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

//! A class to set up an interface to all the Dense Linear Algebra (`DLA`)
//! operations required by `ChASE.`
/*!
    In the class ChaseMpiDLAInterface, the `DLA` functions are only setup as a
   series of `virtual` functions without direct implementation. The
   implementation of these `DLA` will be laterly implemented by a set of derived
   classes targeting different computing architectures. Currently, in `ChASE`,
   we provide multiple derived classes chase::mpi::ChaseMpiDLABlaslapack,
   chase::mpi::ChaseMpiDLABlaslapackSeq,
   chase::mpi::ChaseMpiDLABlaslapackSeqInplace, chase::mpi::ChaseMpiDLACudaSeq
   and chase::mpi::ChaseMpiDLAMultiGPU.

    This **MatrixFreeInterface** provides the Matrix-Matrix multiplication for
   use in ChASE. The core functionality is `apply()`, which performs a `HEMM`.
    - **ASSUMPTION 1**: MatrixFreeInterface assumes that `apply()` is called an
   even number of times. This is always the case when used with algorithm of
   ChASE.
    - **ASSUMPTION 2**: `shiftMatrix` must be called at least once before
   `apply()` is called. Some implementations may use `shift()` to transfer the
   matrix to the GPU).
    - The manner in which the Matrix `H` is loaded into the class is defined by
   the subclass. Further, the size of the vectors `V1` and `V2` must be known to
   any subclass.
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

    // After a call to shiftMatrix(T c) all subsequent calls to apply() and
    //   applyVec() perform the multiplication with a shifted H
    /*!
      This function shift the diagonal of global matrix with a constant value
      `c`. After a call to `shiftMatrix` all subsequent calls to `apply` and
      `applyVec` perform the multiplication with a shift `c`. This is a virtual
      function, its implementation varies differently in different derived
      classes.
      @param c: shift value
    */
    virtual void shiftMatrix(T c, bool isunshift = false) = 0;
    /*!
      Compared to `preApplication` defined previously, this function only
      populates `V1`. After a call to this function the state of V2 is
      undefined. This is a virtual function, its implementation varies
      differently in different derived classes.
      @param V1: a `N * max_block_` rectangular matrix
      @param locked: an integer indicating the number of locked (converged)
      eigenvectors
      @param block: an integer indicating the number of non-locked
      (non-converged) eigenvectors
    */
    virtual void preApplication(T* V1, std::size_t locked,
                                std::size_t block) = 0;

    // Performs V2<- alpha*V1*H + beta*V2 and swap(V1,V2).
    // The first offset vectors of V1 and V2 are not part of the GEMM.
    // The GEMM uses block vectors.
    // block must be smaller or equal to the block passed in preApplication.
    // Including the locked vectors passed into apply we perform, in MATLAB
    // notation
    // V2[:,locked+offset, locked+offset+block]
    //       <- alpha* V1[:,locked+offset,locked+offset+block]*H
    //          + beta*V2[:,locked+offset,locked+offset+block]

    //! Performs `V2<- alpha*V1*H + beta*V2` and `swap(V1,V2)`.
    /*!
      The first `offset` vectors of V1 and V2 are not part of the `HEMM`.
      The `HEMM` uses block vectors. `block` must be smaller or equal to the
      block passed in `preApplication`. Including the `locked` vectors passed
      into apply we perform, in `MATLAB` notation:
      `V2[:,start:end]<-alpha*V1[:,start:end]*H+beta*V2[:,start:end]`, in which
      `start=locked+offset` and `end=locked+offset+block`.
      @param alpha: a scalar times on `V1*H` in `HEMM` operation.
      @param beta: a scalar times on `V2` in `HEMM` operation.
      @param offset: an offset of number vectors which the `HEMM` starting from.
      @param block: number of non-converged eigenvectors, it indicates the
      number of vectors in `V1` and `V2` to perform `HEMM`.
    */
    virtual void initVecs() = 0;
    virtual void initRndVecs() = 0;

    virtual void apply(T alpha, T beta, std::size_t offset, std::size_t block,
                       std::size_t locked) = 0;
    virtual void asynCxHGatherC(std::size_t locked, std::size_t block,
                                bool isCcopied = false) = 0;

    virtual void C2V(T* v1, std::size_t off1, T* v2, std::size_t off2,
                     std::size_t block) = 0;
    virtual void V2C(T* v1, std::size_t off1, T* v2, std::size_t off2,
                     std::size_t block) = 0;

    virtual void Swap(std::size_t i, std::size_t j) = 0;
    // Copies V2, the result of one or more results of apply() to V.
    // block number of vectors are copied.
    // The first locked ( as supplied to preApplication() ) vectors are skipped
    //! Copies `V2`, the result of one or more results of `apply` to `V`.
    /*!
       `block` number of vectors are copied. The first `locked` ( as supplied to
      `preApplication` ) vectors are skipped.
      @param V: the vectors which `V2` are copied to.
      @param block: number of vectors to be copied to `V` from `V2`.
    */
    virtual bool postApplication(T* V, std::size_t block,
                                 std::size_t locked) = 0;

    // Performs a GEMV with alpha=1.0 and beta=0.0
    // Equivalent to calling:
    //
    // preApplication( B, 0, 1 );
    // apply( T(1.0), T(0.0), 0, 1 );
    // postApplication( C, 1 );
    //! Performs a Generalized Matrix Vector Multiplication (`GEMV`) with
    //! `alpha=1.0` and `beta=0.0`.
    /*!
       The operation is `C=H*B`.
       @param B: the vector to be multiplied on `H`.
       @param C: the vector to store the product of `H` and `B`.
    */
    virtual void applyVec(T* B, T* C) = 0;
    // Returns ptr to H, which may be used to populate H.
    virtual int get_nprocs() const = 0;
    virtual void Start() = 0;
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
      1. It performs `A<-approxV^T * workspace`, for both `approxV^T` and
      `workspace`, this operation starts from its column of numbering `locked`.
      `A` is of size `block*block`.
      2. It computes the eigenpairs of `A`, the eigenvalues are stored in
      `ritzv`, and the eigenvectors are overwritten into `A`.
      3. It performs `workspace=A*approxV`, for both `approxV^T` and
      `workspace`, this operation starts from its column of numbering `locked`.
      @param[in] N: the row number of `approxV^T` and `workspace`.
      @param[in] block: the number of vectors are used for these operations
      within `approxV^T` and `workspace`.
      @param[in] approxV: On entry, a `N*max_block_` rectangular matrix of type
      `T`.
      @param[in] locked: number of converged eigenvectors, which are not
      considered for these operations. All the computation in the kernel related
      to `approxV^T` and `workspace` start from the offset `locked * N`.
      @param[in/out] workspace: On entry, a `N*max_block_` rectangular matrix of
      type `T`. On exit, overwritten by `A*approxV`.
      @param[in] One: scalar `T(1)`.
      @param[in] Zero: scalar `T(0)`.
      @param[out] ritzv: a real vector which stores the computed eigenvalues.
    */
    virtual void RR(std::size_t block, std::size_t locked, Base<T>* ritzv) = 0;

    virtual void syherk(char uplo, char trans, std::size_t n, std::size_t k,
                        T* alpha, T* a, std::size_t lda, T* beta, T* c,
                        std::size_t ldc, bool first = true) = 0;

    virtual int potrf(char uplo, std::size_t n, T* a, std::size_t lda) = 0;

    virtual void trsm(char side, char uplo, char trans, char diag,
                      std::size_t m, std::size_t n, T* alpha, T* a,
                      std::size_t lda, T* b, std::size_t ldb,
                      bool first = false) = 0;

    virtual void heevd(int matrix_layout, char jobz, char uplo, std::size_t n,
                       T* a, std::size_t lda, Base<T>* w) = 0;

    virtual void Resd(Base<T>* ritzv, Base<T>* resid, std::size_t locked,
                      std::size_t unconverged) = 0;
    virtual void hhQR(std::size_t locked) = 0;
    virtual void cholQR(std::size_t locked) = 0;
    virtual void getLanczosBuffer(T** V1, T** V2, std::size_t* ld, T** v0,
                                  T** v1, T** w) = 0;
    virtual void getLanczosBuffer2(T** v0, T** v1, T** w) = 0;
    virtual void LanczosDos(std::size_t idx, std::size_t m, T* ritzVc) = 0;
};
} // namespace mpi
} // namespace chase

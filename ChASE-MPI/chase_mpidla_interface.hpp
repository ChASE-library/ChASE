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

    // preApplication prepares internal state to perform the GEMM:
    // V2 <- alpha * H*V1 + beta*V2
    // The first locked number of vectors of V1 are not used in the GEMM.
    //     In ChASE these are the locked vectors.
    // Starting from locked, V1 contains block number of vectors.
    /*!
      `preApplication` prepares internal state to perform the `HEMM`: `V2 <-
      alpha * H*V1 + beta*V2`. The first `locked` number of vectors of `V1` are
      not used in the `HEMM`. In ChASE, these are the locked vectors, which are
      already converged with acceptable tolerance. Starting from the column
      `locked`, `V1` contains `block` number of vectors. This is a virtual
      function, its implementation varies differently in different derived
      classes.
      @param V1: a `N * max_block_` rectangular matrix
      @param V2: a `N * max_block_` rectangular matrix
      @param locked: an integer indicating the number of locked (converged)
      eigenvectors
      @param block: an integer indicating the number of non-locked
      (non-converged) eigenvectors
    */
    virtual void preApplication(T* V1, T* V2, std::size_t locked,
                                std::size_t block) = 0;

    // This function only populates V1.
    // After a call to this function the state of V2 is undefined.

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
    virtual void initRndVecsFromFile(std::string rnd_file) = 0;

    virtual void apply(T alpha, T beta, std::size_t offset, std::size_t block,
                       std::size_t locked) = 0;
    virtual void asynCxHGatherC(std::size_t locked, std::size_t block, bool isCcopied = false) = 0;

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

    // The offsets and sizes of the block of the matrix H that the class uses.
    // In a subclass that uses MPI parallelization of the HEMM offsets may be
    // non-zero for some processes.
    virtual void get_off(std::size_t* xoff, std::size_t* yoff,
                         std::size_t* xlen, std::size_t* ylen) const = 0;

    // Returns ptr to H, which may be used to populate H.
    virtual T* get_H() const = 0;
    virtual std::size_t get_mblocks() const = 0;
    virtual std::size_t get_nblocks() const = 0;
    virtual std::size_t get_m() const = 0;
    virtual std::size_t get_n() const = 0;
    virtual int* get_coord() const = 0;
    virtual void get_offs_lens(std::size_t*& r_offs, std::size_t*& r_lens,
                               std::size_t*& r_offs_l, std::size_t*& c_offs,
                               std::size_t*& c_lens,
                               std::size_t*& c_offs_l) const = 0;

    virtual int get_nprocs() const = 0;
    virtual void Start() = 0;
    virtual void End() = 0;

    // other BLAS and LAPACK routines
    //! Perform a `LAPACK-like` function which returns the value of the 1-norm,
    //! Frobenius norm, infinity-norm, or the largest absolute value of any
    //! element of a general rectangular matrix `A` with scalar type `T`.
    /*!
      @param norm: specifies the value to be returned in `lange`.
      @param m: the number of rows of the matrix A.
      @param n: the number of columns of the matrix A.
      @param A: A is an array of type `T`, dimension `(lda, n)`, the `m` by `n`
      matrix `A`.
      @param lda: the leading dimension of the array `A`.
      \return the value of a required type of norm of given matrix.
    */
    virtual Base<T> lange(char norm, std::size_t m, std::size_t n, T* A,
                          std::size_t lda) = 0;

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

    //! A `BLAS-like` function which forms the Generalized Matrix-Matrix
    //! production operation (`GEMM`).
    /*!
      `gemm_small` performs one of the matrix-matrix operations `C := alpha*op(
   A )*op( B ) + beta*C`, where `op(X)` is one of `op(X) = X` or `op(X) = X**T`.
   `alpha` and `beta` are scalars, and `A`, `B` and `C` are matrices, with
   `op(A)` an `m` by `k` matrix,  `op( B )`  a  `k` by `n` matrix and  `C` an
   `m` by `n` matrix. This function is implemented specifically for the matrices
   with relatively small `m`, `n` and `k`. In the current implementation of
   ChASE, this function performs a normal `gemm` without considering the sizes
   of matrices, but it will be improved in the later verison.
       @param[in] Layout: layout of matrices, currently only column-major layout
   is supported in ChASE.
       @param[in] transa: it specifies the form of `op( A )` to be used in the
   matrix multiplication. If its value is 1, the `op` is `conjugate transpose`
   operation, if its value is 2, the `op` is `transpose` operation and if its
   value is `3`, we have `op (A)=A`.
       @param[in] transb: it specifies the form of `op( B )` to be used in the
   matrix multiplication. If its value is 1, the `op` is `conjugate transpose`
   operation, if its value is 2, the `op` is `transpose` operation and if its
   value is `3`, we have `op (B)=B`.
       @param[in] m:  it  specifies  the number  of rows  of the  matrix `op( A
   )`  and of the  matrix  `C`.
       @param[in] n: it specifies the number of columns of the matrix `op( B )`
   and the number of columns of the matrix `C`.
       @param[in] k: it specifies  the number of columns of the matrix `op( A )`
   and the number of rows of the matrix `op( B )`
       @param[in] alpha: it specifies the scalar `alpha`.
       @param[in] a: an array of type `T`,  dimension `( lda, ka )`, where `ka`
   is `k`  when  `transa = 3`,  and is  `m`  otherwise. Before entry with
   `transa = 3`,  the leading  `m` by `k` part of the array  `a`  must contain
   the matrix  `a`,  otherwise the leading  `k` by `m` part of the array  `a`
   must contain  the matrix `a`.
       @param[in] lda: it specifies the first dimension of `a`. When `transa=3`,
   it must be at least `max(1, m)`, otherwise it must be at least `(1, k)`.
       @param[in] b: an array of type `T`, dimension `( ldb, kb )`, where `kb`
   is `n`  when  `transb=3`,  and is  `k`  otherwise. Before entry with
   `transb=3`,  the leading  `k` by `n` part of the array  `b`  must contain the
   matrix  `b`,  otherwise the leading  `n` by `k`  part of the array  `b`  must
   contain  the matrix `b`.
       @param[in] ldb: it specifies the first dimension of `b`. When `transa=3`,
   it must be at least `max(1, k)`, otherwise it must be at least `(1, n)`.
       @param[in] beta: it specifies the scalar `beta`.
       @param[in/out] `c`: an array of type `T`, dimension `( ldc, n )`.
       @param[in] ldc: it specifies the first dimension of `c`. It must be at
   least `max(1,m)`.
    */
    virtual void gemm(CBLAS_LAYOUT Layout, CBLAS_TRANSPOSE transa,
                      CBLAS_TRANSPOSE transb, std::size_t m, std::size_t n,
                      std::size_t k, T* alpha, T* a, std::size_t lda, T* b,
                      std::size_t ldb, T* beta, T* c, std::size_t ldc) = 0;

    //! A `LAPACK-like` function which computes selected eigenvalues and
    //! eigenvectors of a real symmetric tridiagonal matrix of **DOUBLE
    //! PRECISION**.
    /*!
      The routine computes selected eigenvalues and, optionally, eigenvectors of
      a real symmetric tridiagonal matrix `T`. Any such unreduced matrix has a
      well defined set of pairwise different real eigenvalues, the corresponding
      real eigenvectors are pairwise orthogonal. The spectrum may be computed
      either completely or partially by specifying either an interval `(vl,vu]`
      or a range of indices `il:iu` for the desired eigenvalues.
      @param[in] matrix_layout: layout of matrices, currently only column-major
      layout is supported in ChASE.
      @param[in] jobz: Must be `N` or `V`. If `jobz = N`, then only eigenvalues
      are computed, else if `jobz = V`, then eigenvalues and eigenvectors are
      computed.
      @param[in] range: Must be `A` or `V` or `I`. If `range = A`, the routine
      computes all eigenvalues. If `range = V`, the routine computes all
      eigenvalues in the half-open interval: `(vl, vu]`. If `range = I`, the
      routine computes eigenvalues with indices `il` to `iu`.
      @param[in] n: The order of the matrix `T`.
      @param[in/out] d: array, size `n`. It contains `n` diagonal elements of
      the tridiagonal matrix `T`. On exit, the array d is overwritten.
      @param[in/out] e: array, size `n`. It Contains `(n-1)` off-diagonal
      elements of the tridiagonal matrix `T` in elements `0` to `n-2` of `e`. On
      exit, the array d is overwritten.
      @param[in] vl&vu: If `range = V`, the lower and upper bounds of the
      interval to be searched for eigenvalues. Constraint: `vl<vu`.
      @param[in] il&iu: If `range = I`, the indices in ascending order of the
      smallest and largest eigenvalues to be returned. Constraint: `1≤il≤iu≤n`,
      if `n>0`. If `range = A` or `V`, `il` and `iu` are not referenced.
      @param[out] m: The total number of eigenvalues found, `0≤m≤n`. If `range =
      A`, then `m=n`, and if `range = I`, then `m=iu-il+1`.
      @param[out] w: Array, size `n`. The first `m` elements contain the
      selected eigenvalues in ascending order.
      @param[out] z: Array, size `max(1, ldz*m)` for column major layout. If
      `jobz = V`, and `info = 0`, then the first `m` columns of `z` contain the
      orthonormal eigenvectors of the matrix `T` corresponding to the selected
      eigenvalues, with the `i-th` column of `z` holding the eigenvector
      associated with `w(i)`. If `jobz = N`, then z is not referenced.
      @param[in] ldz: The leading dimension of the output array `z`. If `jobz =
      V`, then `ldz ≥ max(1, n)` for column major layout; `ldz ≥ 1` otherwise.
      @param[in] nzc: The number of eigenvectors to be held in the array `z`. If
      `range = A`, then `nzc≥max(1, n)`; If `range = V`, then `nzc` is greater
      than or equal to the number of eigenvalues in the half-open interval:
      `(vl, vu]`. If `range = I`, then `nzc≥iu-il+1`. If `nzc = -1`, then a
      workspace query is assumed; the routine calculates the number of columns
      of the array `z` that are needed to hold the eigenvectors.
      @param[out] isuppz: Array, size `(2*max(1, m))`. The support of the
      eigenvectors in `z`, that is the indices indicating the nonzero elements
      in `z`. The `i-th` computed eigenvector is nonzero only in elements
      `isuppz[2*i - 2]` through `isuppz[2*i - 1]`. This is relevant in the case
      when the matrix is split. `isuppz` is only accessed when `jobz = V` and
      `n>0`.
      @param[in/out] tryrac: On entry, If `tryrac` is `true`, it indicates that
      the code should check whether the tridiagonal matrix defines its
      eigenvalues to high relative accuracy. On exit, set to `true`. `tryrac` is
      set to `false` if the matrix does not define its eigenvalues to high
      relative accuracy.
    */
    virtual std::size_t stemr(int matrix_layout, char jobz, char range,
                              std::size_t n, double* d, double* e, double vl,
                              double vu, std::size_t il, std::size_t iu, int* m,
                              double* w, double* z, std::size_t ldz,
                              std::size_t nzc, int* isuppz,
                              lapack_logical* tryrac) = 0;

    //! A `LAPACK-like` function which computes selected eigenvalues and
    //! eigenvectors of a real symmetric tridiagonal matrix of **SINGLE
    //! PRECISION**.
    /*!
      The routine computes selected eigenvalues and, optionally, eigenvectors of
      a real symmetric tridiagonal matrix `T`. Any such unreduced matrix has a
      well defined set of pairwise different real eigenvalues, the corresponding
      real eigenvectors are pairwise orthogonal. The spectrum may be computed
      either completely or partially by specifying either an interval `(vl,vu]`
      or a range of indices `il:iu` for the desired eigenvalues.
      @param[in] matrix_layout: layout of matrices, currently only column-major
      layout is supported in ChASE.
      @param[in] jobz: Must be `N` or `V`. If `jobz = N`, then only eigenvalues
      are computed, else if `jobz = V`, then eigenvalues and eigenvectors are
      computed.
      @param[in] range: Must be `A` or `V` or `I`. If `range = A`, the routine
      computes all eigenvalues. If `range = V`, the routine computes all
      eigenvalues in the half-open interval: `(vl, vu]`. If `range = I`, the
      routine computes eigenvalues with indices `il` to `iu`.
      @param[in] n: The order of the matrix `T`.
      @param[in/out] d: array, size `n`. It contains `n` diagonal elements of
      the tridiagonal matrix `T`. On exit, the array d is overwritten.
      @param[in/out] e: array, size `n`. It Contains `(n-1)` off-diagonal
      elements of the tridiagonal matrix `T` in elements `0` to `n-2` of `e`. On
      exit, the array d is overwritten.
      @param[in] vl&vu: If `range = V`, the lower and upper bounds of the
      interval to be searched for eigenvalues. Constraint: `vl<vu`.
      @param[in] il&iu: If `range = I`, the indices in ascending order of the
      smallest and largest eigenvalues to be returned. Constraint: `1≤il≤iu≤n`,
      if `n>0`. If `range = A` or `V`, `il` and `iu` are not referenced.
      @param[out] m: The total number of eigenvalues found, `0≤m≤n`. If `range =
      A`, then `m=n`, and if `range = I`, then `m=iu-il+1`.
      @param[out] w: Array, size `n`. The first `m` elements contain the
      selected eigenvalues in ascending order.
      @param[out] z: Array, size `max(1, ldz*m)` for column major layout. If
      `jobz = V`, and `info = 0`, then the first `m` columns of `z` contain the
      orthonormal eigenvectors of the matrix `T` corresponding to the selected
      eigenvalues, with the `i-th` column of `z` holding the eigenvector
      associated with `w(i)`. If `jobz = N`, then z is not referenced.
      @param[in] ldz: The leading dimension of the output array `z`. If `jobz =
      V`, then `ldz ≥ max(1, n)` for column major layout; `ldz ≥ 1` otherwise.
      @param[in] nzc: The number of eigenvectors to be held in the array `z`. If
      `range = A`, then `nzc≥max(1, n)`; If `range = V`, then `nzc` is greater
      than or equal to the number of eigenvalues in the half-open interval:
      `(vl, vu]`. If `range = I`, then `nzc≥iu-il+1`. If `nzc = -1`, then a
      workspace query is assumed; the routine calculates the number of columns
      of the array `z` that are needed to hold the eigenvectors.
      @param[out] isuppz: Array, size `(2*max(1, m))`. The support of the
      eigenvectors in `z`, that is the indices indicating the nonzero elements
      in `z`. The `i-th` computed eigenvector is nonzero only in elements
      `isuppz[2*i - 2]` through `isuppz[2*i - 1]`. This is relevant in the case
      when the matrix is split. `isuppz` is only accessed when `jobz = V` and
      `n>0`.
      @param[in/out] tryrac: On entry, If `tryrac` is `true`, it indicates that
      the code should check whether the tridiagonal matrix defines its
      eigenvalues to high relative accuracy. On exit, set to `true`. `tryrac` is
      set to `false` if the matrix does not define its eigenvalues to high
      relative accuracy.
    */
    virtual std::size_t stemr(int matrix_layout, char jobz, char range,
                              std::size_t n, float* d, float* e, float vl,
                              float vu, std::size_t il, std::size_t iu, int* m,
                              float* w, float* z, std::size_t ldz,
                              std::size_t nzc, int* isuppz,
                              lapack_logical* tryrac) = 0;

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
                      std::size_t lda, T* b, std::size_t ldb, bool first = false) = 0;

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

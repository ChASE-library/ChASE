#pragma once

#include <cstdlib>
#include "algorithm/types.hpp"

namespace chase {
namespace mpi {

template <class T>
class ChaseMpiDLAInterface{
  public:
    virtual ~ChaseMpiDLAInterface(){};
    virtual Base<T> lange(char norm, std::size_t m, std::size_t n, T* A, std::size_t lda) = 0;
    //QR factorization and construct the unitary marix Q explicitly.
    virtual void gegqr(std::size_t N, std::size_t nevex, T * approxV_, std::size_t LDA) = 0;
    virtual void axpy(std::size_t N, T * alpha, T * x, std::size_t incx, T *y, std::size_t incy) = 0;
    virtual void scal(std::size_t N, T *a, T *x, std::size_t incx) = 0;
    virtual Base<T> nrm2(std::size_t n, T *x, std::size_t incx) = 0;
    virtual T dot(std::size_t n, T* x, std::size_t incx, T* y, std::size_t incy) = 0;
    virtual void gemm_small(CBLAS_LAYOUT Layout, CBLAS_TRANSPOSE transa,
            		 CBLAS_TRANSPOSE transb, std::size_t m,
            		 std::size_t n, std::size_t k, T* alpha,
            		 T* a, std::size_t lda, T* b,
            		 std::size_t ldb, T* beta, T* c, std::size_t ldc) = 0;

    virtual void gemm_large(CBLAS_LAYOUT Layout, CBLAS_TRANSPOSE transa,
                         CBLAS_TRANSPOSE transb, std::size_t m,
                         std::size_t n, std::size_t k, T* alpha,
                         T* a, std::size_t lda, T* b,
                         std::size_t ldb, T* beta, T* c, std::size_t ldc) = 0;

    virtual void RR_kernel(std::size_t N, std::size_t block, T *approxV, std::size_t locked, T *workspace, T One, T Zero, Base<T> *ritzv) = 0;

    virtual std::size_t stemr(int matrix_layout, char jobz, char range, std::size_t n,
                    double* d, double* e, double vl, double vu, std::size_t il, std::size_t iu,
                    int* m, double* w, double* z, std::size_t ldz, std::size_t nzc,
                    int* isuppz, lapack_logical* tryrac) = 0;

    virtual std::size_t stemr(int matrix_layout, char jobz, char range, std::size_t n,
                    float* d, float* e, float vl, float vu, std::size_t il, std::size_t iu,
                    int* m, float* w, float* z, std::size_t ldz, std::size_t nzc,
                    int* isuppz, lapack_logical* tryrac) = 0;

};


}
}

#pragma once
#include "ChASE-MPI/chase_mpidla_interface.hpp"

namespace chase {
namespace mpi {

template <class T>
class ChaseMpiDLA : public ChaseMpiDLAInterface<T>{
  public:
    ChaseMpiDLA(ChaseMpiDLAInterface<T> *dla):
	dla_(dla){
    }
    ~ChaseMpiDLA(){}

    Base<T> lange(char norm, std::size_t m, std::size_t n, T* A, std::size_t lda){
	return dla_->lange(norm, m, n, A, lda);
    }

    void gegqr(std::size_t N, std::size_t nevex, T * approxV, std::size_t LDA){
	dla_->gegqr(N, nevex, approxV, LDA);
    }

    void axpy(std::size_t N, T * alpha, T * x, std::size_t incx, T *y, std::size_t incy){
	dla_->axpy(N, alpha, x, incx, y, incy);
    }

    void scal(std::size_t N, T *a, T *x, std::size_t incx){
	dla_->scal(N, a, x, incx);
    }

    Base<T> nrm2(std::size_t n, T *x, std::size_t incx){
	return dla_->nrm2(n, x, incx);
    }

    T dot(std::size_t n, T* x, std::size_t incx, T* y, std::size_t incy){
	return dla_->dot(n, x, incx, y, incy);
    }

    void gemm_small(CBLAS_LAYOUT Layout, CBLAS_TRANSPOSE transa,
                         CBLAS_TRANSPOSE transb, std::size_t m,
                         std::size_t n, std::size_t k, T* alpha,
                         T* a, std::size_t lda, T* b,
                         std::size_t ldb, T* beta, T* c, std::size_t ldc)
    {
	dla_->gemm_small(Layout, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }

    void gemm_large(CBLAS_LAYOUT Layout, CBLAS_TRANSPOSE transa,
                         CBLAS_TRANSPOSE transb, std::size_t m,
                         std::size_t n, std::size_t k, T* alpha,
                         T* a, std::size_t lda, T* b,
                         std::size_t ldb, T* beta, T* c, std::size_t ldc)
    {
        dla_->gemm_large(Layout, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }

    void RR_kernel(std::size_t N, std::size_t block, T *approxV, std::size_t locked, T *workspace, T One, T Zero, Base<T> *ritzv){
	dla_->RR_kernel(N, block, approxV, locked, workspace, One, Zero, ritzv);	
    }

    std::size_t stemr(int matrix_layout, char jobz, char range, std::size_t n,
                    double* d, double* e, double vl, double vu, std::size_t il, std::size_t iu,
                    int* m, double* w, double* z, std::size_t ldz, std::size_t nzc,
                    int* isuppz, lapack_logical* tryrac){
	return dla_->stemr(matrix_layout, jobz, range, n, d, e, vl, vu, il, iu, m, w, z, ldz, nzc, isuppz, tryrac);
    }

    std::size_t stemr(int matrix_layout, char jobz, char range, std::size_t n,
                    float* d, float* e, float vl, float vu, std::size_t il, std::size_t iu,
                    int* m, float* w, float* z, std::size_t ldz, std::size_t nzc,
                    int* isuppz, lapack_logical* tryrac){
        return dla_->stemr(matrix_layout, jobz, range, n, d, e, vl, vu, il, iu, m, w, z, ldz, nzc, isuppz, tryrac);
   }
  private:
    std::unique_ptr<ChaseMpiDLAInterface<T>> dla_;

}; 

}  // namespace mpi
}  // namespace chase


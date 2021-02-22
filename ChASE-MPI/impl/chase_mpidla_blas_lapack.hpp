#pragma once
#include "ChASE-MPI/blas_templates.hpp"
#include "ChASE-MPI/chase_mpidla_interface.hpp"

namespace chase {
namespace mpi {

template <class T>
class ChaseMpiDLABlasLapack : public ChaseMpiDLAInterface<T> {
   public:
     ChaseMpiDLABlasLapack(){}
     ~ChaseMpiDLABlasLapack(){}

    Base<T> lange(char norm, std::size_t m, std::size_t n, T* A, std::size_t lda){
        return t_lange(norm, m, n, A, lda);
    }

     void gegqr(std::size_t N, std::size_t nevex, T * approxV, std::size_t LDA){
	auto tau = std::unique_ptr<T[]> {
    	  new T[ nevex ]
        };
	t_geqrf(LAPACK_COL_MAJOR, N, nevex, approxV, LDA, tau.get());
        t_gqr(LAPACK_COL_MAJOR, N, nevex, nevex, approxV, LDA, tau.get());
     }

     void axpy(std::size_t N, T * alpha, T * x, std::size_t incx, T *y, std::size_t incy){
	t_axpy(N, alpha, x, incx, y, incy);
     }

     void scal(std::size_t N, T *a, T *x, std::size_t incx){
         t_scal(N, a, x, incx);
     }

     Base<T> nrm2(std::size_t n, T *x, std::size_t incx){
     	 return t_nrm2(n, x, incx);
     }

     T dot(std::size_t n, T* x, std::size_t incx, T* y, std::size_t incy){
         return t_dot(n, x, incx, y, incy);
     }

    void gemm_small(CBLAS_LAYOUT Layout, CBLAS_TRANSPOSE transa,
                         CBLAS_TRANSPOSE transb, std::size_t m,
                         std::size_t n, std::size_t k, T* alpha,
                         T* a, std::size_t lda, T* b,
                         std::size_t ldb, T* beta, T* c, std::size_t ldc)
    {
        t_gemm(Layout, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }

    void gemm_large(CBLAS_LAYOUT Layout, CBLAS_TRANSPOSE transa,
                         CBLAS_TRANSPOSE transb, std::size_t m,
                         std::size_t n, std::size_t k, T* alpha,
                         T* a, std::size_t lda, T* b,
                         std::size_t ldb, T* beta, T* c, std::size_t ldc)
    {
        t_gemm(Layout, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }

    void RR_kernel(std::size_t N, std::size_t block, T *approxV, std::size_t locked, T *workspace, T One, T Zero, Base<T> *ritzv){
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


    std::size_t stemr(int matrix_layout, char jobz, char range, std::size_t n,
                    double* d, double* e, double vl, double vu, std::size_t il, std::size_t iu,
                    int* m, double* w, double* z, std::size_t ldz, std::size_t nzc,
                    int* isuppz, lapack_logical* tryrac){
        return t_stemr<double>(matrix_layout, jobz, range, n, d, e, vl, vu, il, iu, m, w, z, ldz, nzc, isuppz, tryrac);
    }

    std::size_t stemr(int matrix_layout, char jobz, char range, std::size_t n,
                    float* d, float* e, float vl, float vu, std::size_t il, std::size_t iu,
                    int* m, float* w, float* z, std::size_t ldz, std::size_t nzc,
                    int* isuppz, lapack_logical* tryrac){
        return t_stemr<float>(matrix_layout, jobz, range, n, d, e, vl, vu, il, iu, m, w, z, ldz, nzc, isuppz, tryrac);
   }

};

}  // namespace mpi
}  // namespace chase


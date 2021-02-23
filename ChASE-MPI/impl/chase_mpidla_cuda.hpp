#pragma once

#include <assert.h>
#include <cuComplex.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <complex>
#include <cuda_profiler_api.h>

#include "ChASE-MPI/blas_templates.hpp"
#include "ChASE-MPI/chase_mpidla_interface.hpp"

namespace chase {
namespace mpi {

template <class T>
class ChaseMpiDLACUDA : public ChaseMpiDLAInterface<T> {
   public:
     ChaseMpiDLACUDA(){
	MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shmcomm_);
	MPI_Comm_size(shmcomm_, &shmsize_);
	MPI_Comm_rank(shmcomm_, &shmrank_);
	MPI_Comm_rank(MPI_COMM_WORLD, &globalrank_);
	cuda_exec(cudaGetDeviceCount(&num_devices_));
        num_devices_per_rank_ = num_devices_ / shmsize_;
	//std::cout << "num_devices = " << num_devices_ << ", num_devices_per_rank_ = " <<
	//       num_devices_per_rank_ << std::endl;
	cusolverH_ = (cusolverDnHandle_t*) malloc(num_devices_per_rank_ * sizeof(cusolverDnHandle_t));
        cublasH_ = (cublasHandle_t*) malloc(num_devices_per_rank_ * sizeof(cublasHandle_t));

        for (int dev=0; dev < num_devices_per_rank_; dev++) {
            cuda_exec(cudaSetDevice(dev));
            cusolverDnCreate(&cusolverH_[dev]);
            cublasCreate(&cublasH_[dev]);
        }
     }

     ~ChaseMpiDLACUDA(){
	for (int dev=0; dev < num_devices_per_rank_; dev++) {
	    if (cusolverH_[dev]) cusolverDnDestroy(cusolverH_[dev]);
            if (cublasH_[dev]) cublasDestroy(cublasH_[dev]);	
	}
     }
     
    Base<T> lange(char norm, std::size_t m, std::size_t n, T* A, std::size_t lda){
        return t_lange(norm, m, n, A, lda);
    }

    void gegqr(std::size_t N, std::size_t nevex, T * approxV, std::size_t LDA){

        T *d_V = NULL;
    	T *d_tau = NULL;
    	int *devInfo = NULL;
    	T *d_work = NULL;

        int lwork_geqrf = 0;
        int lwork_orgqr = 0;
        int lwork = 0;

        int info_gpu = 0;
        cudaSetDevice(0); 	
	cuda_exec(cudaMalloc ((void**)&d_V  , sizeof(T)*N*nevex));
        cuda_exec(cudaMalloc ((void**)&d_tau  , sizeof(T)*nevex));
        cuda_exec(cudaMalloc ((void**)&devInfo  , sizeof(int)));
        cudaSetDevice(0);	
	cuda_exec(cudaMemcpy(d_V, approxV, sizeof(T)*N*nevex, cudaMemcpyHostToDevice));
        cudaSetDevice(0);
        cusolver_status_ = cusolverDnTgeqrf_bufferSize(
            cusolverH_[0],
            N,
            nevex,
            d_V,
            LDA,
            &lwork_geqrf);
        assert (cusolver_status_ == CUSOLVER_STATUS_SUCCESS);
	cudaSetDevice(0);
	cusolver_status_ = cusolverDnTgqr_bufferSize(
            cusolverH_[0],
            N,
            nevex,
            nevex,
            d_V,
            LDA,
	    d_tau,
            &lwork_orgqr);
        assert (cusolver_status_ == CUSOLVER_STATUS_SUCCESS);

	lwork = (lwork_geqrf > lwork_orgqr)? lwork_geqrf : lwork_orgqr;
        cudaSetDevice(0);
        cuda_exec(cudaMalloc((void**)&d_work, sizeof(T)*lwork));
        cudaSetDevice(0);
        cusolver_status_ = cusolverDnTgeqrf(
            cusolverH_[0],
            N,
            nevex,
            d_V,
            LDA,
            d_tau,
            d_work,
            lwork,
            devInfo);
        assert(CUSOLVER_STATUS_SUCCESS == cusolver_status_);
        cudaSetDevice(0);
	cusolver_status_ = cusolverDnTgqr(
            cusolverH_[0],
            N,
            nevex,
            nevex,
            d_V,
            LDA,
            d_tau,
            d_work,
            lwork,
            devInfo);
        assert(CUSOLVER_STATUS_SUCCESS == cusolver_status_);
        cudaSetDevice(0);
        cuda_exec(cudaMemcpy(approxV, d_V, sizeof(T)*N*nevex, cudaMemcpyDeviceToHost));

	if (d_V    ) cudaFree(d_V);
        if (d_tau  ) cudaFree(d_tau);
        if (devInfo) cudaFree(devInfo);
        if (d_work ) cudaFree(d_work);

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
 
      T *d_V = NULL;
      T *d_W = NULL;
      T *d_A = NULL;
      Base<T> *d_ritz = NULL;
      T *d_work = NULL;
      int *devInfo = NULL;
      int lwork = 0;
      int info_gpu = 0;

      cudaSetDevice(0);
      cuda_exec(cudaMalloc ((void**)&d_V, sizeof(T) * N * block));
      cuda_exec(cudaMalloc ((void**)&d_W, sizeof(T) * N * block));
      cuda_exec(cudaMalloc ((void**)&d_A, sizeof(T) * block * block));
      cudaSetDevice(0);
      cuda_exec(cudaMemcpy(d_V, approxV + locked * N, sizeof(T)* N * block, cudaMemcpyHostToDevice));
      cuda_exec(cudaMemcpy(d_W, workspace + locked * N, sizeof(T)* N * block, cudaMemcpyHostToDevice));
      cudaSetDevice(0);
      cublas_status_ = cublasTgemm(
	cublasH_[0],
	CUBLAS_OP_C,
	CUBLAS_OP_N,
	block,
	block,
	N,
	&One,
        d_V, N,
        d_W, N,
        &Zero,
        d_A, 
	block);
      assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);
      //HEEVD
      cudaSetDevice(0);
      cuda_exec(cudaMalloc ((void**)&d_ritz, sizeof(Base<T>) * block));
      cuda_exec(cudaMalloc ((void**)&devInfo, sizeof(int)));
      cudaSetDevice(0);
      cusolver_status_ = cusolverDnTheevd_bufferSize(
            cusolverH_[0],
            CUSOLVER_EIG_MODE_VECTOR,
	    CUBLAS_FILL_MODE_LOWER,
            block,
            d_A,
            block,
	    d_ritz,
            &lwork);	
      assert (cusolver_status_ == CUSOLVER_STATUS_SUCCESS);
      cudaSetDevice(0);
      cuda_exec(cudaMalloc((void**)&d_work, sizeof(T)*lwork));
      cudaSetDevice(0);
      cusolver_status_ = cusolverDnTheevd(
            cusolverH_[0],
            CUSOLVER_EIG_MODE_VECTOR,
            CUBLAS_FILL_MODE_LOWER,
            block,
            d_A,
            block,
            d_ritz,
	    d_work,
	    lwork,
            devInfo);	
      assert (cusolver_status_ == CUSOLVER_STATUS_SUCCESS);
      cudaSetDevice(0);
      cublas_status_ = cublasTgemm(
        cublasH_[0],
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        N,
        block,
        block,
        &One,
        d_V, N,
        d_A, block,
        &Zero,
        d_W,
        N);
      assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);
      cudaSetDevice(0);
      cuda_exec(cudaMemcpy(approxV + locked * N, d_V, sizeof(T)* N * block, cudaMemcpyDeviceToHost));
      cuda_exec(cudaMemcpy(workspace + locked * N, d_W, sizeof(T)* N * block, cudaMemcpyDeviceToHost));
      cuda_exec(cudaMemcpy(ritzv, d_ritz, sizeof(Base<T>) * block, cudaMemcpyDeviceToHost));

      if (d_A) cudaFree(d_A);
      if (d_W) cudaFree(d_W);
      if (d_V) cudaFree(d_V);
      if (d_ritz) cudaFree(d_ritz);
      if (devInfo) cudaFree(devInfo);
      if (d_work) cudaFree(d_work);
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
   
   private:
	cublasStatus_t cublas_status_ = CUBLAS_STATUS_SUCCESS;
        cusolverStatus_t cusolver_status_ = CUSOLVER_STATUS_SUCCESS;

	cusolverDnHandle_t *cusolverH_ = nullptr;
        cublasHandle_t *cublasH_ = nullptr;

	int num_devices_;

	MPI_Comm shmcomm_;
	int shmsize_;
	int shmrank_;
	int globalrank_;
	int num_devices_per_rank_;
};

}  // namespace mpi
}  // namespace chase


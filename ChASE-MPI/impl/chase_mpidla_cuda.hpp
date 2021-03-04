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
     ChaseMpiDLACUDA(ChaseMpiProperties<T>* matrix_properties){

     	matrix_properties_ = matrix_properties;
	N_ = matrix_properties_->get_N();
        nev_ = matrix_properties_->GetNev();	
        nex_ = matrix_properties_->GetNex();	

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
            cuda_exec(cudaSetDevice(shmrank_*num_devices_per_rank_ + dev));
            cusolverDnCreate(&cusolverH_[dev]);
            cublasCreate(&cublasH_[dev]);
        }

	cudaSetDevice(shmrank_*num_devices_per_rank_);
        cuda_exec(cudaMalloc ((void**)&devInfo_, sizeof(int)));
        cuda_exec(cudaMalloc ((void**)&d_V_  , sizeof(T)*N_*(nev_ + nex_)));
        cuda_exec(cudaMalloc ((void**)&d_return_  , sizeof(T)*(nev_ + nex_)));

        int lwork_geqrf = 0;
        int lwork_orgqr = 0;
        cudaSetDevice(shmrank_*num_devices_per_rank_);
	cusolver_status_ = cusolverDnTgeqrf_bufferSize(
            cusolverH_[0],
            N_,
            nev_ + nex_,
            d_V_,
            N_,
            &lwork_geqrf);
        assert (cusolver_status_ == CUSOLVER_STATUS_SUCCESS);
        cudaSetDevice(shmrank_*num_devices_per_rank_);
	cusolver_status_ = cusolverDnTgqr_bufferSize(
            cusolverH_[0],
            N_,
            nev_ + nex_,
            nev_ + nex_,
            d_V_,
            N_,
            d_return_,
            &lwork_orgqr);
        assert (cusolver_status_ == CUSOLVER_STATUS_SUCCESS);

        lwork_ = (lwork_geqrf > lwork_orgqr)? lwork_geqrf : lwork_orgqr;

	cudaSetDevice(shmrank_*num_devices_per_rank_);
        cuda_exec(cudaMalloc((void**)&d_work_, sizeof(T)*lwork_));

     }

     ~ChaseMpiDLACUDA(){

     	for (int dev=0; dev < num_devices_per_rank_; dev++) {
	    if (cusolverH_[dev]) cusolverDnDestroy(cusolverH_[dev]);
            if (cublasH_[dev]) cublasDestroy(cublasH_[dev]);	
	}

	if (devInfo_) cudaFree(devInfo_);
        if (d_V_) cudaFree(d_V_);
	if (d_return_) cudaFree(d_return_);
        if (d_work_) cudaFree(d_work_);	
     }
     
    Base<T> lange(char norm, std::size_t m, std::size_t n, T* A, std::size_t lda){
        return t_lange(norm, m, n, A, lda);
    }

    void gegqr(std::size_t N, std::size_t nevex, T * approxV, std::size_t LDA){

    	cudaSetDevice(shmrank_*num_devices_per_rank_);
	cuda_exec(cudaMemcpy(d_V_, approxV, sizeof(T)*N*nevex, cudaMemcpyHostToDevice));
        cudaSetDevice(shmrank_*num_devices_per_rank_);
	cusolver_status_ = cusolverDnTgeqrf(
            cusolverH_[0],
            N,
            nevex,
            d_V_,
            LDA,
            d_return_,
            d_work_,
            lwork_,
            devInfo_);
        assert(CUSOLVER_STATUS_SUCCESS == cusolver_status_);
        cudaSetDevice(shmrank_*num_devices_per_rank_);
	cusolver_status_ = cusolverDnTgqr(
            cusolverH_[0],
            N,
            nevex,
            nevex,
            d_V_,
            LDA,
            d_return_,
            d_work_,
            lwork_,
            devInfo_);
        assert(CUSOLVER_STATUS_SUCCESS == cusolver_status_);
        cudaSetDevice(shmrank_*num_devices_per_rank_);
	cuda_exec(cudaMemcpy(approxV, d_V_, sizeof(T)*N*nevex, cudaMemcpyDeviceToHost));
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

        T *d_A_ = NULL;
        T *d_W_ = NULL;

        cudaSetDevice(shmrank_*num_devices_per_rank_);	
        cuda_exec(cudaMalloc ((void**)&d_W_, sizeof(T) * N * block));
        cuda_exec(cudaMalloc ((void**)&d_A_, sizeof(T) * block * block));
        cudaSetDevice(shmrank_*num_devices_per_rank_);
	cuda_exec(cudaMemcpy(d_V_, approxV + locked * N, sizeof(T)* N * block, cudaMemcpyHostToDevice));
        cuda_exec(cudaMemcpy(d_W_, workspace + locked * N, sizeof(T)* N * block, cudaMemcpyHostToDevice));
        cudaSetDevice(shmrank_*num_devices_per_rank_);
	cublas_status_ = cublasTgemm(
	  cublasH_[0],
	  CUBLAS_OP_C,
	  CUBLAS_OP_N,
	  block,
	  block,
	  N,
	  &One,
          d_V_, N,
          d_W_, N,
          &Zero,
          d_A_, 
	  block);
        assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);
        //HEEVD

        cuda_exec(cudaMemcpy(A, d_A_, sizeof(T)* block * block, cudaMemcpyDeviceToHost));	
        t_heevd(LAPACK_COL_MAJOR, 'V', 'L', block, A, block, ritzv);
        cuda_exec(cudaMemcpy(d_A_, A, sizeof(T)* block * block, cudaMemcpyHostToDevice));

	cudaSetDevice(shmrank_*num_devices_per_rank_);
      	cublas_status_ = cublasTgemm(
            cublasH_[0],
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            N,
            block,
            block,
            &One,
            d_V_, N,
            d_A_, block,
            &Zero,
            d_W_,
            N);
      	assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);
        cudaSetDevice(shmrank_*num_devices_per_rank_);
  	cuda_exec(cudaMemcpy(approxV + locked * N, d_V_, sizeof(T)* N * block, cudaMemcpyDeviceToHost));
      	cuda_exec(cudaMemcpy(workspace + locked * N, d_W_, sizeof(T)* N * block, cudaMemcpyDeviceToHost));

      	if (d_A_) cudaFree(d_A_);
      	if (d_W_) cudaFree(d_W_);
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

	int *devInfo_ = NULL;
      	T *d_V_ = NULL;
	T *d_return_ = NULL;
	T *d_work_ = NULL;
        int lwork_ = 0;

	ChaseMpiProperties<T>* matrix_properties_;

	std::size_t N_;
  	std::size_t nev_;
  	std::size_t nex_;
};

}  // namespace mpi
}  // namespace chase


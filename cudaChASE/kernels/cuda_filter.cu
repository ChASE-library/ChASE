
#include <cuda_runtime.h>
#include <cuComplex.h>
#include "../include/cuda_util.h"
#include "cublas_v2.h"
#include <omp.h>

#define BLOCKDIM 256
//#define PROFILE

//A += B
__global__ void add_matrix(cuDoubleComplex* A, int lda, const cuDoubleComplex *B, int ldb, int m, int n) //TODO use 2D blocks
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if(idx < m){
      for(int i=0;i < n ; ++i){
         A[idx + i * lda ].x += B[idx + i * ldb ].x;
         A[idx + i * lda ].y += B[idx + i * ldb ].y;
      }
   }

}

__global__ void zshift_matrix(cuDoubleComplex* A, int lda, int n, double shift, int offset)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if(idx  >= offset  && idx < n) //TODO verify correctness in multi-gpu case
      A[(idx-offset) * lda + idx].x += shift;
}

void cuda_filter(double _Complex *A, double _Complex*x, int n, int m,
            int deg, double lambda_1, double lower, double upper,
            double _Complex*y, gpu_data_t *gpuData)
{
  DEBUG_PRINT("enter filter...\n");

  cuDoubleComplex zero;
  zero.x = 0.0;
  zero.y = 0.0;
  double _Complex *tmp = NULL;
  cuDoubleComplex alpha, beta;
  double c = (upper + lower) / 2;
  double e = (upper - lower) / 2;
  double sigma_1   = e / (lambda_1 - c);
  double sigma     = sigma_1;
  double sigma_new;
  int i = 0;

  //--------------------------------- A = A-cI -------------------------------------
  int num_blocks = (n+(BLOCKDIM-1))/BLOCKDIM;
  //A = A-cI  
  cudaSetDevice(0);
  zshift_matrix<<<num_blocks, BLOCKDIM,0,gpuData->stream0>>>(gpuData->H_d, n, n, -c, 0);
#ifdef MULTI_GPU
  cudaSetDevice(1);
  zshift_matrix<<<num_blocks, BLOCKDIM,0,gpuData->stream1>>>(gpuData->H1_d, n, n, -c, gpuData->work_per_gpu[0]);
#endif
  //--------------------------------------------------------------------------------

  //---------------------------- y = alpha*(A-cI)*x --------------------------------
  alpha.x = sigma_1 / e;
  alpha.y = 0.0;
  beta.x = 0.0;
  beta.y = 0.0;

  //---------------------------- HtD transfer --------------------------------
  cudaSetDevice(0);
  HANDLE_ERROR( cudaMemcpyAsync(gpuData->V_d, x, gpuData->sizeV, cudaMemcpyHostToDevice,gpuData->stream0)); 
  HANDLE_ERROR( cudaMemcpyAsync(gpuData->W_d, y, gpuData->sizeW, cudaMemcpyHostToDevice,gpuData->stream0)); 
#ifdef MULTI_GPU
  cudaSetDevice(1);
  HANDLE_ERROR( cudaMemcpyAsync(gpuData->V1_d, x, gpuData->sizeV, cudaMemcpyHostToDevice,gpuData->stream1)); 
  HANDLE_ERROR( cudaMemcpyAsync(gpuData->W1_d, y, gpuData->sizeW, cudaMemcpyHostToDevice,gpuData->stream1)); 
#endif
  cudaSetDevice(0);
  cudaDeviceSynchronize();
  //--------------------------------------------------------------------------------

#ifdef PROFILE
     cudaDeviceSynchronize();
     double start_time = omp_get_wtime();
#endif

#ifdef MULTI_GPU
  cudaSetDevice(0);
  HANDLE_ERROR_CUBLAS( cublasZgemm( gpuData->handle,
           CUBLAS_OP_N, CUBLAS_OP_N,
           n, m, gpuData->work_per_gpu[0], &alpha,
           gpuData->H_d, n, gpuData->V_d, n, &beta, gpuData->W_d, n) );

  cudaSetDevice(1);
  HANDLE_ERROR_CUBLAS( cublasZgemm(   gpuData->handle1,
           CUBLAS_OP_N, CUBLAS_OP_N,
           n, m, gpuData->work_per_gpu[1], &alpha,
           gpuData->H1_d, n, gpuData->V1_d + gpuData->work_per_gpu[0], n, &zero, gpuData->W1_d, n) );

  HANDLE_ERROR( cudaMemcpyAsync(gpuData->tmp1_d, gpuData->W_d, gpuData->sizeW, cudaMemcpyDeviceToDevice, gpuData->stream0)); 
  cudaEventRecord ( gpuData->event0 , gpuData->stream0 );
  cudaSetDevice(1);
  HANDLE_ERROR( cudaMemcpyAsync(gpuData->tmp0_d, gpuData->W1_d, gpuData->sizeW, cudaMemcpyDeviceToDevice, gpuData->stream1)); 
  cudaEventRecord ( gpuData->event1 , gpuData->stream1 );

  int num_blocks_x = (n+(BLOCKDIM-1))/BLOCKDIM;
  cudaSetDevice(1);
  cudaEventSynchronize ( gpuData->event0);
  add_matrix<<<num_blocks_x, BLOCKDIM,0, gpuData->stream1>>>(gpuData->W1_d, n,
        gpuData->tmp1_d, n, n, m);
  cudaSetDevice(0);
  cudaEventSynchronize ( gpuData->event1);
  add_matrix<<<num_blocks_x, BLOCKDIM,0, gpuData->stream0>>>(gpuData->W_d, n,
        gpuData->tmp0_d, n, n, m);

  cudaDeviceSynchronize();
  cudaSetDevice(1);
#else
  HANDLE_ERROR_CUBLAS( cublasZgemm(   gpuData->handle,
           CUBLAS_OP_N, CUBLAS_OP_N,
           n, m, n, &alpha,
           gpuData->H_d, n, gpuData->V_d, n, &beta, gpuData->W_d, n) );
#endif
//  ZGEMM("N", "N", &n, &m, &n, (const double _Complex*)&alpha, (const double _Complex*)(A), &n,
//        (const double _Complex*)x, &n, (const double _Complex*)&beta, y, &n);
  //--------------------------------------------------------------------------------
#ifdef PROFILE
     cudaDeviceSynchronize();
     //stop timing
     double end_time = omp_get_wtime();
     printf("GFLOPS: %dx%dx%d: %f\n",n,m,n,((float)n*m*n*8)/(end_time - start_time)/1e9);
#endif

  for(i = 2; i <= deg; ++i)
  {
     sigma_new = 1.0 / ( 2.0/sigma_1 - sigma );

     //---------------------- x = alpha(A-cI)y + beta*x ---------------------------
     alpha.x = 2.0*sigma_new / e;
     alpha.y = 0.0;
     beta.x  = -sigma * sigma_new;
     beta.y  = 0.0;

#ifdef PROFILE
     cudaDeviceSynchronize();
     double start_time = omp_get_wtime();
#endif

#ifdef MULTI_GPU
     cudaSetDevice(0);
     HANDLE_ERROR_CUBLAS( cublasZgemm( gpuData->handle,
              CUBLAS_OP_N, CUBLAS_OP_N,
              n, m, gpuData->work_per_gpu[0], &alpha,
              gpuData->H_d, n, gpuData->W_d, n, &beta, gpuData->V_d, n) );

     cudaSetDevice(1);
     HANDLE_ERROR_CUBLAS( cublasZgemm(   gpuData->handle1,
              CUBLAS_OP_N, CUBLAS_OP_N,
              n, m, gpuData->work_per_gpu[1], &alpha,
              gpuData->H1_d, n, gpuData->W1_d + gpuData->work_per_gpu[0], n, &zero, gpuData->V1_d, n) );

     HANDLE_ERROR( cudaMemcpyAsync(gpuData->tmp1_d, gpuData->V_d, gpuData->sizeV, cudaMemcpyDeviceToDevice, gpuData->stream0)); 
     cudaEventRecord ( gpuData->event0 , gpuData->stream0 );
     cudaSetDevice(1);
     HANDLE_ERROR( cudaMemcpyAsync(gpuData->tmp0_d, gpuData->V1_d, gpuData->sizeV, cudaMemcpyDeviceToDevice, gpuData->stream1)); 
     cudaEventRecord ( gpuData->event1 , gpuData->stream1 );

     int num_blocks_x = (n+(BLOCKDIM-1))/BLOCKDIM;
     cudaSetDevice(1);
     cudaEventSynchronize ( gpuData->event0);
     add_matrix<<<num_blocks_x, BLOCKDIM,0, gpuData->stream1>>>(gpuData->V1_d, n,
           gpuData->tmp1_d, n, n, m);
     cudaSetDevice(0);
     cudaEventSynchronize ( gpuData->event1);
     add_matrix<<<num_blocks_x, BLOCKDIM,0, gpuData->stream0>>>(gpuData->V_d, n,
           gpuData->tmp0_d, n, n, m);

     cudaDeviceSynchronize();
     cudaSetDevice(1);

#else
     HANDLE_ERROR_CUBLAS( cublasZgemm(   gpuData->handle,
              CUBLAS_OP_N, CUBLAS_OP_N,
              n, m, n, &alpha,
              gpuData->H_d, n, gpuData->W_d, n, &beta, gpuData->V_d, n) );
#endif

#ifdef PROFILE
     cudaDeviceSynchronize();
     //stop timing
     double end_time = omp_get_wtime();
     printf("GFLOPS: %dx%dx%d: %f\n",n,m,n,((float)n*m*n*8)/(end_time - start_time)/1e9);
#endif

     //      ZGEMM("N", "N", &n, &m, &n, (const double _Complex*)&alpha, (const double _Complex*)(A), &n,
     //            (const double _Complex*)y, &n, (const double _Complex*)&beta, x, &n);
     //----------------------------------------------------------------------------

     tmp = x; x = y; y = tmp; 
     cuDoubleComplex *tmp0 = gpuData->V_d; gpuData->V_d = gpuData->W_d; gpuData->W_d = tmp0;
     cuDoubleComplex *tmp1 = gpuData->V1_d; gpuData->V1_d = gpuData->W1_d; gpuData->W1_d = tmp1;

     sigma = sigma_new;      
  } // for(i = 2; i <= deg; ++i)
  

  //---------------------------- DtH transfer --------------------------------
  cudaSetDevice(0);
  HANDLE_ERROR( cudaMemcpyAsync(x, gpuData->V_d, gpuData->sizeV, cudaMemcpyDeviceToHost,
           gpuData->stream0) ); //TODO remove
  HANDLE_ERROR( cudaMemcpyAsync(y, gpuData->W_d, gpuData->sizeW, cudaMemcpyDeviceToHost,
           gpuData->stream0) );
   cudaDeviceSynchronize(); //TODO remove?
  //--------------------------------------------------------------------------
  
  //-----------------------------------RESTORE-A------------------------------------
  cudaSetDevice(0);
  zshift_matrix<<<num_blocks, BLOCKDIM, 0, gpuData->stream0>>>(gpuData->H_d, n, n, c, 0); 
#ifdef MULTI_GPU
  cudaSetDevice(1);
  zshift_matrix<<<num_blocks, BLOCKDIM,0,gpuData->stream1>>>(gpuData->H1_d, n, n, c, gpuData->work_per_gpu[0]);
#endif
  //--------------------------------------------------------------------------------
  cudaSetDevice(0);

  DEBUG_PRINT("leaving filter...\n");
  return;
}


void cuda_filterModified(double _Complex *A, double _Complex *x, int n, int m, int nev,
                    int M, int *deg, double lambda_1, double lower, double upper,
                    double _Complex *y, int block, gpu_data_t *gpuData)
{
   cuDoubleComplex zero;
   zero.x = 0.0;
   zero.y = 0.0;
   double _Complex *tmp = NULL;
   cuDoubleComplex alpha, beta;
   double c = (upper + lower) / 2;
   double e = (upper - lower) / 2;
   double sigma_1   = e / (lambda_1 - c);
   double sigma     = sigma_1;
   double sigma_new;
   int i, j = 0;
   int opt = 0;

   cudaDeviceSynchronize(); //TODO remove?
   //----------------------------------- A = A-cI ------------------------------------
   int num_blocks = (n+(BLOCKDIM-1))/BLOCKDIM;
   //A = A-cI  
   cudaSetDevice(0);
   zshift_matrix<<<num_blocks, BLOCKDIM,0,gpuData->stream0>>>(gpuData->H_d, n, n, -c, 0);
#ifdef MULTI_GPU
   cudaSetDevice(1);
   zshift_matrix<<<num_blocks, BLOCKDIM,0,gpuData->stream1>>>(gpuData->H1_d, n, n, -c, gpuData->work_per_gpu[0]);
#endif
   //---------------------------------------------------------------------------------

   //------------------------------- y = alpha*(A-cI)*x ------------------------------
   alpha.x = sigma_1 / e;
   alpha.y = 0.0;
   beta.x = 0.0;
   beta.y = 0.0;

   //---------------------------- HtD transfer --------------------------------
   cudaSetDevice(0);
   HANDLE_ERROR( cudaMemcpyAsync(gpuData->V_d, x, gpuData->sizeV, cudaMemcpyHostToDevice,gpuData->stream0)); 
   HANDLE_ERROR( cudaMemcpyAsync(gpuData->W_d, y, gpuData->sizeW, cudaMemcpyHostToDevice,gpuData->stream0)); 
#ifdef MULTI_GPU
   cudaSetDevice(1);
   HANDLE_ERROR( cudaMemcpyAsync(gpuData->V1_d, x, gpuData->sizeV, cudaMemcpyHostToDevice,gpuData->stream1)); 
   HANDLE_ERROR( cudaMemcpyAsync(gpuData->W1_d, y, gpuData->sizeW, cudaMemcpyHostToDevice,gpuData->stream1)); 
#endif
   cudaSetDevice(0);
   cudaDeviceSynchronize(); //TODO remove?
   //--------------------------------------------------------------------------------

#ifdef PROFILE
     cudaDeviceSynchronize();
     double start_time = omp_get_wtime();
#endif

#ifdef MULTI_GPU
   cudaSetDevice(0);
   HANDLE_ERROR_CUBLAS( cublasZgemm( gpuData->handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, m, gpuData->work_per_gpu[0], &alpha,
            gpuData->H_d, n, gpuData->V_d, n, &beta, gpuData->W_d, n) );

   cudaSetDevice(1);
   HANDLE_ERROR_CUBLAS( cublasZgemm(   gpuData->handle1,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, m, gpuData->work_per_gpu[1], &alpha,
            gpuData->H1_d, n, gpuData->V1_d + gpuData->work_per_gpu[0], n, &zero, gpuData->W1_d, n) );

   HANDLE_ERROR( cudaMemcpyAsync(gpuData->tmp1_d, gpuData->W_d, gpuData->sizeW, cudaMemcpyDeviceToDevice, gpuData->stream0)); 
   cudaEventRecord ( gpuData->event0 , gpuData->stream0 );
   cudaSetDevice(1);
   HANDLE_ERROR( cudaMemcpyAsync(gpuData->tmp0_d, gpuData->W1_d, gpuData->sizeW, cudaMemcpyDeviceToDevice, gpuData->stream1)); 
   cudaEventRecord ( gpuData->event1 , gpuData->stream1 );

   int num_blocks_x = (n+(BLOCKDIM-1))/BLOCKDIM;
   cudaSetDevice(1);
   cudaEventSynchronize ( gpuData->event0);
   add_matrix<<<num_blocks_x, BLOCKDIM,0, gpuData->stream1>>>(gpuData->W1_d, n,
         gpuData->tmp1_d, n, n, m);
   cudaSetDevice(0);
   cudaEventSynchronize ( gpuData->event1);
   add_matrix<<<num_blocks_x, BLOCKDIM,0, gpuData->stream0>>>(gpuData->W_d, n,
         gpuData->tmp0_d, n, n, m);

   cudaDeviceSynchronize();
   cudaSetDevice(1);
#else
   HANDLE_ERROR_CUBLAS( cublasZgemm(   gpuData->handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, m, n, &alpha,
            gpuData->H_d, n, gpuData->V_d, n, &beta, gpuData->W_d, n) );
#endif
//   ZGEMM("N", "N", &n, &m, &n, (const double _Complex*)&alpha, (const double _Complex*)(A), &n,
//         (const double _Complex*)x, &n, (const double _Complex*)&beta, y , &n);
   //---------------------------------------------------------------------------------
#ifdef PROFILE
     cudaDeviceSynchronize();
     //stop timing
     double end_time = omp_get_wtime();
     printf("GFLOPS: %dx%dx%d: %f\n",n,m,n,((float)n*m*n*8)/(end_time - start_time)/1e9);
#endif

   while(j < block && deg[j++] == 1)
      ++opt;
   --j;
   m -= opt;


   for(i = 2; i <= M; ++i)
   {
      sigma_new = 1.0 / ( 2.0/sigma_1 - sigma );

      //----------------------- x = alpha(A-cI)y + beta*x ----------------------------
      alpha.x = 2.0*sigma_new / e;
      alpha.y = 0.0;
      beta.x  = -sigma * sigma_new;
      beta.y  = 0.0;

#ifdef PROFILE
     double start_time = omp_get_wtime();
#endif

#ifdef MULTI_GPU
      cudaSetDevice(0);
      HANDLE_ERROR_CUBLAS( cublasZgemm( gpuData->handle,
               CUBLAS_OP_N, CUBLAS_OP_N,
               n, m, gpuData->work_per_gpu[0], &alpha,
               gpuData->H_d, n, gpuData->W_d + n * opt, n, &beta, gpuData->V_d + n * opt, n) );

      cudaSetDevice(1);
      HANDLE_ERROR_CUBLAS( cublasZgemm(   gpuData->handle1,
               CUBLAS_OP_N, CUBLAS_OP_N,
               n, m, gpuData->work_per_gpu[1], &alpha,
               gpuData->H1_d, n, gpuData->W1_d + n * opt + gpuData->work_per_gpu[0], n, &zero, gpuData->V1_d + n * opt, n) );

      HANDLE_ERROR( cudaMemcpyAsync(gpuData->tmp1_d, gpuData->V_d + n * opt, n * m * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice, gpuData->stream0)); 
      cudaEventRecord ( gpuData->event0 , gpuData->stream0 );
      cudaSetDevice(1);
      HANDLE_ERROR( cudaMemcpyAsync(gpuData->tmp0_d, gpuData->V1_d + n * opt, n * m * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice, gpuData->stream1)); 
      cudaEventRecord ( gpuData->event1 , gpuData->stream1 );

      int num_blocks_x = (n+(BLOCKDIM-1))/BLOCKDIM;
      cudaSetDevice(1);
      cudaEventSynchronize ( gpuData->event0);
      add_matrix<<<num_blocks_x, BLOCKDIM,0, gpuData->stream1>>>(gpuData->V1_d + n * opt, n, gpuData->tmp1_d, n, n, m);
      cudaSetDevice(0);
      cudaEventSynchronize ( gpuData->event1);
      add_matrix<<<num_blocks_x, BLOCKDIM,0, gpuData->stream0>>>(gpuData->V_d + n * opt, n, gpuData->tmp0_d, n, n, m);

      cudaDeviceSynchronize();
      cudaSetDevice(1);
#else
      HANDLE_ERROR_CUBLAS( cublasZgemm(   gpuData->handle,
               CUBLAS_OP_N, CUBLAS_OP_N,
               n, m, n, &alpha,
               gpuData->H_d, n, gpuData->W_d + n * opt, n, &beta, gpuData->V_d + n * opt, n) );
#endif
//      ZGEMM("N", "N", &n, &m, &n, (const double _Complex*)&alpha, (const double _Complex*)(A), &n,
//            (const double _Complex*)y+n*opt, &n, (const double _Complex*)&beta, x+n*opt, &n);
      //------------------------------------------------------------------------------
#ifdef PROFILE
     cudaDeviceSynchronize();
     //stop timing
     double end_time = omp_get_wtime();
     printf("GFLOPS: %dx%dx%d: %f\n",n,m,n,((float)n*m*n*8)/(end_time - start_time)/1e9);
#endif

      sigma = sigma_new;      

      //swap ptr
      tmp = x; x = y; y = tmp; //TODO remove?
      cuDoubleComplex *tmp0 = gpuData->V_d; gpuData->V_d = gpuData->W_d; gpuData->W_d = tmp0;
      cuDoubleComplex *tmp1 = gpuData->V1_d; gpuData->V1_d = gpuData->W1_d; gpuData->W1_d = tmp1;

      m += opt;
      while(j < block && deg[j++] == i)
         ++opt;
      --j;
      m -= opt;

   } // for(i = 2; i <= M; ++i)

   //---------------------------- DtH transfer --------------------------------
   cudaSetDevice(0);
   HANDLE_ERROR( cudaMemcpyAsync(x, gpuData->V_d, gpuData->sizeV, cudaMemcpyDeviceToHost,
            gpuData->stream0) ); //TODO remove
   HANDLE_ERROR( cudaMemcpyAsync(y, gpuData->W_d, gpuData->sizeW, cudaMemcpyDeviceToHost,
            gpuData->stream0) );
  //--------------------------------------------------------------------------
  
   cudaDeviceSynchronize(); //TODO remove?

   //----------------------------------RESTORE-A---------------------------------------
   cudaSetDevice(0);
   zshift_matrix<<<num_blocks, BLOCKDIM, 0, gpuData->stream0>>>(gpuData->H_d, n, n, c, 0);
#ifdef MULTI_GPU
   cudaSetDevice(1);
   zshift_matrix<<<num_blocks, BLOCKDIM,0,gpuData->stream1>>>(gpuData->H1_d, n, n, c, gpuData->work_per_gpu[0]);
#endif
   //--------------------------------------------------------------------------------
   cudaSetDevice(0);

   return;
}

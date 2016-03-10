#include "../include/filter.h"
#include <iostream>

#include "cublas_v2.h"
#include <cuda_runtime.h>
#include <cuComplex.h>
#include "cublas_v2.h"
#include <omp.h>
#include <stdio.h>

#define BLOCKDIM 256

__global__ void zshift_matrix(cuDoubleComplex* A, int lda, int n, double shift, int offset)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if(idx  >= offset  && idx < n)
      A[(idx-offset) * lda + idx].x += shift;
}


static void handleError( cudaError_t error, const char *file, int line ) {
    if (error != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( error ), file, line );
        exit( -1 );
    }
}


static void handleError_cublas( cublasStatus_t error, const char *file, int line ) {
    if (error != CUBLAS_STATUS_SUCCESS) {
       if(error == CUBLAS_STATUS_INTERNAL_ERROR)
        printf( "CUBLAS_STATUS_INTERNAL_ERROR in %s at line %d\n", file, line );
       else if(error == CUBLAS_STATUS_EXECUTION_FAILED)
        printf( "CUBLAS_STATUS_EXECUTION_FAILED in %s at line %d\n", file, line );
       else if(error == CUBLAS_STATUS_MAPPING_ERROR)
        printf( "CUBLAS_STATUS_MAPPING_ERROR in %s at line %d\n", file, line );
       else if(error == CUBLAS_STATUS_ARCH_MISMATCH)
        printf( "CUBLAS_STATUS_ARCH_MISMATCH in %s at line %d\n", file, line );
       else if(error == CUBLAS_STATUS_INVALID_VALUE)
        printf( "CUBLAS_STATUS_INVALID_VALUE in %s at line %d\n", file, line );
       else if(error == CUBLAS_STATUS_ALLOC_FAILED)
        printf( "CUBLAS_STATUS_ALLOC_FAILED in %s at line %d\n", file, line );
       else if(error == CUBLAS_STATUS_NOT_INITIALIZED)
        printf( "CUBLAS_STATUS_NOT_INITIALIZED in %s at line %d\n", file, line );
       else
        printf( "Error code: %d in %s at line %d\n", error, file, line );
        exit( -1 );
    }
}

#define HANDLE_ERROR( error ) (handleError( (error) , __FILE__ , __LINE__ ))
#define HANDLE_ERROR_CUBLAS( error ) (handleError_cublas( (error) , __FILE__ , __LINE__ ))



int cuda_filter( MKL_Complex16 *H, MKL_Complex16 *V, int n, int unprocessed,
            int deg, int *degrees, double lambda_1, double lower, double upper,
            MKL_Complex16 *W )
{
  int N = n;
  int Av = 0;
  int num_mult = 0;

  cuDoubleComplex * H_d, *H_dO;
  cuDoubleComplex * V_d, *V_dO;
  cuDoubleComplex * W_d, *W_dO;
  MKL_Complex16 *V0, *W0;
  //cuDoubleComplex * tau_d;
//  double* ritzv_d;
  size_t sizeA, sizeV;

  cublasHandle_t handle;
  cudaStream_t stream;
  cudaEvent_t event;

//  HANDLE_ERROR(cudaSetDeviceFlags(cudaDeviceScheduleYield));
   //create stream
  cudaSetDevice(0);
  cudaStreamCreate(&stream);
  cublasCreate(&handle);
  //all cublas tasks will be scheduled to this stream
  cublasSetStream(handle, stream);

  //magma_init();

  sizeA = N * N * sizeof(cuDoubleComplex);
  sizeV = N * unprocessed * sizeof(cuDoubleComplex);

//   HANDLE_ERROR( cudaMalloc((void**) &(ritzv_d), nevex * sizeof(double)) );  //sync
   HANDLE_ERROR( cudaMalloc((void**) &(H_dO), sizeA) );  //sync
   HANDLE_ERROR( cudaMalloc((void**) &(V_dO), sizeV) );  //sync
   HANDLE_ERROR( cudaMalloc((void**) &(W_dO), sizeV) );  //sync
   cudaEventCreate( &event );

   H_d = H_dO;
   V_d = V_dO;
   W_d = W_dO;
   W0 = W;
   V0 = V;

  cuDoubleComplex alpha, beta;
  double c = (upper + lower) / 2;
  double e = (upper - lower) / 2;
  double sigma_1   = e / (lambda_1 - c);
  double sigma     = sigma_1;
  double sigma_new;

    //---------------------------- y = alpha*(A-cI)*x --------------------------------
  alpha.x = sigma_1 / e;
  alpha.y = 0.0;
  beta.x = 0.0;
  beta.y = 0.0;

  //---------------------------- HtD transfer --------------------------------
  HANDLE_ERROR( cudaMemcpyAsync(H_d, H, sizeA, cudaMemcpyHostToDevice, stream)); //sync
  HANDLE_ERROR( cudaMemcpyAsync(V_d, V, sizeV, cudaMemcpyHostToDevice, stream)); //sync

  //----------------------------------- A = A-cI -------------------------------
  int num_blocks = (n+(BLOCKDIM-1))/BLOCKDIM;
  zshift_matrix<<<num_blocks, BLOCKDIM,0,stream>>>(H_d, n, n, -c, 0);


  HANDLE_ERROR_CUBLAS(
    cublasZgemm(
      handle,
      CUBLAS_OP_N, CUBLAS_OP_N,
      n, unprocessed, n, &alpha,
      H_d, n, V_d, n, &beta, W_d, n
      )
    );

  Av += unprocessed;
  num_mult++;
  while( unprocessed >= 0 && *degrees <= num_mult )
  {
    degrees++; V+=n; W+=n; unprocessed--;
    V_d+=n; W_d+=n;
  };

  for( int i = 2; i <= deg; ++i )
    {
      sigma_new = 1.0 / ( 2.0/sigma_1 - sigma );

      //----------------------- V = alpha(A-cI)W + beta*V ----------------------
      alpha.x = 2.0*sigma_new / e;
      alpha.y = 0.0;
      beta.x  = -sigma * sigma_new;
      beta.y  = 0.0;

      HANDLE_ERROR_CUBLAS(
        cublasZgemm(
          handle,
          CUBLAS_OP_N, CUBLAS_OP_N,
          n, unprocessed, n, &alpha,
          H_d, n, W_d, n, &beta, V_d, n
          )
        );

      sigma = sigma_new;

      std::swap( V, W );
      std::swap( V0, W0 );
      std::swap( V_d, W_d );
      std::swap( V_dO, W_dO );

      Av += unprocessed;
      num_mult++;
      while( unprocessed >= 0 && *degrees <= num_mult )
      {
        degrees++; V+=n; W+=n; unprocessed--;
        V_d+=n; W_d+=n;
      }

    } // for(i = 2; i <= deg; ++i)

  //---------------------------- DtH transfer --------------------------------
    HANDLE_ERROR(
      cudaMemcpyAsync(V0, V_dO, sizeV, cudaMemcpyDeviceToHost, stream) );
    HANDLE_ERROR(
      cudaMemcpyAsync(W0, W_dO, sizeV, cudaMemcpyDeviceToHost, stream) );

  //-----------------------------------RESTORE-A------------------------------------
  //zshift_matrix<<<num_blocks, BLOCKDIM, 0, stream>>>(H_d, n, n, c, 0);

    cudaDeviceSynchronize();

   cudaFree (H_dO);
   cudaFree (V_dO);
   cudaFree (W_dO);
   //magma_finalize();
   cudaStreamDestroy(stream);
   cublasDestroy(handle);


   return Av;
}

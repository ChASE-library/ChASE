#include "../include/filter.h"
#include "../include/cuda_util.h"
#include <iostream>

#define BLOCKDIM 256

__global__ void zshift_matrix(cuDoubleComplex* A, int lda, int n, double shift, int offset)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if(idx  >= offset  && idx < n)
      A[(idx-offset) * lda + idx].x += shift;
}


int cuda_filter( cuDoubleComplex *H, cuDoubleComplex *dV_, int n, int unprocessed,
            int deg, int *degrees, double lambda_1, double lower, double upper,
                 cuDoubleComplex *dW_, cudaStream_t *stream, cublasHandle_t *handle )
{
  int N = n;
  int Av = 0;
  int num_mult = 0;

  size_t sizeV;
  sizeV = N * unprocessed * sizeof(cuDoubleComplex);

  cuDoubleComplex *dW, *dV;
  dW = dW_;
  dV = dV_;

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

  //----------------------------------- A = A-cI -------------------------------
  int num_blocks = (n+(BLOCKDIM-1))/BLOCKDIM;
  zshift_matrix<<<num_blocks, BLOCKDIM,0,*stream>>>(H, n, n, -c, 0);


  HANDLE_ERROR_CUBLAS(
    cublasZgemm(
      *handle,
      CUBLAS_OP_N, CUBLAS_OP_N,
      n, unprocessed, n, &alpha,
      H, n, dV, n, &beta, dW, n
      )
    );

  Av += unprocessed;
  num_mult++;
  while( unprocessed >= 0 && *degrees <= num_mult )
  {
    degrees++; unprocessed--;
    dV+=n; dW+=n;
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
          *handle,
          CUBLAS_OP_N, CUBLAS_OP_N,
          n, unprocessed, n, &alpha,
          H, n, dW, n, &beta, dV, n
          )
        );

      sigma = sigma_new;

      std::swap( dV, dW );
      std::swap( dV_, dW_ );

      Av += unprocessed;
      num_mult++;
      while( unprocessed >= 0 && *degrees <= num_mult )
      {
        degrees++; unprocessed--;
        dV+=n; dW+=n;
      }

    } // for(i = 2; i <= deg; ++i)

  //-----------------------------------RESTORE-A------------------------------------
  zshift_matrix<<<num_blocks, BLOCKDIM, 0, *stream>>>(H, n, n, c, 0);

   return Av;
}

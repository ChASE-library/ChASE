#include "../include/filter.h"
#include <iostream>
#include <mkl.h>
#include <omp.h>
#include <stdio.h>

using namespace std;

void multi_ZGEMM(const char transa, const char transb, const MKL_INT m, const MKL_INT n, const MKL_INT k,
      const MKL_Complex16 alpha, const MKL_Complex16 *A, const MKL_INT lda,
      const MKL_Complex16 *B, const MKL_INT ldb, const MKL_Complex16 beta,
      MKL_Complex16 *C, const MKL_INT ldc)
{
   char signal0,signal1;
   const int matrix_elementsA = m * k;
   const int matrix_elementsB = k * n;
   const int matrix_elementsC = m * n;
   const int k_mic0 = k/2;
   const int k_mic1 = k - k_mic0;
   MKL_Complex16 zero;
   zero.real = 0.0;
   zero.imag = 0.0;

#pragma offload target(mic:0) \
  in(A: length(0) REUSE) \
  in(B: length(matrix_elementsB) REUSE) \
  inout(C: length(matrix_elementsC) REUSE) signal(&signal0)
  {
     ZGEMM(&transa, &transb, &m, &n, &k_mic0, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
  }

  MKL_Complex16 *tmp = new MKL_Complex16[m * n];

#pragma offload target(mic:1) \
  in(A: length(0) REUSE) \
  in(B: length(matrix_elementsB) REUSE) \
  out(tmp: length(matrix_elementsC) alloc_if(1) free_if(1))
  {
     ZGEMM(&transa, &transb, &m, &n, &k_mic1, &alpha, A, &lda, B, &ldb, &zero, tmp, &ldc);
  }
#pragma offload_wait target(mic:0) wait(&signal0)

  //add the two results to the final C (on the host)
#pragma omp parallel for
  for(int i=0;i < n; ++i){ //loop over columns
//#pragma ivdep
     for(int j=0;j < m; ++j){ //loop over rows
        C[i*ldc + j].real += tmp[i*ldc + j].real;
        C[i*ldc + j].imag += tmp[i*ldc + j].imag;
     }
  }

  //update C on both devices
#pragma offload_transfer target(mic:0) in((C):length(matrix_elementsC) REUSE) signal(&signal1)
#pragma offload_transfer target(mic:1) in((C):length(matrix_elementsC) REUSE)
#pragma offload_wait target(mic:0) wait(&signal1)

  delete[] tmp;

}

void filter(MKL_Complex16 *A, MKL_Complex16 *x, int n, int m,
            int deg, double lambda_1, double lower, double upper,
            MKL_Complex16 *y)
{  
  MKL_Complex16 alpha;  
  MKL_Complex16 beta;
  double c = (upper + lower) / 2;
  double e = (upper - lower) / 2;
  double sigma_1   = e / (lambda_1 - c);
  double sigma     = sigma_1;
  double sigma_new;
  int i = 0;

  //--------------------------------- A = A-cI -------------------------------------
#pragma offload target(mic:0) \
  in(A: length(0) REUSE)
  {
#pragma omp parallel for private(i)
#pragma ivdep
     for( i = 0; i < n; ++i )
        A[ i*n + i ].real -= c;
  }
  //--------------------------------------------------------------------------------

  //---------------------------- y = alpha*(A-cI)*x --------------------------------
  alpha.real = sigma_1 / e; alpha.imag = 0.0;
  beta.real = 0.0; beta.imag = 0.0;


  const int matrix_elementsA = n * n;
  const int matrix_elementsX = n * m;
#pragma offload target(mic:0) \
  in(A: length(0) REUSE) \
  in(x: length(matrix_elementsX) REUSE) \
  in(y: length(matrix_elementsX) REUSE) 
  {
     ZGEMM("N", "N", &n, &m, &n, &alpha, (const MKL_Complex16*)(A), &n,
           (const MKL_Complex16*)x, &n, &beta, y, &n);
  }
//  out(y:length(matrix_elements) free_if(1))
  //--------------------------------------------------------------------------------

  for(i = 2; i <= deg; ++i)
    {
      sigma_new = 1.0 / ( 2.0/sigma_1 - sigma );

      //---------------------- x = alpha(A-cI)y + beta*x ---------------------------
      alpha.real = 2.0*sigma_new / e; alpha.imag = 0.0;
      beta.real = -sigma * sigma_new; beta.imag = 0.0;
      double start_time = omp_get_wtime();
#pragma offload target(mic:0) \
      in(A: length(0) REUSE) \
      in(x: length(0) REUSE) \
      in(y: length(0) REUSE) 
      {
         ZGEMM("N", "N", &n, &m, &n, &alpha, (const MKL_Complex16*)(A), &n,
               (const MKL_Complex16*)y, &n, &beta, x, &n);

         MKL_Complex16 *tmp = x;
         x   = y;
         y   = tmp;
      }
#ifdef PROFILE
      double end_time = omp_get_wtime();
      printf("GFLOPS: %f\n",((float)n*m*n*8)/(end_time - start_time)/1e9);
#endif
      //----------------------------------------------------------------------------
      

      sigma = sigma_new;      
    } // for(i = 2; i <= deg; ++i)
 #pragma offload target(mic:0) \
	out(x: length(matrix_elementsX) REUSE) \
	out(y: length(matrix_elementsX) REUSE)
	{
	}

 
  //-----------------------------------RESTORE-A------------------------------------
#pragma offload target(mic:0) \
  in(A: length(0) REUSE)
   {
#pragma omp parallel for private(i)
#pragma ivdep
      for(i = 0; i < n; ++i)
         A[i*n + i].real += c;
   }
  //--------------------------------------------------------------------------------

  return;
}


void filter_multiple_phis(MKL_Complex16 *A, MKL_Complex16 *x, int n, int m,
            int deg, double lambda_1, double lower, double upper,
            MKL_Complex16 *y)
{  
  MKL_Complex16 alpha;  
  MKL_Complex16 beta;
  double c = (upper + lower) / 2;
  double e = (upper - lower) / 2;
  double sigma_1   = e / (lambda_1 - c);
  double sigma     = sigma_1;
  double sigma_new;
  int i = 0;

  //--------------------------------- A = A-cI -------------------------------------
  char signal0,signal1;
#pragma offload target(mic:0) \
  in(A: length(0) REUSE) signal(&signal0)
  {
#pragma omp parallel for private(i)
#pragma ivdep
     for( i = 0; i < n; ++i )
        A[ i*n + i ].real -= c;
  }
#pragma offload target(mic:1) \
  in(A: length(0) REUSE)
  {
#pragma omp parallel for private(i)
#pragma ivdep
     for( i = 0; i < n; ++i )
        A[ i*n + i ].real -= c;
  }
#pragma offload_wait target(mic:0) wait(&signal0)
  //--------------------------------------------------------------------------------

  //---------------------------- y = alpha*(A-cI)*x --------------------------------
  alpha.real = sigma_1 / e; alpha.imag = 0.0;
  beta.real = 0.0; beta.imag = 0.0;



  const int matrix_elementsX = n * m;
  multi_ZGEMM('N', 'N', n, m, n, alpha, (const MKL_Complex16*)(A), n,
        (const MKL_Complex16*)x, n, beta, y, n);
//#pragma offload target(mic:0) \
//  in(A: length(0) REUSE) \
//  in(x: length(matrix_elementsX) REUSE) \
//  in(y: length(matrix_elementsX) REUSE) 
//  {   ZGEMM("N", "N", &n, &m, &n, &alpha, (const MKL_Complex16*)(A), &n,
//        (const MKL_Complex16*)x, &n, &beta, y, &n);
//  }
  //--------------------------------------------------------------------------------

  for(i = 2; i <= deg; ++i)
    {
      sigma_new = 1.0 / ( 2.0/sigma_1 - sigma );

      //---------------------- x = alpha(A-cI)y + beta*x ---------------------------
      alpha.real = 2.0*sigma_new / e; alpha.imag = 0.0;
      beta.real = -sigma * sigma_new; beta.imag = 0.0;
      double start_time = omp_get_wtime();
#pragma offload target(mic:0) \
      in(A: length(0) REUSE) \
      in(x: length(0) REUSE) \
      in(y: length(0) REUSE) 
      {
         ZGEMM("N", "N", &n, &m, &n, &alpha, (const MKL_Complex16*)(A), &n,
               (const MKL_Complex16*)y, &n, &beta, x, &n);

         MKL_Complex16 *tmp = x;
         x   = y;
         y   = tmp;
      }
#ifdef PROFILE
      double end_time = omp_get_wtime();
      printf("GFLOPS: %f\n",((float)n*m*n*8)/(end_time - start_time)/1e9);
#endif
      //----------------------------------------------------------------------------
      

      sigma = sigma_new;      
    } // for(i = 2; i <= deg; ++i)
 #pragma offload target(mic:0) \
	out(x: length(matrix_elementsX) REUSE) \
	out(y: length(matrix_elementsX) REUSE)
	{
	}

 
  //-----------------------------------RESTORE-A------------------------------------
#pragma offload target(mic:0) \
  in(A: length(0) REUSE) signal(&signal1)
   {
#pragma omp parallel for private(i)
#pragma ivdep
      for(i = 0; i < n; ++i)
         A[i*n + i].real += c;
   }
#pragma offload target(mic:1) \
  in(A: length(0) REUSE)
  {
#pragma omp parallel for private(i)
#pragma ivdep
     for( i = 0; i < n; ++i )
        A[ i*n + i ].real += c;
  }
#pragma offload_wait target(mic:0) wait(&signal1)
  //--------------------------------------------------------------------------------

  return;
}

void filterModified(MKL_Complex16 *A, MKL_Complex16 *x, int n, int m, int nev,
                    int M, int *deg, double lambda_1, double lower, double upper,
                    MKL_Complex16 *y, int block)
{  
   MKL_Complex16 alpha;  
   MKL_Complex16 beta;
   double c = (upper + lower) / 2;
   double e = (upper - lower) / 2;
   double sigma_1   = e / (lambda_1 - c);
   double sigma     = sigma_1;
   double sigma_new;
   int i, j = 0;
   int opt = 0;

   //----------------------------------- A = A-cI ------------------------------------
#pragma offload target(mic:0) \
  in(A: length(0) REUSE)
   {
#pragma omp parallel for private(i)
#pragma ivdep
      for( i = 0; i < n ; ++i )
         A[ i*n + i ].real -= c;
   }
   //---------------------------------------------------------------------------------

   //------------------------------- y = alpha*(A-cI)*x ------------------------------
   alpha.real = sigma_1 / e; alpha.imag = 0.0;
   beta.real = 0.0; beta.imag = 0.0;

  const int matrix_elementsA = n * n;
  const int matrix_elementsX = n * m;
#pragma offload target(mic:0) \
  in(A: length(0) REUSE) \
  in(x: length(matrix_elementsX) REUSE) \
  in(y: length(matrix_elementsX) REUSE) 
  {
     ZGEMM("N", "N", &n, &m, &n, &alpha, (const MKL_Complex16*)(A), &n,
           (const MKL_Complex16*)x, &n, &beta, y , &n);
  }
   //---------------------------------------------------------------------------------

   while(j < block && deg[j++] == 1)
      ++opt;
   --j;
   m -= opt;


   for(i = 2; i <= M; ++i)
   {
      sigma_new = 1.0 / ( 2.0/sigma_1 - sigma );

      //----------------------- x = alpha(A-cI)y + beta*x ----------------------------
      alpha.real = 2.0*sigma_new / e; alpha.imag = 0.0;
      beta.real = -sigma * sigma_new; beta.imag = 0.0;

      double start_time = omp_get_wtime();
#pragma offload target(mic:0) \
      in(A: length(0) REUSE) \
      nocopy(x: length(0) REUSE) \
      nocopy(y: length(0) REUSE) 
      {
         ZGEMM("N", "N", &n, &m, &n, &alpha, (const MKL_Complex16*)(A), &n,
               (const MKL_Complex16*)y+n*opt, &n, &beta, x+n*opt, &n);
         
         MKL_Complex16 *tmp = x;
         x   = y;
         y   = tmp;
      }
#ifdef PROFILE
      double end_time = omp_get_wtime();
      printf("GFLOPS: %f\n",((float)n*m*n*8)/(end_time - start_time)/1e9);
#endif
      //------------------------------------------------------------------------------

      sigma = sigma_new;      


      m += opt;
      while(j < block && deg[j++] == i)
         ++opt;
      --j;
      m -= opt;

   } // for(i = 2; i <= M; ++i)

#pragma offload target(mic:0) \
   out(x: length(matrix_elementsX) REUSE) \
   out(y: length(matrix_elementsX) REUSE)
   {
   }

   //----------------------------------RESTORE-A---------------------------------------
#pragma offload target(mic:0) \
  in(A: length(0) REUSE)
   {
#pragma omp parallel for private(i)
#pragma ivdep
      for(i = 0; i < n; ++i)
         A[i*n + i].real += c;
   }
   //----------------------------------------------------------------------------------

   return;
}

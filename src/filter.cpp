#include "../include/filter.h"
#include<iostream>
using namespace std;

void filter(MKL_Complex16 *A, MKL_Complex16 *x, int n, int m,
            int deg, double lambda_1, double lower, double upper,
            MKL_Complex16 *y)
{
  MKL_Complex16 *tmp = NULL;
  MKL_Complex16 alpha;
  MKL_Complex16 beta;
  double c = (upper + lower) / 2;
  double e = (upper - lower) / 2;
  double sigma_1   = e / (lambda_1 - c);
  double sigma     = sigma_1;
  double sigma_new;
  int i = 0;

  //--------------------------------- A = A-cI -------------------------------------
  for( i = 0 ; i < n ; ++i )
    A[ i*n + i ] -= c;
  //--------------------------------------------------------------------------------

  //---------------------------- y = alpha*(A-cI)*x --------------------------------
  alpha = MKL_Complex16 (sigma_1 / e, 0.0);
  beta = MKL_Complex16 (0.0, 0.0);

  ZGEMM("N", "N", &n, &m, &n, &alpha, (const MKL_Complex16*)(A), &n,
        (const MKL_Complex16*)x, &n, &beta, y, &n);
  //--------------------------------------------------------------------------------

  for(i = 2; i <= deg; ++i)
    {
      sigma_new = 1.0 / ( 2.0/sigma_1 - sigma );

      //---------------------- x = alpha(A-cI)y + beta*x ---------------------------
      alpha = MKL_Complex16 (2.0*sigma_new / e, 0.0);
      beta = MKL_Complex16 (-sigma * sigma_new, 0.0);

      ZGEMM("N", "N", &n, &m, &n, &alpha, (const MKL_Complex16*)(A), &n,
            (const MKL_Complex16*)y, &n, &beta, x, &n);
      //----------------------------------------------------------------------------

      tmp = x;
      x   = y;
      y   = tmp;

      sigma = sigma_new;
    } // for(i = 2; i <= deg; ++i)

  //-----------------------------------RESTORE-A------------------------------------
  for(i = 0 ; i < n ; ++i)
    A[i*n + i] += c;
  //--------------------------------------------------------------------------------

  return;
}


void filterModified(MKL_Complex16 *A, MKL_Complex16 *x, int n, int m, int nev,
                    int M, int *deg, double lambda_1, double lower, double upper,
                    MKL_Complex16 *y, int block)
{
  MKL_Complex16 *tmp = NULL;
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
  for( i = 0 ; i < n ; ++i )
    A[ i*n + i ] -= c;
  //---------------------------------------------------------------------------------

  //------------------------------- y = alpha*(A-cI)*x ------------------------------
  alpha = MKL_Complex16 (sigma_1 / e, 0.0);
  beta = MKL_Complex16 (0.0, 0.0);

  ZGEMM("N", "N", &n, &m, &n, &alpha, (const MKL_Complex16*)(A), &n,
        (const MKL_Complex16*)x, &n, &beta, y, &n);
  //---------------------------------------------------------------------------------

   while(j < block && deg[j++] == 1)
      ++opt;
   --j;
   m -= opt;


  for(i = 2; i <= M; ++i)
    {
      sigma_new = 1.0 / ( 2.0/sigma_1 - sigma );

      //----------------------- x = alpha(A-cI)y + beta*x ----------------------------
      alpha = MKL_Complex16 (2.0*sigma_new / e, 0.0);
      beta = MKL_Complex16 (-sigma * sigma_new, 0.0);

      ZGEMM("N", "N", &n, &m, &n, &alpha, (const MKL_Complex16*)(A), &n,
            (const MKL_Complex16*)y, &n, &beta, x, &n);
      //------------------------------------------------------------------------------

      sigma = sigma_new;

      tmp = x;
      x   = y;
      y   = tmp;

      
      m += opt;
      while(j < block && deg[j++] == i)
         ++opt;
      --j;
      m -= opt;

    } // for(i = 2; i <= M; ++i)

  //----------------------------------RESTORE-A---------------------------------------
  for(i = 0 ; i < n ; ++i)
    A[i*n + i] += c;
  //----------------------------------------------------------------------------------

  return;
}

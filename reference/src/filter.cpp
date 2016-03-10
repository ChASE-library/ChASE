#include "../include/filter.h"

int filter( MKL_Complex16 *A, MKL_Complex16 *V, int n, int unprocessed,
            int deg, int *degrees, double lambda_1, double lower, double upper,
            MKL_Complex16 *W )
{
  double c = (upper + lower) / 2;
  double e = (upper - lower) / 2;
  double sigma_1   = e / (lambda_1 - c);
  double sigma     = sigma_1;
  double sigma_new;

  int num_mult = 0;
  int Av = 0;

  //----------------------------------- A = A-cI -------------------------------
  for( auto i = 0 ; i < n ; ++i )
    A[ i*n + i ] -= c;

  //------------------------------- Y = alpha*(A-cI)*V -------------------------
  MKL_Complex16 alpha = MKL_Complex16 (sigma_1 / e, 0.0);
  MKL_Complex16 beta = MKL_Complex16 (0.0, 0.0);

  cblas_zgemm(
    CblasColMajor, CblasNoTrans, CblasNoTrans,
    n, unprocessed, n,
    &alpha,
    A, n,
    V, n,
    &beta,
    W, n
    );

  Av += unprocessed;
  num_mult++;
  while( unprocessed >= 0 && *degrees <= num_mult )
  {
    degrees++; V+=n; W+=n; unprocessed--;
  };

  for( auto i = 2; i <= deg; ++i )
    {
      sigma_new = 1.0 / ( 2.0/sigma_1 - sigma );

      //----------------------- V = alpha(A-cI)W + beta*V ----------------------
      alpha = MKL_Complex16 (2.0*sigma_new / e, 0.0);
      beta = MKL_Complex16 (-sigma * sigma_new, 0.0);

      cblas_zgemm(
        CblasColMajor, CblasNoTrans, CblasNoTrans,
        n, unprocessed, n,
        &alpha,
        A, n,
        W, n,
        &beta,
        V, n
        );

//	TODO
//      cblas_zhemm(
//        CblasColMajor,
//        CblasLeft,
//        CblasLower,
//        n, unprocessed,
//        &alpha,
//        A, n,
//        W, n,
//        &beta,
//        V, n
//        );

      sigma = sigma_new;
      std::swap( V, W );

      Av += unprocessed;
      num_mult++;
      while( unprocessed >= 0 && *degrees <= num_mult )
      {
        degrees++; V+=n; W+=n; unprocessed--;
      }

    } // for(i = 2; i <= deg; ++i)

  //----------------------------------RESTORE-A---------------------------------
  for( auto i = 0 ; i < n ; ++i )
    A[i*n + i] += c;

  return Av;
}

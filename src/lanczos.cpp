#include "../include/lanczos.h"

void lanczos(MKL_Complex16 *B, MKL_Complex16 *v, int n, int blk, int m, double tol, int ctrl, double *ritzv, double *bound)
{


  int M = ( ctrl == 1 ) ? 4*blk : m;

  MKL_Complex16 *v0 = NULL; // Auxiliary vector.

 // Eigenvalues  of the tridiagonal matrix T.
  double *TeValues  =   TeValues = new double[M];
 // Eigenvectors of the tridiagonal matrix T.
  MKL_Complex16 *TeVectors = new MKL_Complex16[m*m];

 // Contains the     diagonal elements of the tridiagonal matrix T.
  double *d = new double[2*m];
// Contains the off-diagonal elements of the tridiagonal matrix T.
  double *e = new double[2*m];

  int tryrac   = 1; // Input parameter for dstemr_.
  int notneeded_m; // Output parameter of dstemr_.
  int iONE = 1;

  MKL_Complex16 alpha;
  MKL_Complex16 beta;
  double real_alpha = 0.0;
  double real_beta  = 0.0;

  int   *isuppz = new int[2*M]; // Input parameter for dstemr_. Not used (but needed).
  MKL_Complex16 *V = new MKL_Complex16[n*M]; // Transformation matrix consisted of Lanczos vectors.

  MKL_Complex16 *G     = NULL;
  if( ctrl == 1 )
  {
    G = new MKL_Complex16[n*n];
  }

  //------------------------------ alpha = v / ||v|| ---------------------------
  real_alpha = DZNRM2( &n, (const MKL_Complex16*)v, &iONE );

  alpha = MKL_Complex16 (1/real_alpha, 0.0);
  ZSCAL( &n, &alpha, (MKL_Complex16*)v, &iONE );
  //----------------------------------------------------------------------------


  MKL_Complex16 *v1 = v; // Auxiliary vector
  // v0 = NULL;

  for(int k = 0; k < M ; )
  {
    //----------------------------- V(:,k) = v1 ------------------------------
    std::memcpy( V+k*n, v1, n * sizeof(MKL_Complex16) );
    //------------------------------------------------------------------------

    //----------------------------- v = B*v1 - bet*v0 ------------------------
    v1 = V+k*n;

    alpha = MKL_Complex16 (1.0,0.0);
    beta = MKL_Complex16 (0.0, 0.0);

    // v := alpha*B*v1 + beta*v
    ZGEMV( "N", &n, &n, &alpha, (const MKL_Complex16*)B, &n,
           (const MKL_Complex16*)v1, &iONE, &beta, v, &iONE );

    //------------------------------------------------------------------------

    if( v0 != NULL )
    {
      beta = MKL_Complex16 (-real_beta,0.0);
      ZAXPY( &n, &beta, (const MKL_Complex16*)v0, &iONE, v, &iONE );
    }

    // alp(ha) = v1'*v;
    ZDOTC( &alpha, &n, (const MKL_Complex16*)v1, &iONE, (const MKL_Complex16*)v, &iONE );
    //------------------------------------------------------------------------

    //------------------------------- v = alpha*v1 + v -----------------------
    alpha *= MKL_Complex16 (-1.0,0.0);

    ZAXPY( &n, &alpha, (const MKL_Complex16*)v1, &iONE, v, &iONE );
    //------------------------------------------------------------------------

    //----------------------------------RESTORE-alpha-------------------------
    alpha *= MKL_Complex16 (-1.0,0.0);
    //------------------------------------------------------------------------

    //--------------------------------- beta = norm(v) -----------------------
    real_beta = DZNRM2( &n, (const MKL_Complex16 *)v, &iONE );
    //------------------------------------------------------------------------

    v0 = v1;

    //----------------------------- v1 = v/beta; (v = v/beta) ----------------
    beta = MKL_Complex16 (1.0/real_beta,0.0);

    ZSCAL( &n, (const MKL_Complex16*)(&beta), v, &iONE );
    v1 = v;
    //------------------------------------------------------------------------

    if( k < m )
    { // Needed for the upper bound.
      d[k] = real(alpha);
      e[k] = real_beta;
    }

    k += 1;
    if( k == m ) // Compute the upper bound.
    {
      char jobz;
      if( ctrl == 1 ) // RANDOM: Compute eigenvalues only.
        jobz = 'N';
      else            // APPROX: eigenvectors
        jobz = 'V';

      LAPACKE_dstemr(
        LAPACK_COL_MAJOR,
        jobz,    'A',
        k,
        d,    e,
        NULL,    NULL,    NULL,    NULL,
        &notneeded_m,
        TeValues,
        (double*)TeVectors,
        M,    k,
        isuppz,    &tryrac
        );

      double Tnorm = std::max( fabs( TeValues[0] ) , fabs( TeValues[k-1] ) );

      if( ctrl == 1 ) // RANDOM
        *bound = Tnorm + fabs(real_beta);
      else   // APPROX
        *bound = Tnorm + fabs(real_beta)*fabs(((double*)TeVectors)[(k-1)*M+k-1]);

    } // if( k == m )

  } // for(int k = 0; k < M ; )

  if( ctrl == 1)
  {
  std::memcpy( G, B, n*n * sizeof(MKL_Complex16) );

  MKL_Complex16 *TAU = new MKL_Complex16[M];
  // [V, ~ ] = qr(V);
  //ZGEQRF( &n, &M, V, &n, (MKL_Complex16*)dwork, zwork, &lzwork, &INFO );
  LAPACKE_zgeqrf(
    LAPACK_COL_MAJOR,
    n, M,
    V, n,
    TAU );
  // G = V'B
  LAPACKE_zunmqr(
    LAPACK_COL_MAJOR,
    'L', 'C',
    n, n, M,
    V, n,
    TAU,
    G, n );
  // G = V'BV
  LAPACKE_zunmqr(
    LAPACK_COL_MAJOR,
    'R', 'N',
    M, n, M,
    V, n,
    TAU,
    G, n );

  delete[] TAU;

  //---------------------------FIND-EIGENVALUES-OF-G----------------------------
//  il = 1; iu = blk;
//  ZHEEVR( "N", "I", "U", &M, G, &n, &vl, &vu, &il, &iu, &tol,
//          &notneeded_m, TeValues, TeVectors, &M, isuppz,
//          zwork, &lzwork, dwork, &ldwork, iwork, &liwork, &INFO );

  LAPACKE_zheevr(
    LAPACK_COL_MAJOR,
    'N', 'I', 'U',
    M,
    G,    n,
    NULL,    NULL,
    1,    blk,
    tol,
    &notneeded_m,
    TeValues,
    TeVectors,
    M,
    isuppz );

  // Eigenvalues stored in TeValues.
  DCOPY( &blk, TeValues, &iONE, ritzv, &iONE); // Copy the eigenvalues to output ritzv.
  //----------------------------------------------------------------------------
  }


  delete[] V;
  //delete[] dwork;
  //delete[] iwork;

  // depending on ctrl, may be NULL (no manual check needed - delete[] checks it)
  delete[] G;
  delete[] TeVectors;

  delete[] TeValues;
  delete[] isuppz;

  delete[] d;
  delete[] e;
}

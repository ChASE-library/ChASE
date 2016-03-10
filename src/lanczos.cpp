#include "../include/lanczos.h"


void lanczos(const MKL_Complex16 *H, int N, int numvec, int m, int nevex, double *upperb,
                bool mode, double *ritzv_, MKL_Complex16 *V_)
{

    std::mt19937 gen(2342.0); // TODO
    std::normal_distribution<> d;

  if( !mode )
  {
    // all we need is the upper bound
    MKL_Complex16 *randomVector = new MKL_Complex16[N];
    for( auto i=0; i < N; ++i)
      randomVector[i] = std::complex<double>( d(gen), d(gen) );

    lanczosM( H,  N,  m, upperb, false,
              randomVector, NULL, NULL );

    delete[] randomVector;
  }
  else
  {
    // we need a bound for lambda1.


    //We will do numvec many Lanczos procedures and save all the eigenvalues,
    //and the first entries of the eigenvectors
    double *Theta = new double[numvec*m]();
    double *Tau = new double[numvec*m]();
    double *ritzV = new double[m*m]();
    MKL_Complex16 *V = new MKL_Complex16[N*m];
    double upperb_ = std::numeric_limits<double>::min();
    double lowerb, lambda;
    for( auto i=0; i < numvec; ++i)
    {
      // Generate random vector
      for( auto k=0; k < N; ++k)
        V[k] = std::complex<double>( d(gen), d(gen) );

      lanczosM ( H, N, m, &upperb_, true, V, Theta+m*i, ritzV );

      for( auto k=i; k < m; ++k)
        Tau[k+m*i] = std::abs(ritzV[k*m])*abs(ritzV[k*m]);
      *upperb = std::max( upperb_, *upperb );
    }


    double *ThetaSorted = new double[numvec*m];
    for( auto k=0; k < numvec*m; ++k )
      ThetaSorted[k] = Theta[k];
    std::sort(ThetaSorted, ThetaSorted + numvec*m, std::less<double>());

    lambda = ThetaSorted[0];
    std::cout << lambda << std::endl;


    double curr, prev = 0;
    const double sigma = 0.25;
    const double threshold = 2*sigma*sigma/10;
    const double search = static_cast<double>(nevex) / static_cast<double>(N) * 1.1;
    // CDF of a Gaussian, erf is a c++11 function
    const auto G = [&] ( double x ) -> double {
      return 0.5 * (1 + erf( x / sqrt( 2*sigma*sigma ) ) );
    };

    for( auto i=0; i < numvec*m; ++i )
    {
      curr = 0;
      for( int j=0; j < numvec*m; ++j )
      {
        if( ThetaSorted[i] < (Theta[j] - threshold ) )
          curr += 0;
        else if( ThetaSorted[i] > (Theta[j] + threshold ) )
          curr += Tau[j] * 1;
        else
          curr += Tau[j] * G(ThetaSorted[i] - Theta[j]);
      }
      curr = curr / numvec;

      if( curr > search )
      {
        if( std::abs( curr - search ) < std::abs( prev - search ) )
          lowerb = ThetaSorted[i];
        else
          lowerb = ThetaSorted[i-1];
        break;
      }
      prev = curr;
    }


    // Now we extract the Eigenvectors that correspond to eigenvalues < lowerb
    int idx = 0;
    for (int i = 0; i < m; ++i) {
      if ( Theta[ (numvec-1)*m + i] > lowerb ) {
        idx = i-1;
        break;
      }
    }

    std::cout << "Obtained " << idx << " vectors from DoS " << m << " " << idx << std::endl; 
    if ( idx > 0 )
    {
      MKL_Complex16 *ritzVc = new MKL_Complex16[m*m]();
      for( auto i=0; i < m*m; ++i)
        ritzVc[i] = MKL_Complex16( ritzV[i], 0);

      MKL_Complex16 alpha = MKL_Complex16(1, 0);
      MKL_Complex16 beta = MKL_Complex16(0, 0);
      cblas_zgemm(
        CblasColMajor,
        CblasNoTrans,
        CblasNoTrans,
        N,
        idx,
        m,
        &alpha,
        V, N,
        ritzVc, m,
        &beta,
        V_, N );
      delete[] ritzVc;
    }


    for( auto i=0; i < nevex; ++i)
//      ritzv_[i] = ThetaSorted[(numvec-1)*m+i];
      ritzv_[i] = lambda;
    ritzv_[nevex-1] = lowerb;


    std::cout << lowerb << " " << lambda << std::endl;

    // Cleanup
    delete[] Theta;
    delete[] Tau;
    delete[] V;
  }
}

// Do m-step lanczos procedure, if requested, return V
void lanczosM( const MKL_Complex16 *H, int n, int m, double *upperb,
                bool ctrl, MKL_Complex16 *V, double *ritzv,  double *ritzV )
{
  double *d = new double[m]();
  double *e = new double[m]();

  // SO C++03 5.3.4[expr.new]/15
  MKL_Complex16 *v0_ = new MKL_Complex16[n]();
  MKL_Complex16 *w_ = new MKL_Complex16[n]();

  MKL_Complex16 *v0 = v0_;
  MKL_Complex16 *w = w_;

  MKL_Complex16 alpha = MKL_Complex16(1.0,0.0);
  MKL_Complex16 beta = MKL_Complex16 (0.0, 0.0);
  MKL_Complex16 One = MKL_Complex16(1.0,0.0);
  MKL_Complex16 Zero = MKL_Complex16(0.0,0.0);


  MKL_Complex16 *v1 = V;
  // ENSURE that v1 has one norm
  double real_alpha = cblas_dznrm2( n, v1, 1 );
  alpha = MKL_Complex16(1/real_alpha, 0.0);
  cblas_zscal( n, &alpha, v1, 1 );
  double real_beta = 0;


  real_beta = 0;

  for( auto k=0; k < m; ++k)
  {
    if( ctrl )
    std::memcpy( V+k*n, v1, n*sizeof(MKL_Complex16) );

    cblas_zgemv(
      CblasColMajor,
      CblasNoTrans,
      n,
      n,
      &One,
      H,      n,
      v1, 1,
      &Zero,
      w, 1 );

    cblas_zdotc_sub(
      n,
      v1, 1,
      w, 1,
      &alpha);

    alpha = -alpha;
    cblas_zaxpy(
      n,
      &alpha,
      v1, 1,
      w, 1);
    alpha = -alpha;

    d[k] = alpha.real();
    if ( k == m-1 )
      break;

    beta = MKL_Complex16( -real_beta, 0);
    cblas_zaxpy(
      n,
      &beta,
      v0, 1,
      w, 1);
    beta = -beta;

    real_beta = cblas_dznrm2( n, w, 1 );
    beta = MKL_Complex16 (1.0/real_beta,0.0);

    cblas_zscal(
      n, &beta, w, 1 );


    e[k] = real_beta;

    std::swap( v1, v0 );
    std::swap( v1, w );

  }

  delete[] w_;
  delete[] v0_;



  int notneeded_m;
  int tryrac =0;
  int *isuppz = new int[2*m];
  if( !ctrl )
  {
  ritzV = new double[m*m];
  ritzv = new double[m];
  }

  LAPACKE_dstemr(
    LAPACK_COL_MAJOR,
    'V',    'A',
    m,
    d,    e,
    NULL,    NULL,    NULL,    NULL,
    &notneeded_m,
    ritzv,
    ritzV,
    m,    m,
    isuppz,    &tryrac
    );

  *upperb = std::max( fabs( ritzv[0] ) , fabs( ritzv[m-1] ) )
    + std::abs(real_beta) * std::abs( ritzV[m*m-1] );

  if( ctrl )
  {
    ;
  }
  else
  {
    delete[] ritzV;
    delete[] ritzv;
  }

  delete[] isuppz;
  delete[] d;
  delete[] e;
}

/*
void lanczos(const MKL_Complex16 *B, MKL_Complex16 *v, int n, int blk, int m,
               double tol, int ctrl, double *ritzv, double *bound)
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
*///

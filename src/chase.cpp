#include "../include/chfsi.h"
#include "../include/lanczos.h"
#include "../include/filter.h"

template <typename T>
void swap_kj(int k, int j, T* array)
{
  T tmp = array[k];
  array[k] = array[j];
  array[j] = tmp;
}

void ColSwap( MKL_Complex16 *V, int N, int i, int j )
{
  MKL_Complex16 *ztmp = new MKL_Complex16[N];
  memcpy(ztmp,  V+N*i, N*sizeof(MKL_Complex16));
  memcpy(V+N*i, V+N*j, N*sizeof(MKL_Complex16));
  memcpy(V+N*j, ztmp,  N*sizeof(MKL_Complex16));
  delete[] ztmp;
}

int calc_degrees( int N, int unconverged,  double upperb, double lowerb, double tol,
                  double *ritzv, double *resid, int *degrees,
                  MKL_Complex16 *V, MKL_Complex16 *W)
{

  double c = (upperb + lowerb) / 2; // Center of the interval.
  double e = (upperb - lowerb) / 2; // Half-length of the interval.
  double rho;

  for( auto i = 0; i < unconverged; ++i )
  {
    double t = (ritzv[i] - c)/e;
    rho = std::max(
      abs( t - sqrt( t*t-1 ) ),
      abs( t + sqrt( t*t-1 ) )
      );

    degrees[i] = ceil(fabs(log(resid[i]/(tol))/log(rho)));
    degrees[i] = std::min( degrees[i] + omp_delta, omp_degmax );
  }

  for (int j = 0; j < unconverged-1; ++j)
    for (int k = j; k < unconverged; ++k)
      if (degrees[k] < degrees[j])
      {
        swap_kj(k, j, degrees);
        swap_kj(k, j, ritzv);
        swap_kj(k, j, resid);
        ColSwap( V, N, k, j );
      }

  return std::min( omp_degmax, degrees[unconverged-1] );
}

int locking( int N, int unconverged, double tol,
             double *ritzv, double *resid, int *degrees,
             MKL_Complex16 *V)
{
  int converged = 0;
  for (int j = 0; j < unconverged; ++j)
  {
    if (resid[j] > tol) continue;
    if (j != converged)
    {
      swap_kj( j, converged, degrees ); // if we filter again
      swap_kj( j, converged, ritzv ); // if we terminate
      ColSwap( V, N, j, converged );
    }
    converged++;
  }
  return converged;
}

void resd( int N, int unconverged,
          double *ritzv, double *resid,
          MKL_Complex16 *H, MKL_Complex16 *V, MKL_Complex16 *W)
{
  MKL_Complex16 alpha = MKL_Complex16 (1.0, 0.0);
  MKL_Complex16 beta  = MKL_Complex16 (0.0, 0.0);

  cblas_zgemm(
    CblasColMajor,    CblasNoTrans,    CblasNoTrans,
    N,    unconverged,    N,
    &alpha,
    H,    N,
    V,    N,
    &beta,
    W,    N
    );

  double norm1, norm2;
    for( int i = 0 ; i < unconverged ; ++i )
    {
      beta  = MKL_Complex16 (-ritzv[i], 0.0);
      cblas_zaxpy( N, &beta, V+N*i, 1, W+N*i, 1);

      norm1 = cblas_dznrm2( N, W+N*i, 1);
      norm2 = cblas_dznrm2( N, V+N*i, 1);

      resid[i] = norm1/norm2;
    }
}

void QR( int N, int nevex, int converged, MKL_Complex16 *W, MKL_Complex16 *tau )
{
  MKL_Complex16 *saveW = new MKL_Complex16[N*converged];
  memcpy( saveW, W, N*converged*sizeof(MKL_Complex16) );
  // TODO: don't make lapacke do the memory management
  // |-> lapack_zgeqrf
  LAPACKE_zgeqrf(
    LAPACK_COL_MAJOR,
    N,    nevex,
    W,    N,
    tau
    );

 LAPACKE_zungqr(
    LAPACK_COL_MAJOR,
    N,    nevex,    nevex,
    W,    N,
    tau
    );


 memcpy( W, saveW, N*converged*sizeof(MKL_Complex16) );
 delete[] saveW;
}

void RR( int N, int block, double *Lambda,
         MKL_Complex16 *H, MKL_Complex16 *V, MKL_Complex16 *W )
{
  MKL_Complex16 * A = new MKL_Complex16[block*block]; // For LAPACK.
  MKL_Complex16 * X = new MKL_Complex16[block*block]; // For LAPACK.

  int *isuppz = new int[2*block];

  MKL_Complex16 alpha = MKL_Complex16 (1.0, 0.0);
  MKL_Complex16 beta  = MKL_Complex16 (0.0, 0.0);

  // V <- H*V
  cblas_zgemm(
    CblasColMajor,    CblasNoTrans,    CblasNoTrans,
    N,    block,    N,
    &alpha,
    H,    N,
    W,    N,
    &beta,
    V,    N
    );

  // A <- W * V
  cblas_zgemm(
    CblasColMajor,    CblasConjTrans,    CblasNoTrans,
    block,    block,    N,
    &alpha,
    W,    N,
    V,    N,
    &beta,
    A,    block
    );

  // TODO try D&C
  int numeigs = 0;
  LAPACKE_zheevr(
    LAPACK_COL_MAJOR,    'V',    'A',    'U',
    block,
    A,    block,
    NULL,    NULL,    NULL,    NULL,
    dlamch_( "S" ),    &numeigs,
    Lambda,
    X,    block,
    isuppz
    );

  cblas_zgemm(
    CblasColMajor,    CblasNoTrans,    CblasNoTrans,
    N,    block,    block,
    &alpha,
    W,    N,
    X,    block,
    &beta,
    V,    N
    );

  delete[] A;
  delete[] X;
  delete[] isuppz;
  //omp_time[END_RR] += (omp_get_wtime() - omp_time[BGN_RR]);
}

// approx solution (or random vexs in V)
void chfsi(MKL_Complex16* const H, int N, MKL_Complex16* V, MKL_Complex16* W,
           double* ritzv_, int nev, const int nex, int deg_, int* const degrees_,
           const double tol, const CHASE_MODE_TYPE int_mode,
           const CHASE_OPT_TYPE int_opt)
{
  MKL_Complex16 *RESULT = W;

  double* resid_ = new double[nev];

  const int nevex = nev + nex; // Block size for the algorithm.
  int unconverged = nev;
  int block = nev+nex;

  int deg = deg_;

  // To store the approximations obtained from lanczos().
  double lowerb, upperb, lambda;

  int * degrees = degrees_;
  double * ritzv = ritzv_;
  double * resid = resid_;

  MKL_Complex16 *V_Full = V;
  MKL_Complex16 *W_Full = W;
  //-----------------------  VALIDATION  -----------------------------
  if ( int_opt != CHASE_OPT_NONE )
    assert(degrees != NULL);

  //-----------------------GENERATE-A-RANDOM-VECTOR-----------------------------
  MKL_Complex16 *randomVector = new MKL_Complex16[N];
  std::mt19937 gen(2342.0); // TODO
  std::normal_distribution<> d;

  for( std::size_t i=0; i < N; ++i)
  {
    randomVector[i] = std::complex<double>( d(gen), d(gen) );
  }
  //----------------------------------------------------------------------------

  int converged = 0; // Number of converged eigenpairs.
  int iteration = 0; // Current iteration.

  int *fake_degrees = new int[nev+nex];
  for( int i = 0; i < nevex; ++i)
    fake_degrees[i] = deg;
  int fake_deg = deg;

  lanczos(H, randomVector, N, nevex, omp_lanczos, tol,
          int_mode == CHASE_MODE_RANDOM,
          (int_mode == CHASE_MODE_RANDOM) ? ritzv : NULL, &upperb);
  delete[] randomVector; // Not needed anymore.


  while( converged < nev && iteration < omp_maxiter)
  {

    lambda = * std::min_element( ritzv_, ritzv_ + nevex );
    lowerb = * std::max_element( ritzv_, ritzv_ + nevex );

    assert( lowerb < upperb );

    std::cout
      << "iteration: " << iteration
      << "\t" << lambda << " " << lowerb
      << " " << unconverged
      << std::endl;

    //-------------------------------------FILTER-----------------------------
    if( int_opt == CHASE_OPT_NONE || iteration == 0 )
    {
      chase_filteredVecs += (unconverged+nex)*deg; // Number of filtered vectors

      filter(H, V, N, unconverged+nex, deg, lambda, lowerb, upperb, W);
    }
    else
    {
      deg = calc_degrees(
        N, unconverged, upperb, lowerb, tol,
        ritzv, resid, degrees,
        V, W );

      for( auto i = 0 ; i < unconverged ; ++i)
        chase_filteredVecs += degrees[i];
      chase_filteredVecs += nex*deg;

      filterModified(H, V, N, unconverged+nex, nev, deg, degrees,
                     lambda, lowerb, upperb, W, unconverged);

    }
    if(deg%2 == 0) // To 'fix' the situation if the filter swapped them.
    {
      MKL_Complex16 *swap = NULL; // Just a pointer for swapping V and W.
      swap = V; V = W; W = swap;
      swap = V_Full; V_Full = W_Full; W_Full = swap;
    }
    // Solution is in W

    //----------------------------------  QR  -------------------------------
    QR( N, nevex, converged, W_Full, V);

    //-------------------------------------RAYLEIGH-RITZ------------------------
    RR(
      N, unconverged+nex,
      ritzv,
      H, V, W );
    // returns eigenvectors in V

    //----------------------------------RESIDUAL-----------------------------

    resd(
      N, unconverged,
      ritzv, resid,
      H, V, W );
    // overwrites W with H*V

    //---------------------------------- LOCKING -----------------------------
    int new_converged = locking(
      N, unconverged, tol,
      ritzv, resid, degrees,
      V );

    //-------------------- update pointers and counts --------------------
    converged += new_converged;
    unconverged -= new_converged;

    memcpy(W, V, N*new_converged*sizeof(MKL_Complex16));

    W += N*new_converged;
    V += N*new_converged;
    resid += new_converged;
    ritzv += new_converged;
    degrees += new_converged;


    //memcpy(W, V, N*nevex*sizeof(MKL_Complex16));
    iteration++;
  } // while ( converged < nev && iteration < omp_maxiter )

  //-----------------------SORT-EIGENPAIRS-ACCORDING-TO-EIGENVALUES-------------
  for( auto i = 0; i < nev-1; ++i )
    for( auto j = i+1; j < nev; ++j )
    {
      if( ritzv_[i] > ritzv_[j] )
      {
        swap_kj( i, j, ritzv_ );
        ColSwap( V_Full, N, i, j );
      }
    }
  //----------------------------------------------------------------------------

  omp_iteration = iteration;

  delete[] resid_;
}

void get_iteration(int* iteration)
{
  *iteration = omp_iteration;
}


void get_time(double* time)
{
  for (int i = 0; i < 5; ++i)
    time[i] = omp_time[5+i];
}

void get_filteredVecs(int* filteredVecs)
{
  *filteredVecs = chase_filteredVecs;
}

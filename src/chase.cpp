#include "../include/chfsi.h"
#include "../include/lanczos.h"
#include "../include/filter.h"

#include <algorithm>
template <typename T>
void swap_kj(int k, int j, T* array)
{
  T tmp = array[k];
  array[k] = array[j];
  array[j] = tmp;
  return;
}

void ColSwap( MKL_Complex16 *V, int N, int i, int j )
{
  MKL_Complex16 *ztmp = new MKL_Complex16[N];
  memcpy(ztmp,  V+N*i, N*sizeof(MKL_Complex16));
  memcpy(V+N*i, V+N*j, N*sizeof(MKL_Complex16));
  memcpy(V+N*j, ztmp,  N*sizeof(MKL_Complex16));
}

int calc_degrees( int N, int converged, int nev, double upperb, double lowerb, double tol,
             double *ritzv, double *rho, int *degrees, double *resid,
             MKL_Complex16 *V)
{

  double c = (upperb + lowerb) / 2; // Center of the interval.
  double e = (upperb - lowerb) / 2; // Half-length of the interval.

  for( auto i=converged; i < nev; ++i )
  {
    double t = (ritzv[i] - c)/e;
    rho[i] = std::max(
      abs( t - sqrt( t*t-1 ) ),
      abs( t + sqrt( t*t-1 ) )
      );

    degrees[i] = ceil(fabs(log(resid[i]/(tol))/log(rho[i])));
    degrees[i] = std::min( degrees[i] + omp_delta, omp_degmax );
  }

  for (int j = converged; j < nev-1; ++j)
    for (int k = j+1; k < nev; ++k)
      if (degrees[k] < degrees[j])
      {
        swap_kj(k, j, degrees);
        ColSwap( V, N, k, j );
      }

  return std::min( omp_degmax, degrees[nev-1] );
}

int locking( int N, int unconverged, double tol,
             double *resid, MKL_Complex16 *V, int *degrees, double *ritzv )
{
  int converged = 0;
  for (int j = 0; j < unconverged; ++j)
  {
    if (resid[j] > tol) continue;
    if (j != converged)
    {
      if (degrees != NULL) swap_kj(j, converged, degrees);
      //swap_perm(j, converged, pi, pi_inv);
      swap_kj( j, converged, ritzv );
      ColSwap( V, N, j, converged );
      swap_kj(j, converged, resid);
    }
    converged++;
  }
  return converged;
}

int resd( int N, int unconverged,
           double *resid, double *ritzv,
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
  //MKL_Complex16 *saveW = new MKL_Complex16[N*converged];
  //memcpy( saveW, W, N*converged );
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

 //memcpy( W, saveW, N*converged );
 //delete[] saveW;
}

void RR( int N, int block,  MKL_Complex16 *H, MKL_Complex16 *W, MKL_Complex16 *V,
         double *Lambda)
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
           double* ritzv, int nev, const int nex, int deg, int* const degrees,
           const double tol, const CHASE_MODE_TYPE int_mode,
           const CHASE_OPT_TYPE int_opt)
{
  MKL_Complex16 *RESULT = W;

  double* resid = new double[nev];
  double* rho = new double[nev];

  const int blk = nev + nex; // Block size for the algorithm.
  int block = nev + nex; // Number of non converged eigenvalues. NOT constant!
  int nevex = nev+nex;

  // To store the approximations obtained from lanczos().
  double lowerb, upperb, lambda;


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


  lanczos(H, randomVector, N, blk, omp_lanczos, tol,
          int_mode == CHASE_MODE_RANDOM,
          (int_mode == CHASE_MODE_RANDOM) ? ritzv : NULL, &upperb);
  delete[] randomVector; // Not needed anymore.


  // rename block to unconverged or something
  // rename V to V_Full, V+N*converged to V, same for W

  while(converged < nev && iteration < omp_maxiter)
  {

    lambda = * std::min_element( ritzv, ritzv + nev + nex );
    lowerb = * std::max_element( ritzv, ritzv + nev + nex );

    // todo check lowerb < upperb

    std::cout << "iteration: " << iteration << std::endl;

    if( int_opt == CHASE_OPT_NONE || iteration == 0 )
    {
      chase_filteredVecs += block*deg; // Number of filtered vectors.

      //-------------------------------------FILTER-----------------------------
      filter(H, V + N*converged, N, block, deg, lambda, lowerb, upperb,
             W + N*converged);

      if(deg%2 == 0) // To 'fix' the situation if the filter swapped them.
      {
        MKL_Complex16 *swap = NULL; // Just a pointer for swapping V and W.
        swap = V; V = W; W = swap;
      }
    }
    else
    {
      deg = calc_degrees(
        N, converged, nev, upperb, lowerb, tol,
        ritzv, rho, degrees, resid,
        V );

      // TODO: this looks fishy on first glance
      for( auto i = converged ; i < nev ; ++i)
      {
        chase_filteredVecs += degrees[i];
      }
      chase_filteredVecs += nex*deg;

      filterModified(H, V + N*converged, N, block, nev, deg, degrees+converged,
                     lambda, lowerb, upperb, W + N*converged, block-nex);

      if(deg%2 == 0) // To 'fix' the situation if the filter swapped them.
      {
        MKL_Complex16 *swap = NULL; // Just a pointer for swapping V and W.
        swap = V; V = W; W = swap;
      }
    }

    //----------------------------------  QR  -------------------------------
    QR( N, blk, converged, W, V+N*converged);

    //-------------------------------------RAYLEIGH-RITZ------------------------
    RR(
      N, block,
      H, W + N*converged, V+N*converged, ritzv + converged );

    resd(
      N, nev-converged,
      resid + converged, ritzv + converged,
      H, V + N*converged, W + N*converged );

    int new_converged = locking(
      N, nev-converged, tol,
      resid+converged, V+N*converged, degrees + converged, ritzv+converged );

    block -= new_converged;
    // TODO
    memcpy(W+N*converged, V+N*converged, N*new_converged*sizeof(MKL_Complex16));
    converged += new_converged;

    iteration++;
  } // while ( converged < nev && iteration < omp_maxiter )

  //-----------------------SORT-EIGENPAIRS-ACCORDING-TO-EIGENVALUES-------------
  for( auto i = 0; i < nev; ++i )
    for( auto j = 0; j < nev; ++j )
    {
      if( ritzv[i] < ritzv[j] )
      {
        swap_kj( i, j, ritzv );
        ColSwap( V, N, i, j );
      }
    }
  //----------------------------------------------------------------------------

  omp_iteration = iteration;

  //TODO
  memcpy( W, V, N*(converged)*sizeof(MKL_Complex16));

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

#include "../include/chase.h"

template <typename T>
void swap_kj(std::size_t k, std::size_t j, T* array);
void check_params(std::size_t N, std::size_t nev, std::size_t nex,
		  const double tol, std::size_t deg_);

void chase(MKL_Complex16* const H, std::size_t N, MKL_Complex16* V, MKL_Complex16* W,
           double* ritzv_, std::size_t nev, const std::size_t nex, std::size_t deg_, std::size_t* const degrees_,
           const double tol_, const CHASE_MODE_TYPE int_mode,
           const CHASE_OPT_TYPE int_opt)
{
  //Parameter check
  check_params(N, nev, nex, tol_, deg_);

  start_clock( TimePtrs::All );

  chase_filtered_vecs = 0;
  const std::size_t nevex = nev + nex; // Block size for the algorithm.
  std::size_t unconverged = nev + nex;

  // To store the approximations obtained from lanczos().
  double lowerb, upperb, lambda;
  double normH = std::max(LAPACKE_zlange(LAPACK_COL_MAJOR, '1', N, N, H, N), 1.0);
  const double tol = tol_ * normH;
  double* resid_ = new double[nevex];

  // store input values
  std::size_t deg = deg_;
  std::size_t * degrees = degrees_;
  double * ritzv = ritzv_;
  double * resid = resid_;

  MKL_Complex16 *V_Full = V;
  MKL_Complex16 *W_Full = W;

  //-------------------------------- VALIDATION --------------------------------
  assert(degrees != NULL);
  deg = std::min( deg, chase_max_deg );
  for( auto i = 0; i < nevex; ++i )
    degrees[i] = deg;

  // --------------------------------- LANCZOS ---------------------------------
  start_clock( TimePtrs::Lanczos );
  bool random = int_mode == CHASE_MODE_RANDOM;
  std::size_t vecLanczos = lanczos(H, N, 4, random? std::max((std::size_t)1,nevex/4) : chase_lanczos_iter, nevex, &upperb,
          random, random? ritzv : NULL, random? V : NULL );

  end_clock( TimePtrs::Lanczos );
  std::size_t locked = 0; // Number of converged eigenpairs.
  std::size_t iteration = 0; // Current iteration.
  lowerb = * std::max_element( ritzv, ritzv + unconverged );

  while( unconverged > nex && iteration < 4*chase_max_iter)
  {
    if(unconverged < nevex || iteration == 0){
      lambda = * std::min_element( ritzv_, ritzv_ + nevex );
      auto tmp = * std::max_element( ritzv, ritzv + unconverged );
      lowerb = (lowerb + tmp ) / 2;
      //upperb = lowerb + std::abs(lowerb - lambda);
    }
#ifdef OUTPUT
    std::cout
      << "iteration: " << iteration
      << "\t" << lambda << " " << lowerb << " " << upperb
      << " " << unconverged
      << std::endl;
#endif
    //    assert( lowerb < upperb );
    if( lowerb > upperb ) {
      std::cout << "ASSERTION FAILURE lowerb > upperb\n";
      lowerb = upperb;
    }
    //--------------------------------- FILTER ---------------------------------
    if( int_opt != CHASE_OPT_NONE && iteration != 0 )
    {
      deg = calc_degrees(
        N, unconverged, nex, upperb, lowerb, tol, ritzv,
        resid, degrees, V, W );
    }


    start_clock( TimePtrs::Filter );
#ifdef CHASE_BUILD_CUDA

    std::size_t Av = cuda_filter(
      H, V, N, unconverged, deg, degrees,
      lambda, lowerb, upperb, W );

#else
    std::size_t Av = filter(
      H, V, N, unconverged, deg, degrees,
      lambda, lowerb, upperb, W );

#endif
    end_clock( TimePtrs::Filter );

    chase_filtered_vecs += Av;

    if( deg % 2 == 0 ) // To 'fix' the situation if the filter swapped them.
    {
      std::swap( V, W );
      std::swap( V_Full, W_Full );
    }
    // Solution is now in W

    //----------------------------------- QR -----------------------------------
    // Orthogonalize W_Full, then copy the locked vector from V_Full to W_Full
    QR( N, nevex, locked, W_Full, V, V_Full);

    // ----------------------------- RAYLEIGH  RITZ ----------------------------
    // returns eigenvectors in V
    RR( N, unconverged, ritzv, H, V, W );

    // -------------------------------- RESIDUAL -------------------------------
    // overwrites W with H*V
    calc_residuals( N, unconverged, normH, ritzv, resid, H, V, W );

    // -------------------------------- LOCKING --------------------------------
    std::size_t new_converged = locking(
      N, unconverged, tol, ritzv, resid, degrees, V );

    // ---------------------------- Update pointers ----------------------------
    // Since we double buffer we need the entire locked portion in W and V
    memcpy( W, V, N*new_converged*sizeof(MKL_Complex16) );

    locked += new_converged;
    unconverged -= new_converged;

    W += N*new_converged;
    V += N*new_converged;
    resid += new_converged;
    ritzv += new_converged;
    degrees += new_converged;

    iteration++;
  } // while ( converged < nev && iteration < omp_maxiter )

  //---------------------SORT-EIGENPAIRS-ACCORDING-TO-EIGENVALUES---------------
  for( auto i = 0; i < nev-1; ++i )
    for( auto j = i+1; j < nev; ++j )
    {
      if( ritzv_[i] > ritzv_[j] )
      {
        swap_kj( i, j, ritzv_ );
        ColSwap( W_Full, N, i, j );
        ColSwap( V_Full, N, i, j );
      }
    }


  chase_iteration_count = iteration;
  delete[] resid_;

  end_clock( TimePtrs::All );
}


std::size_t get_iter_count()
{
  return chase_iteration_count;
}


std::size_t get_filtered_vecs()
{
  return chase_filtered_vecs;
}


template <typename T>
void swap_kj(std::size_t k, std::size_t j, T* array)
{
  T tmp = array[k];
  array[k] = array[j];
  array[j] = tmp;
}


void ColSwap( MKL_Complex16 *V, std::size_t N, std::size_t i, std::size_t j )
{
  MKL_Complex16 *ztmp = new MKL_Complex16[N];
  memcpy(ztmp,  V+N*i, N*sizeof(MKL_Complex16));
  memcpy(V+N*i, V+N*j, N*sizeof(MKL_Complex16));
  memcpy(V+N*j, ztmp,  N*sizeof(MKL_Complex16));
  delete[] ztmp;
}


std::size_t calc_degrees( std::size_t N, std::size_t unconverged, std::size_t nex,  double upperb, double lowerb,
                  double tol, double *ritzv, double *resid, std::size_t *degrees,
                  MKL_Complex16 *V, MKL_Complex16 *W)
{
  start_clock( TimePtrs::Degrees );
  double c = (upperb + lowerb) / 2; // Center of the interval.
  double e = (upperb - lowerb) / 2; // Half-length of the interval.
  double rho;

  for( auto i = 0; i < unconverged-nex; ++i )
  {
    double t = (ritzv[i] - c)/e;
    rho = std::max(
      std::abs( t - sqrt( t*t-1 ) ),
      std::abs( t + sqrt( t*t-1 ) )
      );

    degrees[i] = ceil(std::abs(log(resid[i]/(tol))/log(rho)));
    degrees[i] = std::min(
      degrees[i] + chase_deg_extra,
      chase_max_deg
      );
  }

  for( auto i = unconverged-nex; i < unconverged; ++i )
  {
    degrees[i] = degrees[unconverged-1-nex];
  }

  for( auto  j = 0; j < unconverged-1; ++j )
    for( auto  k = j; k < unconverged; ++k )
      if( degrees[k] < degrees[j] )
      {
        swap_kj(k, j, degrees); // for filter
        swap_kj(k, j, ritzv);
//        swap_kj(k, j, resid);
        ColSwap( V, N, k, j );
      }

  end_clock( TimePtrs::Degrees );
  return degrees[unconverged-1];
}


std::size_t locking( std::size_t N, std::size_t unconverged, double tol,
             double *ritzv, double *resid, std::size_t *degrees,
             MKL_Complex16 *V)
{
  // we build the permutation
  std::vector<int> index(unconverged, 0);
for (int i = 0 ; i != index.size() ; i++) {
    index[i] = i;
}
sort(index.begin(), index.end(),
    [&](const int& a, const int& b) {
        return (ritzv[a] < ritzv[b]);
    }
);


  std::size_t converged = 0;
  for (auto k = 0; k < unconverged; ++k)
  {
    auto j = index[k]; // walk through 
    if (resid[j] > tol) break;
    if (j != converged)
    {
      swap_kj( j, converged, resid ); // if we filter again
      swap_kj( j, converged, ritzv );
      ColSwap( V, N, j, converged );
    }
    converged++;
  }
  end_clock( TimePtrs::Resids_Locking );
  return converged;
}


void calc_residuals(
  std::size_t N, std::size_t unconverged, double norm,
  double *ritzv, double *resid, MKL_Complex16 *H, MKL_Complex16 *V,
  MKL_Complex16 *W)
{
  start_clock( TimePtrs::Resids_Locking );
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
  for( auto i = 0 ; i < unconverged ; ++i )
  {
    beta  = MKL_Complex16 (-ritzv[i], 0.0);
    cblas_zaxpy( N, &beta, V+N*i, 1, W+N*i, 1);

    norm1 = cblas_dznrm2( N, W+N*i, 1);
    resid[i] = norm1/norm;
  }
}


void QR( std::size_t N, std::size_t nevex, std::size_t converged, MKL_Complex16 *W,
         MKL_Complex16 *tau, MKL_Complex16 *saveW )
{
  start_clock( TimePtrs::Qr );
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
  end_clock( TimePtrs::Qr );
}


void RR( std::size_t N, std::size_t block, double *Lambda,
         MKL_Complex16 *H, MKL_Complex16 *V, MKL_Complex16 *W )
{
  start_clock( TimePtrs::Rr );
  MKL_Complex16 * A = new MKL_Complex16[block*block]; // For LAPACK.
  MKL_Complex16 * Z = new MKL_Complex16[block*block]; // For LAPACK.
  int * isuppz = new int[2*block];


  MKL_Complex16 One = MKL_Complex16 (1.0, 0.0);
  MKL_Complex16 Zero  = MKL_Complex16 (0.0, 0.0);
  int m;

  // V <- H*V
  cblas_zgemm(
    CblasColMajor,    CblasNoTrans,    CblasNoTrans,
    N,    block,    N,
    &One,
    H,    N,
    W,    N,
    &Zero,
    V,    N
    );

  // A <- W * V
  cblas_zgemm(
    CblasColMajor,    CblasConjTrans,    CblasNoTrans,
    block,    block,    N,
    &One,
    W,    N,
    V,    N,
    &Zero,
    A,    block
    );

  // LAPACKE_zheevd(
  //   LAPACK_COL_MAJOR,
  //   'V',    'L',
  //   block,
  //   A,    block,
  //   Lambda
  //   );

  LAPACKE_zheevr ( LAPACK_COL_MAJOR, 'V', 'A', 'L', block, A, block,
                   1.0, 1.0, 1, 1, 1e-12, &m, Lambda, Z, block,
                   isuppz );
  cblas_zcopy( block*block, Z, 1, A, 1 );

  cblas_zgemm(
    CblasColMajor,    CblasNoTrans,    CblasNoTrans,
    N,    block,    block,
    &One,
    W,    N,
    A,    block,
    &Zero,
    V,    N
    );

  delete[] A;
  delete[] Z;
  delete[] isuppz;

  end_clock( TimePtrs::Rr );
}

void check_params(std::size_t N, std::size_t nev, std::size_t nex,
		  const double tol, std::size_t deg)
{
  bool abort_flag = false;
  if(tol < 1e-14)
    std::clog << "WARNING: Tolerance too small, may take a while." << std::endl;
  if(deg < 8 || deg > chase_max_deg)
    std::clog << "WARNING: Degree should be between 8 and " << chase_max_deg << "."
              << " (current: " << deg << ")" << std::endl;
  if((double)nex/nev < 0.15 || (double)nex/nev > 0.75)
  {
    std::clog << "WARNING: NEX should be between 0.15*NEV and 0.75*NEV."
              << " (current: " << (double)nex/nev << ")" << std::endl;
    //abort_flag = true;
  }
  if(nev+nex > N)
  {
    std::cerr << "ERROR: NEV+NEX has to be smaller than N." << std::endl;
    abort_flag = true;
  }

  if(abort_flag)
  {
    std::cerr << "Stopping execution." << std::endl;
    exit(-1);
  }
}

extern "C" {
  void c_chase_(MKL_Complex16* H, int *N, MKL_Complex16* V, MKL_Complex16* W,
                double* ritzv, int* nev, int* nex, int* deg,
                double *tol, char* mode, char* opt)
  {
    size_t *degrees = (size_t*) malloc( (*nev+*nex)*sizeof( size_t ) );

    chase( H, static_cast<std::size_t>(*N), V, W, ritzv,
           static_cast<std::size_t>(*nev), static_cast<std::size_t>(*nex),
           *deg, degrees, *tol, *mode, *opt);

    free(degrees);
  }
}

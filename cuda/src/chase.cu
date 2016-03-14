#include "../include/chase.h"
#include "../include/cuda_util.h"
#include <iostream>

template <typename T>
void swap_kj(int k, int j, T* array);

void chase(MKL_Complex16* const H, int N, MKL_Complex16* V, MKL_Complex16* W,
           double* ritzv_, int nev, const int nex, int deg_, int* const degrees_,
           const double tol_, const CHASE_MODE_TYPE int_mode,
           const CHASE_OPT_TYPE int_opt)
{
  start_clock( TimePtrs::All );

  chase_filtered_vecs = 0;
  const int nevex = nev + nex; // Block size for the algorithm.
  int unconverged = nev + nex;

  // To store the approximations obtained from lanczos().
  double lowerb, upperb, lambda;
  double normH = std::max(
    LAPACKE_zlange(LAPACK_COL_MAJOR, '1', N, N, H, N), 1.0 );
  const double tol = tol_ * normH;

  double* resid_ = new double[nevex];

  // store input values
  int deg = deg_;
  int * degrees = degrees_;
  double * ritzv = ritzv_;
  double * resid = resid_;

  MKL_Complex16 *V_ = V;
  MKL_Complex16 *W_ = W;

  cublasHandle_t handle;
  cudaStream_t stream;
  cudaEvent_t event;

  //cudaSetDevice(0);
  cudaStreamCreate(&stream);
  cublasCreate(&handle);
  cublasSetStream(handle, stream);
  cudaEventCreate( &event );

  cuDoubleComplex *dH;
  cuDoubleComplex *dV_, *dV;
  cuDoubleComplex *dW_, *dW;
  cuDoubleComplex *ztmp = NULL;

  std::size_t sizeA = N * N * sizeof(cuDoubleComplex);
  std::size_t sizeV = N * unconverged * sizeof(cuDoubleComplex);

  HANDLE_ERROR( cudaMalloc((void**) &(dH), sizeA) );  //sync

  HANDLE_ERROR( cudaMalloc((void**) &(dV_), sizeV) );  //sync
  HANDLE_ERROR( cudaMalloc((void**) &(dW_), sizeV) );  //sync
  dV = dV_;
  dW = dW_;

  // For zgeqrf, part of the QR orthogonalization
  int nb = magma_get_zgeqrf_nb( N, nevex );
  int zmem_size =
    ( 2 * std::min( N, nevex ) + std::ceil( N/32 ) * 32 ) * nb
    * sizeof(magmaDoubleComplex_ptr);
  HANDLE_ERROR( cudaMalloc((void**) &(ztmp), zmem_size) );

  HANDLE_ERROR( cudaMemcpyAsync(dH, H, sizeA, cudaMemcpyHostToDevice, stream));
  HANDLE_ERROR( cudaMemcpyAsync(dV, V, sizeV, cudaMemcpyHostToDevice, stream));
  HANDLE_ERROR( cudaMemcpyAsync(dW, W, sizeV, cudaMemcpyHostToDevice, stream));

  //-------------------------------- VALIDATION --------------------------------
  assert(degrees != NULL);
  deg = std::min( deg, chase_max_deg );
  for( auto i = 0; i < nevex; ++i )
    degrees[i] = deg;

  // --------------------------------- LANCZOS ---------------------------------
  start_clock( TimePtrs::Lanczos );
  bool random = int_mode == CHASE_MODE_RANDOM;
  int vecLanczos = lanczos(
    H, N, 4, random? nevex/4 : chase_lanczos_iter, nevex,
    &upperb, random, random? ritzv : NULL, random? V : NULL );

  HANDLE_ERROR(
    cudaMemcpyAsync(dV, V, N * vecLanczos * sizeof(cuDoubleComplex),
                    cudaMemcpyHostToDevice, stream)); //sync

  end_clock( TimePtrs::Lanczos );

  int locked = 0; // Number of converged eigenpairs.
  int iteration = 0; // Current iteration.

  while( unconverged > nex && iteration < chase_max_iter)
  {
    lambda = * std::min_element( ritzv_, ritzv_ + nevex );
    lowerb = * std::max_element( ritzv, ritzv + unconverged );

    assert( lowerb < upperb );
    std::cout
      << "iteration: " << iteration
      << "\t" << lambda << " " << lowerb << " " << upperb
      << " " << unconverged
      << std::endl;

    //--------------------------------- FILTER ---------------------------------
    cudaDeviceSynchronize();

    if( int_opt != CHASE_OPT_NONE && iteration != 0 )
    {
      deg = calc_degrees(
       N, unconverged, nex, upperb, lowerb, tol, ritzv,
       resid, degrees, dV, dW, ztmp, &handle );
    }

    start_clock( TimePtrs::Filter );
    int Av = cuda_filter(
      dH, dV, N, unconverged, deg, degrees,
      lambda, lowerb, upperb, dW, &stream, &handle );
    cudaDeviceSynchronize();

    end_clock( TimePtrs::Filter );

    chase_filtered_vecs += Av;

    if( deg % 2 == 0 ) // To 'fix' the situation if the filter swapped them.
    {
      std::swap( V, W );
      std::swap( V_, W_ );
      std::swap( dV, dW );
      std::swap( dV_, dW_ );
    }
    // Solution is now in W

    //----------------------------------- QR -----------------------------------
    // Orthogonalize W_, then copy the locked vector from V_ to W_
    start_clock( TimePtrs::Qr );
    int magma_info;

    magma_zgeqrf_gpu(
      N,      nevex,
      dW_,      N,
      (magmaDoubleComplex*) V,
      ztmp,
      &magma_info
      );

    magma_zungqr_gpu(
      N,      nevex,      nevex,
      dW_,      N,
      (magmaDoubleComplex*) V,
      ztmp,
      nb,
      &magma_info
      );

    HANDLE_ERROR(
      cudaMemcpyAsync(dW_, dV_, N*locked*sizeof( cuDoubleComplex ),
                      cudaMemcpyDeviceToDevice, stream) );

    end_clock( TimePtrs::Qr );

    // ----------------------------- RAYLEIGH  RITZ ----------------------------
    // returns eigenvectors in V
    RR( N, unconverged, ritzv, dH, dV, dW, &handle, ztmp );

    // -------------------------------- RESIDUAL -------------------------------
    // overwrites W with H*V
    // synchronizes CPU and GPU
    calc_residuals( N, unconverged, normH, ritzv, resid, dH, dV, dW, W,
                    &handle, &stream );

    // -------------------------------- LOCKING --------------------------------
    int new_converged = locking(
      N, unconverged, tol, ritzv, resid, degrees, dV, &handle, ztmp );

    // ---------------------------- Update pointers ----------------------------
    // Since we double buffer we need the entire locked portion in W and V
    sizeV = N*new_converged*sizeof(cuDoubleComplex);
    HANDLE_ERROR(
      cudaMemcpyAsync(V, dV, sizeV, cudaMemcpyDeviceToHost, stream) );
    HANDLE_ERROR(
      cudaMemcpyAsync(W, dW, sizeV, cudaMemcpyDeviceToHost, stream) );

    HANDLE_ERROR(
      cudaMemcpyAsync(dW, dV, sizeV, cudaMemcpyDeviceToDevice, stream) );
    memcpy( W, V, N*new_converged*sizeof(MKL_Complex16) );

    locked += new_converged;
    unconverged -= new_converged;

    W += N*new_converged;
    V += N*new_converged;
    dW += N*new_converged;
    dV += N*new_converged;
    resid += new_converged;
    ritzv += new_converged;
    degrees += new_converged;

    iteration++;
  } // while ( converged < nev && iteration < omp_maxiter )

  cudaDeviceSynchronize();
  //---------------------SORT-EIGENPAIRS-ACCORDING-TO-EIGENVALUES---------------
  for( auto i = 0; i < nev-1; ++i )
    for( auto j = i+1; j < nev; ++j )
    {
      if( ritzv_[i] > ritzv_[j] )
      {
        swap_kj( i, j, ritzv_ );
        ColSwap( W_, N, i, j );
        ColSwap( V_, N, i, j );
      }
    }

  chase_iteration_count = iteration;
  delete[] resid_;

  cudaFree (dV_);
  cudaFree (dH);
  cudaFree (dW_);
  cudaFree (ztmp);
  cudaStreamDestroy(stream);
  cublasDestroy(handle);

  end_clock( TimePtrs::All );
}


int get_iter_count()
{
  return chase_iteration_count;
}


int get_filtered_vecs()
{
  return chase_filtered_vecs;
}


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


int calc_degrees( int N, int unconverged, int nex,  double upperb, double lowerb,
                  double tol, double *ritzv, double *resid, int *degrees,
                  cuDoubleComplex *V, cuDoubleComplex *W, cuDoubleComplex *ztmp,
                  cublasHandle_t *handle )
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

        HANDLE_ERROR_CUBLAS(
          cublasZcopy(*handle, N,
                      V+N*j, 1,
                      ztmp, 1
            ));
        HANDLE_ERROR_CUBLAS(
          cublasZcopy(*handle, N,
                      V+N*j, 1,
                      V+N*j, 1
            ));
        HANDLE_ERROR_CUBLAS(
          cublasZcopy(*handle, N,
                      ztmp, 1,
                      V+N*j, 1
            ));
      }
  end_clock( TimePtrs::Degrees );
  return degrees[unconverged-1];
}


int locking( int N, int unconverged, double tol,
             double *ritzv, double *resid, int *degrees,
             cuDoubleComplex *V, cublasHandle_t *handle, cuDoubleComplex *ztmp )
{

  int converged = 0;
  for (int j = 0; j < unconverged; ++j)
  {
    if (resid[j] > tol) continue;
    if (j != converged)
    {
      swap_kj( j, converged, resid ); // if we filter again
      swap_kj( j, converged, ritzv );
      //ColSwap( V, N, j, converged, handle, ztmp );

      HANDLE_ERROR_CUBLAS(
        cublasZcopy(*handle, N,
                    V+N*converged, 1,
                    ztmp, 1
          ));
      HANDLE_ERROR_CUBLAS(
        cublasZcopy(*handle, N,
                    V+N*j, 1,
                    V+N*converged, 1
          ));
      HANDLE_ERROR_CUBLAS(
        cublasZcopy(*handle, N,
                    ztmp, 1,
                    V+N*j, 1
          ));

    }
    converged++;
  }
  end_clock( TimePtrs::Resids_Locking );
//  cudaFree( ztmp );

  return converged;
}

void calc_residuals(
  int N, int unconverged, double norm, double *ritzv, double *resid,
  cuDoubleComplex *dH, cuDoubleComplex *dV, cuDoubleComplex *dW, MKL_Complex16 *W,
  cublasHandle_t *handle, cudaStream_t *stream)
{
  start_clock( TimePtrs::Resids_Locking );
  cuDoubleComplex One = {1.0, 0.0};
  cuDoubleComplex Zero  = {0.0, 0.0};

  HANDLE_ERROR_CUBLAS(
    cublasZgemm(
      *handle,
      CUBLAS_OP_N, CUBLAS_OP_N,
      N, unconverged, N,
      &One,
      dH, N,
      dV, N,
      &Zero,
      dW, N
      )
    );

  for( int i = 0 ; i < unconverged ; ++i )
  {
    cuDoubleComplex beta  = {-ritzv[i], 0.0};
    //      cblas_zaxpy( N, &beta, V+N*i, 1, W+N*i, 1);
    HANDLE_ERROR_CUBLAS(
      cublasZaxpy(*handle, N,
                  &beta,
                  dV+N*i, 1,
                  dW+N*i, 1 ) );

    HANDLE_ERROR_CUBLAS(
      cublasDznrm2(*handle, N,
                   dW+N*i, 1, resid+i ) );
  }

  HANDLE_ERROR(
    cudaMemcpyAsync(W, dW, N*unconverged*sizeof(cuDoubleComplex),
                    cudaMemcpyDeviceToHost, *stream));
  cudaDeviceSynchronize();
  for( int i = 0 ; i < unconverged ; ++i )
  {
    // TODO use zscal
    resid[i] = resid[i]/norm;
  }
}


void RR( int N, int block, double *Lambda,
         cuDoubleComplex *H, cuDoubleComplex *V, cuDoubleComplex *W,
         cublasHandle_t *handle, cuDoubleComplex *dA)
{
  start_clock( TimePtrs::Rr );

  cuDoubleComplex One = {1.0, 0.0};
  cuDoubleComplex Zero  = {0.0, 0.0};

  HANDLE_ERROR_CUBLAS(
    cublasZgemm(
      *handle,
      CUBLAS_OP_N, CUBLAS_OP_N,
      N, block, N,
      &One,
      H, N,
      W, N,
      &Zero,
      V, N
      )
    );

  HANDLE_ERROR_CUBLAS(
    cublasZgemm(
      *handle,
      CUBLAS_OP_C, CUBLAS_OP_N,
      block, block, N,
      &One,
      W, N,
      V, N,
      &Zero,
      dA, block
      )
    );

  int nb = magma_get_zhetrd_nb( block );
  int lwork = std::max( block + block*nb, 2*block + block*block );
  int liwork = 3 + 5*block;
  int lrwork = 1 + 5*block + 2*block*block;
  int info;

  magmaDoubleComplex *wA = new magmaDoubleComplex[block*block];
  magmaDoubleComplex *work = new magmaDoubleComplex[lwork];
  double *rwork = new double[lrwork];
  int *iwork = new int[liwork];

  magma_zheevd_gpu(
    MagmaVec,
    MagmaLower,
    block,
    dA,    block,
    Lambda,
    wA,    block,
    work,    lwork,
    rwork,    lrwork,
    iwork,    liwork,
    &info
    );

  delete[] wA;
  delete[] rwork;
  delete[] work;
  delete[] iwork;

  HANDLE_ERROR_CUBLAS(
    cublasZgemm(
      *handle,
      CUBLAS_OP_N, CUBLAS_OP_N,
      N, block, block,
      &One,
      W, N,
      dA, block,
      &Zero,
      V, N
      )
    );

  end_clock( TimePtrs::Rr );
}

#include "lacpy.cuh"
#include <algorithm>

#define BLK_X 64
#define BLK_Y BLK_X
const std::size_t max_blocks = 65535;

namespace chase
{
namespace linalg
{
namespace internal
{
namespace cuda
{

__device__ inline std::size_t device_min(std::size_t a, std::size_t b) {
    return (a < b) ? a : b;
}

__device__ inline std::size_t device_max(std::size_t a, std::size_t b) {
    return (a > b) ? a : b;
}

static __device__ void dlacpy_full_device(
    std::size_t m, std::size_t n,
    const double *dA, std::size_t ldda,
    double       *dB, std::size_t lddb )
{
    std::size_t ind = blockIdx.x*BLK_X + threadIdx.x;
    std::size_t iby = blockIdx.y*BLK_Y;
    bool full = (iby + BLK_Y <= n);
    if ( ind < m ) {
        dA += ind + iby*ldda;
        dB += ind + iby*lddb;
        if ( full ) {
            #pragma unroll
            for( std::size_t j=0; j < BLK_Y; ++j ) {
                dB[j*lddb] = dA[j*ldda];
            }
        }
        else {
            for( std::size_t j=0; j < BLK_Y && iby+j < n; ++j ) {
                dB[j*lddb] = dA[j*ldda];
            }
        }
    }
}

__global__ void dlacpy_full_kernel(
    std::size_t m, std::size_t n,
    const double *dA, std::size_t ldda,
    double       *dB, std::size_t lddb )
{
    dlacpy_full_device(m, n, dA, ldda, dB, lddb);
}


static __device__ void slacpy_full_device(
    std::size_t m, std::size_t n,
    const float *dA, std::size_t ldda,
    float       *dB, std::size_t lddb )
{
    std::size_t ind = blockIdx.x*BLK_X + threadIdx.x;
    std::size_t iby = blockIdx.y*BLK_Y;
    bool full = (iby + BLK_Y <= n);
    if ( ind < m ) {
        dA += ind + iby*ldda;
        dB += ind + iby*lddb;
        if ( full ) {
            #pragma unroll
            for( std::size_t j=0; j < BLK_Y; ++j ) {
                dB[j*lddb] = dA[j*ldda];
            }
        }
        else {
            for( std::size_t j=0; j < BLK_Y && iby+j < n; ++j ) {
                dB[j*lddb] = dA[j*ldda];
            }
        }
    }
}

__global__ void slacpy_full_kernel(
    std::size_t m, std::size_t n,
    const float *dA, std::size_t ldda,
    float       *dB, std::size_t lddb )
{
    slacpy_full_device(m, n, dA, ldda, dB, lddb);
}


static __device__ void zlacpy_full_device(
    std::size_t m, std::size_t n,
    const cuDoubleComplex *dA, std::size_t ldda,
    cuDoubleComplex       *dB, std::size_t lddb )
{
    std::size_t ind = blockIdx.x*BLK_X + threadIdx.x;
    std::size_t iby = blockIdx.y*BLK_Y;
    bool full = (iby + BLK_Y <= n);
    if ( ind < m ) {
        dA += ind + iby*ldda;
        dB += ind + iby*lddb;
        if ( full ) {
            #pragma unroll
            for( std::size_t j=0; j < BLK_Y; ++j ) {
                dB[j*lddb] = dA[j*ldda];
            }
        }
        else {
            for( std::size_t j=0; j < BLK_Y && iby+j < n; ++j ) {
                dB[j*lddb] = dA[j*ldda];
            }
        }
    }
}

__global__
void zlacpy_full_kernel(
    std::size_t m, std::size_t n,
    const cuDoubleComplex *dA, std::size_t ldda,
    cuDoubleComplex       *dB, std::size_t lddb )
{
    zlacpy_full_device(m, n, dA, ldda, dB, lddb);
}

static __device__ void clacpy_full_device(
    std::size_t m, std::size_t n,
    const cuComplex *dA, std::size_t ldda,
    cuComplex       *dB, std::size_t lddb )
{
    std::size_t ind = blockIdx.x*BLK_X + threadIdx.x;
    std::size_t iby = blockIdx.y*BLK_Y;
    bool full = (iby + BLK_Y <= n);
    if ( ind < m ) {
        dA += ind + iby*ldda;
        dB += ind + iby*lddb;
        if ( full ) {
            #pragma unroll
            for( std::size_t j=0; j < BLK_Y; ++j ) {
                dB[j*lddb] = dA[j*ldda];
            }
        }
        else {
            for( std::size_t j=0; j < BLK_Y && iby+j < n; ++j ) {
                dB[j*lddb] = dA[j*ldda];
            }
        }
    }
}

__global__ void clacpy_full_kernel(
    std::size_t m, std::size_t n,
    const cuComplex *dA, std::size_t ldda,
    cuComplex       *dB, std::size_t lddb )
{
    clacpy_full_device(m, n, dA, ldda, dB, lddb);
}

static __device__ void dlacpy_upper_device(
    std::size_t m, std::size_t n,
    const double *dA, std::size_t ldda,
    double       *dB, std::size_t lddb )
{
    std::size_t ind = blockIdx.x*BLK_X + threadIdx.x;
    std::size_t iby = blockIdx.y*BLK_Y;
    bool full = (iby + BLK_Y <= n);
    if ( ind < m ) {
        dA += ind + iby*ldda;
        dB += ind + iby*lddb;
        std::size_t copyLimit = device_min(ind - iby + 1, static_cast<std::size_t>(BLK_Y) );
        if ( full ) {
            #pragma unroll
            for( std::size_t j=0; j < BLK_Y; ++j ) {
                dB[j*lddb] = (j < copyLimit) ?  dA[j*ldda] : 0.0;
            }
        }
        else {
            for( std::size_t j=0; j < BLK_Y && iby+j < n; ++j ) {
                dB[j*lddb] =  (j < copyLimit) ?  dA[j*ldda] : 0.0;
            }
        }
    }
}

__global__ void dlacpy_upper_kernel(
    std::size_t m, std::size_t n,
    const double *dA, std::size_t ldda,
    double       *dB, std::size_t lddb )
{
    dlacpy_upper_device(m, n, dA, ldda, dB, lddb);
}

static __device__ void slacpy_upper_device(
    std::size_t m, std::size_t n,
    const float *dA, std::size_t ldda,
    float       *dB, std::size_t lddb )
{
    std::size_t ind = blockIdx.x*BLK_X + threadIdx.x;
    std::size_t iby = blockIdx.y*BLK_Y;
    bool full = (iby + BLK_Y <= n);
    if ( ind < m ) {
        dA += ind + iby*ldda;
        dB += ind + iby*lddb;
        std::size_t copyLimit = device_min(ind - iby + 1, static_cast<std::size_t>(BLK_Y) );
        if ( full ) {
            #pragma unroll
            for( std::size_t j=0; j < BLK_Y; ++j ) {
                dB[j*lddb] = (j < copyLimit) ?  dA[j*ldda] : 0.0;
            }
        }
        else {
            for( std::size_t j=0; j < BLK_Y && iby+j < n; ++j ) {
                dB[j*lddb] =  (j < copyLimit) ?  dA[j*ldda] : 0.0;
            }
        }
    }
}

__global__ void slacpy_upper_kernel(
    std::size_t m, std::size_t n,
    const float *dA, std::size_t ldda,
    float       *dB, std::size_t lddb )
{
    slacpy_upper_device(m, n, dA, ldda, dB, lddb);
}

static __device__ void clacpy_upper_device(
    std::size_t m, std::size_t n,
    const cuComplex *dA, std::size_t ldda,
    cuComplex       *dB, std::size_t lddb )
{
    std::size_t ind = blockIdx.x*BLK_X + threadIdx.x;
    std::size_t iby = blockIdx.y*BLK_Y;
    bool full = (iby + BLK_Y <= n);
    if ( ind < m ) {
        dA += ind + iby*ldda;
        dB += ind + iby*lddb;
        std::size_t copyLimit = device_min(ind - iby + 1, static_cast<std::size_t>(BLK_Y) );

        if ( full ) {
            #pragma unroll
            for( std::size_t j=0; j < BLK_Y; ++j ) {
                dB[j*lddb] = (j < copyLimit) ?  dA[j*ldda] : make_cuFloatComplex(0.0f, 0.0f);
            }
        }
        else {
            for( std::size_t j=0; j < BLK_Y && iby+j < n; ++j ) {
                dB[j*lddb] =  (j < copyLimit) ?  dA[j*ldda] : make_cuFloatComplex(0.0f, 0.0f);
            }
        }
    }
}

__global__ void clacpy_upper_kernel(
    std::size_t m, std::size_t n,
    const cuComplex *dA, std::size_t ldda,
    cuComplex       *dB, std::size_t lddb )
{
    clacpy_upper_device(m, n, dA, ldda, dB, lddb);
}

static __device__ void zlacpy_upper_device(
    std::size_t m, std::size_t n,
    const cuDoubleComplex *dA, std::size_t ldda,
    cuDoubleComplex       *dB, std::size_t lddb )
{
    std::size_t ind = blockIdx.x*BLK_X + threadIdx.x;
    std::size_t iby = blockIdx.y*BLK_Y;
    bool full = (iby + BLK_Y <= n);
    if ( ind < m ) {
        dA += ind + iby*ldda;
        dB += ind + iby*lddb;
        std::size_t copyLimit = device_min(ind - iby + 1, static_cast<std::size_t>(BLK_Y) );

        if ( full ) {
            #pragma unroll
            for( std::size_t j=0; j < BLK_Y; ++j ) {
                dB[j*lddb] = (j < copyLimit) ?  dA[j*ldda] : make_cuDoubleComplex(0.0, 0.0);
            }
        }
        else {
            for( std::size_t j=0; j < BLK_Y && iby+j < n; ++j ) {
                dB[j*lddb] =  (j < copyLimit) ?  dA[j*ldda] : make_cuDoubleComplex(0.0, 0.0);
            }
        }
    }
}

__global__ void zlacpy_upper_kernel(
    std::size_t m, std::size_t n,
    const cuDoubleComplex *dA, std::size_t ldda,
    cuDoubleComplex       *dB, std::size_t lddb )
{
    zlacpy_upper_device(m, n, dA, ldda, dB, lddb);
}


static __device__ void dlacpy_lower_device(
    std::size_t m, std::size_t n,
    const double *dA, std::size_t ldda,
    double       *dB, std::size_t lddb )
{
    std::size_t ind = blockIdx.x*BLK_X + threadIdx.x;
    std::size_t iby = blockIdx.y*BLK_Y;
    bool full = (iby + BLK_Y <= n);
    if ( ind < m ) {
        dA += ind + iby*ldda;
        dB += ind + iby*lddb;
        std::size_t copyStart = device_max(ind - iby, static_cast<std::size_t>(0));

        if ( full ) {
            #pragma unroll
            for( std::size_t j=0; j < BLK_Y; ++j ) {
                dB[j*lddb] = (j >= copyStart) ?  dA[j*ldda] : 0.0;
            }
        }
        else {
            for( std::size_t j=0; j < BLK_Y && iby+j < n; ++j ) {
                dB[j*lddb] =  (j >= copyStart) ?  dA[j*ldda] : 0.0;
            }
        }
    }
}

__global__ void dlacpy_lower_kernel(
    std::size_t m, std::size_t n,
    const double *dA, std::size_t ldda,
    double       *dB, std::size_t lddb )
{
    dlacpy_lower_device(m, n, dA, ldda, dB, lddb);
}

static __device__ void slacpy_lower_device(
    std::size_t m, std::size_t n,
    const float *dA, std::size_t ldda,
    float       *dB, std::size_t lddb )
{
    std::size_t ind = blockIdx.x*BLK_X + threadIdx.x;
    std::size_t iby = blockIdx.y*BLK_Y;
    bool full = (iby + BLK_Y <= n);
    if ( ind < m ) {
        dA += ind + iby*ldda;
        dB += ind + iby*lddb;
        std::size_t copyStart = device_max(ind - iby, static_cast<std::size_t>(0));
        if ( full ) {
            #pragma unroll
            for( std::size_t j=0; j < BLK_Y; ++j ) {
                dB[j*lddb] = (j >= copyStart) ?  dA[j*ldda] : 0.0;
            }
        }
        else {
            for( std::size_t j=0; j < BLK_Y && iby+j < n; ++j ) {
                dB[j*lddb] =  (j >= copyStart) ?  dA[j*ldda] : 0.0;
            }
        }
    }
}

__global__ void slacpy_lower_kernel(
    std::size_t m, std::size_t n,
    const float *dA, std::size_t ldda,
    float       *dB, std::size_t lddb )
{
    slacpy_lower_device(m, n, dA, ldda, dB, lddb);
}

static __device__ void clacpy_lower_device(
    std::size_t m, std::size_t n,
    const cuComplex *dA, std::size_t ldda,
    cuComplex       *dB, std::size_t lddb )
{
    std::size_t ind = blockIdx.x*BLK_X + threadIdx.x;
    std::size_t iby = blockIdx.y*BLK_Y;
    bool full = (iby + BLK_Y <= n);
    if ( ind < m ) {
        dA += ind + iby*ldda;
        dB += ind + iby*lddb;
        std::size_t copyStart = device_max(ind - iby, static_cast<std::size_t>(0));

        if ( full ) {
            #pragma unroll
            for( std::size_t j=0; j < BLK_Y; ++j ) {
                dB[j*lddb] = (j >= copyStart) ?  dA[j*ldda] : make_cuFloatComplex(0.0f, 0.0f);
            }
        }
        else {
            for( std::size_t j=0; j < BLK_Y && iby+j < n; ++j ) {
                dB[j*lddb] =  (j >= copyStart) ?  dA[j*ldda] : make_cuFloatComplex(0.0f, 0.0f);
            }
        }
    }
}

__global__ void clacpy_lower_kernel(
    std::size_t m, std::size_t n,
    const cuComplex *dA, std::size_t ldda,
    cuComplex       *dB, std::size_t lddb )
{
    clacpy_lower_device(m, n, dA, ldda, dB, lddb);
}

static __device__ void zlacpy_lower_device(
    std::size_t m, std::size_t n,
    const cuDoubleComplex *dA, std::size_t ldda,
    cuDoubleComplex       *dB, std::size_t lddb )
{
    std::size_t ind = blockIdx.x*BLK_X + threadIdx.x;
    std::size_t iby = blockIdx.y*BLK_Y;
    bool full = (iby + BLK_Y <= n);
    if ( ind < m ) {
        dA += ind + iby*ldda;
        dB += ind + iby*lddb;
        std::size_t copyStart = device_max(ind - iby, static_cast<std::size_t>(0));

        if ( full ) {
            #pragma unroll
            for( std::size_t j=0; j < BLK_Y; ++j ) {
                dB[j*lddb] = (j >= copyStart) ?  dA[j*ldda] : make_cuDoubleComplex(0.0, 0.0);
            }
        }
        else {
            for( std::size_t j=0; j < BLK_Y && iby+j < n; ++j ) {
                dB[j*lddb] =  (j >= copyStart) ?  dA[j*ldda] : make_cuDoubleComplex(0.0, 0.0);
            }
        }
    }
}

__global__ void zlacpy_lower_kernel(
    std::size_t m, std::size_t n,
    const cuDoubleComplex *dA, std::size_t ldda,
    cuDoubleComplex       *dB, std::size_t lddb )
{
    zlacpy_lower_device(m, n, dA, ldda, dB, lddb);
}



void t_lacpy_gpu(char uplo, std::size_t m, std::size_t n, float *dA, std::size_t ldda, float *dB, std::size_t lddb, cudaStream_t stream_ )
{
    #define dA(i_, j_) (dA + (i_) + (j_)*ldda)
    #define dB(i_, j_) (dB + (i_) + (j_)*lddb)
    std::size_t super_NB = max_blocks*BLK_X;
    dim3 super_grid(  (m + super_NB - 1) / super_NB,  (n + super_NB - 1) / super_NB );

    dim3 threads( BLK_X, 1 );
    dim3 grid;

    std::size_t mm, nn;
    if ( uplo == 'L' ) 
    {
        for( std::size_t i=0; i < super_grid.x; ++i ) {
            mm = (i == super_grid.x-1 ? m % super_NB : super_NB);
            grid.x = ( mm + BLK_X - 1) / BLK_X;
            for( std::size_t j=0; j < super_grid.y; ++j ) {  // full row
                nn = (j == super_grid.y-1 ? n % super_NB : super_NB);
                grid.y = ( nn + BLK_X - 1) / BLK_Y;
                if ( i == j ) {  // diagonal super block
                    slacpy_upper_kernel <<< grid, threads, 0, stream_ >>>
                        ( mm, nn, dA(i*super_NB, j*super_NB), ldda, dB(i*super_NB, j*super_NB), lddb );
                }
                else // off diagonal super block
                {   
                    slacpy_full_kernel <<< grid, threads, 0, stream_ >>>
                        ( mm, nn, dA(i*super_NB, j*super_NB), ldda, dB(i*super_NB, j*super_NB), lddb );
                }
            }
        }    
    }
    else if(uplo == 'U')
    {
        for( std::size_t i=0; i < super_grid.x; ++i ) {
            mm = (i == super_grid.x-1 ? m % super_NB : super_NB);
            grid.x = ( mm + BLK_X - 1) / BLK_X;
            for( std::size_t j=0; j < super_grid.y; ++j ) {  // full row
                nn = (j == super_grid.y-1 ? n % super_NB : super_NB);
                grid.y = ( nn + BLK_X - 1) / BLK_Y;
                if ( i == j ) {  // diagonal super block
                    slacpy_lower_kernel <<< grid, threads, 0, stream_ >>>
                        ( mm, nn, dA(i*super_NB, j*super_NB), ldda, dB(i*super_NB, j*super_NB), lddb );
                }
                else // off diagonal super block
                {   
                    slacpy_full_kernel <<< grid, threads, 0, stream_ >>>
                        ( mm, nn, dA(i*super_NB, j*super_NB), ldda, dB(i*super_NB, j*super_NB), lddb );
                }
            }
        }
    }
    else
    {
        for( std::size_t i=0; i < super_grid.x; ++i ) {
            mm = (i == super_grid.x-1 ? m % super_NB : super_NB);
            grid.x = ( mm + BLK_X - 1) / BLK_X;
            for( std::size_t j=0; j < super_grid.y; ++j ) {  // full row
                nn = (j == super_grid.y-1 ? n % super_NB : super_NB);
                grid.y = ( nn + BLK_X - 1) / BLK_Y;
                slacpy_full_kernel <<< grid, threads, 0, stream_ >>>
                    ( mm, nn, dA(i*super_NB, j*super_NB), ldda, dB(i*super_NB, j*super_NB), lddb );
            }
        }
    }
}

void t_lacpy_gpu(char uplo, std::size_t m, std::size_t n, double *dA, std::size_t ldda, double *dB, std::size_t lddb, cudaStream_t stream_ )
{
    #define dA(i_, j_) (dA + (i_) + (j_)*ldda)
    #define dB(i_, j_) (dB + (i_) + (j_)*lddb)	
    std::size_t super_NB = max_blocks*BLK_X;
    dim3 super_grid(  (m + super_NB - 1) / super_NB,  (n + super_NB - 1) / super_NB );     

    dim3 threads( BLK_X, 1 );
    dim3 grid;

    std::size_t mm, nn;
    if ( uplo == 'L' ) 
    {
        for( std::size_t i=0; i < super_grid.x; ++i ) {
            mm = (i == super_grid.x-1 ? m % super_NB : super_NB);
            grid.x = ( mm + BLK_X - 1) / BLK_X;
            for( std::size_t j=0; j < super_grid.y; ++j ) {  // full row
                nn = (j == super_grid.y-1 ? n % super_NB : super_NB);
                grid.y = ( nn + BLK_X - 1) / BLK_Y;
                if ( i == j ) {  // diagonal super block
                    dlacpy_upper_kernel <<< grid, threads, 0, stream_ >>>
                        ( mm, nn, dA(i*super_NB, j*super_NB), ldda, dB(i*super_NB, j*super_NB), lddb );
                }
                else // off diagonal super block
                {   
                    dlacpy_full_kernel <<< grid, threads, 0, stream_ >>>
                        ( mm, nn, dA(i*super_NB, j*super_NB), ldda, dB(i*super_NB, j*super_NB), lddb );
                }
            }
        }    
    }
    else if(uplo == 'U')
    {
        for( std::size_t i=0; i < super_grid.x; ++i ) {
            mm = (i == super_grid.x-1 ? m % super_NB : super_NB);
            grid.x = ( mm + BLK_X - 1) / BLK_X;
            for( std::size_t j=0; j < super_grid.y; ++j ) {  // full row
                nn = (j == super_grid.y-1 ? n % super_NB : super_NB);
                grid.y = ( nn + BLK_X - 1) / BLK_Y;
                if ( i == j ) {  // diagonal super block
                    dlacpy_lower_kernel <<< grid, threads, 0, stream_ >>>
                        ( mm, nn, dA(i*super_NB, j*super_NB), ldda, dB(i*super_NB, j*super_NB), lddb );
                }
                else // off diagonal super block
                {   
                    dlacpy_full_kernel <<< grid, threads, 0, stream_ >>>
                        ( mm, nn, dA(i*super_NB, j*super_NB), ldda, dB(i*super_NB, j*super_NB), lddb );
                }
            }
        }
    }
    else
    {
        for( std::size_t i=0; i < super_grid.x; ++i ) {
            mm = (i == super_grid.x-1 ? m % super_NB : super_NB);
            grid.x = ( mm + BLK_X - 1) / BLK_X;
            for( std::size_t j=0; j < super_grid.y; ++j ) {  // full row
                nn = (j == super_grid.y-1 ? n % super_NB : super_NB);
                grid.y = ( nn + BLK_X - 1) / BLK_Y;
                dlacpy_full_kernel <<< grid, threads, 0, stream_ >>>
                    ( mm, nn, dA(i*super_NB, j*super_NB), ldda, dB(i*super_NB, j*super_NB), lddb );
            }
        }
    }
}	

void t_lacpy_gpu(char uplo, std::size_t m, std::size_t n, std::complex<double> *ddA, std::size_t ldda, std::complex<double> *ddB, std::size_t lddb, cudaStream_t stream_ )
{
    cuDoubleComplex *dA = reinterpret_cast<cuDoubleComplex*>(ddA);
    cuDoubleComplex *dB = reinterpret_cast<cuDoubleComplex*>(ddB);
    #define dA(i_, j_) (dA + (i_) + (j_)*ldda)
    #define dB(i_, j_) (dB + (i_) + (j_)*lddb)
    std::size_t super_NB = max_blocks*BLK_X;
    dim3 super_grid(  (m + super_NB - 1) / super_NB,  (n + super_NB - 1) / super_NB );

    dim3 threads( BLK_X, 1 );
    dim3 grid;

    std::size_t mm, nn;
    if ( uplo == 'L' ) 
    {
        for( std::size_t i=0; i < super_grid.x; ++i ) {
            mm = (i == super_grid.x-1 ? m % super_NB : super_NB);
            grid.x = ( mm + BLK_X - 1) / BLK_X;
            for( std::size_t j=0; j < super_grid.y; ++j ) {  // full row
                nn = (j == super_grid.y-1 ? n % super_NB : super_NB);
                grid.y = ( nn + BLK_X - 1) / BLK_Y;
                if ( i == j ) {  // diagonal super block
                    zlacpy_upper_kernel <<< grid, threads, 0, stream_ >>>
                        ( mm, nn, dA(i*super_NB, j*super_NB), ldda, dB(i*super_NB, j*super_NB), lddb );
                }
                else // off diagonal super block
                {   
                    zlacpy_full_kernel <<< grid, threads, 0, stream_ >>>
                        ( mm, nn, dA(i*super_NB, j*super_NB), ldda, dB(i*super_NB, j*super_NB), lddb );
                }
            }
        }    
    }
    else if(uplo == 'U')
    {
        for( std::size_t i=0; i < super_grid.x; ++i ) {
            mm = (i == super_grid.x-1 ? m % super_NB : super_NB);
            grid.x = ( mm + BLK_X - 1) / BLK_X;
            for( std::size_t j=0; j < super_grid.y; ++j ) {  // full row
                nn = (j == super_grid.y-1 ? n % super_NB : super_NB);
                grid.y = ( nn + BLK_X - 1) / BLK_Y;
                if ( i == j ) {  // diagonal super block
                    zlacpy_lower_kernel <<< grid, threads, 0, stream_ >>>
                        ( mm, nn, dA(i*super_NB, j*super_NB), ldda, dB(i*super_NB, j*super_NB), lddb );
                }
                else // off diagonal super block
                {   
                    zlacpy_full_kernel <<< grid, threads, 0, stream_ >>>
                        ( mm, nn, dA(i*super_NB, j*super_NB), ldda, dB(i*super_NB, j*super_NB), lddb );
                }
            }
        }
    }
    else
    {
        for( std::size_t i=0; i < super_grid.x; ++i ) {
            mm = (i == super_grid.x-1 ? m % super_NB : super_NB);
            grid.x = ( mm + BLK_X - 1) / BLK_X;
            for( std::size_t j=0; j < super_grid.y; ++j ) {  // full row
                nn = (j == super_grid.y-1 ? n % super_NB : super_NB);
                grid.y = ( nn + BLK_X - 1) / BLK_Y;
                zlacpy_full_kernel <<< grid, threads, 0, stream_ >>>
                    ( mm, nn, dA(i*super_NB, j*super_NB), ldda, dB(i*super_NB, j*super_NB), lddb );
            }
        }
    }
}

void t_lacpy_gpu(char uplo, std::size_t m, std::size_t n, std::complex<float> *ddA, std::size_t ldda, std::complex<float> *ddB, std::size_t lddb, cudaStream_t stream_ )
{
    cuComplex *dA = reinterpret_cast<cuComplex*>(ddA);
    cuComplex *dB = reinterpret_cast<cuComplex*>(ddB);
    #define dA(i_, j_) (dA + (i_) + (j_)*ldda)
    #define dB(i_, j_) (dB + (i_) + (j_)*lddb)
    std::size_t super_NB = max_blocks*BLK_X;
    dim3 super_grid(  (m + super_NB - 1) / super_NB,  (n + super_NB - 1) / super_NB );

    dim3 threads( BLK_X, 1 );
    dim3 grid;

    std::size_t mm, nn;
    if ( uplo == 'L' ) 
    {
        for( std::size_t i=0; i < super_grid.x; ++i ) {
            mm = (i == super_grid.x-1 ? m % super_NB : super_NB);
            grid.x = ( mm + BLK_X - 1) / BLK_X;
            for( std::size_t j=0; j < super_grid.y; ++j ) {  // full row
                nn = (j == super_grid.y-1 ? n % super_NB : super_NB);
                grid.y = ( nn + BLK_X - 1) / BLK_Y;
                if ( i == j ) {  // diagonal super block
                    clacpy_upper_kernel <<< grid, threads, 0, stream_ >>>
                        ( mm, nn, dA(i*super_NB, j*super_NB), ldda, dB(i*super_NB, j*super_NB), lddb );
                }
                else // off diagonal super block
                {   
                    clacpy_full_kernel <<< grid, threads, 0, stream_ >>>
                        ( mm, nn, dA(i*super_NB, j*super_NB), ldda, dB(i*super_NB, j*super_NB), lddb );
                }
            }
        }    
    }
    else if(uplo == 'U')
    {
        for( std::size_t i=0; i < super_grid.x; ++i ) {
            mm = (i == super_grid.x-1 ? m % super_NB : super_NB);
            grid.x = ( mm + BLK_X - 1) / BLK_X;
            for( std::size_t j=0; j < super_grid.y; ++j ) {  // full row
                nn = (j == super_grid.y-1 ? n % super_NB : super_NB);
                grid.y = ( nn + BLK_X - 1) / BLK_Y;
                if ( i == j ) {  // diagonal super block
                    clacpy_lower_kernel <<< grid, threads, 0, stream_ >>>
                        ( mm, nn, dA(i*super_NB, j*super_NB), ldda, dB(i*super_NB, j*super_NB), lddb );
                }
                else // off diagonal super block
                {   
                    clacpy_full_kernel <<< grid, threads, 0, stream_ >>>
                        ( mm, nn, dA(i*super_NB, j*super_NB), ldda, dB(i*super_NB, j*super_NB), lddb );
                }
            }
        }
    }
    else
    {
        for( std::size_t i=0; i < super_grid.x; ++i ) {
            mm = (i == super_grid.x-1 ? m % super_NB : super_NB);
            grid.x = ( mm + BLK_X - 1) / BLK_X;
            for( std::size_t j=0; j < super_grid.y; ++j ) {  // full row
                nn = (j == super_grid.y-1 ? n % super_NB : super_NB);
                grid.y = ( nn + BLK_X - 1) / BLK_Y;
                clacpy_full_kernel <<< grid, threads, 0, stream_ >>>
                    ( mm, nn, dA(i*super_NB, j*super_NB), ldda, dB(i*super_NB, j*super_NB), lddb );
            }
        }
    }
}


}
}
}
}
#include "random_normal_distribution.cuh"

#define BLOCKDIM 256
#define GRIDDIM 32

namespace chase
{
namespace linalg
{
namespace internal
{
namespace cuda
{
    // generate `n` random float numbers on GPU
    __global__ void s_normal_kernel(unsigned long long seed, curandStatePhilox4_32_10_t* states,
                                    float* v, std::size_t n)
    {
        std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
        curandStatePhilox4_32_10_t* state = states + tid;
        curand_init(seed, tid, 0, state);

        std::size_t i;
        std::size_t nthreads = gridDim.x * blockDim.x;

        for (i = tid; i < n; i += nthreads)
        {
            v[i] = curand_normal(state);
        }
    }

    // generate `n` random double numbers on GPU
    __global__ void d_normal_kernel(unsigned long long seed, curandStatePhilox4_32_10_t* states,
                                    double* v, std::size_t n)
    {
        std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
        curandStatePhilox4_32_10_t* state = states + tid;
        curand_init(seed, tid, 0, state);

        std::size_t i;
        std::size_t nthreads = gridDim.x * blockDim.x;

        for (i = tid; i < n; i += nthreads)
        {
            v[i] = curand_normal_double(state);
        }
    }
    // generate `n` random complex single numbers on GPU
    __global__ void c_normal_kernel(unsigned long long seed, curandStatePhilox4_32_10_t* states,
                                    cuComplex* v, std::size_t n)
    {
        std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
        curandStatePhilox4_32_10_t* state = states + tid;
        curand_init(seed, tid, 0, state);

        std::size_t i;
        std::size_t nthreads = gridDim.x * blockDim.x;

        for (i = tid; i < n; i += nthreads)
        {
            float rnd = curand_normal(state);
            v[i].x = rnd;
            v[i].y = rnd;
        }
    }

    // generate `n` random complex double numbers on GPU
    __global__ void z_normal_kernel(unsigned long long seed, curandStatePhilox4_32_10_t* states,
                                    cuDoubleComplex* v, std::size_t n)
    {
        std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
        curandStatePhilox4_32_10_t* state = states + tid;
        curand_init(seed, tid, 0, state);

        std::size_t i;
        std::size_t nthreads = gridDim.x * blockDim.x;

        for (i = tid; i < n; i += nthreads)
        {
            double rnd = curand_normal_double(state);
            v[i].x = rnd;
            v[i].y = rnd;
        }
    }


    void chase_rand_normal(unsigned long long seed, curandStatePhilox4_32_10_t* states, float* v,
                        std::size_t n, cudaStream_t stream_)
    {
        s_normal_kernel<<<GRIDDIM, BLOCKDIM, 0, stream_>>>(seed, states, v, n);
    }

    void chase_rand_normal(unsigned long long seed, curandStatePhilox4_32_10_t* states, double* v,
                        std::size_t n, cudaStream_t stream_)
    {
        d_normal_kernel<<<GRIDDIM, BLOCKDIM, 0, stream_>>>(seed, states, v, n);
    }

    void chase_rand_normal(unsigned long long seed, curandStatePhilox4_32_10_t* states,
                        std::complex<float>* v, std::size_t n, cudaStream_t stream_)
    {
        c_normal_kernel<<<GRIDDIM, BLOCKDIM, 0, stream_>>>(
            seed, states, reinterpret_cast<cuComplex*>(v), n);
    }

    void chase_rand_normal(unsigned long long seed, curandStatePhilox4_32_10_t* states,
                        std::complex<double>* v, std::size_t n, cudaStream_t stream_)
    {
        z_normal_kernel<<<GRIDDIM, BLOCKDIM, 0, stream_>>>(
            seed, states, reinterpret_cast<cuDoubleComplex*>(v), n);
    }


}
}
}
}
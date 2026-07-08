// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE


#define BLOCKDIM 256
#define GRIDDIM 32

namespace chase
{
namespace linalg
{
namespace internal
{
namespace CHASE_GPU_DEVICE_NAMESPACE
{
// generate `n` random float numbers on GPU
__global__ void s_normal_kernel(unsigned long long seed,
                                CHASE_GPU_RAND_STATE* states, float* v,
                                std::size_t n)
{
    std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    CHASE_GPU_RAND_STATE* state = states + tid;
    CHASE_GPU_RAND_INIT(seed, tid, 0, state);

    std::size_t i;
    std::size_t nthreads = gridDim.x * blockDim.x;

    for (i = tid; i < n; i += nthreads)
    {
        v[i] = CHASE_GPU_RAND_NORMAL(state);
    }
}

// generate `n` random double numbers on GPU
__global__ void d_normal_kernel(unsigned long long seed,
                                CHASE_GPU_RAND_STATE* states, double* v,
                                std::size_t n)
{
    std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    CHASE_GPU_RAND_STATE* state = states + tid;
    CHASE_GPU_RAND_INIT(seed, tid, 0, state);

    std::size_t i;
    std::size_t nthreads = gridDim.x * blockDim.x;

    for (i = tid; i < n; i += nthreads)
    {
        v[i] = CHASE_GPU_RAND_NORMAL_DOUBLE(state);
    }
}
// generate `n` random complex single numbers on GPU
__global__ void c_normal_kernel(unsigned long long seed,
                                CHASE_GPU_RAND_STATE* states,
                                CHASE_GPU_FLOAT_COMPLEX* v, std::size_t n)
{
    std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    CHASE_GPU_RAND_STATE* state = states + tid;
    CHASE_GPU_RAND_INIT(seed, tid, 0, state);

    std::size_t i;
    std::size_t nthreads = gridDim.x * blockDim.x;

    for (i = tid; i < n; i += nthreads)
    {
        float rnd = CHASE_GPU_RAND_NORMAL(state);
        v[i].x = rnd;
        v[i].y = rnd;
    }
}

// generate `n` random complex double numbers on GPU
__global__ void z_normal_kernel(unsigned long long seed,
                                CHASE_GPU_RAND_STATE* states,
                                CHASE_GPU_DOUBLE_COMPLEX* v, std::size_t n)
{
    std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    CHASE_GPU_RAND_STATE* state = states + tid;
    CHASE_GPU_RAND_INIT(seed, tid, 0, state);

    std::size_t i;
    std::size_t nthreads = gridDim.x * blockDim.x;

    for (i = tid; i < n; i += nthreads)
    {
        double rnd = CHASE_GPU_RAND_NORMAL_DOUBLE(state);
        v[i].x = rnd;
        v[i].y = rnd;
    }
}

void chase_rand_normal(unsigned long long seed,
                       CHASE_GPU_RAND_STATE* states, float* v,
                       std::size_t n, CHASE_GPU_STREAM_T stream_)
{
    s_normal_kernel<<<GRIDDIM, BLOCKDIM, 0, stream_>>>(seed, states, v, n);
}

void chase_rand_normal(unsigned long long seed,
                       CHASE_GPU_RAND_STATE* states, double* v,
                       std::size_t n, CHASE_GPU_STREAM_T stream_)
{
    d_normal_kernel<<<GRIDDIM, BLOCKDIM, 0, stream_>>>(seed, states, v, n);
}

void chase_rand_normal(unsigned long long seed,
                       CHASE_GPU_RAND_STATE* states,
                       std::complex<float>* v, std::size_t n,
                       CHASE_GPU_STREAM_T stream_)
{
    c_normal_kernel<<<GRIDDIM, BLOCKDIM, 0, stream_>>>(
        seed, states, reinterpret_cast<CHASE_GPU_FLOAT_COMPLEX*>(v), n);
}

void chase_rand_normal(unsigned long long seed,
                       CHASE_GPU_RAND_STATE* states,
                       std::complex<double>* v, std::size_t n,
                       CHASE_GPU_STREAM_T stream_)
{
    z_normal_kernel<<<GRIDDIM, BLOCKDIM, 0, stream_>>>(
        seed, states, reinterpret_cast<CHASE_GPU_DOUBLE_COMPLEX*>(v), n);
}

} // namespace CHASE_GPU_DEVICE_NAMESPACE
} // namespace internal
} // namespace linalg
} // namespace chase

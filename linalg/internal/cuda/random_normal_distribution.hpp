#include "random_normal_distribution.cuh"

namespace chase
{
namespace linalg
{
namespace internal
{
namespace cuda
{
    template<typename T>
    void init_random_vectors(T *v, std::size_t n, cudaStream_t *stream_ = nullptr)
    {
        cudaStream_t usedStream = (stream_ == nullptr) ? 0 : *stream_;
        curandStatePhilox4_32_10_t* states_ = NULL;
        cudaMalloc((void**)&states_, sizeof(curandStatePhilox4_32_10_t) * (256 * 32));

	    unsigned long long seed = 24141;
        chase_rand_normal(seed, states_, v, n, usedStream);

        cudaFree(states_);

    }

}
}
}
}
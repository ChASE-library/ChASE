#include "random_normal_distribution.cuh"
#include "Impl/chase_gpu/nvtx.hpp"

namespace chase
{
namespace linalg
{
namespace internal
{
namespace cuda
{
    /**
     * @brief Initializes a vector with random values using the CUDA random number generator.
     * 
     * This function initializes a vector `v` with random values sampled from a normal distribution. 
     * The random number generation is performed on the GPU using CUDA and `curand`, and it can be 
     * executed asynchronously with respect to the provided CUDA stream. If no stream is provided, 
     * the default stream is used.
     * 
     * @tparam T The data type of the vector elements (e.g., `float`, `double`).
     * 
     * @param v Pointer to the vector on the GPU that will be populated with random values.
     * @param n The number of elements in the vector `v`.
     * @param stream_ Optional CUDA stream for asynchronous execution. If `nullptr`, the default stream is used.
     * 
     * @note The function uses `curandStatePhilox4_32_10_t` for managing the random state and a fixed seed value.
     *       It allocates memory on the GPU for the random number generation state and frees it after the operation.
     */    
    template<typename T>
    void init_random_vectors(T *v, std::size_t n, cudaStream_t *stream_ = nullptr)
    {
        SCOPED_NVTX_RANGE();

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
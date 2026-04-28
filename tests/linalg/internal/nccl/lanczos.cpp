// This file is a part of ChASE.
// Copyright (c) 2015-2024, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include <gtest/gtest.h>
#include <algorithm>
#include <complex>
#include <cmath>
#include <cstring>
#include <type_traits>
#include "linalg/internal/cuda/random_normal_distribution.hpp"
#include "linalg/internal/nccl/lanczos.hpp"
#include "grid/mpiGrid2D.hpp"
#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"

// Global static resources that persist across all test suites
namespace {
/** Absolute tolerance: rel_ratio * max(|reference|, 1). */
template <typename B>
B rel_abs_tolerance(B reference, B rel_ratio = B(0.05))
{
    using std::abs;
    return rel_ratio * (std::max)(abs(reference), B(1));
}

/**
 * Reference scale for multi-vector NCCL Lanczos `upperb` (max |Ritz| + |β_last|).
 * For Clement N≈500, M=10 this is ~1.51·(N−1); keep in sync with observed bound.
 */
template <typename B>
B lanczos_multi_upperb_reference(B spectral_radius)
{
    constexpr B kScale = B(1.51);
    return kScale * spectral_radius;
}

    bool resources_initialized = false;
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid;
    cublasHandle_t cublasH;
    cudaStream_t stream;
    int world_rank, world_size;
    std::complex<double>* d_H = nullptr;        // Device memory for H matrix
    std::complex<double>* d_V_data = nullptr;   // Device memory for V matrix
    std::size_t d_H_size = 0;
    std::size_t d_V_data_size = 0;
    std::complex<double>* Clement_z = nullptr;
    std::complex<float>* Clement_c = nullptr;
    double* Clement_d = nullptr;
    float* Clement_s = nullptr;
    std::size_t Clement_size = 0;
    curandStatePhilox4_32_10_t* d_curand_states = nullptr;

template <typename T, typename MV>
void fill_V_cuda_normal_gpu(
    MV& V_,
    const std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>& grid,
    cudaStream_t stream,
    curandStatePhilox4_32_10_t* states)
{
    int mpi_col_rank;
    MPI_Comm_rank(grid->get_col_comm(), &mpi_col_rank);
    const unsigned long long seed =
        1337ULL + static_cast<unsigned long long>(mpi_col_rank);
    const std::size_t n = V_.l_ld() * V_.l_cols();
    cudaStream_t* stream_ptr = &stream;

    if constexpr (std::is_floating_point_v<T>) {
        chase::linalg::internal::cuda::chase_rand_normal(
            seed, states, V_.l_data(), n, stream);
    } else {
        using B = chase::Base<T>;
        B* d_tmp = nullptr;
        CHECK_CUDA_ERROR(
            cudaMalloc(reinterpret_cast<void**>(&d_tmp), n * sizeof(B)));
        chase::linalg::internal::cuda::chase_rand_normal(seed, states, d_tmp,
                                                          n, stream);
        chase::linalg::internal::cuda::copyRealToT(d_tmp, V_.l_data(),
                                                   static_cast<int>(n),
                                                   stream_ptr);
        CHECK_CUDA_ERROR(cudaFree(d_tmp));
    }
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
}
}

template<typename T>
void InitializeClementMatrix(T* Clement, std::size_t N) {
    for (auto i = 0; i < N; ++i) {
        Clement[i + N * i] = T(0);
        if (i != N - 1) {
            T val = std::sqrt(i * (N + 1 - i));
            Clement[i + 1 + N * i] = val;
            Clement[i + N * (i + 1)] = val;
        }
    }
}
template <typename T>
class LanczosNCCLDistTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        // Initialize resources only once for all test suites
        if (!resources_initialized) {
            // Initialize MPI grid
            MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
            MPI_Comm_size(MPI_COMM_WORLD, &world_size);
            ASSERT_EQ(world_size, 4);  // Ensure we're running with 4 processes
            mpi_grid = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(2, 2, MPI_COMM_WORLD);

            // Initialize CUDA resources
            CHECK_CUBLAS_ERROR(cublasCreate(&cublasH));
            CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
            CHECK_CUBLAS_ERROR(cublasSetStream(cublasH, stream));
            CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void**>(&d_curand_states),
                                        sizeof(curandStatePhilox4_32_10_t) * (256 * 32)));
            // Allocate Clement matrices
            Clement_size = N * N;
            Clement_z = new std::complex<double>[Clement_size];
            Clement_c = new std::complex<float>[Clement_size];
            Clement_d = new double[Clement_size];
            Clement_s = new float[Clement_size];

            // Initialize Clement matrices
            InitializeClementMatrix(Clement_z, N);
            InitializeClementMatrix(Clement_c, N);
            InitializeClementMatrix(Clement_d, N);
            InitializeClementMatrix(Clement_s, N);

            std::cout << "Clement_z: initialized" << std::endl;
            std::cout << "Clement_c: initialized" << std::endl;
            std::cout << "Clement_d: initialized" << std::endl;
            std::cout << "Clement_s: initialized" << std::endl;

            resources_initialized = true;
        }
    }

    void SetUp() override {
        // Ensure device memory is allocated if needed
        std::size_t required_H_size = m * m;
        std::size_t required_V_size = m * M;
        
        if (d_H == nullptr || d_H_size < required_H_size) {
            if (d_H != nullptr) {
                CHECK_CUDA_ERROR(cudaFree(d_H));
            }
            CHECK_CUDA_ERROR(cudaMalloc((void**)&d_H, sizeof(std::complex<double>) * required_H_size));
            d_H_size = required_H_size;
        }

        if (d_V_data == nullptr || d_V_data_size < required_V_size) {
            if (d_V_data != nullptr) {
                CHECK_CUDA_ERROR(cudaFree(d_V_data));
            }
            CHECK_CUDA_ERROR(cudaMalloc((void**)&d_V_data, sizeof(std::complex<double>) * required_V_size));
            d_V_data_size = required_V_size;
        }
    }

    void TearDown() override {
        // Don't free device memory here - it will be reused
    }

    static constexpr std::size_t M = 10;
    static constexpr std::size_t numvec = 4;
    static constexpr std::size_t N = 500;
    static constexpr std::size_t m = 250;

    // Accessors for the global resources
    static std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> get_mpi_grid() { return mpi_grid; }
    static cublasHandle_t get_cublas_handle() { return cublasH; }
    static cudaStream_t get_stream() { return stream; }
    static int get_world_rank() { return world_rank; }
    static int get_world_size() { return world_size; }
    static std::complex<double>* get_H_device() { return d_H; }
    static std::complex<double>* get_V_device() { return d_V_data; }
    static std::complex<double>* get_Clement_z() { return Clement_z; }
    static std::complex<float>* get_Clement_c() { return Clement_c; }
    static double* get_Clement_d() { return Clement_d; }
    static float* get_Clement_s() { return Clement_s; }
};

// Add a global test environment to handle resource cleanup at program exit
class ResourceCleanupEnvironment : public ::testing::Environment {
public:
    ~ResourceCleanupEnvironment() override {
        if (resources_initialized) {
            if (d_H != nullptr) {
                CHECK_CUDA_ERROR(cudaFree(d_H));
                d_H = nullptr;
                d_H_size = 0;
            }

            if (d_V_data != nullptr) {
                CHECK_CUDA_ERROR(cudaFree(d_V_data));
                d_V_data = nullptr;
                d_V_data_size = 0;
            }

            if (d_curand_states != nullptr) {
                CHECK_CUDA_ERROR(cudaFree(d_curand_states));
                d_curand_states = nullptr;
            }

            CHECK_CUBLAS_ERROR(cublasDestroy(cublasH));
            CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
            
            // Reset the mpi_grid shared_ptr before program exit
            // This ensures the MpiGrid2D destructor is called only once
            mpi_grid.reset();
            
            resources_initialized = false;
            std::cout << "Resources freed at program exit" << std::endl;
        }
    }
};

using TestTypes = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(LanczosNCCLDistTest, TestTypes);


TYPED_TEST(LanczosNCCLDistTest, mlanczosGPU) {
    using T = TypeParam;  // Get the current type

    // Allocate CPU buffers for this test
    std::vector<chase::Base<T>> ritzv(this->M * this->numvec);
    std::vector<chase::Base<T>> ritzV(this->M * this->M);
    std::vector<chase::Base<T>> Tau(this->M * this->M * this->numvec);

    // Get pointer to the appropriate Clement matrix
    T* Clement = [this]() -> T* {
        if constexpr (std::is_same_v<T, std::complex<double>>) {
            return reinterpret_cast<T*>(this->get_Clement_z());
        } else if constexpr (std::is_same_v<T, std::complex<float>>) {
            return reinterpret_cast<T*>(this->get_Clement_c());
        } else if constexpr (std::is_same_v<T, double>) {
            return reinterpret_cast<T*>(this->get_Clement_d());
        } else {
            return reinterpret_cast<T*>(this->get_Clement_s());
        }
    }();

    auto H_ = chase::distMatrix::BlockBlockMatrix<T, chase::platform::GPU>(this->m, this->m, this->m, reinterpret_cast<T*>(this->get_H_device()), this->get_mpi_grid());
    auto V_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::column, chase::platform::GPU>(this->m, this->M, this->m, reinterpret_cast<T*>(this->get_V_device()), this->get_mpi_grid());
    auto H2_ = chase::distMatrix::BlockBlockMatrix<T, chase::platform::CPU>(this->N, this->N, this->get_mpi_grid());

    fill_V_cuda_normal_gpu<T>(V_, this->get_mpi_grid(), this->get_stream(),
                              d_curand_states);

    auto Clement_ = chase::distMatrix::RedundantMatrix<T, chase::platform::CPU>(this->N, this->N, this->N, Clement, this->get_mpi_grid());
    Clement_.redistributeImpl(&H2_);
    CHECK_CUDA_ERROR(cudaMemcpy(H_.l_data(), H2_.l_data(), sizeof(T) * H2_.l_rows() * H2_.l_cols(), cudaMemcpyHostToDevice));

    chase::Base<T> upperb;
    cudaDeviceSynchronize();
    chase::linalg::internal::cuda_nccl::lanczos(this->get_cublas_handle(),
                                          this->M,
                                          this->numvec,
                                          H_,
                                          V_,
                                          &upperb,
                                          ritzv.data(),
                                          Tau.data(),
                                          ritzV.data());

    cudaDeviceSynchronize();
    {
        using B = chase::Base<T>;
        const B spectral = B(this->N - 1);
        const B ref = lanczos_multi_upperb_reference(spectral);
        EXPECT_NEAR(upperb, ref, rel_abs_tolerance(ref));
    }
}

TYPED_TEST(LanczosNCCLDistTest, lanczosGPU) {
    using T = TypeParam;  // Get the current type

    // Get pointer to the appropriate Clement matrix
    T* Clement = [this]() -> T* {
        if constexpr (std::is_same_v<T, std::complex<double>>) {
            return reinterpret_cast<T*>(this->get_Clement_z());
        } else if constexpr (std::is_same_v<T, std::complex<float>>) {
            return reinterpret_cast<T*>(this->get_Clement_c());
        } else if constexpr (std::is_same_v<T, double>) {
            return reinterpret_cast<T*>(this->get_Clement_d());
        } else {
            return reinterpret_cast<T*>(this->get_Clement_s());
        }
    }();

    auto H_ = chase::distMatrix::BlockBlockMatrix<T, chase::platform::GPU>(this->m, this->m, this->m, reinterpret_cast<T*>(this->get_H_device()), this->get_mpi_grid());
    auto V_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::column, chase::platform::GPU>(this->m, this->M, this->m, reinterpret_cast<T*>(this->get_V_device()), this->get_mpi_grid());
    auto H2_ = chase::distMatrix::BlockBlockMatrix<T, chase::platform::CPU>(this->N, this->N, this->get_mpi_grid());

    fill_V_cuda_normal_gpu<T>(V_, this->get_mpi_grid(), this->get_stream(),
                              d_curand_states);

    auto Clement_ = chase::distMatrix::RedundantMatrix<T>(this->N, this->N, this->N, Clement, this->get_mpi_grid());
    Clement_.redistributeImpl(&H2_);
    CHECK_CUDA_ERROR(cudaMemcpy(H_.l_data(), H2_.l_data(), sizeof(T) * H2_.l_rows() * H2_.l_cols(), cudaMemcpyHostToDevice));

    chase::Base<T> upperb;
    cudaDeviceSynchronize();
    chase::linalg::internal::cuda_nccl::lanczos(this->get_cublas_handle(),
                                          this->M,
                                          H_,
                                          V_,
                                          &upperb);
    cudaDeviceSynchronize();

    {
        using B = chase::Base<T>;
        const B ref = B(this->N - 1);
        EXPECT_NEAR(upperb, ref, rel_abs_tolerance(ref));
    }
}

TYPED_TEST(LanczosNCCLDistTest, mlanczosGPUBlockCyclic){
    using T = TypeParam;  // Get the current type

    // Allocate CPU buffers for this test
    std::vector<chase::Base<T>> ritzv(this->M * this->numvec);
    std::vector<chase::Base<T>> ritzV(this->M * this->M);
    std::vector<chase::Base<T>> Tau(this->M * this->M * this->numvec);

    std::size_t blocksize = 25;

    // Get pointer to the appropriate Clement matrix
    T* Clement = [this]() -> T* {
        if constexpr (std::is_same_v<T, std::complex<double>>) {
            return reinterpret_cast<T*>(this->get_Clement_z());
        } else if constexpr (std::is_same_v<T, std::complex<float>>) {
            return reinterpret_cast<T*>(this->get_Clement_c());
        } else if constexpr (std::is_same_v<T, double>) {
            return reinterpret_cast<T*>(this->get_Clement_d());
        } else {
            return reinterpret_cast<T*>(this->get_Clement_s());
        }
    }();

    auto H_ = chase::distMatrix::BlockCyclicMatrix<T, chase::platform::GPU>(this->N, this->N, this->m, this->m, blocksize, blocksize, this->m, reinterpret_cast<T*>(this->get_H_device()), this->get_mpi_grid());
    auto V_ = chase::distMultiVector::DistMultiVectorBlockCyclic1D<T, chase::distMultiVector::CommunicatorType::column, chase::platform::GPU>(this->N, this->m, this->M, blocksize, this->m, reinterpret_cast<T*>(this->get_V_device()), this->get_mpi_grid());
    auto H2_ = chase::distMatrix::BlockCyclicMatrix<T, chase::platform::CPU>(this->N, this->N, blocksize, blocksize, this->get_mpi_grid());

    fill_V_cuda_normal_gpu<T>(V_, this->get_mpi_grid(), this->get_stream(),
                              d_curand_states);

    auto Clement_ = chase::distMatrix::RedundantMatrix<T>(this->N, this->N, this->N, Clement, this->get_mpi_grid());
    Clement_.redistributeImpl(&H2_);
    CHECK_CUDA_ERROR(cudaMemcpy(H_.l_data(), H2_.l_data(), sizeof(T) * H2_.l_rows() * H2_.l_cols(), cudaMemcpyHostToDevice));

    chase::Base<T> upperb;
    
    cudaDeviceSynchronize();
    chase::linalg::internal::cuda_nccl::lanczos(this->get_cublas_handle(),
                                          this->M,
                                          this->numvec,
                                          H_,
                                          V_,
                                          &upperb,
                                          ritzv.data(),
                                          Tau.data(),
                                          ritzV.data());

    cudaDeviceSynchronize();
    {
        using B = chase::Base<T>;
        const B spectral = B(this->N - 1);
        const B ref = lanczos_multi_upperb_reference(spectral);
        EXPECT_NEAR(upperb, ref, rel_abs_tolerance(ref));
    }
}

TYPED_TEST(LanczosNCCLDistTest, lanczosGPUBlockCyclic){
    using T = TypeParam;  // Get the current type

    std::size_t blocksize = 25;

    // Get pointer to the appropriate Clement matrix
    T* Clement = [this]() -> T* {
        if constexpr (std::is_same_v<T, std::complex<double>>) {
            return reinterpret_cast<T*>(this->get_Clement_z());
        } else if constexpr (std::is_same_v<T, std::complex<float>>) {
            return reinterpret_cast<T*>(this->get_Clement_c());
        } else if constexpr (std::is_same_v<T, double>) {
            return reinterpret_cast<T*>(this->get_Clement_d());
        } else {
            return reinterpret_cast<T*>(this->get_Clement_s());
        }
    }();

    auto H_ = chase::distMatrix::BlockCyclicMatrix<T, chase::platform::GPU>(this->N, this->N, this->m, this->m, blocksize, blocksize, this->m, reinterpret_cast<T*>(this->get_H_device()), this->get_mpi_grid());
    auto V_ = chase::distMultiVector::DistMultiVectorBlockCyclic1D<T, chase::distMultiVector::CommunicatorType::column, chase::platform::GPU>(this->N, this->m, this->M, blocksize, this->m, reinterpret_cast<T*>(this->get_V_device()), this->get_mpi_grid());
    auto H2_ = chase::distMatrix::BlockCyclicMatrix<T, chase::platform::CPU>(this->N, this->N, blocksize, blocksize, this->get_mpi_grid());

    fill_V_cuda_normal_gpu<T>(V_, this->get_mpi_grid(), this->get_stream(),
                              d_curand_states);

    auto Clement_ = chase::distMatrix::RedundantMatrix<T>(this->N, this->N, this->N, Clement, this->get_mpi_grid());
    Clement_.redistributeImpl(&H2_);
    CHECK_CUDA_ERROR(cudaMemcpy(H_.l_data(), H2_.l_data(), sizeof(T) * H2_.l_rows() * H2_.l_cols(), cudaMemcpyHostToDevice));

    chase::Base<T> upperb;
    cudaDeviceSynchronize();
    chase::linalg::internal::cuda_nccl::lanczos(this->get_cublas_handle(),
                                          this->M,
                                          H_,
                                          V_,
                                          &upperb);
    cudaDeviceSynchronize();

    {
        using B = chase::Base<T>;
        const B ref = B(this->N - 1);
        EXPECT_NEAR(upperb, ref, rel_abs_tolerance(ref));
    }
}


// Add this at the end of the file, before main()
::testing::Environment* const resource_env = ::testing::AddGlobalTestEnvironment(new ResourceCleanupEnvironment);
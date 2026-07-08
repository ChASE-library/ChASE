// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include "linalg/internal/gpu/dist_mpi/cuda/lanczos.hpp"
#include "grid/mpiGrid2D.hpp"
#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"
#include "tests/linalg/internal/lanczos_reference.hpp"
#include <cmath>
#include <complex>
#include <cstring>
#include <gtest/gtest.h>
#include <random>

// Global static resources that persist across all test suites
namespace
{
bool resources_initialized = false;
std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>
    mpi_grid;
cublasHandle_t cublasH;
cudaStream_t stream;
int world_rank, world_size;
std::complex<double>* d_H = nullptr;      // Device memory for H matrix
std::complex<double>* d_V_data = nullptr; // Device memory for V matrix
std::size_t d_H_size = 0;
std::size_t d_V_data_size = 0;
std::complex<double>* Reference_z = nullptr;
std::complex<float>* Reference_c = nullptr;
double* Reference_d = nullptr;
float* Reference_s = nullptr;
std::size_t Reference_size = 0;
double Reference_min_z = 0.0;
float Reference_min_c = 0.0f;
double Reference_min_d = 0.0;
float Reference_min_s = 0.0f;
double Reference_max_z = 0.0;
float Reference_max_c = 0.0f;
double Reference_max_d = 0.0;
float Reference_max_s = 0.0f;
} // namespace

template <typename T>
T* GetReferenceHermitianMatrix()
{
    if constexpr (std::is_same_v<T, std::complex<double>>)
    {
        return Reference_z;
    }
    else if constexpr (std::is_same_v<T, std::complex<float>>)
    {
        return Reference_c;
    }
    else if constexpr (std::is_same_v<T, double>)
    {
        return Reference_d;
    }
    else
    {
        return Reference_s;
    }
}

template <typename T>
chase::Base<T> GetReferenceLambdaMin()
{
    if constexpr (std::is_same_v<T, std::complex<double>>)
    {
        return Reference_min_z;
    }
    else if constexpr (std::is_same_v<T, std::complex<float>>)
    {
        return Reference_min_c;
    }
    else if constexpr (std::is_same_v<T, double>)
    {
        return Reference_min_d;
    }
    else
    {
        return Reference_min_s;
    }
}

template <typename T>
chase::Base<T> GetReferenceLambdaMax()
{
    if constexpr (std::is_same_v<T, std::complex<double>>)
    {
        return Reference_max_z;
    }
    else if constexpr (std::is_same_v<T, std::complex<float>>)
    {
        return Reference_max_c;
    }
    else if constexpr (std::is_same_v<T, double>)
    {
        return Reference_max_d;
    }
    else
    {
        return Reference_max_s;
    }
}
template <typename T>
class LanczosCUDAAwareMPIDistTest : public ::testing::Test
{
protected:
    static void SetUpTestSuite()
    {
        // Initialize resources only once for all test suites
        if (!resources_initialized)
        {
            // Initialize MPI grid
            MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
            MPI_Comm_size(MPI_COMM_WORLD, &world_size);
            ASSERT_EQ(world_size, 4); // Ensure we're running with 4 processes
            mpi_grid = std::make_shared<
                chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(
                2, 2, MPI_COMM_WORLD);

            // Initialize CUDA resources
            CHECK_CUBLAS_ERROR(cublasCreate(&cublasH));
            CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
            CHECK_CUBLAS_ERROR(cublasSetStream(cublasH, stream));

            Reference_size = N * N;
            Reference_z = new std::complex<double>[Reference_size];
            Reference_c = new std::complex<float>[Reference_size];
            Reference_d = new double[Reference_size];
            Reference_s = new float[Reference_size];

            chase::tests::lanczos::initializeReferenceHermitianMatrix(
                Reference_z, N);
            chase::tests::lanczos::initializeReferenceHermitianMatrix(
                Reference_c, N);
            chase::tests::lanczos::initializeReferenceHermitianMatrix(
                Reference_d, N);
            chase::tests::lanczos::initializeReferenceHermitianMatrix(
                Reference_s, N);

            chase::tests::lanczos::computeReferenceEigenBounds(
                Reference_z, N, &Reference_min_z, &Reference_max_z);
            chase::tests::lanczos::computeReferenceEigenBounds(
                Reference_c, N, &Reference_min_c, &Reference_max_c);
            chase::tests::lanczos::computeReferenceEigenBounds(
                Reference_d, N, &Reference_min_d, &Reference_max_d);
            chase::tests::lanczos::computeReferenceEigenBounds(
                Reference_s, N, &Reference_min_s, &Reference_max_s);

            std::cout << "Reference Hermitian matrices initialized"
                      << std::endl;

            resources_initialized = true;
        }
    }

    void SetUp() override
    {
        // Ensure device memory is allocated if needed
        std::size_t required_H_size = m * m;
        std::size_t required_V_size = m * M;

        if (d_H == nullptr || d_H_size < required_H_size)
        {
            if (d_H != nullptr)
            {
                CHECK_CUDA_ERROR(cudaFree(d_H));
            }
            CHECK_CUDA_ERROR(cudaMalloc(
                (void**)&d_H, sizeof(std::complex<double>) * required_H_size));
            d_H_size = required_H_size;
        }

        if (d_V_data == nullptr || d_V_data_size < required_V_size)
        {
            if (d_V_data != nullptr)
            {
                CHECK_CUDA_ERROR(cudaFree(d_V_data));
            }
            CHECK_CUDA_ERROR(
                cudaMalloc((void**)&d_V_data,
                           sizeof(std::complex<double>) * required_V_size));
            d_V_data_size = required_V_size;
        }
    }

    void TearDown() override
    {
        // Don't free device memory here - it will be reused
    }

    static constexpr std::size_t M = 10;
    static constexpr std::size_t numvec = 4;
    static constexpr std::size_t N = 500;
    static constexpr std::size_t m = 250;

    // Accessors for the global resources
    static std::shared_ptr<
        chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>
    get_mpi_grid()
    {
        return mpi_grid;
    }
    static cublasHandle_t get_cublas_handle() { return cublasH; }
    static cudaStream_t get_stream() { return stream; }
    static int get_world_rank() { return world_rank; }
    static int get_world_size() { return world_size; }
    static std::complex<double>* get_H_device() { return d_H; }
    static std::complex<double>* get_V_device() { return d_V_data; }
};

// Add a global test environment to handle resource cleanup at program exit
class ResourceCleanupEnvironment : public ::testing::Environment
{
public:
    ~ResourceCleanupEnvironment() override
    {
        if (resources_initialized)
        {
            if (d_H != nullptr)
            {
                CHECK_CUDA_ERROR(cudaFree(d_H));
                d_H = nullptr;
                d_H_size = 0;
            }

            if (d_V_data != nullptr)
            {
                CHECK_CUDA_ERROR(cudaFree(d_V_data));
                d_V_data = nullptr;
                d_V_data_size = 0;
            }

            delete[] Reference_z;
            delete[] Reference_c;
            delete[] Reference_d;
            delete[] Reference_s;
            Reference_z = nullptr;
            Reference_c = nullptr;
            Reference_d = nullptr;
            Reference_s = nullptr;
            Reference_size = 0;

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

using TestTypes =
    ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(LanczosCUDAAwareMPIDistTest, TestTypes);

TYPED_TEST(LanczosCUDAAwareMPIDistTest, mlanczosGPU)
{
    using T = TypeParam; // Get the current type
    int* coords = this->get_mpi_grid().get()->get_coords();

    // Allocate CPU buffers for this test
    std::vector<chase::Base<T>> ritzv(this->M * this->numvec);
    std::vector<chase::Base<T>> ritzV(this->M * this->M);
    std::vector<chase::Base<T>> Tau(this->M * this->M * this->numvec);

    T* reference_matrix = GetReferenceHermitianMatrix<T>();

    auto H_ = chase::distMatrix::BlockBlockMatrix<T, chase::platform::GPU>(
        this->m, this->m, this->m, reinterpret_cast<T*>(this->get_H_device()),
        this->get_mpi_grid());
    auto V_ = chase::distMultiVector::DistMultiVector1D<
        T, chase::distMultiVector::CommunicatorType::column,
        chase::platform::GPU>(this->m, this->M, this->m,
                              reinterpret_cast<T*>(this->get_V_device()),
                              this->get_mpi_grid());
    auto H2_ = chase::distMatrix::BlockBlockMatrix<T, chase::platform::CPU>(
        this->N, this->N, this->get_mpi_grid());

    V_.allocate_cpu_data();
    std::mt19937 gen(1337.0 + coords[0]);
    std::normal_distribution<> d;

    for (auto j = 0; j < V_.l_rows() * V_.l_cols(); j++)
    {
        auto rnd = getRandomT<T>([&]() { return d(gen); });
        V_.cpu_data()[j] = rnd;
    }

    V_.H2D();

    auto Reference_ =
        chase::distMatrix::RedundantMatrix<T, chase::platform::CPU>(
            this->N, this->N, this->N, reference_matrix,
            this->get_mpi_grid());
    Reference_.redistributeImpl(&H2_);
    CHECK_CUDA_ERROR(cudaMemcpy(H_.l_data(), H2_.l_data(),
                                sizeof(T) * H2_.l_rows() * H2_.l_cols(),
                                cudaMemcpyHostToDevice));

    chase::Base<T> upperb;

    chase::linalg::internal::cuda_mpi::lanczos(
        this->get_cublas_handle(), this->M, this->numvec, H_, V_, &upperb,
        ritzv.data(), Tau.data(), ritzV.data());

    chase::tests::lanczos::expectRitzValuesInsideReferenceSpectrum(
        ritzv, this->M, this->numvec, GetReferenceLambdaMin<T>(),
        GetReferenceLambdaMax<T>());
    chase::tests::lanczos::expectLanczosUpperBoundNearReference(
        upperb, GetReferenceLambdaMin<T>(), GetReferenceLambdaMax<T>());
}

TYPED_TEST(LanczosCUDAAwareMPIDistTest, lanczosGPU)
{
    using T = TypeParam; // Get the current type

    int* coords = this->get_mpi_grid().get()->get_coords();

    T* reference_matrix = GetReferenceHermitianMatrix<T>();

    auto H_ = chase::distMatrix::BlockBlockMatrix<T, chase::platform::GPU>(
        this->m, this->m, this->m, reinterpret_cast<T*>(this->get_H_device()),
        this->get_mpi_grid());
    auto V_ = chase::distMultiVector::DistMultiVector1D<
        T, chase::distMultiVector::CommunicatorType::column,
        chase::platform::GPU>(this->m, this->M, this->m,
                              reinterpret_cast<T*>(this->get_V_device()),
                              this->get_mpi_grid());
    auto H2_ = chase::distMatrix::BlockBlockMatrix<T, chase::platform::CPU>(
        this->N, this->N, this->get_mpi_grid());

    V_.allocate_cpu_data();

    std::mt19937 gen(1337.0 + coords[0]);
    std::normal_distribution<> d;

    for (auto j = 0; j < V_.l_rows() * V_.l_cols(); j++)
    {
        auto rnd = getRandomT<T>([&]() { return d(gen); });
        V_.cpu_data()[j] = rnd;
    }

    V_.H2D();

    auto Reference_ = chase::distMatrix::RedundantMatrix<T>(
        this->N, this->N, this->N, reference_matrix, this->get_mpi_grid());
    Reference_.redistributeImpl(&H2_);
    CHECK_CUDA_ERROR(cudaMemcpy(H_.l_data(), H2_.l_data(),
                                sizeof(T) * H2_.l_rows() * H2_.l_cols(),
                                cudaMemcpyHostToDevice));

    chase::Base<T> upperb;

    chase::linalg::internal::cuda_mpi::lanczos(this->get_cublas_handle(),
                                               this->M, H_, V_, &upperb);

    chase::tests::lanczos::expectLanczosUpperBoundNearReference(
        upperb, GetReferenceLambdaMin<T>(), GetReferenceLambdaMax<T>());
}

TYPED_TEST(LanczosCUDAAwareMPIDistTest, mlanczosGPUBlockCyclic)
{
    using T = TypeParam; // Get the current type

    // Allocate CPU buffers for this test
    std::vector<chase::Base<T>> ritzv(this->M * this->numvec);
    std::vector<chase::Base<T>> ritzV(this->M * this->M);
    std::vector<chase::Base<T>> Tau(this->M * this->M * this->numvec);

    int* coords = this->get_mpi_grid().get()->get_coords();
    std::size_t blocksize = 25;

    T* reference_matrix = GetReferenceHermitianMatrix<T>();

    auto H_ = chase::distMatrix::BlockCyclicMatrix<T, chase::platform::GPU>(
        this->N, this->N, this->m, this->m, blocksize, blocksize, this->m,
        reinterpret_cast<T*>(this->get_H_device()), this->get_mpi_grid());
    auto V_ = chase::distMultiVector::DistMultiVectorBlockCyclic1D<
        T, chase::distMultiVector::CommunicatorType::column,
        chase::platform::GPU>(this->N, this->m, this->M, blocksize, this->m,
                              reinterpret_cast<T*>(this->get_V_device()),
                              this->get_mpi_grid());
    auto H2_ = chase::distMatrix::BlockCyclicMatrix<T, chase::platform::CPU>(
        this->N, this->N, blocksize, blocksize, this->get_mpi_grid());

    V_.allocate_cpu_data();
    std::mt19937 gen(1337.0 + coords[0]);
    std::normal_distribution<> d;

    for (auto j = 0; j < V_.l_rows() * V_.l_cols(); j++)
    {
        auto rnd = getRandomT<T>([&]() { return d(gen); });
        V_.cpu_data()[j] = rnd;
    }

    V_.H2D();

    auto Reference_ = chase::distMatrix::RedundantMatrix<T>(
        this->N, this->N, this->N, reference_matrix, this->get_mpi_grid());
    Reference_.redistributeImpl(&H2_);
    CHECK_CUDA_ERROR(cudaMemcpy(H_.l_data(), H2_.l_data(),
                                sizeof(T) * H2_.l_rows() * H2_.l_cols(),
                                cudaMemcpyHostToDevice));

    chase::Base<T> upperb;

    chase::linalg::internal::cuda_mpi::lanczos(
        this->get_cublas_handle(), this->M, this->numvec, H_, V_, &upperb,
        ritzv.data(), Tau.data(), ritzV.data());

    chase::tests::lanczos::expectRitzValuesInsideReferenceSpectrum(
        ritzv, this->M, this->numvec, GetReferenceLambdaMin<T>(),
        GetReferenceLambdaMax<T>());
    chase::tests::lanczos::expectLanczosUpperBoundNearReference(
        upperb, GetReferenceLambdaMin<T>(), GetReferenceLambdaMax<T>());
}

TYPED_TEST(LanczosCUDAAwareMPIDistTest, lanczosGPUBlockCyclic)
{
    using T = TypeParam; // Get the current type

    int* coords = this->get_mpi_grid().get()->get_coords();
    std::size_t blocksize = 25;

    T* reference_matrix = GetReferenceHermitianMatrix<T>();

    auto H_ = chase::distMatrix::BlockCyclicMatrix<T, chase::platform::GPU>(
        this->N, this->N, this->m, this->m, blocksize, blocksize, this->m,
        reinterpret_cast<T*>(this->get_H_device()), this->get_mpi_grid());
    auto V_ = chase::distMultiVector::DistMultiVectorBlockCyclic1D<
        T, chase::distMultiVector::CommunicatorType::column,
        chase::platform::GPU>(this->N, this->m, this->M, blocksize, this->m,
                              reinterpret_cast<T*>(this->get_V_device()),
                              this->get_mpi_grid());
    auto H2_ = chase::distMatrix::BlockCyclicMatrix<T, chase::platform::CPU>(
        this->N, this->N, blocksize, blocksize, this->get_mpi_grid());

    V_.allocate_cpu_data();
    std::mt19937 gen(1337.0 + coords[0]);
    std::normal_distribution<> d;

    for (auto j = 0; j < V_.l_rows() * V_.l_cols(); j++)
    {
        auto rnd = getRandomT<T>([&]() { return d(gen); });
        V_.cpu_data()[j] = rnd;
    }

    V_.H2D();

    auto Reference_ = chase::distMatrix::RedundantMatrix<T>(
        this->N, this->N, this->N, reference_matrix, this->get_mpi_grid());
    Reference_.redistributeImpl(&H2_);
    CHECK_CUDA_ERROR(cudaMemcpy(H_.l_data(), H2_.l_data(),
                                sizeof(T) * H2_.l_rows() * H2_.l_cols(),
                                cudaMemcpyHostToDevice));

    chase::Base<T> upperb;

    chase::linalg::internal::cuda_mpi::lanczos(this->get_cublas_handle(),
                                               this->M, H_, V_, &upperb);

    chase::tests::lanczos::expectLanczosUpperBoundNearReference(
        upperb, GetReferenceLambdaMin<T>(), GetReferenceLambdaMax<T>());
}

// Add this at the end of the file, before main()
::testing::Environment* const resource_env =
    ::testing::AddGlobalTestEnvironment(new ResourceCleanupEnvironment);

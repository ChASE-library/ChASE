// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include "linalg/internal/gpu/backend/error.hpp"
#include "external/amd_gpu/rocblaspp/rocblaspp.hpp"
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include "linalg/internal/gpu/dist_xccl/rccl/lanczos.hpp"
#include "grid/mpiGrid2D.hpp"
#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <gtest/gtest.h>
#include <random>
#include <type_traits>

// Global static resources that persist across all test suites
namespace
{
bool resources_initialized = false;
std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>
    mpi_grid;
rocblas_handle cublasH;
hipStream_t stream;
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
struct IsStdComplex : std::false_type
{
};

template <typename T>
struct IsStdComplex<std::complex<T>> : std::true_type
{
};

template <typename T>
T makeHermitianValue(chase::Base<T> real, chase::Base<T> imag)
{
    if constexpr (IsStdComplex<T>::value)
    {
        return T(real, imag);
    }
    else
    {
        (void)imag;
        return T(real);
    }
}

template <typename T>
T conjugateForHermitian(T value)
{
    if constexpr (IsStdComplex<T>::value)
    {
        return std::conj(value);
    }
    else
    {
        return value;
    }
}

template <typename T>
void InitializeReferenceHermitianMatrix(T* matrix, std::size_t N)
{
    using RealT = chase::Base<T>;

    std::fill(matrix, matrix + N * N, T(0));
    for (std::size_t j = 0; j < N; ++j)
    {
        const RealT uj =
            RealT(1.0) + RealT(0.2) * std::cos(RealT(0.07 * j));
        for (std::size_t i = j; i < N; ++i)
        {
            const RealT ui =
                RealT(1.0) + RealT(0.2) * std::cos(RealT(0.07 * i));
            RealT real = RealT(45.0) * ui * uj / RealT(N);
            RealT imag = RealT(0.0);

            if (i == j)
            {
                real += RealT(5.0) + RealT(25.0 * i) / RealT(N - 1);
            }
            else
            {
                const auto distance =
                    static_cast<RealT>(std::min(i - j, std::size_t(16)));
                real += RealT(0.08) * std::cos(RealT(0.11 * (i + j))) /
                        (RealT(1.0) + distance);
                if constexpr (IsStdComplex<T>::value)
                {
                    imag = RealT(0.04) *
                           std::sin(RealT(0.13 * i + 0.17 * j)) /
                           (RealT(1.0) + distance);
                }
            }

            matrix[i + N * j] = makeHermitianValue<T>(real, imag);
            if (i != j)
            {
                matrix[j + N * i] =
                    conjugateForHermitian(matrix[i + N * j]);
            }
        }
    }
}

template <typename T>
void ComputeReferenceEigenBounds(const T* matrix, std::size_t N,
                                 chase::Base<T>* lambda_min,
                                 chase::Base<T>* lambda_max)
{
    std::vector<T> work(matrix, matrix + N * N);
    std::vector<chase::Base<T>> eigs(N);
    chase::linalg::lapackpp::t_heevd(CblasColMajor, 'N', 'U', N,
                                     work.data(), N, eigs.data());
    *lambda_min = eigs.front();
    *lambda_max = eigs.back();
}

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
void ExpectRitzValuesInsideReferenceSpectrum(
    const std::vector<chase::Base<T>>& ritzv, std::size_t M,
    std::size_t numvec)
{
    using RealT = chase::Base<T>;
    const RealT lambda_min = GetReferenceLambdaMin<T>();
    const RealT lambda_max = GetReferenceLambdaMax<T>();
    const RealT span = std::max(lambda_max - lambda_min, RealT(1.0));
    const RealT slack = RealT(0.05) * span;

    for (std::size_t i = 0; i < numvec; ++i)
    {
        EXPECT_GE(ritzv[i * M], lambda_min - slack);
        EXPECT_LE(ritzv[(i + 1) * M - 1], lambda_max + slack);
    }
}

template <typename T>
void ExpectLanczosUpperBoundNearReference(chase::Base<T> upperb)
{
    using RealT = chase::Base<T>;
    const RealT lambda_min = GetReferenceLambdaMin<T>();
    const RealT lambda_max = GetReferenceLambdaMax<T>();
    const RealT span = std::max(lambda_max - lambda_min, RealT(1.0));

    EXPECT_TRUE(std::isfinite(upperb));
    EXPECT_GE(upperb, lambda_max - RealT(0.02) * span);
    EXPECT_LE(upperb, lambda_max + RealT(0.35) * span);
}

template <typename T>
class LanczosGPURCCLDistTest : public ::testing::Test
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
            CHECK_ROCBLAS_ERROR(rocblas_create_handle(&cublasH));
            CHECK_HIP_ERROR(hipStreamCreate(&stream));
            CHECK_ROCBLAS_ERROR(rocblas_set_stream(cublasH, stream));

            Reference_size = N * N;
            Reference_z = new std::complex<double>[Reference_size];
            Reference_c = new std::complex<float>[Reference_size];
            Reference_d = new double[Reference_size];
            Reference_s = new float[Reference_size];

            InitializeReferenceHermitianMatrix(Reference_z, N);
            InitializeReferenceHermitianMatrix(Reference_c, N);
            InitializeReferenceHermitianMatrix(Reference_d, N);
            InitializeReferenceHermitianMatrix(Reference_s, N);

            ComputeReferenceEigenBounds(Reference_z, N, &Reference_min_z,
                                        &Reference_max_z);
            ComputeReferenceEigenBounds(Reference_c, N, &Reference_min_c,
                                        &Reference_max_c);
            ComputeReferenceEigenBounds(Reference_d, N, &Reference_min_d,
                                        &Reference_max_d);
            ComputeReferenceEigenBounds(Reference_s, N, &Reference_min_s,
                                        &Reference_max_s);

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
                CHECK_HIP_ERROR(hipFree(d_H));
            }
            CHECK_HIP_ERROR(hipMalloc(
                (void**)&d_H, sizeof(std::complex<double>) * required_H_size));
            d_H_size = required_H_size;
        }

        if (d_V_data == nullptr || d_V_data_size < required_V_size)
        {
            if (d_V_data != nullptr)
            {
                CHECK_HIP_ERROR(hipFree(d_V_data));
            }
            CHECK_HIP_ERROR(
                hipMalloc((void**)&d_V_data,
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
    static rocblas_handle get_cublas_handle() { return cublasH; }
    static hipStream_t get_stream() { return stream; }
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
                CHECK_HIP_ERROR(hipFree(d_H));
                d_H = nullptr;
                d_H_size = 0;
            }

            if (d_V_data != nullptr)
            {
                CHECK_HIP_ERROR(hipFree(d_V_data));
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

            CHECK_ROCBLAS_ERROR(rocblas_destroy_handle(cublasH));
            CHECK_HIP_ERROR(hipStreamDestroy(stream));

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
TYPED_TEST_SUITE(LanczosGPURCCLDistTest, TestTypes);

TYPED_TEST(LanczosGPURCCLDistTest, mlanczosGPU)
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
    CHECK_HIP_ERROR(hipMemcpy(H_.l_data(), H2_.l_data(),
                                sizeof(T) * H2_.l_rows() * H2_.l_cols(),
                                hipMemcpyHostToDevice));

    chase::Base<T> upperb;

    chase::linalg::internal::hip_rccl::lanczos(
        this->get_cublas_handle(), this->M, this->numvec, H_, V_, &upperb,
        ritzv.data(), Tau.data(), ritzV.data());

    ExpectRitzValuesInsideReferenceSpectrum<T>(ritzv, this->M, this->numvec);
    ExpectLanczosUpperBoundNearReference<T>(upperb);
}

TYPED_TEST(LanczosGPURCCLDistTest, lanczosGPU)
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
    CHECK_HIP_ERROR(hipMemcpy(H_.l_data(), H2_.l_data(),
                                sizeof(T) * H2_.l_rows() * H2_.l_cols(),
                                hipMemcpyHostToDevice));

    chase::Base<T> upperb;

    chase::linalg::internal::hip_rccl::lanczos(this->get_cublas_handle(),
                                               this->M, H_, V_, &upperb);

    ExpectLanczosUpperBoundNearReference<T>(upperb);
}

TYPED_TEST(LanczosGPURCCLDistTest, mlanczosGPUBlockCyclic)
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
    CHECK_HIP_ERROR(hipMemcpy(H_.l_data(), H2_.l_data(),
                                sizeof(T) * H2_.l_rows() * H2_.l_cols(),
                                hipMemcpyHostToDevice));

    chase::Base<T> upperb;

    chase::linalg::internal::hip_rccl::lanczos(
        this->get_cublas_handle(), this->M, this->numvec, H_, V_, &upperb,
        ritzv.data(), Tau.data(), ritzV.data());

    ExpectRitzValuesInsideReferenceSpectrum<T>(ritzv, this->M, this->numvec);
    ExpectLanczosUpperBoundNearReference<T>(upperb);
}

TYPED_TEST(LanczosGPURCCLDistTest, lanczosGPUBlockCyclic)
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
    CHECK_HIP_ERROR(hipMemcpy(H_.l_data(), H2_.l_data(),
                                sizeof(T) * H2_.l_rows() * H2_.l_cols(),
                                hipMemcpyHostToDevice));

    chase::Base<T> upperb;

    chase::linalg::internal::hip_rccl::lanczos(this->get_cublas_handle(),
                                               this->M, H_, V_, &upperb);

    ExpectLanczosUpperBoundNearReference<T>(upperb);
}

// Add this at the end of the file, before main()
::testing::Environment* const resource_env =
    ::testing::AddGlobalTestEnvironment(new ResourceCleanupEnvironment);

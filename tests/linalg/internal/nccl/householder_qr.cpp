// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
// Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include "linalg/internal/nccl/nccl_kernels.hpp"
#include "grid/mpiGrid2D.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"
#include "tests/linalg/internal/mpi/TestConditions.hpp"
#include "tests/linalg/internal/utils.hpp"
#include <complex>
#include <cstdlib>
#include <gtest/gtest.h>
#include <string>

namespace
{
bool resources_initialized = false;
std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>
    mpi_grid;
cublasHandle_t cublasH = nullptr;
cudaStream_t stream = nullptr;
int world_rank = 0, world_size = 0;

/** Set env for the duration of a scope; restore previous value on destruction. */
class ScopedEnvVar
{
public:
    ScopedEnvVar(const char* name, const char* value) : name_(name)
    {
        const char* p = std::getenv(name_);
        had_prev_ = (p != nullptr);
        if (had_prev_)
            prev_ = p;
        setenv(name_, value, 1);
    }
    ~ScopedEnvVar()
    {
        if (had_prev_)
            setenv(name_, prev_.c_str(), 1);
        else
            unsetenv(name_);
    }
    ScopedEnvVar(const ScopedEnvVar&) = delete;
    ScopedEnvVar& operator=(const ScopedEnvVar&) = delete;

private:
    const char* name_;
    std::string prev_;
    bool had_prev_;
};
} // namespace

template <typename T>
class HouseholderNCCLDistTest : public ::testing::Test
{
protected:
    static void SetUpTestSuite()
    {
        if (!resources_initialized)
        {
            MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
            MPI_Comm_size(MPI_COMM_WORLD, &world_size);
            ASSERT_EQ(world_size, 4);
            mpi_grid = std::make_shared<
                chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(
                4, 1, MPI_COMM_WORLD);
            CHECK_CUBLAS_ERROR(cublasCreate(&cublasH));
            CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
            CHECK_CUBLAS_ERROR(cublasSetStream(cublasH, stream));
            resources_initialized = true;
        }
    }

    static constexpr std::size_t N = 100;
    static constexpr std::size_t n = 50;

    static auto grid()
    {
        return mpi_grid;
    }
    static cublasHandle_t cublas() { return cublasH; }
    static int rank() { return world_rank; }
};

class HouseholderNCCLResourceCleanup : public ::testing::Environment
{
public:
    ~HouseholderNCCLResourceCleanup() override
    {
        if (resources_initialized)
        {
            CHECK_CUBLAS_ERROR(cublasDestroy(cublasH));
            CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
            mpi_grid.reset();
            resources_initialized = false;
        }
    }
};

using HouseholderNCCLTypes =
    ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(HouseholderNCCLDistTest, HouseholderNCCLTypes);

TYPED_TEST(HouseholderNCCLDistTest, houseQR1FormQCond10)
{
    using T = TypeParam;
    auto machineEpsilon = MachineEpsilon<T>::value();

    auto V_ = chase::distMultiVector::DistMultiVector1D<
        T, chase::distMultiVector::CommunicatorType::column,
        chase::platform::GPU>(this->N, this->n, this->grid());

    const std::size_t xlen = V_.l_rows();
    const std::size_t xoff =
        static_cast<std::size_t>(this->rank()) * (this->N / 4u);

    std::vector<T> Vh(xlen * this->n);
    read_vectors(Vh.data(), GetQRFileName<T>() + "cond_10.bin", xoff, xlen,
                 this->N, this->n, 0);
    CHECK_CUDA_ERROR(cudaMemcpy(V_.l_data(), Vh.data(),
                                sizeof(T) * xlen * this->n,
                                cudaMemcpyHostToDevice));

    chase::linalg::internal::cuda_nccl::houseQR1_formQ(
        this->cublas(), V_, nullptr, 0, nullptr);

    CHECK_CUDA_ERROR(cudaMemcpy(Vh.data(), V_.l_data(),
                                sizeof(T) * xlen * this->n,
                                cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    const auto orth =
        orthogonality<T>(xlen, this->n, Vh.data(), V_.l_ld(),
                         this->grid()->get_col_comm());
    ASSERT_NEAR(orth, machineEpsilon, machineEpsilon * 25);
}

// Exercise HouseQRTuning (struct overrides defaults; same path as production).
TYPED_TEST(HouseholderNCCLDistTest, houseQR1FormQWithTuningStruct)
{
    using T = TypeParam;
    auto machineEpsilon = MachineEpsilon<T>::value();

    auto V_ = chase::distMultiVector::DistMultiVector1D<
        T, chase::distMultiVector::CommunicatorType::column,
        chase::platform::GPU>(this->N, this->n, this->grid());

    const std::size_t xlen = V_.l_rows();
    const std::size_t xoff =
        static_cast<std::size_t>(this->rank()) * (this->N / 4u);

    std::vector<T> Vh(xlen * this->n);
    read_vectors(Vh.data(), GetQRFileName<T>() + "cond_10.bin", xoff, xlen,
                 this->N, this->n, 0);
    CHECK_CUDA_ERROR(cudaMemcpy(V_.l_data(), Vh.data(),
                                sizeof(T) * xlen * this->n,
                                cudaMemcpyHostToDevice));

    chase::linalg::internal::cuda_nccl::HouseQRTuning tuning{};
    tuning.outer_block_nb = 24;
    tuning.panel_sub_nb = 6;
    tuning.formq_chunks = 2;
    tuning.timing_blocking = 0;

    chase::linalg::internal::cuda_nccl::houseQR1_formQ(
        this->cublas(), V_, nullptr, 0, &tuning);

    CHECK_CUDA_ERROR(cudaMemcpy(Vh.data(), V_.l_data(),
                                sizeof(T) * xlen * this->n,
                                cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    const auto orth =
        orthogonality<T>(xlen, this->n, Vh.data(), V_.l_ld(),
                         this->grid()->get_col_comm());
    ASSERT_NEAR(orth, machineEpsilon, machineEpsilon * 25);
}

// CHASE_QR_CHECK_ORTHO=1 enables nccl_validate_orthogonality inside houseQR1_formQ.
TYPED_TEST(HouseholderNCCLDistTest, envCheckOrthoRuns)
{
    using T = TypeParam;
    auto machineEpsilon = MachineEpsilon<T>::value();

    ScopedEnvVar check_ortho("CHASE_QR_CHECK_ORTHO", "1");

    auto V_ = chase::distMultiVector::DistMultiVector1D<
        T, chase::distMultiVector::CommunicatorType::column,
        chase::platform::GPU>(this->N, this->n, this->grid());

    const std::size_t xlen = V_.l_rows();
    const std::size_t xoff =
        static_cast<std::size_t>(this->rank()) * (this->N / 4u);

    std::vector<T> Vh(xlen * this->n);
    read_vectors(Vh.data(), GetQRFileName<T>() + "cond_10.bin", xoff, xlen,
                 this->N, this->n, 0);
    CHECK_CUDA_ERROR(cudaMemcpy(V_.l_data(), Vh.data(),
                                sizeof(T) * xlen * this->n,
                                cudaMemcpyHostToDevice));

    ASSERT_NO_THROW(chase::linalg::internal::cuda_nccl::houseQR1_formQ(
        this->cublas(), V_, nullptr, 0, nullptr));

    CHECK_CUDA_ERROR(cudaMemcpy(Vh.data(), V_.l_data(),
                                sizeof(T) * xlen * this->n,
                                cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    const auto orth =
        orthogonality<T>(xlen, this->n, Vh.data(), V_.l_ld(),
                         this->grid()->get_col_comm());
    ASSERT_NEAR(orth, machineEpsilon, machineEpsilon * 25);
}

// CHASE_QR_SUB_NB / CHASE_QR_OUTER_BLOCK_NB are read when tuning is null.
TYPED_TEST(HouseholderNCCLDistTest, envOuterBlockAndSubNb)
{
    using T = TypeParam;
    auto machineEpsilon = MachineEpsilon<T>::value();

    ScopedEnvVar sub_nb("CHASE_QR_SUB_NB", "4");
    ScopedEnvVar outer_nb("CHASE_QR_OUTER_BLOCK_NB", "16");

    auto V_ = chase::distMultiVector::DistMultiVector1D<
        T, chase::distMultiVector::CommunicatorType::column,
        chase::platform::GPU>(this->N, this->n, this->grid());

    const std::size_t xlen = V_.l_rows();
    const std::size_t xoff =
        static_cast<std::size_t>(this->rank()) * (this->N / 4u);

    std::vector<T> Vh(xlen * this->n);
    read_vectors(Vh.data(), GetQRFileName<T>() + "cond_1e4.bin", xoff, xlen,
                 this->N, this->n, 0);
    CHECK_CUDA_ERROR(cudaMemcpy(V_.l_data(), Vh.data(),
                                sizeof(T) * xlen * this->n,
                                cudaMemcpyHostToDevice));

    chase::linalg::internal::cuda_nccl::houseQR1_formQ(
        this->cublas(), V_, nullptr, 0, nullptr);

    CHECK_CUDA_ERROR(cudaMemcpy(Vh.data(), V_.l_data(),
                                sizeof(T) * xlen * this->n,
                                cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    const auto orth =
        orthogonality<T>(xlen, this->n, Vh.data(), V_.l_ld(),
                         this->grid()->get_col_comm());
    EXPECT_GT(orth, machineEpsilon);
    EXPECT_LT(orth, chase::Base<T>(1));
}

::testing::Environment* const householder_nccl_env =
    ::testing::AddGlobalTestEnvironment(new HouseholderNCCLResourceCleanup);

// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
// Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include "linalg/internal/cuda_aware_mpi/cuda_mpi_kernels.hpp"
#include "linalg/internal/cuda_aware_mpi/householder_qr.hpp"
#include "grid/mpiGrid2D.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"
#include "tests/linalg/internal/mpi/TestConditions.hpp"
#include "tests/linalg/internal/utils.hpp"
#include <complex>
#include <gtest/gtest.h>

namespace
{
bool resources_initialized = false;
std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>
    mpi_grid;
cublasHandle_t cublasH = nullptr;
cudaStream_t stream = nullptr;
int world_rank = 0, world_size = 0;
} // namespace

template <typename T>
class HouseholderCUDAMPIDistTest : public ::testing::Test
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

    static std::shared_ptr<
        chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>
    grid()
    {
        return mpi_grid;
    }
    static cublasHandle_t cublas() { return cublasH; }
    static int rank() { return world_rank; }
};

class HouseholderCUDAMPIResourceCleanup : public ::testing::Environment
{
public:
    ~HouseholderCUDAMPIResourceCleanup() override
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

using HouseholderCUDAMPITypes =
    ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(HouseholderCUDAMPIDistTest, HouseholderCUDAMPITypes);

TYPED_TEST(HouseholderCUDAMPIDistTest, houseQR1FormQCond10)
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

    chase::linalg::internal::cuda_mpi::HouseQRTuning tuning;
    chase::linalg::internal::cuda_mpi::houseQR1_formQ(
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

TYPED_TEST(HouseholderCUDAMPIDistTest, houseQR1FormQCond1e4)
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
    read_vectors(Vh.data(), GetQRFileName<T>() + "cond_1e4.bin", xoff, xlen,
                 this->N, this->n, 0);
    CHECK_CUDA_ERROR(cudaMemcpy(V_.l_data(), Vh.data(),
                                sizeof(T) * xlen * this->n,
                                cudaMemcpyHostToDevice));

    // Householder QR is backward-stable: Q^H Q ≈ I regardless of conditioning.
    chase::linalg::internal::cuda_mpi::houseQR1_formQ(
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

::testing::Environment* const householder_cuda_mpi_env =
    ::testing::AddGlobalTestEnvironment(
        new HouseholderCUDAMPIResourceCleanup);

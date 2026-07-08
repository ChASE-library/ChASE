// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include "linalg/internal/gpu/backend/error.hpp"
#include "grid/mpiGrid2D.hpp"
#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"
#include "linalg/internal/gpu/dist_mpi/hip/pseudo_hermitian_rayleighRitz.hpp"
#include "linalg/matrix/matrix.hpp"
#include "tests/linalg/internal/utils.hpp"
#include <algorithm>
#include <complex>
#include <gtest/gtest.h>
#include <memory>
#include <rocblas/rocblas.h>
#include <type_traits>

namespace
{
bool resources_initialized = false;
std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>
    mpi_grid;
rocblas_handle rocblasH;
} // namespace

template <typename T>
class PseudoHermitianRayleighRitzHIPAwareMPIDistTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);

        ASSERT_EQ(world_size, 4);

        if (!resources_initialized)
        {
            mpi_grid = std::make_shared<
                chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(
                2, 2, MPI_COMM_WORLD);
            CHECK_ROCBLAS_ERROR(rocblas_create_handle(&rocblasH));
            resources_initialized = true;
        }
    }

    int world_rank = 0;
    int world_size = 0;

    std::size_t N_tiny = 10;
    std::size_t N = 200;

    static rocblas_handle get_rocblas_handle() { return rocblasH; }
    static std::shared_ptr<
        chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>
    get_mpi_grid()
    {
        return mpi_grid;
    }
};

class ResourceCleanupEnvironment : public ::testing::Environment
{
public:
    ~ResourceCleanupEnvironment() override
    {
        if (resources_initialized)
        {
            CHECK_ROCBLAS_ERROR(rocblas_destroy_handle(rocblasH));
            mpi_grid.reset();
            resources_initialized = false;
        }
    }
};

::testing::Environment* const env =
    ::testing::AddGlobalTestEnvironment(new ResourceCleanupEnvironment());

using TestTypes =
    ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(PseudoHermitianRayleighRitzHIPAwareMPIDistTest, TestTypes);

enum class PseudoRayleighRitzVariant
{
    HostLapackGeev,
    PotrfHeevd
};

template <typename T>
void fillDistributedIdentity(
    chase::distMultiVector::DistMultiVector1D<
        T, chase::distMultiVector::CommunicatorType::column,
        chase::platform::GPU>& V)
{
    V.allocate_cpu_data();
    const std::size_t g_off = V.g_off();

    for (std::size_t i = 0; i < V.l_rows(); ++i)
    {
        V.cpu_data()[g_off * V.cpu_ld() + i * (V.cpu_ld() + 1)] = T(1.0);
    }

    V.H2D();
}

template <typename T>
void runPseudoHermitianRayleighRitz(std::size_t N, chase::Base<T> tolerance,
                                    PseudoRayleighRitzVariant variant)
{
    auto H = chase::distMatrix::PseudoHermitianBlockBlockMatrix<
        T, chase::platform::GPU>(N, N, mpi_grid);
    H.allocate_cpu_data();
    H.readFromBinaryFile(N == 10 ? GetBSE_TinyMatrix<T>() : GetBSE_Matrix<T>());
    H.H2D();

    auto exact_eigs = chase::matrix::Matrix<T>(N, 1);
    exact_eigs.readFromBinaryFile(N == 10 ? GetBSE_TinyEigs<T>()
                                          : GetBSE_Eigs<T>());

    auto V1 = chase::distMultiVector::DistMultiVector1D<
        T, chase::distMultiVector::CommunicatorType::column,
        chase::platform::GPU>(N, N, mpi_grid);
    auto V2 = chase::distMultiVector::DistMultiVector1D<
        T, chase::distMultiVector::CommunicatorType::column,
        chase::platform::GPU>(N, N, mpi_grid);
    fillDistributedIdentity(V1);
    fillDistributedIdentity(V2);

    auto W1 = chase::distMultiVector::DistMultiVector1D<
        T, chase::distMultiVector::CommunicatorType::row, chase::platform::GPU>(
        N, N, mpi_grid);
    auto W2 = chase::distMultiVector::DistMultiVector1D<
        T, chase::distMultiVector::CommunicatorType::row, chase::platform::GPU>(
        N, N, mpi_grid);

    auto ritzv =
        chase::distMatrix::RedundantMatrix<chase::Base<T>, chase::platform::GPU>(
            N, 1, mpi_grid);
    ritzv.allocate_cpu_data();

    int* devInfo = nullptr;
    CHECK_HIP_ERROR(hipMalloc(reinterpret_cast<void**>(&devInfo), sizeof(int)));

    if (variant == PseudoRayleighRitzVariant::HostLapackGeev)
    {
        chase::linalg::internal::hip_mpi::pseudo_hermitian_rayleighRitz(
            rocblasH, rocblasH, nullptr, H, V1, V2, W1, W2, ritzv, 0, N,
            devInfo, nullptr, 0, nullptr, 0, nullptr);
    }
    else
    {
        chase::linalg::internal::hip_mpi::pseudo_hermitian_rayleighRitz_v2(
            rocblasH, rocblasH, nullptr, H, V1, V2, W1, W2, ritzv, 0, N,
            devInfo, nullptr, 0, nullptr);
    }

    CHECK_HIP_ERROR(hipFree(devInfo));

    if (variant == PseudoRayleighRitzVariant::PotrfHeevd)
    {
        std::sort(ritzv.cpu_data(), ritzv.cpu_data() + N);
    }

    for (std::size_t i = 0; i < N; ++i)
    {
        EXPECT_NEAR(ritzv.cpu_data()[i],
                    chase::Base<T>(std::real(exact_eigs.data()[i])),
                    tolerance);
    }
}

TYPED_TEST(PseudoHermitianRayleighRitzHIPAwareMPIDistTest,
           TinyPseudoHermitianRayleighRitz)
{
    using T = TypeParam;
    runPseudoHermitianRayleighRitz<T>(
        this->N_tiny, 200 * MachineEpsilon<chase::Base<T>>::value(),
        PseudoRayleighRitzVariant::PotrfHeevd);
}

TYPED_TEST(PseudoHermitianRayleighRitzHIPAwareMPIDistTest,
           PseudoHermitianRayleighRitz)
{
    using T = TypeParam;
    chase::Base<T> tolerance = 1.0e-9;
    if constexpr (std::is_same<T, float>::value ||
                  std::is_same<T, std::complex<float>>::value)
    {
        tolerance = 1.0e-3;
    }

    runPseudoHermitianRayleighRitz<T>(this->N, tolerance,
                                      PseudoRayleighRitzVariant::PotrfHeevd);
}

TYPED_TEST(PseudoHermitianRayleighRitzHIPAwareMPIDistTest,
           TinyPseudoHermitianRayleighRitzHostLapackGeev)
{
    using T = TypeParam;
    runPseudoHermitianRayleighRitz<T>(
        this->N_tiny, 500 * MachineEpsilon<chase::Base<T>>::value(),
        PseudoRayleighRitzVariant::HostLapackGeev);
}

TYPED_TEST(PseudoHermitianRayleighRitzHIPAwareMPIDistTest,
           PseudoHermitianRayleighRitzHostLapackGeev)
{
    using T = TypeParam;
    chase::Base<T> tolerance = 1.0e-9;
    if constexpr (std::is_same<T, float>::value ||
                  std::is_same<T, std::complex<float>>::value)
    {
        tolerance = 1.0e-3;
    }

    runPseudoHermitianRayleighRitz<T>(
        this->N, tolerance, PseudoRayleighRitzVariant::HostLapackGeev);
}

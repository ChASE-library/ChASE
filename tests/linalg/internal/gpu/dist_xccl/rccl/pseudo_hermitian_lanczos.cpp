// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include "linalg/internal/gpu/backend/error.hpp"
#include "grid/mpiGrid2D.hpp"
#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"
#include "linalg/internal/gpu/dist_xccl/rccl/pseudo_hermitian_lanczos.hpp"
#include "linalg/matrix/matrix.hpp"
#include "tests/linalg/internal/utils.hpp"
#include <cmath>
#include <complex>
#include <cstring>
#include <gtest/gtest.h>
#include <memory>
#include <random>
#include <rocblas/rocblas.h>
#include <vector>

namespace
{
bool resources_initialized = false;
std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>
    mpi_grid;
rocblas_handle rocblasH;
} // namespace

template <typename T>
class PseudoHermitianLanczosGPURCCLDistTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);

        ritzv_tiny.resize(M_tiny * numvec_tiny);
        ritzV_tiny.resize(M_tiny * M_tiny);
        Tau_tiny.resize(M_tiny * M_tiny * numvec_tiny);

        ritzv.resize(M * numvec);
        ritzV.resize(M * M);
        Tau.resize(M * M * numvec);

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

    std::size_t M_tiny = 4;
    std::size_t numvec_tiny = 2;
    std::size_t N_tiny = 10;
    std::vector<chase::Base<T>> ritzv_tiny;
    std::vector<chase::Base<T>> ritzV_tiny;
    std::vector<chase::Base<T>> Tau_tiny;

    std::size_t M = 20; // Does not work for M = N... same as NVIDIA test.
    std::size_t numvec = 4;
    std::size_t N = 200;
    std::vector<chase::Base<T>> ritzv;
    std::vector<chase::Base<T>> ritzV;
    std::vector<chase::Base<T>> Tau;

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
TYPED_TEST_SUITE(PseudoHermitianLanczosGPURCCLDistTest, TestTypes);

template <typename T>
void fillRandomVector(
    chase::distMultiVector::DistMultiVector1D<
        T, chase::distMultiVector::CommunicatorType::column,
        chase::platform::GPU>& V,
    int row_coord)
{
    V.allocate_cpu_data();

    std::mt19937 gen(1337.0 + row_coord);
    std::normal_distribution<> dist;

    for (std::size_t j = 0; j < V.l_rows() * V.l_cols(); ++j)
    {
        V.cpu_data()[j] = getRandomT<T>([&]() { return dist(gen); });
    }

    V.H2D();
}

TYPED_TEST(PseudoHermitianLanczosGPURCCLDistTest,
           tinyPseudoHermitianLanczos)
{
    using T = TypeParam;

    int* coords = this->get_mpi_grid()->get_coords();

    auto H_ = chase::distMatrix::PseudoHermitianBlockBlockMatrix<
        T, chase::platform::GPU>(this->N_tiny, this->N_tiny,
                                 this->get_mpi_grid());
    H_.allocate_cpu_data();
    H_.readFromBinaryFile(GetBSE_TinyMatrix<T>());
    H_.H2D();

    chase::matrix::Matrix<T> exact_eigsl_H(this->N_tiny, 1);
    exact_eigsl_H.readFromBinaryFile(GetBSE_TinyEigs<T>());

    auto V_ = chase::distMultiVector::DistMultiVector1D<
        T, chase::distMultiVector::CommunicatorType::column,
        chase::platform::GPU>(this->N_tiny, this->M_tiny,
                              this->get_mpi_grid());
    fillRandomVector(V_, coords[0]);

    chase::Base<T> upperb;

    chase::linalg::internal::hip_rccl::pseudo_hermitian_lanczos(
        this->get_rocblas_handle(), this->M_tiny, this->numvec_tiny, H_, V_,
        &upperb, this->ritzv_tiny.data(), this->Tau_tiny.data(),
        this->ritzV_tiny.data());

    for (std::size_t i = 0; i < this->numvec_tiny; ++i)
    {
        EXPECT_GT(this->ritzv_tiny[i * this->M_tiny],
                  std::real(exact_eigsl_H.data()[0]));
        EXPECT_LT(this->ritzv_tiny[(i + 1) * this->M_tiny - 1],
                  std::real(exact_eigsl_H.data()[this->N_tiny - 1]));
    }
}

TYPED_TEST(PseudoHermitianLanczosGPURCCLDistTest,
           PseudoHermitianLanczos)
{
    using T = TypeParam;

    int* coords = this->get_mpi_grid()->get_coords();

    auto H_ = chase::distMatrix::PseudoHermitianBlockBlockMatrix<
        T, chase::platform::GPU>(this->N, this->N, this->get_mpi_grid());
    H_.allocate_cpu_data();
    H_.readFromBinaryFile(GetBSE_Matrix<T>());
    H_.H2D();

    chase::matrix::Matrix<T> exact_eigsl_H(this->N, 1);
    exact_eigsl_H.readFromBinaryFile(GetBSE_Eigs<T>());

    auto V_ = chase::distMultiVector::DistMultiVector1D<
        T, chase::distMultiVector::CommunicatorType::column,
        chase::platform::GPU>(this->N, this->M, this->get_mpi_grid());
    fillRandomVector(V_, coords[0]);

    chase::Base<T> upperb;

    chase::linalg::internal::hip_rccl::pseudo_hermitian_lanczos(
        this->get_rocblas_handle(), this->M, this->numvec, H_, V_, &upperb,
        this->ritzv.data(), this->Tau.data(), this->ritzV.data());

    for (std::size_t i = 0; i < this->numvec; ++i)
    {
        EXPECT_GT(this->ritzv[i * this->M],
                  std::real(exact_eigsl_H.data()[0]));
        EXPECT_LT(this->ritzv[(i + 1) * this->M - 1],
                  std::real(exact_eigsl_H.data()[this->N - 1]));
    }
}

TYPED_TEST(PseudoHermitianLanczosGPURCCLDistTest,
           tinySimplifiedPseudoHermitianLanczos)
{
    using T = TypeParam;

    int* coords = this->get_mpi_grid()->get_coords();

    auto H_ = chase::distMatrix::PseudoHermitianBlockBlockMatrix<
        T, chase::platform::GPU>(this->N_tiny, this->N_tiny,
                                 this->get_mpi_grid());
    H_.allocate_cpu_data();
    H_.readFromBinaryFile(GetBSE_TinyMatrix<T>());
    H_.H2D();

    chase::matrix::Matrix<T> exact_eigsl_H(this->N_tiny, 1);
    exact_eigsl_H.readFromBinaryFile(GetBSE_TinyEigs<T>());

    auto V_ = chase::distMultiVector::DistMultiVector1D<
        T, chase::distMultiVector::CommunicatorType::column,
        chase::platform::GPU>(this->N_tiny, this->M_tiny, this->get_mpi_grid());
    fillRandomVector(V_, coords[0]);

    chase::Base<T> upperb;

    chase::linalg::internal::hip_rccl::pseudo_hermitian_lanczos(
        this->get_rocblas_handle(), this->M_tiny, H_, V_, &upperb);

    EXPECT_LT(this->ritzv_tiny[this->M_tiny - 1],
              std::real(exact_eigsl_H.data()[this->N_tiny - 1]));
}

TYPED_TEST(PseudoHermitianLanczosGPURCCLDistTest,
           SimplifiedPseudoHermitianLanczos)
{
    using T = TypeParam;

    int* coords = this->get_mpi_grid()->get_coords();

    auto H_ = chase::distMatrix::PseudoHermitianBlockBlockMatrix<
        T, chase::platform::GPU>(this->N, this->N, this->get_mpi_grid());
    H_.allocate_cpu_data();
    H_.readFromBinaryFile(GetBSE_Matrix<T>());
    H_.H2D();

    chase::matrix::Matrix<T> exact_eigsl_H(this->N, 1);
    exact_eigsl_H.readFromBinaryFile(GetBSE_Eigs<T>());

    auto V_ = chase::distMultiVector::DistMultiVector1D<
        T, chase::distMultiVector::CommunicatorType::column,
        chase::platform::GPU>(this->N, this->M, this->get_mpi_grid());
    fillRandomVector(V_, coords[0]);

    chase::Base<T> upperb;

    chase::linalg::internal::hip_rccl::pseudo_hermitian_lanczos(
        this->get_rocblas_handle(), this->M, H_, V_, &upperb);

    EXPECT_LT(this->ritzv[this->M - 1],
              std::real(exact_eigsl_H.data()[this->N - 1]));
}

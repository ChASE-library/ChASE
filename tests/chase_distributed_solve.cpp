// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include <complex>
#include <gtest/gtest.h>
#include <mpi.h>
#include <random>
#include <type_traits>
#include <vector>

#include "algorithm/algorithm.hpp"
#include "grid/mpiGrid2D.hpp"
#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"
#ifdef HAS_CUDA
#include "Impl/pchase_gpu/pchase_gpu.hpp"
#endif
#include "Impl/pchase_cpu/pchase_cpu.hpp"

using namespace chase;

#ifdef HAS_CUDA
using BackendType = chase::grid::backend::NCCL;
#endif

template <typename T>
static chase::Base<T> getResidualToleranceDist()
{
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, std::complex<float>>)
        return 1e-3f;
    else
        return 1e-8;
}

template <typename T, typename ARCH>
void setup_distributed_clement(
    std::size_t N, std::size_t nev, std::size_t nex, std::size_t blocksize,
    Base<T> perturb, int world_size,
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>&
        mpi_grid,
    std::shared_ptr<chase::distMatrix::RedundantMatrix<T, ARCH>>& Clement,
    std::shared_ptr<chase::distMatrix::BlockCyclicMatrix<T, ARCH>>& Hmat,
    std::shared_ptr<chase::distMultiVector::DistMultiVectorBlockCyclic1D<
        T, chase::distMultiVector::CommunicatorType::column, ARCH>>& Vec,
    std::vector<Base<T>>& Lambda, T*& Clement_data)
{
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int dims[2] = {0, 0};
    MPI_Dims_create(world_size, 2, dims);
    mpi_grid = std::make_shared<
        chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(
        dims[0], dims[1], MPI_COMM_WORLD);
    int* grid_dims = mpi_grid->get_dims();
    int* coords = mpi_grid->get_coords();

    Lambda.resize(nev + nex, 0);
    Clement = std::make_shared<chase::distMatrix::RedundantMatrix<T, ARCH>>(
        N, N, mpi_grid);
    Hmat = std::make_shared<chase::distMatrix::BlockCyclicMatrix<T, ARCH>>(
        N, N, blocksize, blocksize, mpi_grid);
    Vec = std::make_shared<chase::distMultiVector::DistMultiVectorBlockCyclic1D<
        T, chase::distMultiVector::CommunicatorType::column, ARCH>>(
        N, nev + nex, blocksize, mpi_grid);

    if constexpr (std::is_same<ARCH, chase::platform::GPU>::value)
    {
        Clement->allocate_cpu_data();
        Hmat->allocate_cpu_data();
        Vec->allocate_cpu_data();
        Clement_data = Clement->cpu_data();
    }
    else
    {
        Clement_data = Clement->l_data();
    }

    for (std::size_t i = 0; i < N; ++i)
    {
        Clement_data[i + N * i] = 0;
        if (i != N - 1)
            Clement_data[i + 1 + N * i] =
                std::sqrt(static_cast<double>(i * (N + 1 - i)));
        if (i != N - 1)
            Clement_data[i + N * (i + 1)] =
                std::sqrt(static_cast<double>(i * (N + 1 - i)));
    }
    std::mt19937 gen(42);
    std::normal_distribution<> d;
    for (std::size_t i = 1; i < N; ++i)
    {
        for (std::size_t j = 1; j < i; ++j)
        {
            T ep;
            if constexpr (std::is_same_v<T, std::complex<float>> ||
                          std::is_same_v<T, std::complex<double>>)
            {
                ep = T(static_cast<chase::Base<T>>(d(gen)),
                       static_cast<chase::Base<T>>(d(gen))) *
                     perturb;
                Clement_data[j + N * i] += ep;
                Clement_data[i + N * j] += std::conj(ep);
            }
            else
            {
                ep = T(static_cast<chase::Base<T>>(d(gen)) * perturb);
                Clement_data[j + N * i] += ep;
                Clement_data[i + N * j] += ep;
            }
        }
    }
}

// Setup for BlockBlockMatrix + DistMultiVector1D (no blocksize).
template <typename T, typename ARCH>
void setup_distributed_clement_blockblock(
    std::size_t N, std::size_t nev, std::size_t nex, Base<T> perturb,
    int world_size,
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>&
        mpi_grid,
    std::shared_ptr<chase::distMatrix::RedundantMatrix<T, ARCH>>& Clement,
    std::shared_ptr<chase::distMatrix::BlockBlockMatrix<T, ARCH>>& Hmat,
    std::shared_ptr<chase::distMultiVector::DistMultiVector1D<
        T, chase::distMultiVector::CommunicatorType::column, ARCH>>& Vec,
    std::vector<Base<T>>& Lambda, T*& Clement_data)
{
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int dims[2] = {0, 0};
    MPI_Dims_create(world_size, 2, dims);
    mpi_grid = std::make_shared<
        chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(
        dims[0], dims[1], MPI_COMM_WORLD);
    (void)world_rank;

    Lambda.resize(nev + nex, 0);
    Clement = std::make_shared<chase::distMatrix::RedundantMatrix<T, ARCH>>(
        N, N, mpi_grid);
    Hmat = std::make_shared<chase::distMatrix::BlockBlockMatrix<T, ARCH>>(
        N, N, mpi_grid);
    Vec = std::make_shared<chase::distMultiVector::DistMultiVector1D<
        T, chase::distMultiVector::CommunicatorType::column, ARCH>>(
        N, nev + nex, mpi_grid);

    if constexpr (std::is_same<ARCH, chase::platform::GPU>::value)
    {
        Clement->allocate_cpu_data();
        Hmat->allocate_cpu_data();
        Vec->allocate_cpu_data();
        Clement_data = Clement->cpu_data();
    }
    else
    {
        Clement_data = Clement->l_data();
    }

    for (std::size_t i = 0; i < N; ++i)
    {
        Clement_data[i + N * i] = 0;
        if (i != N - 1)
            Clement_data[i + 1 + N * i] =
                std::sqrt(static_cast<double>(i * (N + 1 - i)));
        if (i != N - 1)
            Clement_data[i + N * (i + 1)] =
                std::sqrt(static_cast<double>(i * (N + 1 - i)));
    }
    std::mt19937 gen(42);
    std::normal_distribution<> d;
    for (std::size_t i = 1; i < N; ++i)
    {
        for (std::size_t j = 1; j < i; ++j)
        {
            T ep;
            if constexpr (std::is_same_v<T, std::complex<float>> ||
                          std::is_same_v<T, std::complex<double>>)
            {
                ep = T(static_cast<chase::Base<T>>(d(gen)),
                       static_cast<chase::Base<T>>(d(gen))) *
                     perturb;
                Clement_data[j + N * i] += ep;
                Clement_data[i + N * j] += std::conj(ep);
            }
            else
            {
                ep = T(static_cast<chase::Base<T>>(d(gen)) * perturb);
                Clement_data[j + N * i] += ep;
                Clement_data[i + N * j] += ep;
            }
        }
    }
}

// Fixture for pChASECPU: CPU matrix/vector types only. Parameterized over scalar type.
template <typename T>
class ChaseDistributedSolveCPUTest : public ::testing::Test
{
public:
    using BlockCyclicMatrixCPU =
        chase::distMatrix::BlockCyclicMatrix<T, chase::platform::CPU>;
    using DistMultiVectorCPU =
        chase::distMultiVector::DistMultiVectorBlockCyclic1D<
            T, chase::distMultiVector::CommunicatorType::column,
            chase::platform::CPU>;

protected:
    void SetUp() override
    {
        MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
        N_ = 256;
        nev_ = 24;
        nex_ = 16;
        blocksize_ = 32;
        perturb_ = static_cast<chase::Base<T>>(1e-6);
        setup_distributed_clement<T, chase::platform::CPU>(
            N_, nev_, nex_, blocksize_, perturb_, world_size_, mpi_grid_,
            Clement_, Hmat_, Vec_, Lambda_, Clement_data_);
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank_);
    }

    int world_rank_;
    int world_size_;
    std::size_t N_;
    std::size_t nev_;
    std::size_t nex_;
    std::size_t blocksize_;
    chase::Base<T> perturb_;
    std::vector<chase::Base<T>> Lambda_;
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>
        mpi_grid_;
    std::shared_ptr<chase::distMatrix::RedundantMatrix<T, chase::platform::CPU>>
        Clement_;
    std::shared_ptr<BlockCyclicMatrixCPU> Hmat_;
    std::shared_ptr<DistMultiVectorCPU> Vec_;
    T* Clement_data_;
};

using DistSolveCPUTestTypes =
    ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(ChaseDistributedSolveCPUTest, DistSolveCPUTestTypes);

TYPED_TEST(ChaseDistributedSolveCPUTest, pChaseCPU_Solve_Clement_Converges)
{
    using T = TypeParam;
    ChaseBase<T>* single =
        new chase::Impl::pChASECPU<typename TestFixture::BlockCyclicMatrixCPU,
                                    typename TestFixture::DistMultiVectorCPU>(
            this->nev_, this->nex_, this->Hmat_.get(), this->Vec_.get(),
            this->Lambda_.data());

    auto& config = single->GetConfig();
    config.SetTol(1e-10);
    config.SetDeg(16);
    config.SetOpt(true);
    config.SetApprox(false);
    config.SetMaxIter(25);

    this->Clement_->redistributeImpl(this->Hmat_.get());

    ASSERT_NO_THROW(chase::Solve(single));

    if (this->world_rank_ == 0)
    {
        chase::Base<T>* resid = single->GetResid();
        ASSERT_NE(resid, nullptr);
        chase::Base<T> tol = getResidualToleranceDist<T>();
        for (std::size_t i = 0; i < std::min(std::size_t(5), this->nev_); ++i)
        {
            EXPECT_TRUE(std::isfinite(this->Lambda_[i]))
                << "Lambda[" << i << "] is not finite";
            EXPECT_TRUE(std::isfinite(resid[i]))
                << "resid[" << i << "] is not finite";
            EXPECT_LT(resid[i], tol)
                << "resid[" << i << "] = " << resid[i] << " should be below " << tol;
        }
    }
    delete single;
}

// Fixture for pChASECPU with BlockBlockMatrix + DistMultiVector1D. Parameterized over scalar type.
template <typename T>
class ChaseDistributedSolveCPUBlockBlockTest : public ::testing::Test
{
public:
    using BlockBlockMatrixCPU =
        chase::distMatrix::BlockBlockMatrix<T, chase::platform::CPU>;
    using DistMultiVector1DCPU =
        chase::distMultiVector::DistMultiVector1D<
            T, chase::distMultiVector::CommunicatorType::column,
            chase::platform::CPU>;

protected:
    void SetUp() override
    {
        MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
        N_ = 256;
        nev_ = 24;
        nex_ = 16;
        perturb_ = static_cast<chase::Base<T>>(1e-6);
        setup_distributed_clement_blockblock<T, chase::platform::CPU>(
            N_, nev_, nex_, perturb_, world_size_, mpi_grid_, Clement_, Hmat_,
            Vec_, Lambda_, Clement_data_);
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank_);
    }

    int world_rank_;
    int world_size_;
    std::size_t N_;
    std::size_t nev_;
    std::size_t nex_;
    chase::Base<T> perturb_;
    std::vector<chase::Base<T>> Lambda_;
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>
        mpi_grid_;
    std::shared_ptr<chase::distMatrix::RedundantMatrix<T, chase::platform::CPU>>
        Clement_;
    std::shared_ptr<BlockBlockMatrixCPU> Hmat_;
    std::shared_ptr<DistMultiVector1DCPU> Vec_;
    T* Clement_data_;
};

TYPED_TEST_SUITE(ChaseDistributedSolveCPUBlockBlockTest, DistSolveCPUTestTypes);

TYPED_TEST(ChaseDistributedSolveCPUBlockBlockTest, pChaseCPU_Solve_Clement_BlockBlock_Converges)
{
    using T = TypeParam;
    ChaseBase<T>* single =
        new chase::Impl::pChASECPU<typename TestFixture::BlockBlockMatrixCPU,
                                    typename TestFixture::DistMultiVector1DCPU>(
            this->nev_, this->nex_, this->Hmat_.get(), this->Vec_.get(),
            this->Lambda_.data());

    auto& config = single->GetConfig();
    config.SetTol(1e-10);
    config.SetDeg(16);
    config.SetOpt(true);
    config.SetApprox(false);
    config.SetMaxIter(25);

    this->Clement_->redistributeImpl(this->Hmat_.get());

    ASSERT_NO_THROW(chase::Solve(single));

    if (this->world_rank_ == 0)
    {
        chase::Base<T>* resid = single->GetResid();
        ASSERT_NE(resid, nullptr);
        chase::Base<T> tol = getResidualToleranceDist<T>();
        for (std::size_t i = 0; i < std::min(std::size_t(5), this->nev_); ++i)
        {
            EXPECT_TRUE(std::isfinite(this->Lambda_[i]))
                << "Lambda[" << i << "] is not finite";
            EXPECT_TRUE(std::isfinite(resid[i]))
                << "resid[" << i << "] is not finite";
            EXPECT_LT(resid[i], tol)
                << "resid[" << i << "] = " << resid[i] << " should be below " << tol;
        }
    }
    delete single;
}

#ifdef HAS_CUDA
// Fixture for pChASEGPU: GPU matrix/vector types only. Parameterized over scalar type.
template <typename T>
class ChaseDistributedSolveGPUTest : public ::testing::Test
{
public:
    using BlockCyclicMatrixGPU =
        chase::distMatrix::BlockCyclicMatrix<T, chase::platform::GPU>;
    using DistMultiVectorGPU =
        chase::distMultiVector::DistMultiVectorBlockCyclic1D<
            T, chase::distMultiVector::CommunicatorType::column,
            chase::platform::GPU>;

protected:
    void SetUp() override
    {
        MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
        N_ = 256;
        nev_ = 24;
        nex_ = 16;
        blocksize_ = 32;
        perturb_ = static_cast<chase::Base<T>>(1e-6);
        setup_distributed_clement<T, chase::platform::GPU>(
            N_, nev_, nex_, blocksize_, perturb_, world_size_, mpi_grid_,
            Clement_, Hmat_, Vec_, Lambda_, Clement_data_);
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank_);
    }

    int world_rank_;
    int world_size_;
    std::size_t N_;
    std::size_t nev_;
    std::size_t nex_;
    std::size_t blocksize_;
    chase::Base<T> perturb_;
    std::vector<chase::Base<T>> Lambda_;
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>
        mpi_grid_;
    std::shared_ptr<chase::distMatrix::RedundantMatrix<T, chase::platform::GPU>>
        Clement_;
    std::shared_ptr<BlockCyclicMatrixGPU> Hmat_;
    std::shared_ptr<DistMultiVectorGPU> Vec_;
    T* Clement_data_;
};

using DistSolveGPUTestTypes =
    ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(ChaseDistributedSolveGPUTest, DistSolveGPUTestTypes);

TYPED_TEST(ChaseDistributedSolveGPUTest, pChaseGPU_Solve_Clement_Converges)
{
    using T = TypeParam;
    ChaseBase<T>* single =
        new chase::Impl::pChASEGPU<typename TestFixture::BlockCyclicMatrixGPU,
                                    typename TestFixture::DistMultiVectorGPU,
                                    BackendType>(
            this->nev_, this->nex_, this->Hmat_.get(), this->Vec_.get(),
            this->Lambda_.data());

    auto& config = single->GetConfig();
    config.SetTol(1e-10);
    config.SetDeg(16);
    config.SetOpt(true);
    config.SetApprox(false);
    config.SetMaxIter(25);

    this->Clement_->redistributeImpl(this->Hmat_.get());

    ASSERT_NO_THROW(chase::Solve(single));

    if (this->world_rank_ == 0)
    {
        chase::Base<T>* resid = single->GetResid();
        ASSERT_NE(resid, nullptr);
        chase::Base<T> tol = getResidualToleranceDist<T>();
        for (std::size_t i = 0; i < std::min(std::size_t(5), this->nev_); ++i)
        {
            EXPECT_TRUE(std::isfinite(this->Lambda_[i]))
                << "Lambda[" << i << "] is not finite";
            EXPECT_TRUE(std::isfinite(resid[i]))
                << "resid[" << i << "] is not finite";
            EXPECT_LT(resid[i], tol)
                << "resid[" << i << "] = " << resid[i] << " should be below " << tol;
        }
    }
    delete single;
}

// Fixture for pChASEGPU with BlockBlockMatrix + DistMultiVector1D. Parameterized over scalar type.
template <typename T>
class ChaseDistributedSolveGPUBlockBlockTest : public ::testing::Test
{
public:
    using BlockBlockMatrixGPU =
        chase::distMatrix::BlockBlockMatrix<T, chase::platform::GPU>;
    using DistMultiVector1DGPU =
        chase::distMultiVector::DistMultiVector1D<
            T, chase::distMultiVector::CommunicatorType::column,
            chase::platform::GPU>;

protected:
    void SetUp() override
    {
        MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
        N_ = 256;
        nev_ = 24;
        nex_ = 16;
        perturb_ = static_cast<chase::Base<T>>(1e-6);
        setup_distributed_clement_blockblock<T, chase::platform::GPU>(
            N_, nev_, nex_, perturb_, world_size_, mpi_grid_, Clement_, Hmat_,
            Vec_, Lambda_, Clement_data_);
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank_);
    }

    int world_rank_;
    int world_size_;
    std::size_t N_;
    std::size_t nev_;
    std::size_t nex_;
    chase::Base<T> perturb_;
    std::vector<chase::Base<T>> Lambda_;
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>
        mpi_grid_;
    std::shared_ptr<chase::distMatrix::RedundantMatrix<T, chase::platform::GPU>>
        Clement_;
    std::shared_ptr<BlockBlockMatrixGPU> Hmat_;
    std::shared_ptr<DistMultiVector1DGPU> Vec_;
    T* Clement_data_;
};

TYPED_TEST_SUITE(ChaseDistributedSolveGPUBlockBlockTest, DistSolveGPUTestTypes);

TYPED_TEST(ChaseDistributedSolveGPUBlockBlockTest, pChaseGPU_Solve_Clement_BlockBlock_Converges)
{
    using T = TypeParam;
    ChaseBase<T>* single =
        new chase::Impl::pChASEGPU<typename TestFixture::BlockBlockMatrixGPU,
                                    typename TestFixture::DistMultiVector1DGPU,
                                    BackendType>(
            this->nev_, this->nex_, this->Hmat_.get(), this->Vec_.get(),
            this->Lambda_.data());

    auto& config = single->GetConfig();
    config.SetTol(1e-10);
    config.SetDeg(16);
    config.SetOpt(true);
    config.SetApprox(false);
    config.SetMaxIter(25);

    this->Clement_->redistributeImpl(this->Hmat_.get());

    ASSERT_NO_THROW(chase::Solve(single));

    if (this->world_rank_ == 0)
    {
        chase::Base<T>* resid = single->GetResid();
        ASSERT_NE(resid, nullptr);
        chase::Base<T> tol = getResidualToleranceDist<T>();
        for (std::size_t i = 0; i < std::min(std::size_t(5), this->nev_); ++i)
        {
            EXPECT_TRUE(std::isfinite(this->Lambda_[i]))
                << "Lambda[" << i << "] is not finite";
            EXPECT_TRUE(std::isfinite(resid[i]))
                << "resid[" << i << "] is not finite";
            EXPECT_LT(resid[i], tol)
                << "resid[" << i << "] = " << resid[i] << " should be below " << tol;
        }
    }
    
    delete single;
}
#endif

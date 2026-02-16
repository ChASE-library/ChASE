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
#ifdef HAS_CUDA
#include "Impl/chase_gpu/chase_gpu.hpp"
#endif
#include "Impl/chase_cpu/chase_cpu.hpp"

using namespace chase;

template <typename T>
static chase::Base<T> getResidualTolerance()
{
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, std::complex<float>>)
        return 1e-3f;
    else
        return 1e-8;
}

// Unit test based on examples/noinput.cpp: single-node Chase (ChaseCPU / ChaseGPU)
// with in-memory Clement matrix and chase::Solve. Runs under MPI with 1 rank.
// Parameterized over float, double, std::complex<float>, std::complex<double>.
template <typename T>
class ChaseSerialSolveTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank != 0)
            return;

        N_ = 256;
        LDH_ = N_;
        nev_ = 24;
        nex_ = 16;
        perturb_ = static_cast<chase::Base<T>>(1e-6);

        V_.resize(N_ * (nev_ + nex_), T(0));
        Lambda_.resize(nev_ + nex_, 0);
        H_.resize(N_ * LDH_, T(0));

        std::mt19937 gen(42);
        std::normal_distribution<> d;

        for (std::size_t i = 0; i < N_; ++i)
        {
            H_[i + N_ * i] = T(0);
            if (i != N_ - 1)
                H_[i + 1 + LDH_ * i] =
                    static_cast<T>(std::sqrt(static_cast<double>(i * (N_ + 1 - i))));
            if (i != N_ - 1)
                H_[i + LDH_ * (i + 1)] =
                    static_cast<T>(std::sqrt(static_cast<double>(i * (N_ + 1 - i))));
        }
        for (std::size_t i = 1; i < N_; ++i)
        {
            for (std::size_t j = 1; j < i; ++j)
            {
                T ep;
                if constexpr (std::is_same_v<T, std::complex<float>> ||
                              std::is_same_v<T, std::complex<double>>)
                {
                    ep = T(static_cast<chase::Base<T>>(d(gen)),
                           static_cast<chase::Base<T>>(d(gen))) *
                         perturb_;
                    H_[j + LDH_ * i] += ep;
                    H_[i + LDH_ * j] += std::conj(ep);
                }
                else
                {
                    ep = T(static_cast<chase::Base<T>>(d(gen)) * perturb_);
                    H_[j + LDH_ * i] += ep;
                    H_[i + LDH_ * j] += ep;
                }
            }
        }
    }

    std::size_t N_;
    std::size_t LDH_;
    std::size_t nev_;
    std::size_t nex_;
    chase::Base<T> perturb_;
    std::vector<T> V_;
    std::vector<chase::Base<T>> Lambda_;
    std::vector<T> H_;
};

using SerialSolveTestTypes =
    ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(ChaseSerialSolveTest, SerialSolveTestTypes);

TYPED_TEST(ChaseSerialSolveTest, ChaseCPU_Solve_Clement_Converges)
{
    using T = TypeParam;
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank != 0)
        return;

    chase::ChaseBase<T>* single = nullptr;
    chase::Impl::ChASECPU<T> cpu_single(
        static_cast<std::size_t>(this->N_), this->nev_, this->nex_,
        this->H_.data(), this->LDH_, this->V_.data(), this->N_,
        this->Lambda_.data());
    single = &cpu_single;

    auto& config = single->GetConfig();
    config.SetTol(1e-10);
    config.SetDeg(16);
    config.SetOpt(true);
    config.SetApprox(false);
    config.SetMaxIter(25);

    ASSERT_NO_THROW(chase::Solve(single));

    chase::Base<T>* resid = single->GetResid();
    ASSERT_NE(resid, nullptr);

    chase::Base<T> tol = getResidualTolerance<T>();
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

#ifdef HAS_CUDA
TYPED_TEST(ChaseSerialSolveTest, ChaseGPU_Solve_Clement_Converges)
{
    using T = TypeParam;
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank != 0)
        return;

    chase::ChaseBase<T>* single = nullptr;
    chase::Impl::ChASEGPU<T> gpu_single(
        static_cast<std::size_t>(this->N_), this->nev_, this->nex_,
        this->H_.data(), this->LDH_, this->V_.data(), this->N_,
        this->Lambda_.data());
    single = &gpu_single;

    auto& config = single->GetConfig();
    config.SetTol(1e-10);
    config.SetDeg(16);
    config.SetOpt(true);
    config.SetApprox(false);
    config.SetMaxIter(25);

    ASSERT_NO_THROW(chase::Solve(single));

    chase::Base<T>* resid = single->GetResid();
    ASSERT_NE(resid, nullptr);

    chase::Base<T> tol = getResidualTolerance<T>();
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
#endif

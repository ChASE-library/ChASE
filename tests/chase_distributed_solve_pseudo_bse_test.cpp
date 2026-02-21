// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include <algorithm>
#include <complex>
#include <fstream>
#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mpi.h>
#include <string>
#include <vector>

#include "algorithm/algorithm.hpp"
#include "grid/mpiGrid2D.hpp"
#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"
#include "linalg/matrix/matrix.hpp"
#include "Impl/pchase_cpu/pchase_cpu.hpp"

using namespace chase;

// Scalar type for BSE (same as serial test).
using T_PseudoBSE = std::complex<double>;

// Resolve path to a file in BSE_matrices (build dir or source dir).
static std::string get_BSE_path(const char* filename)
{
#ifdef CHASE_TEST_BSE_DIR
    std::string p = CHASE_TEST_BSE_DIR;
    if (!p.empty() && p.back() != '/')
        p += '/';
    p += filename;
    std::ifstream f(p, std::ios::binary);
    if (f.good())
        return p;
#endif
    const char* candidates[] = {
        "tests/linalg/internal/BSE_matrices/",
        "../linalg/internal/BSE_matrices/",
    };
    for (const char* prefix : candidates)
    {
        std::string p = std::string(prefix) + filename;
        std::ifstream f(p, std::ios::binary);
        if (f.good())
            return p;
    }
    return std::string(candidates[0]) + filename;
}

// Build sorted H² eigenvalues from direct-solver H eigenvalues (λ → λ², ascending).
static void sorted_H2_from_eigs(const T_PseudoBSE* eigs_H, std::size_t n,
                                 std::vector<chase::Base<T_PseudoBSE>>& eig_H2_sorted)
{
    eig_H2_sorted.resize(n);
    for (std::size_t i = 0; i < n; ++i)
        eig_H2_sorted[i] = std::real(eigs_H[i]) * std::real(eigs_H[i]);
    std::sort(eig_H2_sorted.begin(), eig_H2_sorted.end());
}

// nev smallest positive eigenvalues from full H spectrum (for reference).
static void nev_smallest_positive_eigs(const T_PseudoBSE* eigs_H, std::size_t n,
                                       std::size_t nev,
                                       std::vector<chase::Base<T_PseudoBSE>>& out)
{
    std::vector<chase::Base<T_PseudoBSE>> pos;
    for (std::size_t i = 0; i < n; ++i)
    {
        chase::Base<T_PseudoBSE> r = std::real(eigs_H[i]);
        if (r > chase::Base<T_PseudoBSE>(0))
            pos.push_back(r);
    }
    std::sort(pos.begin(), pos.end());
    out.clear();
    for (std::size_t i = 0; i < nev && i < pos.size(); ++i)
        out.push_back(pos[i]);
}

// pChASECPU Solve_pseudo with same BSE matrix as serial test (cdouble_random_BSE.bin).
// Eigenvalues (Lambda/ritzv) and residuals are in RedundantMatrix → valid on all ranks.
// Eigenvectors are in DistMultiVectorBlockCyclic1D → distributed by rows; only rank 0
// does printing and reference comparison.
class ChaseDistributedSolvePseudoBSECPUTest : public ::testing::Test
{
public:
    using PseudoHMatrixCPU =
        chase::distMatrix::PseudoHermitianBlockCyclicMatrix<T_PseudoBSE,
                                                            chase::platform::CPU>;
    using DistMultiVectorCPU =
        chase::distMultiVector::DistMultiVectorBlockCyclic1D<
            T_PseudoBSE, chase::distMultiVector::CommunicatorType::column,
            chase::platform::CPU>;

protected:
    void SetUp() override
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size_);

        N_ = 200;
        nev_ = 20;
        nex_ = 20;
        nevex_ = nev_ + nex_;
        blocksize_ = 32;

        int dims[2] = {0, 0};
        MPI_Dims_create(world_size_, 2, dims);
        mpi_grid_ = std::make_shared<
            chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(
            dims[0], dims[1], MPI_COMM_WORLD);

        matrix_path_ = get_BSE_path("cdouble_random_BSE.bin");
        int file_ok = 0;
        if (world_rank_ == 0)
        {
            std::ifstream f(matrix_path_, std::ios::binary);
            file_ok = f.good() ? 1 : 0;
        }
        MPI_Bcast(&file_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (file_ok == 0)
            GTEST_SKIP() << "BSE matrix not found: " << matrix_path_;

        Lambda_.resize(2 * nevex_, chase::Base<T_PseudoBSE>(0));
        Hmat_ = std::make_shared<PseudoHMatrixCPU>(N_, N_, blocksize_, blocksize_,
                                                   mpi_grid_);
        Vec_ = std::make_shared<DistMultiVectorCPU>(
            N_, 2 * nevex_, blocksize_, mpi_grid_);
        Hmat_->readFromBinaryFile(matrix_path_);
    }

    int world_rank_;
    int world_size_;
    std::size_t N_;
    std::size_t nev_;
    std::size_t nex_;
    std::size_t nevex_;
    std::size_t blocksize_;
    std::string matrix_path_;
    std::vector<chase::Base<T_PseudoBSE>> Lambda_;
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>
        mpi_grid_;
    std::shared_ptr<PseudoHMatrixCPU> Hmat_;
    std::shared_ptr<DistMultiVectorCPU> Vec_;
};

TEST_F(ChaseDistributedSolvePseudoBSECPUTest, pChaseCPU_SolvePseudo_BSE_Converges)
{
    ChaseBase<T_PseudoBSE>* single =
        new chase::Impl::pChASECPU<PseudoHMatrixCPU, DistMultiVectorCPU>(
            nev_, nex_, Hmat_.get(), Vec_.get(), Lambda_.data());

    auto& config = single->GetConfig();
    config.SetTol(1e-10);
    config.SetDeg(20);
    config.SetOpt(true);
    config.SetApprox(false);
    config.SetMaxIter(25);
    config.SetNumLanczos(40);
    config.SetLanczosIter(50);

    if (world_rank_ == 0)
        std::cout << "[Solve_pseudo BSE] n=" << N_ << " nev=" << nev_
                  << " nex=" << nex_ << " nevex=" << nevex_ << std::endl;

    if (world_rank_ == 0)
        std::cout << "[Solve_pseudo BSE] calling Solve_pseudo..." << std::endl;
    ASSERT_NO_THROW(chase::Solve_pseudo(single));
    if (world_rank_ == 0)
        std::cout << "[Solve_pseudo BSE] Solve_pseudo returned." << std::endl;

    // Ritz values and residuals are in RedundantMatrix → same on all ranks.
    std::size_t nev_out = single->GetNev();
    chase::Base<T_PseudoBSE>* ritzv = single->GetRitzv();
    chase::Base<T_PseudoBSE>* resid = single->GetResid();
    ASSERT_NE(resid, nullptr);

    if (world_rank_ == 0)
    {
        // H² bounds from direct-solver eigenvalues (reference)
        std::string eigs_path = get_BSE_path("eigs_cdouble_random_BSE.bin");
        std::ifstream eigs_probe(eigs_path, std::ios::binary);
        if (eigs_probe.good())
        {
            eigs_probe.close();
            chase::matrix::Matrix<T_PseudoBSE> eigs_H(N_, 1);
            eigs_H.readFromBinaryFile(eigs_path);
            std::vector<chase::Base<T_PseudoBSE>> eig_H2_sorted;
            sorted_H2_from_eigs(eigs_H.data(), N_, eig_H2_sorted);

            chase::Base<T_PseudoBSE> lambda_1_exact = eig_H2_sorted[0];
            chase::Base<T_PseudoBSE> b_sup_exact = eig_H2_sorted[N_ - 1];
            std::size_t idx_nevnex = 2 * nevex_ - 1;
            chase::Base<T_PseudoBSE> mu_nevnex_exact =
                (idx_nevnex < N_) ? eig_H2_sorted[idx_nevnex] : eig_H2_sorted[N_ - 1];

            std::cout << "[Solve_pseudo BSE] H² bounds from direct solver: lambda_1 = "
                      << std::scientific << std::setprecision(6) << lambda_1_exact
                      << "  lower(mu_nevnex) = " << std::setprecision(6)
                      << mu_nevnex_exact << "  b_sup = " << std::setprecision(6)
                      << b_sup_exact << std::endl;
        }
        else
        {
            std::cout << "[Solve_pseudo BSE] eigs file not found, skipping exact bounds."
                      << std::endl;
        }

        std::cout << "[Solve_pseudo BSE] Requested positive eigenvalues: nev = "
                  << nev_out << " (buffer size = " << single->GetRitzvBlockSize()
                  << "). First nev columns of V = eigenvectors (distributed by rows).\n";

        std::cout << "[Solve_pseudo BSE] First nev from Lambda_.data() (user ritzv buffer):\n";
        for (std::size_t i = 0; i < nev_out; ++i)
            std::cout << "  lambda[" << i << "] = " << std::fixed << std::setprecision(8)
                      << Lambda_.data()[i] << "\n";
        std::cout << "[Solve_pseudo BSE] First nev from GetResid():\n";
        for (std::size_t i = 0; i < nev_out; ++i)
            std::cout << "  residual[" << i << "] = " << std::scientific << std::setprecision(6)
                      << resid[i] << "\n";

        std::vector<chase::Base<T_PseudoBSE>> ref_pos;
        std::string eigs_path_ref = get_BSE_path("eigs_cdouble_random_BSE.bin");
        std::ifstream eigs_ref(eigs_path_ref, std::ios::binary);
        if (eigs_ref.good())
        {
            eigs_ref.close();
            chase::matrix::Matrix<T_PseudoBSE> eigs_H_ref(N_, 1);
            eigs_H_ref.readFromBinaryFile(eigs_path_ref);
            nev_smallest_positive_eigs(eigs_H_ref.data(), N_, nev_, ref_pos);
        }

        std::cout << "[Solve_pseudo BSE] Computed vs reference (nev smallest positive):\n"
                  << "    i     lambda_computed   lambda_ref        residual      |err|\n";
        std::cout << std::fixed << std::setprecision(8);
        for (std::size_t i = 0; i < nev_out; ++i)
        {
            chase::Base<T_PseudoBSE> lam = ritzv[i];
            chase::Base<T_PseudoBSE> res = resid[i];
            chase::Base<T_PseudoBSE> ref = (i < ref_pos.size()) ? ref_pos[i] : chase::Base<T_PseudoBSE>(-1);
            chase::Base<T_PseudoBSE> err = (i < ref_pos.size())
                                              ? std::abs(lam - ref)
                                              : chase::Base<T_PseudoBSE>(-1);
            std::cout << "  " << std::setw(3) << i << "  " << std::setw(16) << lam
                      << "  " << std::setw(16)
                      << (ref >= 0 ? ref : std::numeric_limits<chase::Base<T_PseudoBSE>>::quiet_NaN())
                      << "  " << std::scientific << std::setw(12) << res << "  "
                      << std::scientific
                      << (err >= 0 ? err : std::numeric_limits<chase::Base<T_PseudoBSE>>::quiet_NaN())
                      << "\n";
        }

        std::cout << "[Solve_pseudo BSE] Residuals from GetResid() (eigenvectors are "
                     "distributed; no recomputation from V here).\n"
                  << "    i   residual   status\n";
    }

    const chase::Base<T_PseudoBSE> tol = config.GetTol();
    for (std::size_t i = 0; i < nev_out; ++i)
    {
        EXPECT_TRUE(std::isfinite(Lambda_[i]))
            << "Lambda[" << i << "] is not finite";
        EXPECT_TRUE(std::isfinite(resid[i]))
            << "resid[" << i << "] is not finite";
        EXPECT_LE(resid[i], tol)
            << "resid[" << i << "] = " << resid[i] << " should be <= " << tol;
    }

    if (world_rank_ == 0)
    {
        for (std::size_t i = 0; i < nev_out; ++i)
        {
            bool ok = (resid[i] <= tol);
            std::cout << "  " << std::setw(3) << i << "  " << std::scientific
                      << std::setprecision(6) << resid[i] << "  " << (ok ? "ok" : "FAIL") << "\n";
        }
    }

    delete single;
}

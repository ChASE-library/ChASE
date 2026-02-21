// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include "algorithm/algorithm.hpp"
#include "Impl/chase_cpu/chase_cpu.hpp"
#include "linalg/matrix/matrix.hpp"
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

using namespace chase;

using T = std::complex<double>;

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
static void sorted_H2_from_eigs(const T* eigs_H, std::size_t n,
                                 std::vector<chase::Base<T>>& eig_H2_sorted)
{
    eig_H2_sorted.resize(n);
    for (std::size_t i = 0; i < n; ++i)
        eig_H2_sorted[i] = std::real(eigs_H[i]) * std::real(eigs_H[i]);
    std::sort(eig_H2_sorted.begin(), eig_H2_sorted.end());
}

// nev smallest positive eigenvalues from full H spectrum (for reference).
static void nev_smallest_positive_eigs(const T* eigs_H, std::size_t n,
                                       std::size_t nev,
                                       std::vector<chase::Base<T>>& out)
{
    std::vector<chase::Base<T>> pos;
    for (std::size_t i = 0; i < n; ++i)
    {
        chase::Base<T> r = std::real(eigs_H[i]);
        if (r > chase::Base<T>(0))
            pos.push_back(r);
    }
    std::sort(pos.begin(), pos.end());
    out.clear();
    for (std::size_t i = 0; i < nev && i < pos.size(); ++i)
        out.push_back(pos[i]);
}

// Unit test: Solve_pseudo with BSE matrix from CHASE_TEST_BSE_DIR.
// No expectations yet; use printing to verify setup step by step as
// solve_pseudo is implemented.
class ChaseSerialSolvePseudoBSETest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank != 0)
            return;

        n_ = 200;  // BSE matrix size (cdouble_random_BSE.bin)
        nev_ = 20;
        nex_ = 20;
        nevex_ = nev_ + nex_;

        // Pseudo-Hermitian subspace size is 2*nevex
        V_.resize(n_ * (2 * nevex_), T(0));
        Lambda_.resize(2 * nevex_, chase::Base<T>(0));

        Hmat_ = std::make_unique<chase::matrix::PseudoHermitianMatrix<T>>(n_, n_);
        std::string matrix_path = get_BSE_path("cdouble_random_BSE.bin");
        std::ifstream probe(matrix_path, std::ios::binary);
        if (!probe.good())
        {
            GTEST_SKIP() << "BSE matrix not found: " << matrix_path;
        }
        probe.close();
        Hmat_->readFromBinaryFile(matrix_path);
    }

    std::size_t n_;
    std::size_t nev_;
    std::size_t nex_;
    std::size_t nevex_;
    std::vector<T> V_;
    std::vector<chase::Base<T>> Lambda_;
    std::unique_ptr<chase::matrix::PseudoHermitianMatrix<T>> Hmat_;
};

TEST_F(ChaseSerialSolvePseudoBSETest, SolvePseudo_BSE_Matrix)
{
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank != 0)
        return;

    std::cout << "[Solve_pseudo BSE] n=" << n_ << " nev=" << nev_
              << " nex=" << nex_ << " nevex=" << nevex_ << std::endl;

    chase::ChaseBase<T>* single = nullptr;
    chase::Impl::ChASECPU<T, chase::matrix::PseudoHermitianMatrix<T>> cpu_single(
        n_, nev_, nex_, Hmat_.get(), V_.data(), n_, Lambda_.data());
    single = &cpu_single;

    auto& config = single->GetConfig();
    config.SetTol(1e-10);
    config.SetDeg(20);
    config.SetOpt(true);
    config.SetApprox(false);
    config.SetMaxIter(25);
    config.SetNumLanczos(40);
    config.SetLanczosIter(50);

    std::cout << "[Solve_pseudo BSE] calling Solve_pseudo..." << std::endl;
    ASSERT_NO_THROW(chase::Solve_pseudo(single));
    std::cout << "[Solve_pseudo BSE] Solve_pseudo returned." << std::endl;

    // Print three H² bounds from direct-solver eigenvalues (reference)
    std::string eigs_path = get_BSE_path("eigs_cdouble_random_BSE.bin");
    std::ifstream eigs_probe(eigs_path, std::ios::binary);
    if (eigs_probe.good())
    {
        eigs_probe.close();
        chase::matrix::Matrix<T> eigs_H(n_, 1);
        eigs_H.readFromBinaryFile(eigs_path);
        std::vector<chase::Base<T>> eig_H2_sorted;
        sorted_H2_from_eigs(eigs_H.data(), n_, eig_H2_sorted);

        chase::Base<T> lambda_1_exact = eig_H2_sorted[0];
        chase::Base<T> b_sup_exact = eig_H2_sorted[n_ - 1];
        std::size_t idx_nevnex = 2 * nevex_ - 1;
        chase::Base<T> mu_nevnex_exact =
            (idx_nevnex < n_) ? eig_H2_sorted[idx_nevnex] : eig_H2_sorted[n_ - 1];

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

    // After solve_pseudo(): first GetNev() Ritz values = smallest nev positive eigenvalues;
    // first GetNev() columns of V = corresponding eigenvectors (same permutation applied).
    std::size_t nev_out = single->GetNev();
    chase::Base<T>* ritzv = single->GetRitzv();
    chase::Base<T>* resid = single->GetResid();
    std::cout << "[Solve_pseudo BSE] Requested positive eigenvalues: nev = "
              << nev_out << " (buffer size = " << single->GetRitzvBlockSize()
              << "). First nev columns of V = eigenvectors for these eigenvalues.\n";

    // First nev from user buffer (Lambda_.data()) and from residuals (copy to vector for .data()).
    std::cout << "[Solve_pseudo BSE] First nev from Lambda_.data() (user ritzv buffer):\n";
    for (std::size_t i = 0; i < nev_out; ++i)
        std::cout << "  lambda[" << i << "] = " << std::fixed << std::setprecision(8)
                  << Lambda_.data()[i] << "\n";
    std::vector<chase::Base<T>> residuals(nev_out);
    for (std::size_t i = 0; i < nev_out; ++i)
        residuals[i] = resid[i];
    std::cout << "[Solve_pseudo BSE] First nev from residuals.data() (from GetResid()):\n";
    for (std::size_t i = 0; i < nev_out; ++i)
        std::cout << "  residual[" << i << "] = " << std::scientific << std::setprecision(6)
                  << residuals.data()[i] << "\n";

    std::vector<chase::Base<T>> ref_pos;
    std::string eigs_path_ref = get_BSE_path("eigs_cdouble_random_BSE.bin");
    std::ifstream eigs_ref(eigs_path_ref, std::ios::binary);
    if (eigs_ref.good())
    {
        eigs_ref.close();
        chase::matrix::Matrix<T> eigs_H_ref(n_, 1);
        eigs_H_ref.readFromBinaryFile(eigs_path_ref);
        nev_smallest_positive_eigs(eigs_H_ref.data(), n_, nev_, ref_pos);
    }

    std::cout << "[Solve_pseudo BSE] Computed vs reference (nev smallest positive):\n"
              << "    i     lambda_computed   lambda_ref        residual      |err|\n";
    std::cout << std::fixed << std::setprecision(8);
    for (std::size_t i = 0; i < nev_out; ++i)
    {
        chase::Base<T> lam = ritzv[i];
        chase::Base<T> res = resid[i];
        chase::Base<T> ref = (i < ref_pos.size()) ? ref_pos[i] : chase::Base<T>(-1);
        chase::Base<T> err = (i < ref_pos.size())
                                 ? std::abs(lam - ref)
                                 : chase::Base<T>(-1);
        std::cout << "  " << std::setw(3) << i << "  " << std::setw(16) << lam
                  << "  " << std::setw(16)
                  << (ref >= 0 ? ref : std::numeric_limits<chase::Base<T>>::quiet_NaN())
                  << "  " << std::scientific << std::setw(12) << res << "  "
                  << std::scientific
                  << (err >= 0 ? err : std::numeric_limits<chase::Base<T>>::quiet_NaN())
                  << "\n";
    }

    // Recompute residuals from ritzv and V pointers: ||H*v_j - lambda_j*v_j||
    std::vector<chase::Base<T>> resids_recomputed(nev_out);
    const T* H = Hmat_->data();
    const std::size_t ldh = Hmat_->ld();
    for (std::size_t j = 0; j < nev_out; ++j)
    {
        const T* v = V_.data() + j * n_;
        const chase::Base<T> lam = Lambda_.data()[j];
        std::vector<T> w(n_, T(0));
        for (std::size_t i = 0; i < n_; ++i)
            for (std::size_t k = 0; k < n_; ++k)
                w[i] += H[i + k * ldh] * v[k];
        for (std::size_t i = 0; i < n_; ++i)
            w[i] -= lam * v[i];
        chase::Base<T> nrm = 0;
        for (std::size_t i = 0; i < n_; ++i)
            nrm += std::norm(w[i]);
        resids_recomputed[j] = std::sqrt(nrm);
    }
    const chase::Base<T> tol = config.GetTol();
    std::cout << "[Solve_pseudo BSE] Recomputed residuals (||H*v - lambda*v||) from "
                 "Lambda_.data() and V_.data():\n"
              << "    i   residual_recomputed   residual_from_GetResid()   status\n";
    for (std::size_t i = 0; i < nev_out; ++i)
    {
        chase::Base<T> r_re = resids_recomputed[i];
        chase::Base<T> r_get = resid[i];
        bool ok = (r_re <= tol);
        std::cout << "  " << std::setw(3) << i << "  " << std::scientific
                  << std::setprecision(6) << std::setw(18) << r_re << "  "
                  << std::setw(18) << r_get << "  " << (ok ? "ok" : "FAIL") << "\n";
        EXPECT_LE(r_re, tol) << "Recomputed residual[" << i << "] = " << r_re
                             << " > tol = " << tol;
    }
}

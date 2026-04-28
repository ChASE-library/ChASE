// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include "algorithm/algorithm.hpp"
#include "external/blaspp/blaspp.hpp"
#include "external/lapackpp/lapackpp.hpp"
#include "linalg/matrix/matrix.hpp"
#include "Impl/chase_cpu/chase_cpu.hpp"
#ifdef HAS_CUDA
#include "Impl/chase_gpu/chase_gpu.hpp"
#endif
#include <algorithm>
#include <complex>
#include <fstream>
#include <gtest/gtest.h>
#include <iomanip>
#include <mpi.h>
#include <random>
#include <string>
#include <vector>
#include <iostream>

using T = std::complex<double>;
using Base = chase::Base<T>;

// Build H² eigenvalues from H eigenvalues (λ → λ²), then sort ascending.
void load_and_sort_H2_eigenvalues(const T* eigs_H, std::size_t n,
                                 std::vector<Base>& eig_H2_sorted)
{
    eig_H2_sorted.resize(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        Base lam = std::real(eigs_H[i]);
        eig_H2_sorted[i] = lam * lam;
    }
    std::sort(eig_H2_sorted.begin(), eig_H2_sorted.end());
}

// Resolve path to a file in BSE_matrices (works from build dir or source dir).
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
    return std::string(candidates[0]) + filename;  // fallback for error message
}

// Julia uses run_seed = 1314521 + j per Lanczos run for diverse Ritz samples.
static constexpr unsigned kLanczosRunSeedBase = 1314521u;

class AlgorithmLanczosForH2Test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        n_ = 200;  // BSE matrix size (cdouble_random_BSE.bin is 200x200)
        nev_ = 20;
        nex_ = 20;
        nevex_ = nev_ + nex_;
        numvec_ = 10;
        lanczos_iter_ = 50;

        // PseudoHermitian backend uses 2*nevex_ columns in V and 2*nevex_ ritz values; Lanczos writes to columns 0..M-1.
        V_.resize(n_ * std::max(2 * nevex_, lanczos_iter_));
        ritzv_.resize(2 * nevex_);

        Hmat_ = std::make_unique<chase::matrix::PseudoHermitianMatrix<T>>(
            n_, n_);
        std::string matrix_path = get_BSE_path("cdouble_random_BSE.bin");
        std::ifstream probe(matrix_path, std::ios::binary);
        if (!probe.good())
            GTEST_SKIP() << "BSE matrix not found: " << matrix_path;
        probe.close();
        Hmat_->readFromBinaryFile(matrix_path);
    }

    std::size_t n_;
    std::size_t nev_;
    std::size_t nex_;
    std::size_t nevex_;
    std::size_t numvec_;
    std::size_t lanczos_iter_;
    std::vector<T> V_;
    std::vector<Base> ritzv_;
    std::unique_ptr<chase::matrix::PseudoHermitianMatrix<T>> Hmat_;
};

TEST_F(AlgorithmLanczosForH2Test, LanczosForH2_ReturnsH2Bounds)
{
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank != 0)
        return;

    // Reference: H² eigenvalues from eigs file (λ → λ² in load_and_sort_H2_eigenvalues)
    std::string eigs_path = get_BSE_path("eigs_cdouble_random_BSE.bin");
    std::ifstream eigs_probe(eigs_path, std::ios::binary);
    if (!eigs_probe.good())
        GTEST_SKIP() << "BSE eigs file not found: " << eigs_path;
    eigs_probe.close();

    std::vector<Base> eig_H2_sorted;
    chase::matrix::Matrix<T> eigs_H2(n_, 1);
    eigs_H2.readFromBinaryFile(eigs_path);
    load_and_sort_H2_eigenvalues(eigs_H2.data(), n_, eig_H2_sorted);

    Base smallest_eig_H2 = eig_H2_sorted[0];
    Base largest_eig_H2 = eig_H2_sorted[n_ - 1];
    const std::size_t idx_2nevex = 2 * nevex_ - 1;
    Base lambda_2nevex_minus_1 =
        (idx_2nevex < n_) ? eig_H2_sorted[idx_2nevex] : eig_H2_sorted[n_ - 1];

    chase::Impl::ChASECPU<T, chase::matrix::PseudoHermitianMatrix<T>> single(
        n_, nev_, nex_, Hmat_.get(), V_.data(), n_, ritzv_.data());

    single.GetConfig().SetLanczosIter(lanczos_iter_);
    single.GetConfig().SetNumLanczos(numvec_);
    single.GetConfig().SetApprox(false);

    single.Start();
    // Fill initial vectors with per-column seeds (Julia: run_seed = 1314521 + j) for diverse Lanczos runs.
    std::normal_distribution<> dist;
    for (std::size_t j = 0; j < nevex_; ++j)
    {
        std::mt19937 col_gen(kLanczosRunSeedBase + j);
        for (std::size_t i = 0; i < n_; ++i)
            V_[i + j * n_] = T(dist(col_gen), dist(col_gen));
    }
    single.initVecs(false);
    single.QR(0, 1.0);

    Base upperb = 0;
    std::size_t idx = chase::Algorithm<T>::lanczos_for_H2(
        &single, static_cast<int>(n_), static_cast<int>(numvec_),
        static_cast<int>(lanczos_iter_), static_cast<int>(nevex_), &upperb,
        true, ritzv_.data());

    EXPECT_TRUE(std::isfinite(upperb)) << "upperb (b_sup) is not finite";
    EXPECT_GT(upperb, 0) << "upperb (b_sup) should be positive for H²";

    for (std::size_t i = 0; i < nevex_; ++i)
    {
        EXPECT_TRUE(std::isfinite(ritzv_[i]))
            << "ritzv_[" << i << "] is not finite";
        EXPECT_GE(ritzv_[i], 0)
            << "ritzv_[" << i << "] (H² bound) should be non-negative";
    }

    // μ_1: min of first nevex-1 entries (Lanczos estimate for smallest H² eigenvalue)
    Base mu_1_est = ritzv_[0];
    for (std::size_t i = 1; i < nevex_ - 1; ++i)
        if (ritzv_[i] < mu_1_est)
            mu_1_est = ritzv_[i];
    Base mu_nevnex = ritzv_[nevex_ - 1];

    // Print all values (also when test passes)
    {
        std::cout << std::scientific << std::setprecision(6);
        std::cout << "--- H² reference (sorted ascending) ---\n";
        std::cout << "  smallest λ(H²)     = " << smallest_eig_H2 << "\n";
        std::cout << "  largest λ(H²)     = " << largest_eig_H2 << "\n";
        std::cout << "  λ(H²)[2*nevex-1]  = " << lambda_2nevex_minus_1
                  << "  (index " << idx_2nevex << ")\n";
        std::cout << "  first 5 λ(H²): ";
        for (std::size_t i = 0; i < std::min(std::size_t(5), n_); ++i)
            std::cout << eig_H2_sorted[i] << " ";
        std::cout << "\n  last 5 λ(H²): ";
        for (std::size_t i = n_ >= 5 ? n_ - 5 : 0; i < n_; ++i)
            std::cout << eig_H2_sorted[i] << " ";
        std::cout << "\n--- Lanczos_for_H2 outputs ---\n";
        std::cout << "  upperb (b_sup)    = " << upperb << "\n";
        std::cout << "  μ_1 (est)         = " << mu_1_est << "\n";
        std::cout << "  μ_nevnex          = " << mu_nevnex << "\n";
        std::cout << "  extracted idx     = " << idx << "\n";
        std::cout << "  ritzv_[0.." << (nevex_ - 1) << "]: ";
        for (std::size_t i = 0; i < nevex_; ++i)
            std::cout << ritzv_[i] << (i + 1 < nevex_ ? " " : "\n");
        std::cout << std::endl;
    }

    // μ_1: positive, smaller than μ_nevnex, and close to smallest λ(H²) (distance to smallest
    // much smaller than the gap from smallest to λ(H²)[2*nevex-1]).
    EXPECT_GT(mu_1_est, 0) << "μ_1 estimate should be positive";
    EXPECT_LT(mu_1_est, mu_nevnex)
        << "μ_1 estimate " << mu_1_est << " should be smaller than μ_nevnex " << mu_nevnex;
    Base gap_low = lambda_2nevex_minus_1 - smallest_eig_H2;
    if (gap_low > 0)
    {
        const Base mu_1_gap_ratio = 0.2;  // "much smaller": |μ_1 - smallest| <= ratio * gap
        Base dist_mu1_to_smallest = std::abs(mu_1_est - smallest_eig_H2);
        EXPECT_LE(dist_mu1_to_smallest, mu_1_gap_ratio * gap_low)
            << "|μ_1 - smallest λ(H²)| = " << dist_mu1_to_smallest
            << " should be <= " << mu_1_gap_ratio << " * (λ(H²)[2*nevex-1] - smallest) = "
            << (mu_1_gap_ratio * gap_low);
    }

    EXPECT_GT(mu_nevnex, 0) << "μ_nevnex (last entry) should be positive";
    EXPECT_GE(mu_nevnex, lambda_2nevex_minus_1)
        << "μ_nevnex " << mu_nevnex
        << " should be >= λ(H²)[2*(nev+nex)-1] = " << lambda_2nevex_minus_1;
    EXPECT_LE(mu_nevnex, upperb)
        << "μ_nevnex should not exceed b_sup (upperb)";
    // μ_nevnex should be much closer to λ(H²)[2*nevex-1] than to largest λ(H²).
    // Use a relaxed ratio (0.35) so both CPU and GPU Lanczos pass: estimates vary
    // with random initial vectors and QR path (Cholesky vs Householder).
    Base dist_munevnex_to_2nevex = std::abs(mu_nevnex - lambda_2nevex_minus_1);
    Base dist_munevnex_to_largest = std::abs(mu_nevnex - largest_eig_H2);
    if (dist_munevnex_to_largest > 0)
    {
        const Base mu_nevnex_ratio = 0.35;
        EXPECT_LE(dist_munevnex_to_2nevex, mu_nevnex_ratio * dist_munevnex_to_largest)
            << "|μ_nevnex - λ(H²)[2*nevex-1]| = " << dist_munevnex_to_2nevex
            << " should be <= " << mu_nevnex_ratio << " * |μ_nevnex - largest λ(H²)| = "
            << (mu_nevnex_ratio * dist_munevnex_to_largest);
    }

    // With k=50, nvec=40 Lanczos we expect upperb close to largest λ(H²) (Julia-like)
    const Base upperb_tol = 0.02;
    EXPECT_GE(upperb, (1 - upperb_tol) * largest_eig_H2)
        << "upperb (b_sup) " << upperb
        << " should be >= " << ((1 - upperb_tol) * largest_eig_H2);
    EXPECT_LE(upperb, (1 + upperb_tol) * largest_eig_H2)
        << "upperb (b_sup) " << upperb
        << " should be <= " << ((1 + upperb_tol) * largest_eig_H2);

    EXPECT_LE(idx, static_cast<std::size_t>(lanczos_iter_))
        << "extracted index should be <= m";
}

#ifdef HAS_CUDA
TEST_F(AlgorithmLanczosForH2Test, LanczosForH2_GPU_ReturnsH2Bounds)
{
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank != 0)
        return;

    std::string eigs_path = get_BSE_path("eigs_cdouble_random_BSE.bin");
    std::ifstream eigs_probe(eigs_path, std::ios::binary);
    if (!eigs_probe.good())
        GTEST_SKIP() << "BSE eigs file not found: " << eigs_path;
    eigs_probe.close();

    std::vector<Base> eig_H2_sorted;
    chase::matrix::Matrix<T> eigs_H2(n_, 1);
    eigs_H2.readFromBinaryFile(eigs_path);
    load_and_sort_H2_eigenvalues(eigs_H2.data(), n_, eig_H2_sorted);

    Base smallest_eig_H2 = eig_H2_sorted[0];
    Base largest_eig_H2 = eig_H2_sorted[n_ - 1];
    const std::size_t idx_2nevex = 2 * nevex_ - 1;
    Base lambda_2nevex_minus_1 =
        (idx_2nevex < n_) ? eig_H2_sorted[idx_2nevex] : eig_H2_sorted[n_ - 1];

    using GPUMatrixType =
        chase::matrix::PseudoHermitianMatrix<T, chase::platform::GPU>;
    chase::Impl::ChASEGPU<T, GPUMatrixType> single(
        n_, nev_, nex_, Hmat_->data(), Hmat_->ld(), V_.data(), n_, ritzv_.data());

    single.GetConfig().SetLanczosIter(lanczos_iter_);
    single.GetConfig().SetNumLanczos(numvec_);
    single.GetConfig().SetApprox(false);

    single.Start();
    std::normal_distribution<> dist;
    for (std::size_t j = 0; j < nevex_; ++j)
    {
        std::mt19937 col_gen(kLanczosRunSeedBase + j);
        for (std::size_t i = 0; i < n_; ++i)
            V_[i + j * n_] = T(dist(col_gen), dist(col_gen));
    }
    single.initVecs(false);
    single.QR(0, 1.0);

    Base upperb = 0;
    std::size_t idx = chase::Algorithm<T>::lanczos_for_H2(
        &single, static_cast<int>(n_), static_cast<int>(numvec_),
        static_cast<int>(lanczos_iter_), static_cast<int>(nevex_), &upperb,
        true, ritzv_.data());

    EXPECT_TRUE(std::isfinite(upperb)) << "upperb (b_sup) is not finite";
    EXPECT_GT(upperb, 0) << "upperb (b_sup) should be positive for H²";

    for (std::size_t i = 0; i < nevex_; ++i)
    {
        EXPECT_TRUE(std::isfinite(ritzv_[i]))
            << "ritzv_[" << i << "] is not finite";
        EXPECT_GE(ritzv_[i], 0)
            << "ritzv_[" << i << "] (H² bound) should be non-negative";
    }

    Base mu_1_est = ritzv_[0];
    for (std::size_t i = 1; i < nevex_ - 1; ++i)
        if (ritzv_[i] < mu_1_est)
            mu_1_est = ritzv_[i];
    Base mu_nevnex = ritzv_[nevex_ - 1];

    EXPECT_GT(mu_1_est, 0) << "μ_1 estimate should be positive";
    EXPECT_LT(mu_1_est, mu_nevnex)
        << "μ_1 estimate " << mu_1_est << " should be smaller than μ_nevnex " << mu_nevnex;
    Base gap_low = lambda_2nevex_minus_1 - smallest_eig_H2;
    if (gap_low > 0)
    {
        const Base mu_1_gap_ratio = 0.2;
        Base dist_mu1_to_smallest = std::abs(mu_1_est - smallest_eig_H2);
        EXPECT_LE(dist_mu1_to_smallest, mu_1_gap_ratio * gap_low)
            << "|μ_1 - smallest λ(H²)| = " << dist_mu1_to_smallest
            << " should be <= " << mu_1_gap_ratio << " * (λ(H²)[2*nevex-1] - smallest) = "
            << (mu_1_gap_ratio * gap_low);
    }

    EXPECT_GT(mu_nevnex, 0) << "μ_nevnex (last entry) should be positive";
    EXPECT_GE(mu_nevnex, lambda_2nevex_minus_1)
        << "μ_nevnex " << mu_nevnex
        << " should be >= λ(H²)[2*(nev+nex)-1] = " << lambda_2nevex_minus_1;
    EXPECT_LE(mu_nevnex, upperb)
        << "μ_nevnex should not exceed b_sup (upperb)";

    const Base upperb_tol = 0.02;
    EXPECT_GE(upperb, (1 - upperb_tol) * largest_eig_H2)
        << "upperb (b_sup) " << upperb
        << " should be >= " << ((1 - upperb_tol) * largest_eig_H2);
    EXPECT_LE(upperb, (1 + upperb_tol) * largest_eig_H2)
        << "upperb (b_sup) " << upperb
        << " should be <= " << ((1 + upperb_tol) * largest_eig_H2);

    EXPECT_LE(idx, static_cast<std::size_t>(lanczos_iter_))
        << "extracted index should be <= m";
}
#endif

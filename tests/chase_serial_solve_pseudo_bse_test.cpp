// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include "algorithm/algorithm.hpp"
#include "Impl/chase_cpu/chase_cpu.hpp"
#ifdef HAS_CUDA
#include "Impl/chase_gpu/chase_gpu.hpp"
#endif
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

enum class Backend { CPU, GPU };

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
// Supports both CPU and GPU backends (GPU when HAS_CUDA is defined).
//
// Notes on CPU vs GPU output:
// - H² bounds (lambda_1, mu_nevnex, b_sup) before the 1st iteration can differ slightly:
//   they come from Lanczos for H², which uses random initial vectors. CPU and GPU use
//   different RNGs (host vs device), so the Lanczos runs differ and the estimates differ.
// - "CholeskyQR doesn't work, Householder QR will be used" may appear on GPU (not CPU)
//   when the filter subspace V is very ill-conditioned (e.g. cond(V) ~ 1e20). The GPU
//   Cholesky path (cuBLAS/cuSOLVER) can fail numerically in that regime; the fallback
//   to Householder QR is correct and both backends still converge.
class ChaseSerialSolvePseudoBSETest : public ::testing::TestWithParam<Backend>
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

TEST_P(ChaseSerialSolvePseudoBSETest, SolvePseudo_BSE_Matrix)
{
    const Backend backend = GetParam();
#ifdef HAS_CUDA
    if (backend == Backend::GPU) { /* GPU test runs when HAS_CUDA */ }
#else
    if (backend == Backend::GPU)
        GTEST_SKIP() << "GPU backend not built (no HAS_CUDA)";
#endif

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank != 0)
        return;

    const bool use_gpu = (backend == Backend::GPU);
    const char* mode_str = use_gpu ? "GPU" : "CPU";
    std::cout << "[Solve_pseudo BSE " << mode_str << "] n=" << n_ << " nev=" << nev_
              << " nex=" << nex_ << " nevex=" << nevex_ << std::endl;

    chase::ChaseBase<T>* single = nullptr;
    if (backend == Backend::CPU)
    {
        auto* cpu_single = new chase::Impl::ChASECPU<T, chase::matrix::PseudoHermitianMatrix<T>>(
            n_, nev_, nex_, Hmat_.get(), V_.data(), n_, Lambda_.data());
        single = cpu_single;
    }
#ifdef HAS_CUDA
    else
    {
        using GPUMatrixType =
            chase::matrix::PseudoHermitianMatrix<T, chase::platform::GPU>;
        auto* gpu_single = new chase::Impl::ChASEGPU<T, GPUMatrixType>(
            n_, nev_, nex_, Hmat_->data(), Hmat_->ld(), V_.data(), n_, Lambda_.data());
        single = gpu_single;
    }
#endif

    auto& config = single->GetConfig();
    config.SetTol(1e-10);
    config.SetDeg(20);
    config.SetOpt(true);
    config.SetApprox(false);
    config.SetMaxIter(25);
    config.SetNumLanczos(10);
    config.SetLanczosIter(50);

    std::cout << "[Solve_pseudo BSE " << mode_str << "] calling Solve_pseudo..." << std::endl;
    ASSERT_NO_THROW(chase::Solve_pseudo(single));
    std::cout << "[Solve_pseudo BSE " << mode_str << "] Solve_pseudo returned." << std::endl;

    std::size_t nev_out = single->GetNev();
    chase::Base<T>* ritzv = single->GetRitzv();
    chase::Base<T>* resid = single->GetResid();
    ASSERT_NE(resid, nullptr);

    const chase::Base<T> tol = config.GetTol();
    for (std::size_t i = 0; i < nev_out; ++i)
    {
        EXPECT_TRUE(std::isfinite(Lambda_.data()[i]))
            << "Lambda[" << i << "] is not finite";
        EXPECT_TRUE(std::isfinite(resid[i]))
            << "resid[" << i << "] is not finite";
        EXPECT_LE(resid[i], tol)
            << "resid[" << i << "] = " << resid[i] << " should be <= " << tol;
    }

    // Recompute residuals from user pointers: eigenvalues Lambda_.data() (same as GetRitzv()),
    // eigenvectors V_.data(). For GPU, ChASEGPU::End() copies eigenvectors back to V_.data()
    // so both backends have valid CPU data here. Use explicit loops: res_j = ||H*v_j - lambda_j*v_j||.
    const T* H = Hmat_->data();
    const std::size_t ldh = Hmat_->ld();
    const std::size_t ldv = n_;
    std::vector<chase::Base<T>> resids_recomputed(nev_out);
    for (std::size_t j = 0; j < nev_out; ++j)
    {
        const T* v = V_.data() + j * ldv;
        const chase::Base<T> lam = Lambda_.data()[j];
        std::vector<T> w(n_, T(0));
        for (std::size_t i = 0; i < n_; ++i)
        {
            for (std::size_t k = 0; k < n_; ++k)
                w[i] += H[i + k * ldh] * v[k];
            w[i] -= lam * v[i];
        }
        chase::Base<T> nrm = 0;
        for (std::size_t i = 0; i < n_; ++i)
            nrm += std::norm(w[i]);
        resids_recomputed[j] = std::sqrt(nrm);
    }
    for (std::size_t i = 0; i < nev_out; ++i)
        EXPECT_LE(resids_recomputed[i], tol)
            << "Recomputed residual[" << i << "] (||H*v - lambda*v||) = "
            << resids_recomputed[i] << " > tol = " << tol;

    // Optional: print H² reference bounds (eigs file on disk; same for both backends)
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
        std::cout << "[Solve_pseudo BSE " << mode_str << "] H² bounds (ref): lambda_1 = "
                  << std::scientific << lambda_1_exact
                  << "  mu_nevnex = " << mu_nevnex_exact
                  << "  b_sup = " << b_sup_exact << std::endl;
    }

    // Free backend-specific object (no-op for stack-based CPU in old design; here we used new)
    if (backend == Backend::CPU)
        delete static_cast<chase::Impl::ChASECPU<T, chase::matrix::PseudoHermitianMatrix<T>>*>(single);
#ifdef HAS_CUDA
    else
    {
        using GPUMatrixType =
            chase::matrix::PseudoHermitianMatrix<T, chase::platform::GPU>;
        delete static_cast<chase::Impl::ChASEGPU<T, GPUMatrixType>*>(single);
    }
#endif
}

// Instantiate: CPU always; CPU+GPU when HAS_CUDA
#ifdef HAS_CUDA
INSTANTIATE_TEST_SUITE_P(CPU_and_GPU, ChaseSerialSolvePseudoBSETest,
                         ::testing::Values(Backend::CPU, Backend::GPU));
#else
INSTANTIATE_TEST_SUITE_P(CPU_only, ChaseSerialSolvePseudoBSETest,
                         ::testing::Values(Backend::CPU));
#endif

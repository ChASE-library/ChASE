// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include <algorithm>
#include <complex>
#include <cstring>
#include <fstream>
#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <mpi.h>
#include <string>
#include <vector>

#include "algorithm/algorithm.hpp"
#include "grid/mpiGrid2D.hpp"
#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"
#include "linalg/matrix/matrix.hpp"
#include "Impl/pchase_cpu/pchase_cpu.hpp"
#ifdef HAS_CUDA
#include "Impl/pchase_gpu/pchase_gpu.hpp"
#endif

using namespace chase;

#ifdef HAS_CUDA
using PseudoBSEBackendType = chase::grid::backend::NCCL;
#endif

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

// Backend for parameterized test: CPU or GPU (pChASEGPU with NCCL).
enum class PseudoBSEBackend { CPU, GPU };

// pChASE Solve_pseudo with BSE matrix (cdouble_random_BSE.bin).
// This file uses block-cyclic matrix + DistMultiVectorBlockCyclic1D. Both pChASECPU and
// pChASEGPU currently throw for pseudo-Hermitian + block-cyclic 1D MV; the test catches
// that std::runtime_error and GTEST_SKIP() until block-cyclic pseudo-Hermitian is supported.
// Note: First-iteration Ritz values and residuals can differ between CPU and GPU
// because (1) initial vectors use different RNGs (CPU: std::mt19937, GPU: curand)
// and different seed mapping (CPU: 1337+coords_[0], GPU: 1337+mpi_col_rank); (2) QR
// and Rayleigh-Ritz use different code paths (MPI+LAPACK vs NCCL+cusolver). For
// bitwise-identical iteration history, one would need the same initial V (e.g. CPU
// fill then copy to GPU) and/or CPU fallback when cusolver HEEVD fails.
// Eigenvalues (Lambda/ritzv) and residuals are in RedundantMatrix → valid on all ranks.
// Eigenvectors are in DistMultiVectorBlockCyclic1D → distributed by rows; only rank 0
// does printing and reference comparison.
class ChaseDistributedSolvePseudoBSETest : public ::testing::TestWithParam<PseudoBSEBackend>
{
public:
    using PseudoHMatrixCPU =
        chase::distMatrix::PseudoHermitianBlockCyclicMatrix<T_PseudoBSE,
                                                            chase::platform::CPU>;
    using DistMultiVectorCPU =
        chase::distMultiVector::DistMultiVectorBlockCyclic1D<
            T_PseudoBSE, chase::distMultiVector::CommunicatorType::column,
            chase::platform::CPU>;
#ifdef HAS_CUDA
    using PseudoHMatrixGPU =
        chase::distMatrix::PseudoHermitianBlockCyclicMatrix<T_PseudoBSE,
                                                            chase::platform::GPU>;
    using DistMultiVectorGPU =
        chase::distMultiVector::DistMultiVectorBlockCyclic1D<
            T_PseudoBSE, chase::distMultiVector::CommunicatorType::column,
            chase::platform::GPU>;
#endif

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
    // Keep matrix/vector alive for the lifetime of single (backend-specific).
    std::shared_ptr<PseudoHMatrixCPU> Hmat_cpu_;
    std::shared_ptr<DistMultiVectorCPU> Vec_cpu_;
#ifdef HAS_CUDA
    std::shared_ptr<PseudoHMatrixGPU> Hmat_gpu_;
    std::shared_ptr<DistMultiVectorGPU> Vec_gpu_;
#endif
};

TEST_P(ChaseDistributedSolvePseudoBSETest, SolvePseudo_BSE_Converges)
{
    const PseudoBSEBackend backend = GetParam();
#ifndef HAS_CUDA
    if (backend == PseudoBSEBackend::GPU)
        GTEST_SKIP() << "GPU backend not built (no HAS_CUDA)";
#endif

    const bool use_gpu = (backend == PseudoBSEBackend::GPU);
    const char* mode_str = use_gpu ? "GPU/NCCL" : "CPU";

    ChaseBase<T_PseudoBSE>* single = nullptr;
    if (backend == PseudoBSEBackend::CPU)
    {
        Hmat_cpu_ = std::make_shared<PseudoHMatrixCPU>(N_, N_, blocksize_, blocksize_,
                                                        mpi_grid_);
        Vec_cpu_ = std::make_shared<DistMultiVectorCPU>(N_, 2 * nevex_, blocksize_,
                                                        mpi_grid_);
        Hmat_cpu_->readFromBinaryFile(matrix_path_);
        try
        {
            single = new chase::Impl::pChASECPU<PseudoHMatrixCPU, DistMultiVectorCPU>(
                nev_, nex_, Hmat_cpu_.get(), Vec_cpu_.get(), Lambda_.data());
        }
        catch (const std::runtime_error& e)
        {
            const char* msg = e.what();
            if (msg && std::strstr(msg, "block-cyclic"))
                GTEST_SKIP() << e.what();
            throw;
        }
    }
#ifdef HAS_CUDA
    else
    {
        Hmat_gpu_ = std::make_shared<PseudoHMatrixGPU>(N_, N_, blocksize_, blocksize_,
                                                       mpi_grid_);
        Vec_gpu_ = std::make_shared<DistMultiVectorGPU>(N_, 2 * nevex_, blocksize_,
                                                        mpi_grid_);
        // GPU matrices need CPU buffer for readFromBinaryFile (cpu_data() throws
        // if not allocated). Allocate, load, then copy to device.
        Hmat_gpu_->allocate_cpu_data();
        Hmat_gpu_->readFromBinaryFile(matrix_path_);
        Hmat_gpu_->H2D();
        try
        {
            single = new chase::Impl::pChASEGPU<PseudoHMatrixGPU, DistMultiVectorGPU,
                                                PseudoBSEBackendType>(
                nev_, nex_, Hmat_gpu_.get(), Vec_gpu_.get(), Lambda_.data());
        }
        catch (const std::runtime_error& e)
        {
            const char* msg = e.what();
            if (msg && std::strstr(msg, "block-cyclic"))
                GTEST_SKIP() << e.what();
            throw;
        }
    }
#endif

    auto& config = single->GetConfig();
    config.SetTol(1e-10);
    config.SetDeg(20);
    config.SetOpt(true);
    config.SetApprox(false);
    config.SetMaxIter(25);
    config.SetNumLanczos(10);
    config.SetLanczosIter(40);

    if (world_rank_ == 0)
        std::cout << "[Solve_pseudo BSE " << mode_str << "] n=" << N_ << " nev=" << nev_
                  << " nex=" << nex_ << " nevex=" << nevex_ << std::endl;

    if (world_rank_ == 0)
        std::cout << "[Solve_pseudo BSE " << mode_str << "] calling Solve_pseudo..."
                  << std::endl;
    ASSERT_NO_THROW(chase::Solve_pseudo(single));
    if (world_rank_ == 0)
        std::cout << "[Solve_pseudo BSE " << mode_str << "] Solve_pseudo returned."
                  << std::endl;

    std::size_t nev_out = single->GetNev();
    chase::Base<T_PseudoBSE>* ritzv = single->GetRitzv();
    chase::Base<T_PseudoBSE>* resid = single->GetResid();
    ASSERT_NE(resid, nullptr);

    if (world_rank_ == 0)
    {
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

            std::cout << "[Solve_pseudo BSE " << mode_str
                      << "] H² bounds from direct solver: lambda_1 = "
                      << std::scientific << std::setprecision(6) << lambda_1_exact
                      << "  lower(mu_nevnex) = " << std::setprecision(6)
                      << mu_nevnex_exact << "  b_sup = " << std::setprecision(6)
                      << b_sup_exact << std::endl;
        }
        else
        {
            std::cout << "[Solve_pseudo BSE " << mode_str
                      << "] eigs file not found, skipping exact bounds." << std::endl;
        }

        std::cout << "[Solve_pseudo BSE " << mode_str
                  << "] Requested positive eigenvalues: nev = " << nev_out
                  << " (buffer size = " << single->GetRitzvBlockSize()
                  << "). First nev columns of V = eigenvectors (distributed by rows).\n";

        std::cout << "[Solve_pseudo BSE " << mode_str
                  << "] First nev from Lambda_.data() (user ritzv buffer):\n";
        for (std::size_t i = 0; i < nev_out; ++i)
            std::cout << "  lambda[" << i << "] = " << std::fixed << std::setprecision(8)
                      << Lambda_.data()[i] << "\n";
        std::cout << "[Solve_pseudo BSE " << mode_str << "] First nev from GetResid():\n";
        for (std::size_t i = 0; i < nev_out; ++i)
            std::cout << "  residual[" << i << "] = " << std::scientific
                      << std::setprecision(6) << resid[i] << "\n";

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

        std::cout << "[Solve_pseudo BSE " << mode_str
                  << "] Computed vs reference (nev smallest positive):\n"
                  << "    i     lambda_computed   lambda_ref        residual      |err|\n";
        std::cout << std::fixed << std::setprecision(8);
        for (std::size_t i = 0; i < nev_out; ++i)
        {
            chase::Base<T_PseudoBSE> lam = ritzv[i];
            chase::Base<T_PseudoBSE> res = resid[i];
            chase::Base<T_PseudoBSE> ref =
                (i < ref_pos.size()) ? ref_pos[i] : chase::Base<T_PseudoBSE>(-1);
            chase::Base<T_PseudoBSE> err =
                (i < ref_pos.size()) ? std::abs(lam - ref) : chase::Base<T_PseudoBSE>(-1);
            std::cout << "  " << std::setw(3) << i << "  " << std::setw(16) << lam
                      << "  " << std::setw(16)
                      << (ref >= 0 ? ref
                                   : std::numeric_limits<chase::Base<T_PseudoBSE>>::quiet_NaN())
                      << "  " << std::scientific << std::setw(12) << res << "  "
                      << std::scientific
                      << (err >= 0 ? err
                                   : std::numeric_limits<chase::Base<T_PseudoBSE>>::quiet_NaN())
                      << "\n";
        }

        std::cout << "[Solve_pseudo BSE " << mode_str
                  << "] Residuals from GetResid() (eigenvectors distributed; no "
                     "recomputation from V here).\n"
                  << "    i   residual   status\n";
    }

    const chase::Base<T_PseudoBSE> tol = config.GetTol();
    for (std::size_t i = 0; i < nev_out; ++i)
    {
        EXPECT_TRUE(std::isfinite(Lambda_.data()[i]))
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
                      << std::setprecision(6) << resid[i] << "  " << (ok ? "ok" : "FAIL")
                      << "\n";
        }
    }

    if (backend == PseudoBSEBackend::CPU)
        delete static_cast<chase::Impl::pChASECPU<PseudoHMatrixCPU, DistMultiVectorCPU>*>(
            single);
#ifdef HAS_CUDA
    else
        delete static_cast<chase::Impl::pChASEGPU<PseudoHMatrixGPU, DistMultiVectorGPU,
                                                 PseudoBSEBackendType>*>(single);
#endif
}

// Instantiate: CPU; and GPU when HAS_CUDA. With block-cyclic MV both may GTEST_SKIP until supported.
#ifdef HAS_CUDA
INSTANTIATE_TEST_SUITE_P(CPU_and_GPU, ChaseDistributedSolvePseudoBSETest,
                         ::testing::Values(PseudoBSEBackend::CPU, PseudoBSEBackend::GPU));
#else
INSTANTIATE_TEST_SUITE_P(CPU_only, ChaseDistributedSolvePseudoBSETest,
                         ::testing::Values(PseudoBSEBackend::CPU));
#endif

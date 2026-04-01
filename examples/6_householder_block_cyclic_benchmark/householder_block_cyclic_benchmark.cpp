// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include "popl.hpp"

#include <algorithm>
#include <cctype>
#include <complex>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <mpi.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand_kernel.h>

#include "algorithm/logger.hpp"
#include "grid/mpiGrid2D.hpp"
#include "Impl/chase_gpu/cuda_utils.hpp"
#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"
#include "linalg/internal/cuda/random_normal_distribution.cuh"
#include "linalg/internal/nccl/nccl_kernels.hpp"

using ARCH = chase::platform::GPU;
using BackendType = chase::grid::backend::NCCL;

namespace
{
constexpr int kRandBlockDim = 256;
constexpr int kRandGridDim = 32;

struct BenchConfig
{
    std::size_t N = 0;
    std::size_t ncols = 256;
    std::size_t mb = 64;
    std::size_t nb = 32;
    std::size_t warmup = 1;
    std::size_t iters = 5;
    std::string path_in;
    unsigned long long seed = 1337ull;
    bool use_input_matrix = false;
};

template <typename T>
void fill_vectors_normal_gpu(
    chase::distMultiVector::DistMultiVectorBlockCyclic1D<
        T, chase::distMultiVector::CommunicatorType::column, ARCH>& V,
    curandStatePhilox4_32_10_t* states, unsigned long long seed_base,
    MPI_Comm col_comm)
{
    int mpi_col_rank = 0;
    MPI_Comm_rank(col_comm, &mpi_col_rank);
    const unsigned long long seed = seed_base + static_cast<unsigned long long>(mpi_col_rank);

    chase::linalg::internal::cuda::chase_rand_normal(
        seed, states, V.l_data(), V.l_ld() * V.l_cols(), (cudaStream_t)0);
}

template <typename T>
void fill_vectors_from_matrix_binary(
    const std::string& path_in,
    chase::distMultiVector::DistMultiVectorBlockCyclic1D<
        T, chase::distMultiVector::CommunicatorType::column, ARCH>& V,
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid,
    std::size_t N, std::size_t mb)
{
    auto H = chase::distMatrix::BlockCyclicMatrix<T, ARCH>(N, N, mb, mb, mpi_grid);
    H.allocate_cpu_data();
    V.allocate_cpu_data();

    H.readFromBinaryFile(path_in);

    const std::size_t rows = std::min(H.l_rows(), V.l_rows());
    const std::size_t cols = std::min(H.l_cols(), V.l_cols());

    for (std::size_t j = 0; j < cols; ++j)
    {
        for (std::size_t i = 0; i < rows; ++i)
        {
            V.cpu_data()[i + j * V.l_ld()] = H.cpu_data()[i + j * H.l_ld()];
        }
    }
    V.H2D();
}

template <typename T>
int run_bench(const BenchConfig& cfg)
{
    int world_rank = 0, world_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (cfg.N == 0 || cfg.ncols == 0)
    {
        if (world_rank == 0)
            std::cerr << "N and ncols must be > 0\n";
        return EXIT_FAILURE;
    }

    const int rows = world_size;
    const int cols = 1;
    auto mpi_grid =
        std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(
            rows, cols, MPI_COMM_WORLD);

    auto Vec = chase::distMultiVector::DistMultiVectorBlockCyclic1D<
        T, chase::distMultiVector::CommunicatorType::column, ARCH>(
        cfg.N, cfg.ncols, cfg.mb, mpi_grid);

    cublasHandle_t cublasH = nullptr;
    CHECK_CUBLAS_ERROR(cublasCreate(&cublasH));

    curandStatePhilox4_32_10_t* states = nullptr;
    CHECK_CUDA_ERROR(cudaMalloc(&states, kRandBlockDim * kRandGridDim *
                                             sizeof(curandStatePhilox4_32_10_t)));

    auto init_vectors = [&](unsigned long long seed_shift) {
        if (cfg.use_input_matrix)
        {
            fill_vectors_from_matrix_binary<T>(cfg.path_in, Vec, mpi_grid, cfg.N, cfg.mb);
        }
        else
        {
            fill_vectors_normal_gpu<T>(
                Vec, states, cfg.seed + seed_shift, mpi_grid->get_col_comm());
        }
    };

    if (world_rank == 0)
    {
        std::cout << "Householder block-cyclic benchmark (NCCL GPU)\n"
                  << "N=" << cfg.N
                  << " ncols=" << cfg.ncols
                  << " mb=" << cfg.mb
                  << " nb=" << cfg.nb
                  << " warmup=" << cfg.warmup
                  << " iters=" << cfg.iters
                  << " grid=" << rows << "x" << cols
                  << (cfg.use_input_matrix ? " input=binary\n" : " input=normal\n");
    }

    for (std::size_t w = 0; w < cfg.warmup; ++w)
    {
        init_vectors(w);
        MPI_Barrier(MPI_COMM_WORLD);
        chase::linalg::internal::cuda_nccl::houseQR1_formQ(cublasH, Vec, nullptr, 0,
                                                            cfg.nb);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }

    std::vector<double> timings(cfg.iters, 0.0);
    for (std::size_t it = 0; it < cfg.iters; ++it)
    {
        init_vectors(cfg.warmup + it);
        MPI_Barrier(MPI_COMM_WORLD);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        const double t0 = MPI_Wtime();
        chase::linalg::internal::cuda_nccl::houseQR1_formQ(cublasH, Vec, nullptr, 0,
                                                            cfg.nb);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        const double t1 = MPI_Wtime();
        double local = t1 - t0;
        double global = 0.0;
        MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        timings[it] = global;
    }

    if (world_rank == 0)
    {
        const double min_t = *std::min_element(timings.begin(), timings.end());
        const double max_t = *std::max_element(timings.begin(), timings.end());
        const double avg_t =
            std::accumulate(timings.begin(), timings.end(), 0.0) /
            static_cast<double>(timings.size());

        std::cout << std::fixed << std::setprecision(6)
                  << "houseQR1_formQ wall(s): min=" << min_t
                  << " avg=" << avg_t
                  << " max=" << max_t << "\n";
    }

    CHECK_CUDA_ERROR(cudaFree(states));
    CHECK_CUBLAS_ERROR(cublasDestroy(cublasH));
    return EXIT_SUCCESS;
}
} // namespace

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int world_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    popl::OptionParser desc("Householder Block-Cyclic NCCL GPU benchmark");
    auto help = desc.add<popl::Switch>("h", "help", "show this message");

    BenchConfig conf;
    std::string dtype = "z";

    desc.add<popl::Value<std::size_t>, popl::Attribute::required>("", "n",
        "global matrix/vector length N", 0, &conf.N);
    desc.add<popl::Value<std::size_t>>("", "ncols",
        "number of vectors (panel width benchmarked)", 256, &conf.ncols);
    desc.add<popl::Value<std::size_t>>("", "mb", "block-cyclic block size", 64, &conf.mb);
    desc.add<popl::Value<std::size_t>>("", "nb",
        "Householder blocked nb (houseQR1_formQ)", 32, &conf.nb);
    desc.add<popl::Value<std::size_t>>("", "warmup", "warmup iterations", 1, &conf.warmup);
    desc.add<popl::Value<std::size_t>>("", "iters", "timed iterations", 5, &conf.iters);
    desc.add<popl::Value<unsigned long long>>("", "seed", "base RNG seed", 1337ull, &conf.seed);
    desc.add<popl::Value<std::string>>("", "path_in",
        "optional binary matrix path; first ncols columns used as vectors",
        "", &conf.path_in);
    desc.add<popl::Value<std::string>>("", "dtype",
        "data type: s=float, d=double, c=complex<float>, z=complex<double>",
        "z", &dtype);
    try
    {
        desc.parse(argc, argv);
        if (help->count() == 1)
        {
            if (world_rank == 0)
                std::cout << desc << "\n";
            MPI_Finalize();
            return 0;
        }
    }
    catch (const std::exception& e)
    {
        if (world_rank == 0)
            std::cerr << "Argument parse error: " << e.what() << "\n" << desc
                      << "\n";
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    conf.use_input_matrix = !conf.path_in.empty();
    if (dtype.empty())
        dtype = "z";
    dtype[0] = static_cast<char>(std::tolower(dtype[0]));

    int rc = EXIT_FAILURE;
    if (dtype == "s")
    {
        rc = run_bench<float>(conf);
    }
    else if (dtype == "d")
    {
        rc = run_bench<double>(conf);
    }
    else if (dtype == "c")
    {
        rc = run_bench<std::complex<float>>(conf);
    }
    else if (dtype == "z")
    {
        rc = run_bench<std::complex<double>>(conf);
    }
    else
    {
        if (world_rank == 0)
            std::cerr << "Unsupported --dtype='" << dtype
                      << "'. Use one of: s, d, c, z.\n";
        rc = EXIT_FAILURE;
    }

    MPI_Finalize();
    return rc;
}


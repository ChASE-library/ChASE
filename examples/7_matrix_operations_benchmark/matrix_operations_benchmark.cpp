// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include "grid/mpiGrid2D.hpp"
#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"
#include "linalg/internal/gpu/backend/context.hpp"
#include "linalg/internal/gpu/backend/error.hpp"
#include "linalg/internal/gpu/backend/runtime.hpp"
#include "linalg/internal/gpu/dist_mpi/impl/mpi_collectives.hpp"

#if defined(HAS_CUDA)
#include "linalg/internal/gpu/dist_mpi/cuda/hemm.hpp"
#if defined(HAS_NCCL)
#include "linalg/internal/gpu/dist_xccl/nccl/hemm.hpp"
#include "grid/nccl_utils.hpp"
#endif
#elif defined(HAS_HIP)
#include "linalg/internal/gpu/dist_mpi/hip/hemm.hpp"
#if defined(HAS_RCCL)
#include "linalg/internal/gpu/dist_xccl/rccl/hemm.hpp"
#include "grid/nccl_utils.hpp"
#endif
#else
#error "matrix_operations_benchmark requires CUDA or HIP support"
#endif

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace
{
using T = double;

#if defined(HAS_CUDA)
using Backend = chase::linalg::internal::gpu_backend::cuda_backend;
using MpiKernelNamespace = chase::linalg::internal::cuda_mpi;
#if defined(HAS_NCCL)
using XcclKernelNamespace = chase::linalg::internal::cuda_nccl;
#define CHASE_MATRIX_BENCH_HAS_XCCL 1
#endif

const char* backend_name()
{
    return "CUDA-aware MPI";
}
#elif defined(HAS_HIP)
using Backend = chase::linalg::internal::gpu_backend::hip_backend;
using MpiKernelNamespace = chase::linalg::internal::hip_mpi;
#if defined(HAS_RCCL)
using XcclKernelNamespace = chase::linalg::internal::hip_rccl;
#define CHASE_MATRIX_BENCH_HAS_XCCL 1
#endif

const char* backend_name()
{
    return "HIP-aware MPI";
}
#endif

using HandleTraits =
    chase::linalg::internal::gpu_backend::context_handle_traits<Backend>;
using BlasHandle = typename HandleTraits::blas_handle_type;

BlasHandle create_blas_handle()
{
    BlasHandle handle = nullptr;
    HandleTraits::create_blas(&handle);
    return handle;
}

void destroy_blas_handle(BlasHandle handle)
{
    HandleTraits::destroy_blas(handle);
}

struct Options
{
    std::size_t n = 2000;
    std::size_t nvec = 64;
    std::size_t mb = 128;
    int grid_rows = 0;
    int grid_cols = 0;
    std::string grid_major = "col";
    std::string comm_backend = "mpi";
    int iters = 10;
    int warmup = 2;
    std::size_t allreduce_count = 0;
};

void print_usage(const char* argv0)
{
    std::cout
        << "Usage: " << argv0 << " [options]\n"
        << "Options:\n"
        << "  --n <N>                 Global square matrix size (default 2000)\n"
        << "  --nvec <K>              Number of multivector columns (default 64)\n"
        << "  --mb <B>                Block size for block-cyclic layout (default 128)\n"
        << "  --grid-rows <Pr>        2D grid rows (default MPI_Dims_create)\n"
        << "  --grid-cols <Pc>        2D grid cols (default MPI_Dims_create)\n"
        << "  --grid-major <row|col>  MPI Cartesian rank ordering (default col)\n"
        << "  --comm <mpi|xccl|both>  Benchmark GPU-aware MPI, NCCL/RCCL, or both\n"
        << "                           (default mpi)\n"
        << "  --iters <I>             Timed iterations (default 10)\n"
        << "  --warmup <W>            Warm-up iterations (default 2)\n"
        << "  --allreduce-count <C>   Device elements in row/column allreduce\n"
        << "                           (default nvec*(nvec+1)/2)\n"
        << "  --help                  Print this message\n";
}

std::size_t read_size_arg(int& i, int argc, char** argv)
{
    if (i + 1 >= argc)
    {
        throw std::runtime_error(std::string("missing value for ") + argv[i]);
    }
    return static_cast<std::size_t>(std::stoull(argv[++i]));
}

int read_int_arg(int& i, int argc, char** argv)
{
    if (i + 1 >= argc)
    {
        throw std::runtime_error(std::string("missing value for ") + argv[i]);
    }
    return std::stoi(argv[++i]);
}

Options parse_options(int argc, char** argv)
{
    Options opts;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg(argv[i]);
        if (arg == "--help")
        {
            print_usage(argv[0]);
            MPI_Finalize();
            std::exit(EXIT_SUCCESS);
        }
        else if (arg == "--n")
        {
            opts.n = read_size_arg(i, argc, argv);
        }
        else if (arg == "--nvec")
        {
            opts.nvec = read_size_arg(i, argc, argv);
        }
        else if (arg == "--mb")
        {
            opts.mb = read_size_arg(i, argc, argv);
        }
        else if (arg == "--grid-rows")
        {
            opts.grid_rows = read_int_arg(i, argc, argv);
        }
        else if (arg == "--grid-cols")
        {
            opts.grid_cols = read_int_arg(i, argc, argv);
        }
        else if (arg == "--grid-major")
        {
            if (i + 1 >= argc)
            {
                throw std::runtime_error("missing value for --grid-major");
            }
            opts.grid_major = argv[++i];
        }
        else if (arg == "--comm")
        {
            if (i + 1 >= argc)
            {
                throw std::runtime_error("missing value for --comm");
            }
            opts.comm_backend = argv[++i];
        }
        else if (arg == "--iters")
        {
            opts.iters = read_int_arg(i, argc, argv);
        }
        else if (arg == "--warmup")
        {
            opts.warmup = read_int_arg(i, argc, argv);
        }
        else if (arg == "--allreduce-count")
        {
            opts.allreduce_count = read_size_arg(i, argc, argv);
        }
        else
        {
            throw std::runtime_error("unknown option: " + arg);
        }
    }

    if (opts.n == 0 || opts.nvec == 0 || opts.mb == 0 || opts.iters <= 0 ||
        opts.warmup < 0)
    {
        throw std::runtime_error("n, nvec, mb, iters must be positive");
    }
    if (opts.grid_major != "row" && opts.grid_major != "col")
    {
        throw std::runtime_error("--grid-major must be 'row' or 'col'");
    }
    if (opts.comm_backend != "mpi" && opts.comm_backend != "xccl" &&
        opts.comm_backend != "both")
    {
        throw std::runtime_error("--comm must be 'mpi', 'xccl', or 'both'");
    }
#ifndef CHASE_MATRIX_BENCH_HAS_XCCL
    if (opts.comm_backend == "xccl" || opts.comm_backend == "both")
    {
        throw std::runtime_error(
            "--comm xccl/both requested, but this build has no NCCL/RCCL "
            "distributed GPU collective support");
    }
#endif

    if (opts.allreduce_count == 0)
    {
        opts.allreduce_count = opts.nvec * (opts.nvec + 1) / 2;
    }

    return opts;
}

std::shared_ptr<chase::grid::MpiGrid2DBase>
make_grid(const Options& opts, MPI_Comm comm)
{
    int size = 1;
    MPI_Comm_size(comm, &size);
    int dims[2] = {opts.grid_rows, opts.grid_cols};

    if (dims[0] == 0 && dims[1] == 0)
    {
        MPI_Dims_create(size, 2, dims);
    }
    else if (dims[0] == 0)
    {
        if (size % dims[1] != 0)
        {
            throw std::runtime_error("world size is not divisible by grid cols");
        }
        dims[0] = size / dims[1];
    }
    else if (dims[1] == 0)
    {
        if (size % dims[0] != 0)
        {
            throw std::runtime_error("world size is not divisible by grid rows");
        }
        dims[1] = size / dims[0];
    }

    if (dims[0] * dims[1] != size)
    {
        throw std::runtime_error("grid_rows * grid_cols must equal MPI size");
    }
    if (dims[0] < dims[1])
    {
        throw std::runtime_error(
            "ChASE MpiGrid2D currently requires grid rows >= grid cols; "
            "use --grid-major row|col to compare rank ordering within a "
            "valid Pr x Pc shape");
    }

    if (opts.grid_major == "row")
    {
        return std::make_shared<
            chase::grid::MpiGrid2D<chase::grid::GridMajor::RowMajor>>(
            dims[0], dims[1], comm);
    }

    return std::make_shared<
        chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(
            dims[0], dims[1], comm);
}

template <typename MatrixType>
void fill_matrix(MatrixType& matrix, T value)
{
    matrix.allocate_cpu_data();
    for (std::size_t j = 0; j < matrix.l_cols(); ++j)
    {
        for (std::size_t i = 0; i < matrix.l_rows(); ++i)
        {
            matrix.cpu_data()[j * matrix.cpu_ld() + i] = value;
        }
    }
    matrix.H2D();
}

double source_value(std::size_t global_row, std::size_t col)
{
    return 1.0 + static_cast<double>(global_row) * 0.001 +
           static_cast<double>(col) * 0.000001;
}

template <typename MultiVectorType>
std::vector<std::size_t> local_global_rows(MultiVectorType& vec)
{
    std::vector<std::size_t> rows(vec.l_rows());
    if constexpr (chase::distMultiVector::is_block_cyclic_1d_multivector<
                      MultiVectorType>::value)
    {
        auto global_offs = vec.m_contiguous_global_offs();
        auto local_offs = vec.m_contiguous_local_offs();
        auto lens = vec.m_contiguous_lens();
        for (std::size_t block = 0; block < lens.size(); ++block)
        {
            for (std::size_t i = 0; i < lens[block]; ++i)
            {
                rows[local_offs[block] + i] = global_offs[block] + i;
            }
        }
    }
    else
    {
        for (std::size_t i = 0; i < vec.l_rows(); ++i)
        {
            rows[i] = vec.g_off() + i;
        }
    }
    return rows;
}

template <typename MultiVectorType>
void fill_source_multivector(MultiVectorType& vec)
{
    vec.allocate_cpu_data();
    const auto rows = local_global_rows(vec);
    for (std::size_t j = 0; j < vec.l_cols(); ++j)
    {
        for (std::size_t i = 0; i < vec.l_rows(); ++i)
        {
            vec.cpu_data()[j * vec.cpu_ld() + i] = source_value(rows[i], j);
        }
    }
    vec.H2D();
}

template <typename MultiVectorType>
void fill_constant_multivector(MultiVectorType& vec, T value)
{
    vec.allocate_cpu_data();
    for (std::size_t j = 0; j < vec.l_cols(); ++j)
    {
        for (std::size_t i = 0; i < vec.l_rows(); ++i)
        {
            vec.cpu_data()[j * vec.cpu_ld() + i] = value;
        }
    }
    vec.H2D();
}

template <typename MultiVectorType>
bool check_constant(MultiVectorType& vec, T expected, std::size_t offset,
                    std::size_t sub_size, double tol)
{
    vec.D2H();
    bool ok = true;
    for (std::size_t j = offset; j < offset + sub_size; ++j)
    {
        for (std::size_t i = 0; i < vec.l_rows(); ++i)
        {
            const double diff =
                std::abs(vec.cpu_data()[j * vec.cpu_ld() + i] - expected);
            if (diff > tol)
            {
                ok = false;
            }
        }
    }
    return ok;
}

template <typename MultiVectorType>
bool check_source_values(MultiVectorType& vec, double tol)
{
    vec.D2H();
    const auto rows = local_global_rows(vec);
    bool ok = true;
    for (std::size_t j = 0; j < vec.l_cols(); ++j)
    {
        for (std::size_t i = 0; i < vec.l_rows(); ++i)
        {
            const double expected = source_value(rows[i], j);
            const double diff =
                std::abs(vec.cpu_data()[j * vec.cpu_ld() + i] - expected);
            if (diff > tol)
            {
                ok = false;
            }
        }
    }
    return ok;
}

void print_result(int rank, const std::string& name, double seconds,
                  int iters, bool correctness)
{
    double max_seconds = 0.0;
    MPI_Reduce(&seconds, &max_seconds, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);
    int local_ok = correctness ? 1 : 0;
    int global_ok = 0;
    MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    if (rank == 0)
    {
        std::cout << std::left << std::setw(34) << name
                  << " total_s=" << std::setw(12) << std::setprecision(6)
                  << std::fixed << max_seconds
                  << " avg_ms=" << std::setw(12)
                  << (max_seconds * 1000.0 / static_cast<double>(iters))
                  << " correctness=" << (global_ok ? "OK" : "FAILED")
                  << "\n";
    }
}

void print_payload(int rank, const std::string& name, std::size_t count)
{
    const unsigned long long local_count =
        static_cast<unsigned long long>(count);
    unsigned long long min_count = 0;
    unsigned long long max_count = 0;
    MPI_Reduce(&local_count, &min_count, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(&local_count, &max_count, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0,
               MPI_COMM_WORLD);

    if (rank == 0)
    {
        std::cout << std::left << std::setw(34) << name
                  << " min_count=" << min_count
                  << " max_count=" << max_count << "\n";
    }
}

std::size_t redistribution_segment_count(std::size_t n, std::size_t src_mb,
                                         std::size_t dst_mb,
                                         std::size_t src_dim,
                                         std::size_t dst_dim)
{
    if (n == 0)
    {
        return 0;
    }

    std::size_t count = 1;
    std::size_t current_src = 0;
    std::size_t current_dst = 0;
    for (std::size_t i = 1; i < n; ++i)
    {
        const std::size_t src = (i / src_mb) % src_dim;
        const std::size_t dst = (i / dst_mb) % dst_dim;
        if (src != current_src || dst != current_dst)
        {
            ++count;
            current_src = src;
            current_dst = dst;
        }
    }
    return count;
}

void print_segment_stats(int rank, const std::string& name,
                         std::size_t col_to_row_segments,
                         std::size_t row_to_col_segments)
{
    if (rank == 0)
    {
        std::cout << std::left << std::setw(34) << name
                  << " c2r_segments=" << col_to_row_segments
                  << " r2c_segments=" << row_to_col_segments
                  << " total_segments="
                  << (col_to_row_segments + row_to_col_segments) << "\n";
    }
}

template <typename KernelNamespace, typename MatrixType,
          typename ColumnVectorType, typename RowVectorType>
void benchmark_hemm_directions(const std::string& name, BlasHandle blas_handle,
                               MatrixType& H, ColumnVectorType& Vcol,
                               RowVectorType& Wrow, ColumnVectorType& Vout,
                               const Options& opts, int rank)
{
    fill_matrix(H, T(1.0));
    fill_constant_multivector(Vcol, T(2.0));
    fill_constant_multivector(Wrow, T(0.0));
    fill_constant_multivector(Vout, T(0.0));

    T one = T(1.0);
    T zero = T(0.0);

    auto run_col_to_row = [&]() {
        KernelNamespace::MatrixMultiplyMultiVectors(
            blas_handle, &one, H, Vcol, &zero, Wrow, 0, opts.nvec);
    };
    auto run_row_to_col = [&]() {
        KernelNamespace::MatrixMultiplyMultiVectors(
            blas_handle, &one, H, Wrow, &zero, Vout, 0, opts.nvec);
    };

    for (int i = 0; i < opts.warmup; ++i)
    {
        run_col_to_row();
    }
    chase::linalg::internal::gpu_backend::device_synchronize<Backend>();
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    for (int i = 0; i < opts.iters; ++i)
    {
        run_col_to_row();
    }
    chase::linalg::internal::gpu_backend::device_synchronize<Backend>();
    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    const bool w_ok = check_constant(Wrow, T(2.0 * opts.n), 0, opts.nvec,
                                     1.0e-8 * static_cast<double>(opts.n));
    print_result(rank, name + "-col-to-row", t1 - t0, opts.iters, w_ok);

    for (int i = 0; i < opts.warmup; ++i)
    {
        run_row_to_col();
    }
    chase::linalg::internal::gpu_backend::device_synchronize<Backend>();
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int i = 0; i < opts.iters; ++i)
    {
        run_row_to_col();
    }
    chase::linalg::internal::gpu_backend::device_synchronize<Backend>();
    MPI_Barrier(MPI_COMM_WORLD);
    t1 = MPI_Wtime();

    const bool v_ok =
        check_constant(Vout, T(2.0 * opts.n * opts.n), 0, opts.nvec,
                       1.0e-8 * static_cast<double>(opts.n * opts.n));
    print_result(rank, name + "-row-to-col", t1 - t0, opts.iters, v_ok);
}

template <typename ColumnVectorType, typename RowVectorType>
void benchmark_redistribute_pair_mpi(const std::string& name,
                                     ColumnVectorType& Vcol,
                                     RowVectorType& Wrow,
                                     ColumnVectorType& Vback,
                                     const Options& opts, int rank)
{
    int* dims = Vcol.getMpiGrid()->get_dims();
    const auto col_to_row_segments = redistribution_segment_count(
        opts.n, Vcol.mb(), Wrow.mb(), static_cast<std::size_t>(dims[0]),
        static_cast<std::size_t>(dims[1]));
    const auto row_to_col_segments = redistribution_segment_count(
        opts.n, Wrow.mb(), Vback.mb(), static_cast<std::size_t>(dims[1]),
        static_cast<std::size_t>(dims[0]));
    print_segment_stats(rank, name + "-segments", col_to_row_segments,
                        row_to_col_segments);

    fill_source_multivector(Vcol);
    fill_constant_multivector(Wrow, T(-1.0));
    fill_constant_multivector(Vback, T(-2.0));

    auto run_once = [&]() {
        Vcol.redistributeImpl(&Wrow, 0, opts.nvec);
        Wrow.redistributeImpl(&Vback, 0, opts.nvec);
    };

    for (int i = 0; i < opts.warmup; ++i)
    {
        run_once();
    }
    chase::linalg::internal::gpu_backend::device_synchronize<Backend>();
    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();
    for (int i = 0; i < opts.iters; ++i)
    {
        run_once();
    }
    chase::linalg::internal::gpu_backend::device_synchronize<Backend>();
    MPI_Barrier(MPI_COMM_WORLD);
    const double t1 = MPI_Wtime();

    const bool row_ok = check_source_values(Wrow, 1.0e-12);
    const bool col_ok = check_source_values(Vback, 1.0e-12);
    print_result(rank, name, t1 - t0, opts.iters, row_ok && col_ok);
}

#ifdef CHASE_MATRIX_BENCH_HAS_XCCL
template <typename ColumnVectorType, typename RowVectorType>
void benchmark_redistribute_pair_xccl(const std::string& name,
                                      ColumnVectorType& Vcol,
                                      RowVectorType& Wrow,
                                      ColumnVectorType& Vback,
                                      const Options& opts, int rank)
{
    int* dims = Vcol.getMpiGrid()->get_dims();
    const auto col_to_row_segments = redistribution_segment_count(
        opts.n, Vcol.mb(), Wrow.mb(), static_cast<std::size_t>(dims[0]),
        static_cast<std::size_t>(dims[1]));
    const auto row_to_col_segments = redistribution_segment_count(
        opts.n, Wrow.mb(), Vback.mb(), static_cast<std::size_t>(dims[1]),
        static_cast<std::size_t>(dims[0]));
    print_segment_stats(rank, name + "-segments", col_to_row_segments,
                        row_to_col_segments);

    fill_source_multivector(Vcol);
    fill_constant_multivector(Wrow, T(-1.0));
    fill_constant_multivector(Vback, T(-2.0));

    auto run_once = [&]() {
        Vcol.redistributeImplAsync(&Wrow, 0, opts.nvec);
        Wrow.redistributeImplAsync(&Vback, 0, opts.nvec);
    };

    for (int i = 0; i < opts.warmup; ++i)
    {
        run_once();
    }
    chase::linalg::internal::gpu_backend::device_synchronize<Backend>();
    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();
    for (int i = 0; i < opts.iters; ++i)
    {
        run_once();
    }
    chase::linalg::internal::gpu_backend::device_synchronize<Backend>();
    MPI_Barrier(MPI_COMM_WORLD);
    const double t1 = MPI_Wtime();

    const bool row_ok = check_source_values(Wrow, 1.0e-12);
    const bool col_ok = check_source_values(Vback, 1.0e-12);
    print_result(rank, name, t1 - t0, opts.iters, row_ok && col_ok);
}
#endif

void benchmark_allreduce_mpi(const std::string& name, MPI_Comm comm,
                             std::size_t count, const Options& opts, int rank)
{
    if (count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::runtime_error("allreduce count exceeds MPI int range");
    }

    std::vector<T> host(count, T(1.0));
    T* send_device = nullptr;
    T* recv_device = nullptr;
    chase::linalg::internal::gpu_backend::device_malloc<Backend>(&send_device,
                                                                 count);
    chase::linalg::internal::gpu_backend::device_malloc<Backend>(&recv_device,
                                                                 count);
    std::unique_ptr<T, chase::linalg::internal::gpu_backend::
                           runtime_device_deleter<Backend>>
        send_ptr(send_device);
    std::unique_ptr<T, chase::linalg::internal::gpu_backend::
                           runtime_device_deleter<Backend>>
        recv_ptr(recv_device);
    chase::linalg::internal::gpu_backend::copy_to_device<Backend>(
        send_device, host.data(), count);

    auto run_once = [&]() {
        chase::linalg::internal::gpu_mpi::deviceSynchronizeForMpi();
        MPI_Allreduce(send_device, recv_device, static_cast<int>(count),
                      chase::mpi::getMPI_Type<T>(), MPI_SUM, comm);
        chase::linalg::internal::gpu_mpi::deviceSynchronizeForMpi();
    };

    for (int i = 0; i < opts.warmup; ++i)
    {
        run_once();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();
    for (int i = 0; i < opts.iters; ++i)
    {
        run_once();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double t1 = MPI_Wtime();

    int comm_size = 1;
    MPI_Comm_size(comm, &comm_size);
    std::vector<T> result(count);
    chase::linalg::internal::gpu_backend::copy_to_host<Backend>(
        result.data(), recv_device, count);
    bool ok = true;
    for (std::size_t i = 0; i < count; ++i)
    {
        if (std::abs(result[i] - static_cast<T>(comm_size)) > 1.0e-12)
        {
            ok = false;
            break;
        }
    }
    print_result(rank, name, t1 - t0, opts.iters, ok);
}

#ifdef CHASE_MATRIX_BENCH_HAS_XCCL
void benchmark_allreduce_xccl(const std::string& name, MPI_Comm mpi_comm,
                              ncclComm_t xccl_comm, std::size_t count,
                              const Options& opts, int rank)
{
    std::vector<T> host(count, T(1.0));
    T* send_device = nullptr;
    T* recv_device = nullptr;
    chase::linalg::internal::gpu_backend::device_malloc<Backend>(&send_device,
                                                                 count);
    chase::linalg::internal::gpu_backend::device_malloc<Backend>(&recv_device,
                                                                 count);
    std::unique_ptr<T, chase::linalg::internal::gpu_backend::
                           runtime_device_deleter<Backend>>
        send_ptr(send_device);
    std::unique_ptr<T, chase::linalg::internal::gpu_backend::
                           runtime_device_deleter<Backend>>
        recv_ptr(recv_device);
    chase::linalg::internal::gpu_backend::copy_to_device<Backend>(
        send_device, host.data(), count);

    auto run_once = [&]() {
        CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper(
            send_device, recv_device, count, ncclSum, xccl_comm));
    };

    for (int i = 0; i < opts.warmup; ++i)
    {
        run_once();
    }
    chase::linalg::internal::gpu_backend::device_synchronize<Backend>();
    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();
    for (int i = 0; i < opts.iters; ++i)
    {
        run_once();
    }
    chase::linalg::internal::gpu_backend::device_synchronize<Backend>();
    MPI_Barrier(MPI_COMM_WORLD);
    const double t1 = MPI_Wtime();

    int comm_size = 1;
    MPI_Comm_size(mpi_comm, &comm_size);
    std::vector<T> result(count);
    chase::linalg::internal::gpu_backend::copy_to_host<Backend>(
        result.data(), recv_device, count);
    bool ok = true;
    for (std::size_t i = 0; i < count; ++i)
    {
        if (std::abs(result[i] - static_cast<T>(comm_size)) > 1.0e-12)
        {
            ok = false;
            break;
        }
    }
    print_result(rank, name, t1 - t0, opts.iters, ok);
}
#endif

template <typename ColumnVectorType, typename RowVectorType>
void benchmark_hemm_payload_allreduces_mpi(const std::string& name,
                                           ColumnVectorType& Vcol,
                                           RowVectorType& Wrow,
                                           ColumnVectorType& Vback,
                                           const Options& opts, int rank)
{
    const std::size_t col_to_row_count = Wrow.l_ld() * opts.nvec;
    const std::size_t row_to_col_count = Vback.l_ld() * opts.nvec;

    print_payload(rank, name + "-col-to-row-payload", col_to_row_count);
    benchmark_allreduce_mpi(name + "-col-to-row-allreduce",
                            Vcol.getMpiGrid()->get_col_comm(),
                            col_to_row_count, opts, rank);

    print_payload(rank, name + "-row-to-col-payload", row_to_col_count);
    benchmark_allreduce_mpi(name + "-row-to-col-allreduce",
                            Vcol.getMpiGrid()->get_row_comm(),
                            row_to_col_count, opts, rank);
}

#ifdef CHASE_MATRIX_BENCH_HAS_XCCL
template <typename ColumnVectorType, typename RowVectorType>
void benchmark_hemm_payload_allreduces_xccl(const std::string& name,
                                            ColumnVectorType& Vcol,
                                            RowVectorType& Wrow,
                                            ColumnVectorType& Vback,
                                            const Options& opts, int rank)
{
    const std::size_t col_to_row_count = Wrow.l_ld() * opts.nvec;
    const std::size_t row_to_col_count = Vback.l_ld() * opts.nvec;

    print_payload(rank, name + "-col-to-row-payload", col_to_row_count);
    benchmark_allreduce_xccl(name + "-col-to-row-allreduce",
                             Vcol.getMpiGrid()->get_col_comm(),
                             Vcol.getMpiGrid()->get_nccl_col_comm(),
                             col_to_row_count, opts, rank);

    print_payload(rank, name + "-row-to-col-payload", row_to_col_count);
    benchmark_allreduce_xccl(name + "-row-to-col-allreduce",
                             Vcol.getMpiGrid()->get_row_comm(),
                             Vcol.getMpiGrid()->get_nccl_row_comm(),
                             row_to_col_count, opts, rank);
}
#endif

} // namespace

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    try
    {
        const Options opts = parse_options(argc, argv);
        auto grid = make_grid(opts, MPI_COMM_WORLD);
        BlasHandle blas_handle = create_blas_handle();

        int* dims = grid->get_dims();
        int* coords = grid->get_coords();
        if (rank == 0)
        {
            std::cout << "ChASE distributed GPU matrix operations benchmark\n"
                      << "backend=" << backend_name() << " ranks=" << size
                      << " grid=" << dims[0] << "x" << dims[1]
                      << " grid_major=" << opts.grid_major
                      << " comm=" << opts.comm_backend
                      << " n=" << opts.n << " nvec=" << opts.nvec
                      << " mb=" << opts.mb << " iters=" << opts.iters
                      << " warmup=" << opts.warmup
                      << " allreduce_count=" << opts.allreduce_count << "\n";
            std::cout << std::string(96, '-') << "\n";
        }
        MPI_Barrier(MPI_COMM_WORLD);

        if (rank == 0)
        {
            std::cout << "rank coordinate sample: rank0=(" << coords[0] << ","
                      << coords[1] << ")\n";
        }

        const bool run_mpi = opts.comm_backend == "mpi" ||
                             opts.comm_backend == "both";
        const bool run_xccl = opts.comm_backend == "xccl" ||
                              opts.comm_backend == "both";
#ifdef CHASE_MATRIX_BENCH_HAS_XCCL
        if (run_xccl)
        {
            (void)grid->get_nccl_row_comm();
            (void)grid->get_nccl_col_comm();
            MPI_Barrier(MPI_COMM_WORLD);
        }
#endif

        using BlockMatrix =
            chase::distMatrix::BlockBlockMatrix<T, chase::platform::GPU>;
        using BlockColumnVector = chase::distMultiVector::DistMultiVector1D<
            T, chase::distMultiVector::CommunicatorType::column,
            chase::platform::GPU>;
        using BlockRowVector = chase::distMultiVector::DistMultiVector1D<
            T, chase::distMultiVector::CommunicatorType::row,
            chase::platform::GPU>;

        BlockMatrix H_block(opts.n, opts.n, grid);
        BlockColumnVector V_block_col(opts.n, opts.nvec, grid);
        BlockRowVector W_block_row(opts.n, opts.nvec, grid);
        BlockColumnVector V_block_back(opts.n, opts.nvec, grid);

        if (run_mpi)
        {
            benchmark_hemm_directions<MpiKernelNamespace>(
                "mpi-hemm-block", blas_handle, H_block, V_block_col,
                W_block_row, V_block_back, opts, rank);

            benchmark_allreduce_mpi("mpi-allreduce-row-comm",
                                    grid->get_row_comm(),
                                    opts.allreduce_count, opts, rank);
            benchmark_allreduce_mpi("mpi-allreduce-column-comm",
                                    grid->get_col_comm(),
                                    opts.allreduce_count, opts, rank);
            benchmark_hemm_payload_allreduces_mpi(
                "mpi-hemm-block-payload", V_block_col, W_block_row,
                V_block_back, opts, rank);

            benchmark_redistribute_pair_mpi("mpi-redistribute-block-c2r2c",
                                            V_block_col, W_block_row,
                                            V_block_back, opts, rank);
        }

#ifdef CHASE_MATRIX_BENCH_HAS_XCCL
        if (run_xccl)
        {
            benchmark_hemm_directions<XcclKernelNamespace>(
                "xccl-hemm-block", blas_handle, H_block, V_block_col,
                W_block_row, V_block_back, opts, rank);

            benchmark_allreduce_xccl("xccl-allreduce-row-comm",
                                     grid->get_row_comm(),
                                     grid->get_nccl_row_comm(),
                                     opts.allreduce_count, opts, rank);
            benchmark_allreduce_xccl("xccl-allreduce-column-comm",
                                     grid->get_col_comm(),
                                     grid->get_nccl_col_comm(),
                                     opts.allreduce_count, opts, rank);
            benchmark_hemm_payload_allreduces_xccl(
                "xccl-hemm-block-payload", V_block_col, W_block_row,
                V_block_back, opts, rank);

            benchmark_redistribute_pair_xccl("xccl-redistribute-block-c2r2c",
                                             V_block_col, W_block_row,
                                             V_block_back, opts, rank);
        }
#endif

        using CyclicMatrix =
            chase::distMatrix::BlockCyclicMatrix<T, chase::platform::GPU>;
        using CyclicColumnVector =
            chase::distMultiVector::DistMultiVectorBlockCyclic1D<
                T, chase::distMultiVector::CommunicatorType::column,
                chase::platform::GPU>;
        using CyclicRowVector =
            chase::distMultiVector::DistMultiVectorBlockCyclic1D<
                T, chase::distMultiVector::CommunicatorType::row,
                chase::platform::GPU>;

        CyclicMatrix H_cyclic(opts.n, opts.n, opts.mb, opts.mb, grid);
        CyclicColumnVector V_cyclic_col(opts.n, opts.nvec, opts.mb, grid);
        CyclicRowVector W_cyclic_row(opts.n, opts.nvec, opts.mb, grid);
        CyclicColumnVector V_cyclic_back(opts.n, opts.nvec, opts.mb, grid);

        if (run_mpi)
        {
            benchmark_hemm_directions<MpiKernelNamespace>(
                "mpi-hemm-blockcyclic", blas_handle, H_cyclic, V_cyclic_col,
                W_cyclic_row, V_cyclic_back, opts, rank);
            benchmark_hemm_payload_allreduces_mpi(
                "mpi-hemm-blockcyclic-payload", V_cyclic_col, W_cyclic_row,
                V_cyclic_back, opts, rank);
            benchmark_redistribute_pair_mpi(
                "mpi-redistribute-blockcyclic-c2r2c", V_cyclic_col,
                W_cyclic_row, V_cyclic_back, opts, rank);
        }

#ifdef CHASE_MATRIX_BENCH_HAS_XCCL
        if (run_xccl)
        {
            benchmark_hemm_directions<XcclKernelNamespace>(
                "xccl-hemm-blockcyclic", blas_handle, H_cyclic, V_cyclic_col,
                W_cyclic_row, V_cyclic_back, opts, rank);
            benchmark_hemm_payload_allreduces_xccl(
                "xccl-hemm-blockcyclic-payload", V_cyclic_col, W_cyclic_row,
                V_cyclic_back, opts, rank);
            benchmark_redistribute_pair_xccl(
                "xccl-redistribute-blockcyclic-c2r2c", V_cyclic_col,
                W_cyclic_row, V_cyclic_back, opts, rank);
        }
#endif

        destroy_blas_handle(blas_handle);
    }
    catch (const std::exception& e)
    {
        std::cerr << "[rank " << rank << "] " << e.what() << "\n";
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}

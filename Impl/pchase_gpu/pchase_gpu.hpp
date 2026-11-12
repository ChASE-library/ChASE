// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "algorithm/chaseBase.hpp"
#include "external/cpu/lapackpp/lapackpp.hpp"
#include "grid/mpiGrid2D.hpp"
#include "grid/default_distributed_backend.hpp"
#include "grid/mpiTypes.hpp"
#include "Impl/chase_gpu/nvtx.hpp"
#if defined(HAS_HIP) && !defined(HAS_CUDA)
#include "linalg/internal/gpu/dist_mpi/hip/hip_mpi_kernels.hpp"
#include "linalg/internal/gpu/dist_mpi/hip/hip_mpi_utils.hpp"
#elif defined(HAS_CUDA)
#include "linalg/internal/gpu/dist_mpi/cuda/cuda_mpi_kernels.hpp"
#endif
#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"
#include "linalg/internal/gpu/backend/gpu_backend.hpp"
#include "linalg/matrix/matrix.hpp"
#include <cstring>
#include <iomanip>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#ifdef HAS_RCCL
#include "linalg/internal/gpu/dist_xccl/rccl/rccl_kernels.hpp"
#elif defined(HAS_NCCL)
#include "linalg/internal/gpu/dist_xccl/nccl/nccl_kernels.hpp"
#endif

#if defined(HAS_RCCL) || defined(HAS_NCCL)
#define CHASE_PCHASE_GPU_HAS_GPU_COLLECTIVES 1
#endif
#ifdef HAS_SCALAPACK
#include "external/scalapack/scalapackpp/scalapackpp.hpp"
#endif
#include "algorithm/types.hpp"
#include "linalg/internal/mpi/cholqr.hpp"

#include "../../linalg/internal/typeTraits.hpp"
#include "Impl/pchase_cpu/pchase_cpu.hpp"
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#ifdef CHASE_ENABLE_OZAKI
#include "external/gemmul8pp/gemmul8pp.hpp"
#include "external/gemmul8pp/utilities.hpp"
#include <cmath>
#include <algorithm>
#endif

using namespace chase::linalg;

template <typename Backend>
struct MGPUKernelNamspaceSelector;

// Specialization for MPI backend
template <>
struct MGPUKernelNamspaceSelector<chase::grid::backend::MPI>
{
#if defined(HAS_HIP) && !defined(HAS_CUDA)
    using type = chase::linalg::internal::hip_mpi;
#else
    using type = chase::linalg::internal::cuda_mpi; // Use the struct as a type
#endif
    template <typename GridType>
    static auto getColCommunicator(GridType* grid)
    {
        return grid->get_col_comm(); // MPI communicator
    }
};

#if defined(CHASE_PCHASE_GPU_HAS_GPU_COLLECTIVES)
// Specialization for NCCL / RCCL GPU collective backend
template <>
struct MGPUKernelNamspaceSelector<chase::grid::backend::NCCL>
{
#if defined(HAS_RCCL)
    using type = chase::linalg::internal::hip_rccl;
#else
    using type = chase::linalg::internal::cuda_nccl;
#endif
    template <typename GridType>
    static auto getColCommunicator(GridType* grid)
    {
        return grid->get_nccl_col_comm(); // NCCL/RCCL communicator
    }
};

#endif
namespace chase
{
namespace Impl
{

namespace gpu_backend = chase::linalg::internal::gpu_backend;

/**
 * @page pChASEGPU
 *
 * @section intro_sec Introduction
 * This class implements the GPU-based parallel version of the Chase algorithm
 * using GPU-aware MPI or GPU collectives for efficient multi-GPU
 * communication. The class inherits from `ChaseBase` and is designed for
 * solving large-scale generalized eigenvalue problems on distributed GPU
 * environments. It operates on matrix and multi-vector data types, supporting
 * configurations where matrix data and eigenvectors are mapped to a shared MPI
 * grid for distributed computation.
 *
 * @section constructor_sec Constructors and Destructor
 * The constructor for `pChASEGPU` initializes essential components, including
 * the Hamiltonian matrix, input multi-vectors, result multi-vectors, and
 * configurations for backend BLAS/SOLVER libraries. It also sets up memory
 * allocations for GPU-based operations and verifies that the matrix is square
 * and that the matrix and eigenvectors share the same MPI grid.
 *
 * @section members_sec Private Members
 * The private members include matrix and multi-vector data, configuration
 * settings, and GPU-specific resources like BLAS and SOLVER handles.
 * Additional private members manage MPI-related properties, including process
 * rank and grid coordinates, as well as device memory allocations used for
 * intermediate computations.
 */

/**
 * @brief GPU-based parallel Chase algorithm with NCCL.
 *
 * This class provides an implementation of the Chase algorithm for solving
 * generalized eigenvalue problems on GPU-based platforms. It leverages NVIDIA's
 * GPU-aware MPI or GPU collectives for efficient GPU-to-GPU communication and
 * backend SOLVER/BLAS dispatch for GPU-based linear algebra operations.
 * Designed for distributed GPU environments, it operates on matrix and
 * multi-vector data types with MPI parallelism.
 *
 * @tparam MatrixType The matrix type, such as
 * `chase::distMatrix::BlockBlockMatrix` or
 *                    `chase::distMatrix::BlockCyclicMatrix`.
 * @tparam InputMultiVectorType The input multi-vector type, typically used for
 * storing eigenvectors.
 */
template <typename MatrixType, typename InputMultiVectorType,
          typename CommBackend = chase::grid::backend::DefaultDistributedGpu>
class pChASEGPU : public ChaseBase<typename MatrixType::value_type>
{
    using T = typename MatrixType::value_type;
    using ResultMultiVectorType =
        typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type;
    // using backend = typename chase::grid::backend::MPI;
    using backend = CommBackend;
    //    using kernelNamespace = typename
    //    MGPUKernelNamspaceSelector<backend>::type;
    using kernelNamespace = typename MGPUKernelNamspaceSelector<backend>::type;
    using gpu = chase::linalg::internal::gpu_backend::active_stack;
    using solver_dispatch = gpu::solver;
    using blas_dispatch = gpu::blas;
    using execution_context = gpu::execution_context;
    using runtime_type = gpu::runtime;
    using kernel_dispatch = gpu::kernels;
    using stream_type = typename gpu::stream_type;
    using event_type = typename gpu::event_type;
    using rand_state_type = typename gpu::rand_state_type;
    using blas_handle_type = typename gpu::blas_handle;
    using solver_handle_type = typename gpu::solver_handle;
    using solver_params_type = typename gpu::solver_params;
    using runtime_device_deleter = gpu::device_deleter;

public:
    /**
     * @brief Constructs the `pChASEGPU` object.
     *
     * Initializes the Chase algorithm parameters, matrix data, multi-vector
     * data, and GPU-specific resources. Sets up memory allocations for
     * intermediate data and configures backend BLAS/SOLVER handles for GPU
     * computation.
     *
     * @param nev Number of eigenvalues to compute.
     * @param nex Number of extra vectors for iterative refinement.
     * @param H Pointer to the Hamiltonian matrix.
     * @param V Pointer to the input multi-vector (typically for eigenvectors).
     * @param ritzv Pointer to the Ritz values, used for storing eigenvalues.
     */
    pChASEGPU(std::size_t nev, std::size_t nex, MatrixType* H,
              InputMultiVectorType* V, chase::Base<T>* ritzv)
        : nev_(nev), nex_(nex), nevex_(nev + nex),
          N_(H->g_rows()), gpu_context_(false),
          config_(H->g_rows(), nev, nex)
    {
        SCOPED_NVTX_RANGE();

        if (H->g_rows() != H->g_cols())
        {
            throw std::runtime_error(
                "ChASE requires the matrix solved to be squared");
        }

        if (H->getMpiGrid() != V->getMpiGrid())
        {
            throw std::runtime_error("ChASE requires the matrix and "
                                     "eigenvectors mapped to same MPI grid");
        }

        // Pseudo-Hermitian support currently assumes block-block data layout.
        // Block-cyclic 1D multivectors are not supported for pseudo-Hermitian.
        if constexpr (std::is_same<typename MatrixType::hermitian_type,
                                   chase::matrix::PseudoHermitian>::value)
        {
            if constexpr (chase::distMultiVector::is_block_cyclic_1d_multivector<
                              InputMultiVectorType>::value)
            {
                throw std::runtime_error(
                    "Pseudo-Hermitian pChASEGPU currently supports only "
                    "block-block distributed multivectors (block-cyclic 1D not "
                    "supported).");
            }
        }

        N_ = H->g_rows();
        Hmat_ = H;
        V1_ = V;
        V2_ = V1_->template clone2<InputMultiVectorType>();
        W1_ = V1_->template clone2<ResultMultiVectorType>();
        W2_ = V1_->template clone2<ResultMultiVectorType>();

        if constexpr (std::is_same<typename MatrixType::hermitian_type,
                                   chase::matrix::PseudoHermitian>::value)
        {
            is_sym_ = false;
            is_pseudoHerm_ = true;
        }
        else
        {
            is_sym_ = true;
            is_pseudoHerm_ = false;
        }

        const std::size_t block_size =
            is_pseudoHerm_ ? 2 * nevex_ : nevex_;
        ritzv_ = std::make_unique<chase::distMatrix::RedundantMatrix<
            chase::Base<T>, chase::platform::GPU>>(
            block_size, 1, block_size, ritzv, Hmat_->getMpiGrid_shared_ptr());
        resid_ = std::make_unique<chase::distMatrix::RedundantMatrix<
            chase::Base<T>, chase::platform::GPU>>(
            block_size, 1, Hmat_->getMpiGrid_shared_ptr());

        MPI_Comm_rank(Hmat_->getMpiGrid()->get_comm(), &my_rank_);
        MPI_Comm_size(Hmat_->getMpiGrid()->get_comm(), &nprocs_);
        coords_ = Hmat_->getMpiGrid()->get_coords();
        dims_ = Hmat_->getMpiGrid()->get_dims();

        stream_ = gpu_context_.stream();
        blasH_ = gpu_context_.blas();
        solverH_ = gpu_context_.solver();
        devInfo_ = gpu_context_.dev_info();


#ifdef XGEEV_EXISTS
#ifdef CHASE_OUTPUT
        if (my_rank_ == 0)
        {
            std::ostringstream oss;
            oss << "XGEEV ACTIVATED !\n";
            chase::GetLogger().Log(chase::LogLevel::Debug, "linalg", oss.str(), my_rank_);
        }
#endif
#endif

        runtime_type::device_malloc(reinterpret_cast<void**>(&d_return_),
                                    sizeof(T) * block_size);

        gpu_backend::SolverWorkspacePlan workspace_plan;

#if defined(HAS_HIP) && !defined(HAS_CUDA)
        // HIP distributed Householder QR uses ScaLAPACK; skip geqrf/gqr sizing.
#else
        gpu_backend::plan_query_qr_workspace<solver_dispatch, T>(
            workspace_plan, solverH_, N_, block_size, V1_->l_data(),
            V1_->l_ld(), d_return_);
#endif

        if constexpr (std::is_same<typename MatrixType::hermitian_type,
                                   chase::matrix::PseudoHermitian>::value)
        {
#if defined(XGEEV_EXISTS) && !(defined(HAS_HIP) && !defined(HAS_CUDA))
            A_ = std::make_unique<
                chase::distMatrix::RedundantMatrix<T, chase::platform::GPU>>(
                3 * 2 * nevex_, 2 * nevex_, Hmat_->getMpiGrid_shared_ptr());

            solver_dispatch::check_status(
                solver_dispatch::create_params(&params_));

            gpu_backend::plan_query_geev_workspace<solver_dispatch, T>(
                workspace_plan, solverH_, params_, block_size, A_->l_data(),
                A_->l_ld(), V2_->l_data(), V1_->l_ld());
            lhwork_ = workspace_plan.host_geev_lwork;

#ifdef CHASE_OUTPUT
            if (my_rank_ == 0)
            {
                std::ostringstream oss;
                oss << "GEEV GPU WORKSPACE SIZE = "
                    << workspace_plan.geev_device_lwork
                    << ", GEEV CPU WORKSPACE SIZE = " << lhwork_ << "\n";
                chase::GetLogger().Log(chase::LogLevel::Debug, "linalg",
                                       oss.str(), my_rank_);
            }
#endif
            h_work_ = std::unique_ptr<T[]>(new T[lhwork_]);
#else
            A_ = std::make_unique<
                chase::distMatrix::RedundantMatrix<T, chase::platform::GPU>>(
                2 * 2 * nevex_, 2 * nevex_, Hmat_->getMpiGrid_shared_ptr());

            gpu_backend::plan_query_heevd_workspace<solver_dispatch, T>(
                workspace_plan, solverH_, block_size, A_->l_data(), A_->l_ld(),
                ritzv_->l_data());
#ifdef CHASE_OUTPUT
            if (my_rank_ == 0)
            {
                std::ostringstream oss;
                oss << "HEEVD GPU WORKSPACE SIZE = "
                    << workspace_plan.heevd_lwork << "\n";
                chase::GetLogger().Log(chase::LogLevel::Debug, "linalg",
                                       oss.str(), my_rank_);
            }
#endif
#endif
        }
        else
        {
            A_ = std::make_unique<
                chase::distMatrix::RedundantMatrix<T, chase::platform::GPU>>(
                nevex_, nevex_, Hmat_->getMpiGrid_shared_ptr());

            gpu_backend::plan_query_heevd_workspace<solver_dispatch, T>(
                workspace_plan, solverH_, nevex_, A_->l_data(), A_->l_ld(),
                ritzv_->l_data());
        }

        gpu_backend::plan_query_potrf_workspace<solver_dispatch, T>(
            workspace_plan, solverH_, block_size, A_->l_data(), A_->l_ld());
        gpu_backend::plan_include_packed_triangular(workspace_plan, block_size);

        lwork_ = workspace_plan.device_element_count();
        gpu_context_.ensure_solver_plan<T>(workspace_plan);
        d_work_ = gpu_context_.template work<T>();

        runtime_type::device_malloc(
            reinterpret_cast<void**>(&states_),
            sizeof(rand_state_type) * (256 * 32));

        std::vector<std::size_t> diag_xoffs, diag_yoffs;

        if constexpr (std::is_same<typename MatrixType::matrix_type,
                                   chase::distMatrix::BlockBlock>::value)
        {
            std::size_t* g_offs = Hmat_->g_offs();

            for (auto j = 0; j < Hmat_->l_cols(); j++)
            {
                for (auto i = 0; i < Hmat_->l_rows(); i++)
                {
                    if (g_offs[0] + i == g_offs[1] + j)
                    {
                        diag_xoffs.push_back(i);
                        diag_yoffs.push_back(j);
                    }
                }
            }
        }
        else if constexpr (std::is_same<typename MatrixType::matrix_type,
                                        chase::distMatrix::BlockCyclic>::value)
        {
            auto m_contiguous_global_offs = Hmat_->m_contiguous_global_offs();
            auto n_contiguous_global_offs = Hmat_->n_contiguous_global_offs();
            auto m_contiguous_local_offs = Hmat_->m_contiguous_local_offs();
            auto n_contiguous_local_offs = Hmat_->n_contiguous_local_offs();
            auto m_contiguous_lens = Hmat_->m_contiguous_lens();
            auto n_contiguous_lens = Hmat_->n_contiguous_lens();
            auto mblocks = Hmat_->mblocks();
            auto nblocks = Hmat_->nblocks();

            for (std::size_t j = 0; j < nblocks; j++)
            {
                for (std::size_t i = 0; i < mblocks; i++)
                {
                    for (std::size_t q = 0; q < n_contiguous_lens[j]; q++)
                    {
                        for (std::size_t p = 0; p < m_contiguous_lens[i]; p++)
                        {
                            if (q + n_contiguous_global_offs[j] ==
                                p + m_contiguous_global_offs[i])
                            {
                                diag_xoffs.push_back(
                                    p + m_contiguous_local_offs[i]);
                                diag_yoffs.push_back(
                                    q + n_contiguous_local_offs[j]);
                            }
                        }
                    }
                }
            }
        }
        else
        {
            throw std::runtime_error("Matrix type is not supported");
        }

        diag_cnt = diag_xoffs.size();

        runtime_type::device_malloc(reinterpret_cast<void**>(&d_diag_xoffs),
                                    sizeof(std::size_t) * diag_cnt);
        runtime_type::device_malloc(reinterpret_cast<void**>(&d_diag_yoffs),
                                    sizeof(std::size_t) * diag_cnt);

        runtime_type::memcpy_h2d(d_diag_xoffs, diag_xoffs.data(),
                                 sizeof(std::size_t) * diag_cnt);
        runtime_type::memcpy_h2d(d_diag_yoffs, diag_yoffs.data(),
                                 sizeof(std::size_t) * diag_cnt);

#ifdef QR_DOUBLE_PRECISION
        if constexpr (std::is_same<T, std::complex<float>>::value ||
                      std::is_same<T, float>::value)
        {
            if (!V1_->isDoublePrecisionEnabled())
            {
                V1_->enableDoublePrecision();
            }

            runtime_type::device_malloc(
                reinterpret_cast<void**>(&d_work_d),
                sizeof(typename chase::ToDoublePrecisionTrait<T>::Type) *
                    lwork_);
        }
        if constexpr (std::is_same<T, std::complex<float>>::value ||
                      std::is_same<T, float>::value)
        {
            if (!A_->isDoublePrecisionEnabled())
            {
                A_->enableDoublePrecision();
            }
        }
#elif RR_DOUBLE_PRECISION
        if constexpr (std::is_same<T, std::complex<float>>::value ||
                      std::is_same<T, float>::value)
        {
            if (!A_->isDoublePrecisionEnabled())
            {
                A_->enableDoublePrecision();
            }
        }
#endif

#if defined(HAS_NCCL) && defined(CHASE_GPU_CONSTRUCTOR_BENCHMARK_WARMUP)
        // NCCL benchmark warm-up (optional; CMake: CHASE_GPU_CONSTRUCTOR_BENCHMARK_WARMUP=ON,
        // not combined with Release/RelWithDebInfo). Adds measurable startup cost.
        if constexpr (std::is_same<backend, chase::grid::backend::NCCL>::value)
        {
            if (my_rank_ == 0)
            {
                std::cerr
                    << "[ChASE][WARN] pChASEGPU constructor NCCL benchmark warm-up is enabled "
                       "(extra cholQR/Lanczos work at startup). To disable: configure with "
                       "-DCHASE_GPU_CONSTRUCTOR_BENCHMARK_WARMUP=OFF, or build with "
                       "CMAKE_BUILD_TYPE=Release or RelWithDebInfo (warm-up is never enabled "
                       "for those build types).\n";
                std::cerr.flush();
            }

            event_type start, stop;
            runtime_type::event_create(&start);
            runtime_type::event_create(&stop);

            std::size_t warmup_rows = 1;
            std::size_t warmup_cols = 1;

            // Warm up cholQR1 (SYHERK/AllReduce/Cholesky/TRSM on column comm)
            for (int i = 0; i < 3; i++)
            {
                runtime_type::event_record(start);

                int info = kernelNamespace::cholQR1(
                    blasH_, solverH_, V1_->l_rows(), V1_->l_cols(),
                    V1_->l_data(), V1_->l_ld(),
                    MGPUKernelNamspaceSelector<backend>::getColCommunicator(
                        V1_->getMpiGrid()),
                    d_work_, lwork_, A_->l_data());

                runtime_type::event_record(stop);
                runtime_type::event_synchronize(stop);

                float ms = 0.0f;
                runtime_type::event_elapsed_time(&ms, start, stop);

                std::ostringstream oss;
                oss << "[cholQR1 warm-up " << (i+1) << "/3] 1x1 matrix: "
                    << ms << " ms (info=" << info << ")\n";
                chase::GetLogger().Log(chase::LogLevel::Debug, "linalg", oss.str(), my_rank_);
            }

            // Warm up Lanczos (MatMul + redistribute P2P/AlltoAll patterns)
            chase::Base<T> dummy_upperb = 1.0;
            for (int i = 0; i < 3; i++)
            {
                runtime_type::event_record(start);
                stream_type saved_stream = nullptr;
                blas_dispatch::check_status(
                    blas_dispatch::get_stream(blasH_, &saved_stream));
                // Minimal Lanczos: 1 iteration to warm up redistribute patterns
                //kernelNamespace::lanczos_dispatch(blasH_, 1, *Hmat_, *V1_, &dummy_upperb);
                std::vector<Base<T>> Theta(40 * 20);
                std::vector<Base<T>> Tau(40 * 20);
                std::vector<Base<T>> ritzV(40 * 40);
                kernelNamespace::lanczos_dispatch(blasH_, 40, 20, *Hmat_, *V1_, &dummy_upperb, Theta.data(), Tau.data(), ritzV.data());

                blas_dispatch::check_status(
                    blas_dispatch::set_stream(blasH_, saved_stream));
                runtime_type::event_record(stop);
                runtime_type::event_synchronize(stop);

                float ms = 0.0f;
                runtime_type::event_elapsed_time(&ms, start, stop);
#ifdef CHASE_OUTPUT
                if (my_rank_ == 0)
                {
                    std::ostringstream oss;
                    oss << "[Lanczos warm-up " << (i+1) << "/3] 1 iteration: "
                        << ms << " ms\n";
                    chase::GetLogger().Log(chase::LogLevel::Debug, "linalg", oss.str(), my_rank_);
                }
#endif
            }

            runtime_type::event_destroy(start);
            runtime_type::event_destroy(stop);
        }
#endif // HAS_NCCL && CHASE_GPU_CONSTRUCTOR_BENCHMARK_WARMUP

#ifdef CHASE_ENABLE_OZAKI
        char const* ozaki_disable = std::getenv("CHASE_DISABLE_OZAKI");
        if (ozaki_disable == nullptr || std::atoi(ozaki_disable) == 0)
        {
            runtime_type::device_malloc(reinterpret_cast<void**>(&d_one_norm_H_),
                                        sizeof(chase::Base<T>));
            runtime_type::device_malloc(reinterpret_cast<void**>(&d_inf_norm_H_),
                                        sizeof(chase::Base<T>));
            ozaki_save_memory_ = chase::linalg::gemmul8pp::has_memory_saving();
            char const* env = std::getenv("CHASE_OZAKI_SKIP_SCALING");
            if (env != nullptr)
                ozaki_skip_scaling_ = static_cast<bool>(std::atoi(env));
            env = std::getenv("CHASE_OZAKI_DISABLE_SAVE_MEMORY");
            if ((env != nullptr && std::atoi(env) != 0) || ozaki_skip_scaling_ || !ozaki_save_memory_)
                ozaki_save_memory_ = false;
            else
                initOzaki_II_allocate();
        }
#endif
    }

    pChASEGPU(const pChASEGPU&) = delete;

    ~pChASEGPU()
    {
        SCOPED_NVTX_RANGE();

        if (d_return_)
            runtime_type::device_free(d_return_);
        if (states_)
            runtime_type::device_free(states_);
        if (d_diag_xoffs)
            runtime_type::device_free(d_diag_xoffs);
        if (d_diag_yoffs)
            runtime_type::device_free(d_diag_yoffs);

#ifdef QR_DOUBLE_PRECISION
        if constexpr (std::is_same<T, std::complex<float>>::value ||
                      std::is_same<T, float>::value)
        {
            if (V1_->isDoublePrecisionEnabled())
            {
                V1_->disableDoublePrecision();
            }

            runtime_type::device_free(d_work_d);
        }
        if constexpr (std::is_same<T, std::complex<float>>::value ||
                      std::is_same<T, float>::value)
        {
            if (A_->isDoublePrecisionEnabled())
            {
                A_->disableDoublePrecision();
            }
        }
#elif RR_DOUBLE_PRECISION
        if constexpr (std::is_same<T, std::complex<float>>::value ||
                      std::is_same<T, float>::value)
        {
            if (A_->isDoublePrecisionEnabled())
            {
                A_->disableDoublePrecision();
            }
        }
#endif

#ifdef CHASE_ENABLE_OZAKI
    chase::linalg::gemmul8pp::clear_config(blasH_);
    if (ozaki_workspace_)
        runtime_type::device_free(ozaki_workspace_);
    if (d_one_norm_H_)
        runtime_type::device_free(d_one_norm_H_);
    if (d_inf_norm_H_)
        runtime_type::device_free(d_inf_norm_H_);
#endif
    }

    std::size_t GetN() const override { return N_; }

    std::size_t GetNev() override { return nev_; }

    std::size_t GetNex() override { return nex_; }

    std::size_t GetRitzvBlockSize() const override
    {
        return is_pseudoHerm_ ? 2 * nevex_ : nevex_;
    }

    std::size_t GetLanczosIter() override { return lanczosIter_; }

    std::size_t GetNumLanczos() override { return numLanczos_; }

    chase::Base<T>* GetRitzv() override { return ritzv_->cpu_data(); }
    chase::Base<T>* GetResid() override
    {
        resid_->allocate_cpu_data();
        return resid_->cpu_data();
    }
    ChaseConfig<T>& GetConfig() override { return config_; }
    int get_nprocs() override { return nprocs_; }
    int get_rank() override { return my_rank_; }

    void loadProblemFromFile(std::string filename)
    {
        SCOPED_NVTX_RANGE();
        Hmat_->readFromBinaryFile(filename);
    }

    void saveProblemToFile(std::string filename)
    {
        Hmat_->saveToBinaryFile(filename);
    }

#ifdef CHASE_OUTPUT
    //! Print some intermediate infos during the solving procedure
    void Output(LogLevel level, std::string str,
                const char* category = "algorithm") override
    {
        chase::GetLogger().Log(level, category, str, get_rank());
    }
#endif
    bool checkSymmetryEasy() override
    {
        SCOPED_NVTX_RANGE();

        is_sym_ = kernelNamespace::checkSymmetryEasy(blasH_, *Hmat_);
        return is_sym_;
    }

    bool isSym() override { return is_sym_; }

    bool checkPseudoHermicityEasy() override
    {
        SCOPED_NVTX_RANGE();
        // is_pseudoHerm_= 0;
        return is_pseudoHerm_;
    }

    bool isPseudoHerm() override { return is_pseudoHerm_; }

    void symOrHermMatrix(char uplo) override
    {
        SCOPED_NVTX_RANGE();

        kernelNamespace::symOrHermMatrix(uplo, *Hmat_);
    }

    void Start() override { locked_ = 0; }

    void initVecs(bool random) override
    {
        SCOPED_NVTX_RANGE();

        if (random)
        {
            int mpi_col_rank;
            MPI_Comm_rank(Hmat_->getMpiGrid()->get_col_comm(), &mpi_col_rank);
            unsigned long long seed = 1337 + mpi_col_rank;

            kernel_dispatch::fill_random(seed, states_, V1_->l_data(),
                                         V1_->l_ld() * V1_->l_cols());

            // Pseudo-Hermitian: dampen components in the global lower half of rows
            // (same split as l_half()); standard Hermitian init leaves vectors unchanged.
            if (is_pseudoHerm_)
            {
                const std::size_t m_lower = V1_->l_rows() - V1_->l_half();
                if (m_lower > 0)
                {
                    const T scale = T(0.001);
                    kernel_dispatch::scale_lower_block_rows(
                        V1_->l_data(), V1_->l_ld(), V1_->l_half(), m_lower,
                        V1_->l_cols(), scale);
                }
            }
        }


        kernel_dispatch::lacpy(
            'A', V1_->l_rows(), V1_->l_cols(), V1_->l_data(), V1_->l_ld(),
            V2_->l_data(), V2_->l_ld());

        Hmat_->H2D();
        next_ = NextOp::bAc;
    }

#ifdef CHASE_ENABLE_OZAKI
    unsigned GetOzakiNumModuli() const override { return ozaki_num_moduli_; }
    void SetOzakiNumModuli(unsigned num_moduli) override
    {
        ozaki_num_moduli_ = num_moduli;
        ozaki_skip_scaling_activated_ = false;
    }
    void Activate_HighPrecision_Fallback() override
    {
        if (ozaki_fp_hemm_fallback || ozaki_fallback_disabled_)
            return;
        if (ozaki_fallback_num_moduli_ < 0)
            ozaki_fp_hemm_fallback = true;
        else
            SetOzakiNumModuli(
                static_cast<unsigned>(ozaki_fallback_num_moduli_));
    }
    bool IsOzakiFpHemmFallback() const override
    {
        return ozaki_fp_hemm_fallback;
    }

    void initOzaki_II_allocate()
    {
        chase::linalg::gemmul8pp::set_memory_saving(blasH_, ozaki_save_memory_);
        chase::linalg::gemmul8pp::set_max_worksize(blasH_, ozaki_max_worksize_);

        std::size_t needed = 0;
        if (ozaki_save_memory_)
        {
            needed = ozaki_max_worksize_;
        }
        else
        {
            std::size_t const worksize_nn = chase::linalg::gemmul8pp::workSize<T>(Hmat_->l_rows(), V1_->l_cols(), Hmat_->l_cols(), ozaki_num_moduli_, true, true);
            std::size_t const worksize_cn = chase::linalg::gemmul8pp::workSize<T>(Hmat_->l_cols(), V1_->l_cols(), Hmat_->l_rows(), ozaki_num_moduli_, true, true);
            needed = std::max(worksize_nn, worksize_cn);
        }

        if (ozaki_workspace_ != nullptr && ozaki_workspace_size_ >= needed)
            return;

        if (ozaki_workspace_ != nullptr)
        {
            runtime_type::device_free(ozaki_workspace_);
            ozaki_workspace_ = nullptr;
            ozaki_workspace_size_ = 0;
        }

        ozaki_workspace_size_ = needed;

        size_t free_bytes = 0, total_bytes = 0;
        CHECK_CUDA_ERROR(cudaMemGetInfo(&free_bytes, &total_bytes));
        if (needed >= free_bytes)
        {
            ozaki_fp_hemm_fallback = true;
            ozaki_workspace_size_ = 0;
            return;
        }

        void* ozaki_workspace_tmp = nullptr;
        runtime_type::device_malloc(&ozaki_workspace_tmp, ozaki_workspace_size_);
        ozaki_workspace_ = reinterpret_cast<int8_t*>(ozaki_workspace_tmp);
    }

    // Tolerance Aware Flexible-Accuracy moduli number selectionfor Ozaki-II for Hermitian matrices only. 
    //
    // WARNING : The parallel reduction algorithm is assumed exact. 
    // The global E^f quantity takes the max local E^f quantity across ranks, and multiplies by p = max(pr, pc).
    // 
    // The working moduli number is the smallest with ||E^f|| < tol.
    // The fallback moduli number is the smallest count with Safe Res < tol, active when residual stagnation happens.
    // If none qualifies, then max moduli number is used. 
    // If none qualifies, and CHASE_OZAKI_FP64_FALLBACK is set, then FP64-GEMM is used.
    //
    // Ā = (H − cI) / e, with c = (upper + lower) / 2 and e = (upper − lower) / 2.
    // rho_A = ||H−cI||_∞ / |e| stands in for ||Ā||_2. Columns of V are taken as unit.
    // Complex H uses ef_scale = 2 (see https://arxiv.org/abs/2609.05419 Eq. 12); real H keeps ef_scale = 1.
    // ||E^f|| is ef_scale * Ozaki-II (1,2)-norm upper bound at each moduli count, and
    // ef_scale * the IEEE γ_n bound at the FP64 slot.
    //
    // ||E^p|| = ||E^f|| * u2_model_sum(degree, λ_1, lower, upper).
    // Safe Res = ||E^p|| * |upper − λ_1|.
    //
    // TODO : 
    // - account for reduction error.
    // - cite the paper.
    // - double-check the ef_scale = 2 in complex case. 
    // - replace the variation via coprime moduli number by required mantissa bits q. 
    void initOzaki_II_Select_Moduli_Number(std::size_t degree, chase::Base<T> lambda_1, chase::Base<T> lower, chase::Base<T> upper, double* bounds, double* ep_bounds, double eigen_distance)
    {
        if (ozaki_fp_hemm_fallback)
            return;

        unsigned const moduli_number_min = chase::linalg::gemmul8pp::kMinNumModuliChase<T>;
        unsigned const moduli_number_max = chase::linalg::gemmul8pp::kMaxNumModuli<T>;
        unsigned const fp64_idx = moduli_number_max - moduli_number_min + 1;

        double const u2 = chase::linalg::gemmul8pp::u2_model_sum(degree, static_cast<double>(lambda_1), static_cast<double>(lower), static_cast<double>(upper));

        chase::Base<T> const c = (upper + lower) / chase::Base<T>(2);
        chase::Base<T> const e = (upper - lower) / chase::Base<T>(2);

        // Global diagonal only. A rank that owns none of it leaves its block unchanged.
        kernelNamespace::shiftDiagonal(*Hmat_, d_diag_xoffs, d_diag_yoffs, diag_cnt, -c);

        // Probe Ā = (H-cI)/e via inv_e in A sft/pack (no device A/=e).
        double const abs_e = std::abs(static_cast<double>(e));
        double const inv_e = 1.0 / abs_e;
        double nrm2_a = 0.0;
        double max2_b = 0.0;
        chase::linalg::gemmul8pp::ozaki_compute_scaling_factors_error_bound(blasH_, Hmat_->l_data(), Hmat_->l_ld(), Hmat_->l_rows(), Hmat_->l_cols(), V1_->l_data(), V1_->l_ld(), V1_->l_cols(), ozaki_workspace_, ozaki_workspace_size_, &nrm2_a, &max2_b, inv_e);

        // ||Ā||_2 <= sqrt(||H−cI||_1 · ||H−cI||_∞) / |e| on this rank's block.
        chase::Base<T> one_norm_H = chase::Base<T>(0);
        chase::Base<T> inf_norm_H = chase::Base<T>(0);
        kernel_dispatch::matrix_one_norm(Hmat_->l_data(), Hmat_->l_rows(), Hmat_->l_cols(), Hmat_->l_ld(), d_one_norm_H_, stream_);
        kernel_dispatch::matrix_inf_norm(Hmat_->l_data(), Hmat_->l_rows(), Hmat_->l_cols(), Hmat_->l_ld(), d_inf_norm_H_, stream_);
        runtime_type::memcpy_d2h(&one_norm_H, d_one_norm_H_, sizeof(chase::Base<T>));
        runtime_type::memcpy_d2h(&inf_norm_H, d_inf_norm_H_, sizeof(chase::Base<T>));

        kernelNamespace::shiftDiagonal(*Hmat_, d_diag_xoffs, d_diag_yoffs, diag_cnt, c);

        double const rho_A = std::sqrt(static_cast<double>(one_norm_H) * static_cast<double>(inf_norm_H)) / abs_e;
        std::size_t const k_bound = std::max(Hmat_->l_rows(), Hmat_->l_cols());
        // Complex GEMM: local ||E^f|| is twice the real bound. ||E^p|| follows.
        constexpr double ef_scale = (std::is_same<T, std::complex<float>>::value || std::is_same<T, std::complex<double>>::value) ? 2.0 : 1.0;
        for (unsigned selected_moduli_number = moduli_number_max; selected_moduli_number >= moduli_number_min; --selected_moduli_number)
        {
            bounds[selected_moduli_number - moduli_number_min] = ef_scale * chase::linalg::gemmul8pp::ozaki_gemm_error_bound<T>(rho_A, nrm2_a, max2_b, k_bound, selected_moduli_number);
            ep_bounds[selected_moduli_number - moduli_number_min] = bounds[selected_moduli_number - moduli_number_min] * u2;
        }
        bounds[fp64_idx] = ef_scale * chase::linalg::gemmul8pp::ieee_gemm_error_bound_12(rho_A, k_bound);
        ep_bounds[fp64_idx] = bounds[fp64_idx] * u2;

        // Local bounds above. One max across ranks, then p = max(pr, pc).
        unsigned const n_bounds = fp64_idx + 1;
        MPI_Allreduce(MPI_IN_PLACE, bounds, static_cast<int>(n_bounds), MPI_DOUBLE, MPI_MAX, Hmat_->getMpiGrid()->get_comm());
        int const* grid_dims = Hmat_->getMpiGrid()->get_dims();
        double const p = static_cast<double>(std::max(grid_dims[0], grid_dims[1]));
        for (unsigned i = 0; i < n_bounds; ++i)
        {
            bounds[i] *= p;
            ep_bounds[i] = bounds[i] * u2;
        }

        if (!ozaki_fallback_disabled_)
        {
            double const tol_d = static_cast<double>(GetConfig().GetTol());
            ozaki_num_moduli_ = moduli_number_max;
            ozaki_fallback_num_moduli_ = static_cast<int>(moduli_number_max);
            // Descending so the last hit is the cheapest slice under tol.
            for (unsigned selected_moduli_number = moduli_number_max; selected_moduli_number >= moduli_number_min; --selected_moduli_number)
            {
                if (bounds[selected_moduli_number - moduli_number_min] < tol_d)
                    ozaki_num_moduli_ = selected_moduli_number;
                if ((ep_bounds[selected_moduli_number - moduli_number_min] * eigen_distance) < tol_d)
                    ozaki_fallback_num_moduli_ = static_cast<int>(selected_moduli_number);
            }
            // Exact-size workspace may need to grow after bound-based moduli.
            if (!ozaki_save_memory_)
                initOzaki_II_allocate();

            // FP-GEMM only when no slice meets Safe Res < tol and it is at least as accurate.
            char const* env = std::getenv("CHASE_OZAKI_FP64_FALLBACK");
            bool const fp64_fallback_enabled = (env != nullptr && std::atoi(env) != 0);
            double const fallback_safe_res = ep_bounds[ozaki_fallback_num_moduli_ - moduli_number_min] * eigen_distance;
            if (fp64_fallback_enabled && fallback_safe_res >= tol_d && fallback_safe_res >= (ep_bounds[fp64_idx] * eigen_distance))
                ozaki_fallback_num_moduli_ = -1;
        }
        else
        {
            ozaki_fallback_num_moduli_ = -1;
        }
    }

#ifdef CHASE_OUTPUT
    void initOzaki_II_print(unsigned moduli_number_min, unsigned moduli_number_max, double const* bounds, double const* ep_bounds, double eigen_distance)
    {
        if (my_rank_ != 0)
            return;

        unsigned const fp64_idx = moduli_number_max - moduli_number_min + 1;
        char const* env = std::getenv("CHASE_OZAKI_FP64_FALLBACK");
        bool const fp64_fallback_enabled = (env != nullptr && std::atoi(env) != 0);

        size_t free_bytes = 0, total_bytes = 0;
        CHECK_CUDA_ERROR(cudaMemGetInfo(&free_bytes, &total_bytes));

        std::ostringstream oss;
        double const gib = 1024.0 * 1024.0 * 1024.0;
        std::string const status = ozaki_fp_hemm_fallback ? "disabled (FP HEMM fallback)" : "enabled";
        std::string const fallback = ozaki_fallback_disabled_ ? "off (fixed)" : (ozaki_fallback_num_moduli_ < 0 ? "FP-GEMM" : std::to_string(ozaki_fallback_num_moduli_));
        std::string const save_memory = !chase::linalg::gemmul8pp::has_memory_saving() ? "off (not available in GEMMul8)" : (ozaki_save_memory_ ? "on" : "off");
        std::ostringstream workspace_ss;
        workspace_ss << std::fixed << std::setprecision(2) << ozaki_workspace_size_ / gib << " GiB";
        std::ostringstream free_ss;
        free_ss << std::fixed << std::setprecision(2) << free_bytes / gib << " GiB";
        std::string const path = chase::linalg::gemmul8pp::gemmul8_root();

        std::size_t const key_w = 15;
        std::size_t val_w = status.size();
        val_w = std::max(val_w, fallback.size());
        val_w = std::max(val_w, std::size_t(3)); // fp64_fallback on/off
        val_w = std::max(val_w, save_memory.size());
        val_w = std::max(val_w, workspace_ss.str().size());
        val_w = std::max(val_w, free_ss.str().size());
        val_w = std::max(val_w, path.size());
        val_w = std::max(val_w, std::size_t(8));

        auto const settings_hline = [&]() { oss << "  +" << std::string(key_w + 2, '-') << "+" << std::string(val_w + 2, '-') << "+\n"; };
        auto const settings_row = [&](char const* key, std::string const& val) { oss << "  | " << std::left << std::setw(static_cast<int>(key_w)) << key << " | " << std::left << std::setw(static_cast<int>(val_w)) << val << " |\n"; };

        oss << "Ozaki-II GEMMul8\n";
        settings_hline();
        settings_row("status", status);
        settings_row("num_moduli", std::to_string(ozaki_num_moduli_));
        settings_row("fallback_moduli", fallback);
        settings_row("fp64_fallback", fp64_fallback_enabled ? "on" : "off");
        settings_row("skip_scaling", ozaki_skip_scaling_ ? "on" : "off");
        settings_row("save_memory", save_memory);
        settings_row("workspace", workspace_ss.str());
        settings_row("gpu_free", free_ss.str());
        settings_row("path", path);
        settings_hline();

        if (!ozaki_fp_hemm_fallback)
        {
            auto const hline = [&]() {
                oss << "  +------------+";
                for (unsigned selected_moduli_number = moduli_number_min; selected_moduli_number <= moduli_number_max; ++selected_moduli_number) oss << "----------+";
                oss << "----------+\n";
            };
            hline();
            oss << "  | " << std::left << std::setw(10) << "num_moduli" << " |";
            for (unsigned selected_moduli_number = moduli_number_min; selected_moduli_number <= moduli_number_max; ++selected_moduli_number) oss << std::right << std::setw(9) << selected_moduli_number << " |";
            oss << std::right << std::setw(9) << "FP64" << " |\n";
            hline();
            oss << std::scientific << std::setprecision(2);
            oss << "  | " << std::left << std::setw(10) << "||E^f|| <" << " |";
            for (unsigned selected_moduli_number = moduli_number_min; selected_moduli_number <= moduli_number_max; ++selected_moduli_number) oss << std::right << std::setw(9) << bounds[selected_moduli_number - moduli_number_min] << " |";
            oss << std::right << std::setw(9) << bounds[fp64_idx] << " |\n";
            oss << "  | " << std::left << std::setw(10) << "||E^p|| <" << " |";
            for (unsigned selected_moduli_number = moduli_number_min; selected_moduli_number <= moduli_number_max; ++selected_moduli_number) oss << std::right << std::setw(9) << ep_bounds[selected_moduli_number - moduli_number_min] << " |";
            oss << std::right << std::setw(9) << ep_bounds[fp64_idx] << " |\n";
            oss << "  | " << std::left << std::setw(10) << "Safe Res <" << " |";
            for (unsigned selected_moduli_number = moduli_number_min; selected_moduli_number <= moduli_number_max; ++selected_moduli_number) oss << std::right << std::setw(9) << (ep_bounds[selected_moduli_number - moduli_number_min] * eigen_distance) << " |";
            oss << std::right << std::setw(9) << (ep_bounds[fp64_idx] * eigen_distance) << " |\n";
            hline();
        }
        oss << std::endl;
        chase::GetLogger().Log(ozaki_fp_hemm_fallback ? chase::LogLevel::Warn : chase::LogLevel::Info, "linalg", oss.str(), 0);
    }
#endif

    void initOzaki_II(std::size_t degree, chase::Base<T> lambda_1, chase::Base<T> lower, chase::Base<T> upper) override
    {
        SCOPED_NVTX_RANGE();

        char* env = std::getenv("CHASE_DISABLE_OZAKI");
        if (env != nullptr && std::atoi(env) != 0)
        {
            ozaki_fp_hemm_fallback = true;
            if (my_rank_ == 0)
            {
                chase::GetLogger().Log(
                    chase::LogLevel::Info, "linalg",
                    "Ozaki-II disabled by CHASE_DISABLE_OZAKI; GEMM uses cuBLAS\n",
                    0);
            }
            return;
        }

        env = std::getenv("CHASE_OZAKI_NUM_MODULI");
        ozaki_fallback_disabled_ = (env != nullptr);
        if (ozaki_fallback_disabled_)
        {
            ozaki_num_moduli_ = static_cast<unsigned>(std::atoi(env));
        }
        else
        {
            double tol = GetConfig().GetTol();
            ozaki_num_moduli_ = static_cast<unsigned>(std::min(static_cast<int>(chase::linalg::gemmul8pp::kMaxNumModuli<T>), 4 + static_cast<int>(std::ceil(-std::log10(tol)))));
        }

        ozaki_save_memory_ = chase::linalg::gemmul8pp::has_memory_saving();
        env = std::getenv("CHASE_OZAKI_SKIP_SCALING");
        if (env != nullptr)
            ozaki_skip_scaling_ = static_cast<bool>(std::atoi(env));
        env = std::getenv("CHASE_OZAKI_DISABLE_SAVE_MEMORY");
        if ((env != nullptr && std::atoi(env) != 0) || ozaki_skip_scaling_)
            ozaki_save_memory_ = false;

        // save_memory: allocated in constructor. Exact size: after moduli selection.
        if (!ozaki_save_memory_ || ozaki_workspace_ == nullptr)
            initOzaki_II_allocate();

        unsigned const moduli_number_min = chase::linalg::gemmul8pp::kMinNumModuliChase<T>;
        unsigned const moduli_number_max = chase::linalg::gemmul8pp::kMaxNumModuli<T>;
        double bounds[chase::linalg::gemmul8pp::kMaxNumModuliFp64 - chase::linalg::gemmul8pp::kMinNumModuliChaseFp32 + 2] = {};
        double ep_bounds[chase::linalg::gemmul8pp::kMaxNumModuliFp64 - chase::linalg::gemmul8pp::kMinNumModuliChaseFp32 + 2] = {};
        // Safe Res := ||E^p||_{1,2} * |λ_n − λ_1|.
        double const eigen_distance = std::abs(static_cast<double>(upper) - static_cast<double>(lambda_1));
        initOzaki_II_Select_Moduli_Number(degree, lambda_1, lower, upper, bounds, ep_bounds, eigen_distance);

#ifdef CHASE_OUTPUT
        initOzaki_II_print(moduli_number_min, moduli_number_max, bounds, ep_bounds, eigen_distance);
#endif
    }

#endif

    void ReinitColumns(std::size_t base_col, std::size_t const* col_indices,
                      std::size_t n_indices) override
    {
        SCOPED_NVTX_RANGE();
        if (n_indices == 0)
            return;
        int mpi_col_rank;
        MPI_Comm_rank(Hmat_->getMpiGrid()->get_col_comm(), &mpi_col_rank);
        unsigned long long base_seed = 1337 + static_cast<unsigned long long>(mpi_col_rank);
        for (std::size_t c = 0; c < n_indices; ++c)
        {
            std::size_t j = base_col + col_indices[c];
            unsigned long long seed = base_seed + static_cast<unsigned long long>(j) * 1000uLL;
            kernel_dispatch::fill_random(seed, states_,
                                         V1_->l_data() + j * V1_->l_ld(),
                                         V1_->l_rows());
        }
        for (std::size_t c = 0; c < n_indices; ++c)
        {
            std::size_t j = base_col + col_indices[c];
            kernel_dispatch::lacpy(
                'A', V1_->l_rows(), 1, V1_->l_data() + j * V1_->l_ld(),
                V1_->l_ld(), V2_->l_data() + j * V2_->l_ld(), V2_->l_ld());
        }
    }

    void Lanczos(std::size_t m, chase::Base<T>* upperb) override
    {
        SCOPED_NVTX_RANGE();

        lanczosIter_ = m;
        numLanczos_ = 1;

        stream_type saved_stream = nullptr;
        blas_dispatch::check_status(
            blas_dispatch::get_stream(blasH_, &saved_stream));
        kernelNamespace::lanczos_dispatch(blasH_, m, *Hmat_, *V1_, upperb);
        blas_dispatch::check_status(
            blas_dispatch::set_stream(blasH_, saved_stream));
    }

    void Lanczos(std::size_t M, std::size_t numvec, chase::Base<T>* upperb,
                 chase::Base<T>* ritzv, chase::Base<T>* Tau,
                 chase::Base<T>* ritzV) override
    {

        SCOPED_NVTX_RANGE();

        lanczosIter_ = M;
        numLanczos_ = numvec;

        stream_type saved_stream = nullptr;
        blas_dispatch::check_status(
            blas_dispatch::get_stream(blasH_, &saved_stream));
        kernelNamespace::lanczos_dispatch(blasH_, M, numvec, *Hmat_, *V1_,
                                          upperb, ritzv, Tau, ritzV);
        blas_dispatch::check_status(
            blas_dispatch::set_stream(blasH_, saved_stream));
    }

    void LanczosDos(std::size_t idx, std::size_t m, T* ritzVc) override
    {
        SCOPED_NVTX_RANGE();

        T One = T(1.0);
        T Zero = T(0.0);

        std::unique_ptr<T, runtime_device_deleter> d_ritzVc_ptr = nullptr;
        T* d_ritzVc_;
        runtime_type::device_malloc(reinterpret_cast<void**>(&d_ritzVc_),
                                    m * idx * sizeof(T));
        d_ritzVc_ptr.reset(d_ritzVc_);
        d_ritzVc_ = d_ritzVc_ptr.get();

        runtime_type::memcpy_h2d(d_ritzVc_, ritzVc, m * idx * sizeof(T));

        blas_dispatch::check_status(blas_dispatch::gemm(
            blasH_, blas_dispatch::op_n, blas_dispatch::op_n, V1_->l_rows(), idx, m, &One,
            V1_->l_data(), V1_->l_ld(), d_ritzVc_, m, &Zero, V2_->l_data(),
            V2_->l_ld()));

        kernel_dispatch::lacpy('A', V2_->l_rows(), m,
                                               V2_->l_data(), V2_->l_ld(),
                                               V1_->l_data(), V1_->l_ld());
    }

    void Shift(T c, bool isunshift = false) override
    {
        SCOPED_NVTX_RANGE();

        if (isunshift)
        {
            next_ = NextOp::bAc;
        }

        kernelNamespace::shiftDiagonal(*Hmat_, d_diag_xoffs, d_diag_yoffs,
                                       diag_cnt, std::real(c));
#ifdef CHASE_ENABLE_OZAKI
        ozaki_skip_scaling_activated_ = false;
#endif

#ifdef ENABLE_MIXED_PRECISION
        if constexpr (std::is_same<T, double>::value ||
                      std::is_same<T, std::complex<double>>::value)
        {
            auto min = *std::min_element(resid_->cpu_data() + locked_,
                                         resid_->cpu_data() + nev_);
            bool shouldEnableSP = (min > 1e-3 && !isunshift);
            auto updatePrecision = [&](auto& mat, bool copyback = false)
            {
                if (shouldEnableSP)
                {
                    mat->enableSinglePrecision();
                }
                else if (mat->isSinglePrecisionEnabled())
                {
                    mat->disableSinglePrecision(copyback);
                }
            };

            // Update precision for all matrices
            updatePrecision(Hmat_);
            updatePrecision(V1_, true); // Special case for V1_
            updatePrecision(W1_);

            // Message on enabling single precision
            if (shouldEnableSP && my_rank_ == 0 && !isunshift)
            {
                chase::GetLogger().Log(chase::LogLevel::Info, "linalg",
                    "Enable Single Precision in Filter", my_rank_);
            }
        }
#endif
    }

    void HEMM(std::size_t block, T alpha, T beta, std::size_t offset_left,
              std::size_t offset_right = 0) override
    {
        std::size_t ncols =
            (offset_right < block) ? (block - offset_right) : std::size_t(0);
        if (ncols == 0)
        {
            next_ = (next_ == NextOp::bAc) ? NextOp::cAb : NextOp::bAc;
            return;
        }
#ifdef ENABLE_MIXED_PRECISION
        if constexpr (std::is_same<T, double>::value ||
                      std::is_same<T, std::complex<double>>::value)
        {
            using singlePrecisionT =
                typename chase::ToSinglePrecisionTrait<T>::Type;
            auto min = *std::min_element(resid_->cpu_data() + locked_,
                                         resid_->cpu_data() + nev_);

            if (min > 1e-3)
            {
                auto Hmat_sp = Hmat_->getSinglePrecisionMatrix();
                auto V1_sp = V1_->getSinglePrecisionMatrix();
                auto W1_sp = W1_->getSinglePrecisionMatrix();
                singlePrecisionT alpha_sp =
                    static_cast<singlePrecisionT>(alpha);
                singlePrecisionT beta_sp = static_cast<singlePrecisionT>(beta);

                if (next_ == NextOp::bAc)
                {
                    kernelNamespace::template MatrixMultiplyMultiVectors<
                        singlePrecisionT>(blasH_, &alpha_sp, *Hmat_sp, *V1_sp,
                                          &beta_sp, *W1_sp,
                                          offset_left + locked_, ncols);
                    next_ = NextOp::cAb;
                }
                else
                {
                    kernelNamespace::template MatrixMultiplyMultiVectors<
                        singlePrecisionT>(blasH_, &alpha_sp, *Hmat_sp, *W1_sp,
                                          &beta_sp, *V1_sp,
                                          offset_left + locked_, ncols);
                    next_ = NextOp::bAc;
                }
            }
            else
            {
                if (next_ == NextOp::bAc)
                {
                    kernelNamespace::MatrixMultiplyMultiVectors(
                        blasH_, &alpha, *Hmat_, *V1_, &beta, *W1_,
                        offset_left + locked_, ncols);
                    next_ = NextOp::cAb;
                }
                else
                {
                    kernelNamespace::MatrixMultiplyMultiVectors(
                        blasH_, &alpha, *Hmat_, *W1_, &beta, *V1_,
                        offset_left + locked_, ncols);
                    next_ = NextOp::bAc;
                }
            }
        }
        else
#endif

        {
            if (next_ == NextOp::bAc)
            {
                kernelNamespace::MatrixMultiplyMultiVectors(
                    blasH_, &alpha, *Hmat_, *V1_, &beta, *W1_,
                    offset_left + locked_, ncols);
                next_ = NextOp::cAb;
            }
            else
            {
                kernelNamespace::MatrixMultiplyMultiVectors(
                    blasH_, &alpha, *Hmat_, *W1_, &beta, *V1_,
                    offset_left + locked_, ncols);
                next_ = NextOp::bAc;
            }
        }
    }

#ifdef CHASE_ENABLE_OZAKI
    void HEMM_Ozaki(std::size_t block, T alpha, T beta, std::size_t offset_left,
                    std::size_t offset_right = 0) override
    {
#if !defined(HAS_CUDA)
        HEMM(block, alpha, beta, offset_left, offset_right);
#else
        if(ozaki_fp_hemm_fallback)
        {
            HEMM(block, alpha, beta, offset_left, offset_right);
            return;
        }

        std::size_t ncols =
            (offset_right < block) ? (block - offset_right) : std::size_t(0);
        if (ncols == 0)
        {
            next_ = (next_ == NextOp::bAc) ? NextOp::cAb : NextOp::bAc;
            return;
        }
#ifdef ENABLE_MIXED_PRECISION
        if constexpr (std::is_same<T, double>::value ||
                      std::is_same<T, std::complex<double>>::value)
        {
            using singlePrecisionT =
                typename chase::ToSinglePrecisionTrait<T>::Type;
            auto min = *std::min_element(resid_->cpu_data() + locked_,
                                         resid_->cpu_data() + nev_);

            if (min > 1e-3)
            {
                auto Hmat_sp = Hmat_->getSinglePrecisionMatrix();
                auto V1_sp = V1_->getSinglePrecisionMatrix();
                auto W1_sp = W1_->getSinglePrecisionMatrix();
                singlePrecisionT alpha_sp =
                    static_cast<singlePrecisionT>(alpha);
                singlePrecisionT beta_sp = static_cast<singlePrecisionT>(beta);

                if (next_ == NextOp::bAc)
                {
                    kernelNamespace::template MatrixMultiplyMultiVectors<
                        singlePrecisionT>(blasH_, &alpha_sp, *Hmat_sp, *V1_sp,
                                          &beta_sp, *W1_sp,
                                          offset_left + locked_, ncols);
                    next_ = NextOp::cAb;
                }
                else
                {
                    kernelNamespace::template MatrixMultiplyMultiVectors<
                        singlePrecisionT>(blasH_, &alpha_sp, *Hmat_sp, *W1_sp,
                                          &beta_sp, *V1_sp,
                                          offset_left + locked_, ncols);
                    next_ = NextOp::bAc;
                }
            }
            else
            {
                if (next_ == NextOp::bAc)
                {
                    kernelNamespace::MatrixMultiplyMultiVectorsWithOzakiII(
                        blasH_, &alpha, *Hmat_, *V1_, &beta, *W1_,
                        offset_left + locked_, ncols, ozaki_num_moduli_, ozaki_workspace_);
                    next_ = NextOp::cAb;
                }
                else
                {
                    kernelNamespace::MatrixMultiplyMultiVectorsWithOzakiII(
                        blasH_, &alpha, *Hmat_, *W1_, &beta, *V1_,
                        offset_left + locked_, ncols, ozaki_num_moduli_, ozaki_workspace_);
                    next_ = NextOp::bAc;
                }
            }
        }
        else
#endif
        {
            if (next_ == NextOp::bAc)
            {
                kernelNamespace::MatrixMultiplyMultiVectorsWithOzakiII(
                    blasH_, &alpha, *Hmat_, *V1_, &beta, *W1_,
                    offset_left + locked_, ncols, ozaki_num_moduli_, ozaki_workspace_);
                next_ = NextOp::cAb;
            }
            else
            {
                kernelNamespace::MatrixMultiplyMultiVectorsWithOzakiII(
                    blasH_, &alpha, *Hmat_, *W1_, &beta, *V1_,
                    offset_left + locked_, ncols, ozaki_num_moduli_, ozaki_workspace_);
                next_ = NextOp::bAc;
            }
        }
#endif
    }
#endif

    void HEMM_H2(std::size_t block, T alpha, T beta, T gamma,
                 std::size_t offset_left,
                 std::size_t offset_right = 0) override
    {
        SCOPED_NVTX_RANGE();
        std::size_t ncols =
            (offset_right < block) ? (block - offset_right) : std::size_t(0);
        if (ncols == 0)
        {
            next_ = (next_ == NextOp::bAc) ? NextOp::cAb : NextOp::bAc;
            return;
        }
        const std::size_t col0 = offset_left + locked_;
        T one = T(1);
        T zero = T(0);

        if (next_ == NextOp::bAc)
        {
            kernelNamespace::MatrixMultiplyMultiVectors(
                blasH_, &one, *Hmat_, *V1_, &zero, *W1_, col0, ncols);
            kernelNamespace::MatrixMultiplyMultiVectors(
                blasH_, &alpha, *Hmat_, *W1_, &beta, *V2_, col0, ncols);
            chase::linalg::internal::active_gpu_serial::batchedAxpyScalar(
                gamma, V1_->l_data() + col0 * V1_->l_ld(),
                V2_->l_data() + col0 * V2_->l_ld(),
                static_cast<int>(V2_->l_rows()), static_cast<int>(ncols),
                static_cast<int>(V1_->l_ld()), static_cast<int>(V2_->l_ld()));
            next_ = NextOp::cAb;
        }
        else
        {
            kernelNamespace::MatrixMultiplyMultiVectors(
                blasH_, &one, *Hmat_, *V2_, &zero, *W1_, col0, ncols);
            kernelNamespace::MatrixMultiplyMultiVectors(
                blasH_, &alpha, *Hmat_, *W1_, &beta, *V1_, col0, ncols);
            chase::linalg::internal::active_gpu_serial::batchedAxpyScalar(
                gamma, V2_->l_data() + col0 * V2_->l_ld(),
                V1_->l_data() + col0 * V1_->l_ld(),
                static_cast<int>(V1_->l_rows()), static_cast<int>(ncols),
                static_cast<int>(V2_->l_ld()), static_cast<int>(V1_->l_ld()));
            next_ = NextOp::bAc;
        }
    }

    void FilterPhaseEnd() override
    {
        runtime_type::device_synchronize();
    }

    void ApplyKconjugate(std::size_t block) override {
        if constexpr (std::is_same<typename MatrixType::hermitian_type,
            chase::matrix::PseudoHermitian>::value)
        {
            V1_->Kconjugate(block, locked_);
            // Symmetric locking: layout [locked_ | first half | second half | locked_].
            // Rows: 0..N/2-1 = upper block, N/2..N-1 = lower block.
            // Active first half [locked_, locked_+block), second half [locked_+block, locked_+2*block).
            // K-conjugation: set second-half cols = conjugate(first-half) with block swap.

            //First, we need to know to whom each rank has to send and receive data.
        }
    }

    void QR(chase::Base<T> cond) override
    {
        SCOPED_NVTX_RANGE();
        // Create CUDA events for detailed timing
        event_type qr_start, qr_flip_sign, qr_cholqr_start, qr_cholqr_end, qr_lacpy_start, qr_end;
        runtime_type::event_create(&qr_start);
        runtime_type::event_create(&qr_flip_sign);
        runtime_type::event_create(&qr_cholqr_start);
        runtime_type::event_create(&qr_cholqr_end);
        runtime_type::event_create(&qr_lacpy_start);
        runtime_type::event_create(&qr_end);
        runtime_type::event_record(qr_start);

        float time_hh_core_ms = 0.0f;
        float time_hh_fallback_core_ms = 0.0f;
        float time_chol_shifted_ms = 0.0f;
        float time_chol1_ms = 0.0f;
        float time_chol2_ms = 0.0f;
        float time_copy_to_ms = 0.0f;
        float time_copy_back_ms = 0.0f;

        auto measure_cuda = [&](float& acc_ms, auto&& fn) {
            event_type ev_a, ev_b;
            runtime_type::event_create(&ev_a);
            runtime_type::event_create(&ev_b);
            runtime_type::event_record(ev_a);
            fn();
            runtime_type::event_record(ev_b);
            runtime_type::event_synchronize(ev_b);
            float ms = 0.0f;
            runtime_type::event_elapsed_time(&ms, ev_a, ev_b);
            acc_ms += ms;
            runtime_type::event_destroy(ev_a);
            runtime_type::event_destroy(ev_b);
        };

        int disable = config_.DoCholQR() ? 0 : 1;
        char* cholddisable = getenv("CHASE_DISABLE_CHOLQR");
        if (cholddisable)
        {
            disable = std::atoi(cholddisable);
        }

        int info = 1;

        if constexpr (std::is_same<typename MatrixType::hermitian_type,
                                   chase::matrix::PseudoHermitian>::value)
        {
            kernel_dispatch::lacpy('A', V1_->l_rows(), locked_,
            V1_->l_data() + (V1_->l_cols() - locked_ ) * V1_->l_ld(), V1_->l_ld(), V2_->l_data() + locked_ * V2_->l_ld(),
            V1_->l_ld());

            /* The right eigenvectors are not orthonormal in the QH case, but
             * S-orthonormal. Therefore, we S-orthonormalize the locked vectors
             * against the current subspace By flipping the sign of the lower
             * part of the locked vectors. First, we need to copy the unconverged
               vectors to the end of the vector space*/

            kernel_dispatch::lacpy('A', V1_->l_rows(), V1_->l_cols() - 2*locked_,
            V1_->l_data() + locked_ * V1_->l_ld(), V1_->l_ld(),
            V2_->l_data() + 2 * locked_ * V2_->l_ld(), V2_->l_ld());

            V1_->swap_l_data_ptr(*V2_);

            kernelNamespace::flipLowerHalfMatrixSign(*V1_, 0, 2*locked_);

            runtime_type::event_record(qr_flip_sign);
        }
        else
        {
            runtime_type::event_record(qr_flip_sign);
        }
#ifdef ChASE_DISPLAY_COND_V_SVD
        if constexpr (std::is_same<typename MatrixType::matrix_type,
                                   chase::distMatrix::BlockCyclic>::value &&
                      std::is_same<typename MatrixType::hermitian_type,
                                   chase::matrix::Hermitian>::value)
        {
            static constexpr chase::distMultiVector::CommunicatorType
                communicator_type = InputMultiVectorType::communicator_type;
            auto V_tmp = std::make_unique<
                chase::distMultiVector::DistMultiVectorBlockCyclic1D<
                    T, communicator_type, chase::platform::CPU>>(
                V1_->g_rows(), V1_->g_cols() - locked_, V1_->mb(),
                V1_->getMpiGrid_shared_ptr());
            runtime_type::memcpy_d2h(
                V_tmp->l_data(), V1_->l_data() + locked_ * V1_->l_ld(),
                (V1_->g_cols() - locked_) * V1_->l_ld() * sizeof(T));
            // auto cond_v = kernelNamespace::computeConditionNumber(*V_tmp);
            auto cond_v =
                chase::linalg::internal::cpu_mpi::computeConditionNumber(
                    *V_tmp);

            if (my_rank_ == 0)
            {
                std::cout << "Exact condition number of V from SVD: " << cond_v
                          << std::endl;
            }
        }
#endif
        unsigned int householder_nb = 32u;
        if (const char* hh_nb_env = std::getenv("CHASE_HOUSEHOLDER_NB"))
        {
            const long parsed = std::strtol(hh_nb_env, nullptr, 10);
            if (parsed > 0)
                householder_nb = static_cast<unsigned int>(parsed);
        }

#if defined(HAS_NCCL)
        chase::linalg::internal::cuda_nccl::HouseQRTuning hh_qr_tuning{};
        hh_qr_tuning.outer_block_nb = static_cast<int>(householder_nb);
#endif
        auto call_houseqr_formq = [&](auto& vec, auto* work_ptr) {
#if defined(HAS_HIP) && !defined(HAS_CUDA)
            (void)work_ptr;
            kernelNamespace::houseHoulderQR(vec);
#else
#if defined(HAS_NCCL)
            if constexpr (std::is_same<backend, chase::grid::backend::NCCL>::value)
            {
                kernelNamespace::houseQR1_formQ(blasH_, vec, work_ptr, lwork_,
                                                &hh_qr_tuning);
            }
            else
#endif
            {
                kernelNamespace::houseQR1_formQ(blasH_, vec, work_ptr, lwork_,
                                                householder_nb);
            }
#endif
        };

        // CholQR when disable==0 (always), or disable==1 with cond==1.0 (CholQR1 path).
        if (disable == 1 && cond != static_cast<Base<T>>(1.0))
        {
#ifdef QR_DOUBLE_PRECISION
            if constexpr (std::is_same<T, std::complex<float>>::value ||
                              std::is_same<T, float>::value)
            {
                measure_cuda(time_copy_to_ms, [&]() { V1_->copyTo(); });
                auto V1_d = V1_->getDoublePrecisionMatrix();

                if constexpr (chase::distMultiVector::is_block_cyclic_1d_multivector<InputMultiVectorType>::value)
                {
#if defined(HAS_NCCL)
                    if constexpr (std::is_same<backend,
                                  chase::grid::backend::NCCL>::value)
                    {
#ifdef CHASE_OUTPUT
                        if (my_rank_ == 0)
                        {
                            chase::GetLogger().Log(
                                chase::LogLevel::Debug, "linalg",
                                "Householder QR: NCCL GPU (block-cyclic 1D).\n",
                                0);
                        }
#endif
                        measure_cuda(time_hh_core_ms, [&]() {
                            call_houseqr_formq(
                                *V1_d,
                                reinterpret_cast<typename chase::ToDoublePrecisionTrait<T>::Type*>(d_work_d));
                        });
                    }
                    else
#endif
                    {
#ifdef HAS_SCALAPACK
#ifdef CHASE_OUTPUT
                        if (my_rank_ == 0)
                        {
                            chase::GetLogger().Log(
                                chase::LogLevel::Warn, "linalg",
                                "[Warning]: Using ScaLAPACK Householder QR for block-cyclic 1D (MPI / non-NCCL build).\n",
                                0);
                        }
#endif
                        measure_cuda(time_hh_core_ms, [&]() {
                            kernelNamespace::houseHoulderQR(*V1_d);
                        });
#else
                        throw std::runtime_error(
                            "Block-cyclic 1D distributed Householder QR requires "
                            "NCCL (backend::NCCL) or ScaLAPACK.\n");
#endif
                    }
                }
                else
                {
                    call_houseqr_formq(
                        *V1_d,
                        reinterpret_cast<typename chase::ToDoublePrecisionTrait<T>::Type*>(
                            d_work_d));
                }
                measure_cuda(time_copy_back_ms, [&]() { V1_->copyback(); });
            }
            else
#endif
            {
                if constexpr (chase::distMultiVector::is_block_cyclic_1d_multivector<InputMultiVectorType>::value)
                {
#if defined(HAS_NCCL)
                    if constexpr (std::is_same<backend,
                                  chase::grid::backend::NCCL>::value)
                    {
#ifdef CHASE_OUTPUT
                        if (my_rank_ == 0)
                        {
                            chase::GetLogger().Log(
                                chase::LogLevel::Debug, "linalg",
                                "Householder QR: NCCL GPU (block-cyclic 1D).\n",
                                0);
                        }
#endif
                        measure_cuda(time_hh_core_ms, [&]() {
                            call_houseqr_formq(*V1_, d_work_);
                        });
                    }
                    else
#endif
                    {
#ifdef HAS_SCALAPACK
#ifdef CHASE_OUTPUT
                        if (my_rank_ == 0)
                        {
                            chase::GetLogger().Log(
                                chase::LogLevel::Warn, "linalg",
                                "[Warning]: Using ScaLAPACK Householder QR for block-cyclic 1D (MPI / non-NCCL build).\n",
                                0);
                        }
#endif
                        measure_cuda(time_hh_core_ms, [&]() {
                            kernelNamespace::houseHoulderQR(*V1_);
                        });
#else
                        throw std::runtime_error(
                            "Block-cyclic 1D distributed Householder QR requires "
                            "NCCL (backend::NCCL) or ScaLAPACK.\n");
#endif
                    }
                }
                else
                {
                    measure_cuda(time_hh_core_ms, [&]() {
                        call_houseqr_formq(*V1_, d_work_);
                    });
                }

            }
        }
        else
        {
            Base<T> cond_threshold_upper = (sizeof(Base<T>) == 8) ? 1e8 : 1e4;
            Base<T> cond_threshold_lower = (sizeof(Base<T>) == 8) ? 2e1 : 1e1;

            char* chol_threshold = getenv("CHASE_CHOLQR1_THLD");
            if (chol_threshold)
            {
                cond_threshold_lower = std::atof(chol_threshold);
            }

#ifdef CHASE_OUTPUT
            if (my_rank_ == 0)
            {
                std::ostringstream oss;
                oss << std::setprecision(2) << "cond(V): " << cond;
                chase::GetLogger().Log(chase::LogLevel::Info, "linalg",
                    oss.str(), my_rank_);
            }
#endif

            int info = 1;

            runtime_type::event_record(qr_cholqr_start);

            if (cond > cond_threshold_upper)
            {
#ifdef CHASE_OUTPUT
                /*
                if(my_rank_ == 0){
                    std::cout << "Entering Shifted Cholesky QR 2" << std::endl;
                }*/
#endif
#ifdef QR_DOUBLE_PRECISION
                if constexpr (std::is_same<T, std::complex<float>>::value ||
                              std::is_same<T, float>::value)
                {
                    measure_cuda(time_copy_to_ms, [&]() { V1_->copyTo(); });

                    auto V1_d = V1_->getDoublePrecisionMatrix();
                    auto A_d = A_->getDoublePrecisionMatrix();
                    measure_cuda(time_chol_shifted_ms, [&]() {
                        info = kernelNamespace::shiftedcholQR2(
                            blasH_, solverH_, V1_->g_rows(), V1_->l_rows(),
                            V1_->l_cols(), V1_d->l_data(), V1_->l_ld(),
                            // V1_->getMpiGrid()->get_nccl_col_comm(),
                            MGPUKernelNamspaceSelector<backend>::getColCommunicator(
                                V1_->getMpiGrid()),
                            reinterpret_cast<
                                typename chase::ToDoublePrecisionTrait<T>::Type*>(
                                d_work_d),
                            lwork_, A_d->l_data());
                    });
                    measure_cuda(time_copy_back_ms, [&]() { V1_->copyback(); });
                }
                else
                {
#endif
                    if constexpr (std::is_same<T, std::complex<float>>::value ||
                                std::is_same<T, float>::value)
                    {
                        measure_cuda(time_chol_shifted_ms, [&]() {
                            info = kernelNamespace::shiftedcholQR2(
                                blasH_, solverH_, V1_->g_rows(), V1_->l_rows(),
                                V1_->l_cols(), V1_->l_data(), V1_->l_ld(),
                                // V1_->getMpiGrid()->get_nccl_col_comm(),
                                MGPUKernelNamspaceSelector<backend>::getColCommunicator(
                                    V1_->getMpiGrid()),
                                d_work_, lwork_, A_->l_data());
                        });
                    }else
                    {
                        measure_cuda(time_chol_shifted_ms, [&]() {
                            info = kernelNamespace::shiftedcholQR2(
                                blasH_, solverH_, V1_->g_rows(), V1_->l_rows(),
                                V1_->l_cols(), V1_->l_data(), V1_->l_ld(),
                                // V1_->getMpiGrid()->get_nccl_col_comm(),
                                MGPUKernelNamspaceSelector<backend>::getColCommunicator(
                                    V1_->getMpiGrid()),
                                d_work_, lwork_, A_->l_data());

                        });
                    }
#ifdef QR_DOUBLE_PRECISION
                }
#endif
            }
            else if (cond < cond_threshold_lower)
            {
                /*if constexpr (std::is_same<T, std::complex<float>>::value ||
                std::is_same<T, float>::value)
                {
                    if(V1_->isDoublePrecisionEnabled())
                    {
                        V1_->disableDoublePrecision();
                    }
                    if(A_->isDoublePrecisionEnabled())
                    {
                        A_->disableDoublePrecision();
                    }
                }
                */

#ifdef CHASE_OUTPUT
                if (0)
                {
#ifdef QR_DOUBLE_PRECISION
                    std::cout << "QR_RR_DOUBLE ACTVIATED" << std::endl;
#else
                    std::cout << "QR_RR_DOUBLE DISABLED" << std::endl;
#endif
                    std::cout << "Entering Cholesky QR 1" << std::endl;
                }
#endif
#ifdef QR_DOUBLE_PRECISION
                if constexpr (std::is_same<T, std::complex<float>>::value ||
                              std::is_same<T, float>::value)
                {
                    measure_cuda(time_copy_to_ms, [&]() { V1_->copyTo(); });

                    auto V1_d = V1_->getDoublePrecisionMatrix();
                    auto A_d = A_->getDoublePrecisionMatrix();
                    measure_cuda(time_chol1_ms, [&]() {
                        info = kernelNamespace::cholQR1(
                            blasH_, solverH_, V1_->l_rows(), V1_->l_cols(),
                            V1_d->l_data(), V1_->l_ld(),
                            // V1_->getMpiGrid()->get_nccl_col_comm(),
                            MGPUKernelNamspaceSelector<backend>::getColCommunicator(
                                V1_->getMpiGrid()),
                            reinterpret_cast<
                                typename chase::ToDoublePrecisionTrait<T>::Type*>(
                                d_work_d),
                            lwork_, A_d->l_data());
                    });
                    measure_cuda(time_copy_back_ms, [&]() { V1_->copyback(); });
                }
                else
                {
#endif
                    measure_cuda(time_chol1_ms, [&]() {
                        info = kernelNamespace::cholQR1(
                            blasH_, solverH_, V1_->l_rows(), V1_->l_cols(),
                            V1_->l_data(), V1_->l_ld(),
                            // V1_->getMpiGrid()->get_nccl_col_comm(),
                            MGPUKernelNamspaceSelector<backend>::getColCommunicator(
                                V1_->getMpiGrid()),
                            d_work_, lwork_, A_->l_data());
                    });
#ifdef QR_DOUBLE_PRECISION
                }
#endif
            }
            else
            {
                /*if constexpr (std::is_same<T, std::complex<float>>::value ||
                std::is_same<T, float>::value)
                {
                    if(V1_->isDoublePrecisionEnabled())
                    {
                        V1_->disableDoublePrecision();
                    }
                    if(A_->isDoublePrecisionEnabled())
                    {
                        A_->disableDoublePrecision();
                    }
                }*/

#ifdef CHASE_OUTPUT
                /*
                if(my_rank_ == 0){
                    std::cout << "Entering Cholesky QR 2" << std::endl;
                }*/
#endif
#ifdef QR_DOUBLE_PRECISION
                if constexpr (std::is_same<T, std::complex<float>>::value ||
                              std::is_same<T, float>::value)
                {
                    measure_cuda(time_copy_to_ms, [&]() { V1_->copyTo(); });

                    auto V1_d = V1_->getDoublePrecisionMatrix();
                    auto A_d = A_->getDoublePrecisionMatrix();
                    measure_cuda(time_chol2_ms, [&]() {
                        info = kernelNamespace::cholQR2(
                            blasH_, solverH_, V1_->l_rows(), V1_->l_cols(),
                            V1_d->l_data(), V1_->l_ld(),
                            // V1_->getMpiGrid()->get_nccl_col_comm(),
                            MGPUKernelNamspaceSelector<backend>::getColCommunicator(
                                V1_->getMpiGrid()),
                            reinterpret_cast<
                                typename chase::ToDoublePrecisionTrait<T>::Type*>(
                                d_work_d),
                            lwork_, A_d->l_data());
                    });
                    measure_cuda(time_copy_back_ms, [&]() { V1_->copyback(); });
                }
                else
                {
#endif
                    measure_cuda(time_chol2_ms, [&]() {
                        info = kernelNamespace::cholQR2(
                            blasH_, solverH_, V1_->l_rows(), V1_->l_cols(),
                            V1_->l_data(), V1_->l_ld(),
                            // V1_->getMpiGrid()->get_nccl_col_comm(),
                            MGPUKernelNamspaceSelector<backend>::getColCommunicator(
                                V1_->getMpiGrid()),
                            d_work_, lwork_, A_->l_data());
                    });
#ifdef QR_DOUBLE_PRECISION
                }
#endif
            }

            runtime_type::event_record(qr_cholqr_end);

            if (info != 0)
            {
#ifdef CHASE_OUTPUT
                if (my_rank_ == 0)
                {
                    chase::GetLogger().Log(chase::LogLevel::Warn, "linalg",
                        "CholeskyQR doesn't work, Householder QR will be used.\n",
                        my_rank_);
                }
#endif
#ifdef QR_DOUBLE_PRECISION
                if constexpr (std::is_same<T, std::complex<float>>::value ||
                              std::is_same<T, float>::value)
                {
                    measure_cuda(time_copy_to_ms, [&]() { V1_->copyTo(); });
                    auto V1_d = V1_->getDoublePrecisionMatrix();
                    {
                        if constexpr (chase::distMultiVector::is_block_cyclic_1d_multivector<InputMultiVectorType>::value)
                        {
#if defined(HAS_NCCL)
                            if constexpr (std::is_same<backend,
                                          chase::grid::backend::NCCL>::value)
                            {
#ifdef CHASE_OUTPUT
                                if (my_rank_ == 0)
                                {
                                    chase::GetLogger().Log(
                                        chase::LogLevel::Debug, "linalg",
                                        "Householder QR (CholQR fallback): NCCL GPU (block-cyclic 1D).\n",
                                        0);
                                }
#endif
                                measure_cuda(time_hh_fallback_core_ms, [&]() {
                                    call_houseqr_formq(
                                        *V1_d,
                                        reinterpret_cast<typename chase::ToDoublePrecisionTrait<T>::Type*>(d_work_d));
                                });
                            }
                            else
#endif
                            {
#ifdef HAS_SCALAPACK
#ifdef CHASE_OUTPUT
                                if (my_rank_ == 0)
                                {
                                    chase::GetLogger().Log(
                                        chase::LogLevel::Warn, "linalg",
                                        "[Warning]: Using ScaLAPACK Householder QR for block-cyclic 1D (MPI / non-NCCL build).\n",
                                        0);
                                }
#endif
                                measure_cuda(time_hh_fallback_core_ms, [&]() {
                                    kernelNamespace::houseHoulderQR(*V1_d);
                                });
#else
                                throw std::runtime_error(
                                    "Block-cyclic 1D distributed Householder QR requires "
                                    "NCCL (backend::NCCL) or ScaLAPACK.\n");
#endif
                            }
                        }
                        else
                        {
                            measure_cuda(time_hh_fallback_core_ms, [&]() {
                                call_houseqr_formq(
                                    *V1_d,
                                    reinterpret_cast<typename chase::ToDoublePrecisionTrait<T>::Type*>(d_work_d));
                            });
                        }
                    }
                    measure_cuda(time_copy_back_ms, [&]() { V1_->copyback(); });
                }
                else
#endif
                {
                    {
                        if constexpr (chase::distMultiVector::is_block_cyclic_1d_multivector<InputMultiVectorType>::value)
                        {
#if defined(HAS_NCCL)
                            if constexpr (std::is_same<backend,
                                          chase::grid::backend::NCCL>::value)
                            {
#ifdef CHASE_OUTPUT
                                if (my_rank_ == 0)
                                {
                                    chase::GetLogger().Log(
                                        chase::LogLevel::Debug, "linalg",
                                        "Householder QR (CholQR fallback): NCCL GPU (block-cyclic 1D).\n",
                                        0);
                                }
#endif
                                measure_cuda(time_hh_fallback_core_ms, [&]() {
                                    call_houseqr_formq(*V1_, d_work_);
                                });
                            }
                            else
#endif
                            {
#ifdef HAS_SCALAPACK
#ifdef CHASE_OUTPUT
                                if (my_rank_ == 0)
                                {
                                    chase::GetLogger().Log(
                                        chase::LogLevel::Warn, "linalg",
                                        "[Warning]: Using ScaLAPACK Householder QR for block-cyclic 1D (MPI / non-NCCL build).\n",
                                        0);
                                }
#endif
                                measure_cuda(time_hh_fallback_core_ms, [&]() {
                                    kernelNamespace::houseHoulderQR(*V1_);
                                });
#else
                                throw std::runtime_error(
                                    "Block-cyclic 1D distributed Householder QR requires "
                                    "NCCL (backend::NCCL) or ScaLAPACK.\n");
#endif
                            }
                        }
                        else
                        {
                            measure_cuda(time_hh_fallback_core_ms, [&]() {
                                call_houseqr_formq(*V1_, d_work_);
                            });
                        }
                    }
                }
            }
        }

        runtime_type::event_record(qr_lacpy_start);

        std::size_t unconverged_cols = GetRitzvBlockSize() - locked_;

        if constexpr (std::is_same<typename MatrixType::hermitian_type,
            chase::matrix::PseudoHermitian>::value)
        {
            unconverged_cols = GetRitzvBlockSize() - 2*locked_;

            kernel_dispatch::lacpy('A', V1_->l_rows(), V1_->l_cols() - 2*locked_,
            V1_->l_data() + 2 * locked_ * V1_->l_ld(),V1_->l_ld(),
            V2_->l_data() + locked_ * V2_->l_ld(), V2_->l_ld());

            V1_->swap_l_data_ptr(*V2_);

            kernel_dispatch::lacpy('A', V2_->l_rows(), locked_,
            V1_->l_data(), V1_->l_ld(),
            V2_->l_data(), V2_->l_ld());

            kernel_dispatch::lacpy('A', V2_->l_rows(), locked_,
            V1_->l_data() + (V1_->l_cols() - locked_ ) * V1_->l_ld(), V1_->l_ld(),
            V2_->l_data() + (V2_->l_cols() - locked_ ) * V2_->l_ld(), V2_->l_ld());

        }else{
            unconverged_cols = GetRitzvBlockSize() - locked_;

            kernel_dispatch::lacpy('A', V2_->l_rows(), locked_,
            V2_->l_data(), V2_->l_ld(),
            V1_->l_data(), V1_->l_ld());
        }

        kernel_dispatch::lacpy(
            'A', V2_->l_rows(), unconverged_cols,
            V1_->l_data() + V1_->l_ld() * locked_, V1_->l_ld(),
            V2_->l_data() + V2_->l_ld() * locked_, V2_->l_ld());

        runtime_type::event_record(qr_end);
        runtime_type::event_synchronize(qr_end);

        runtime_type::device_synchronize();

        // Calculate and print detailed timings
        float time_flip_sign = 0.0f, time_cholqr = 0.0f, time_lacpy = 0.0f, time_total = 0.0f;
        runtime_type::event_elapsed_time(&time_flip_sign, qr_start, qr_flip_sign);
        runtime_type::event_elapsed_time(&time_cholqr, qr_cholqr_start, qr_cholqr_end);
        runtime_type::event_elapsed_time(&time_lacpy, qr_lacpy_start, qr_end);
        runtime_type::event_elapsed_time(&time_total, qr_start, qr_end);
#ifdef CHASE_OUTPUT
        if (my_rank_ == 0)
        {
            std::ostringstream oss;
            oss << std::setprecision(6) << std::fixed;
            oss << "\n[QR Timing Rank 0] Total: " << time_total/1000.0 << " s, "
                << "FlipSign: " << time_flip_sign/1000.0 << " s, "
                << "CholQR: " << time_cholqr/1000.0 << " s, "
                << "Lacpy: " << time_lacpy/1000.0 << " s\n";
            oss << "  Breakdown: copyTo=" << time_copy_to_ms/1000.0 << " s, "
                << "copyBack=" << time_copy_back_ms/1000.0 << " s, "
                << "HH(core)=" << time_hh_core_ms/1000.0 << " s, "
                << "HH(fallback)=" << time_hh_fallback_core_ms/1000.0 << " s, "
                << "ShiftedCholQR2=" << time_chol_shifted_ms/1000.0 << " s, "
                << "CholQR1=" << time_chol1_ms/1000.0 << " s, "
                << "CholQR2=" << time_chol2_ms/1000.0 << " s\n";
            chase::GetLogger().Log(chase::LogLevel::Debug, "linalg", oss.str(), my_rank_);
        }
#endif
        // Cleanup events
        runtime_type::event_destroy(qr_start);
        runtime_type::event_destroy(qr_flip_sign);
        runtime_type::event_destroy(qr_cholqr_start);
        runtime_type::event_destroy(qr_cholqr_end);
        runtime_type::event_destroy(qr_lacpy_start);
        runtime_type::event_destroy(qr_end);
    }

    void RR(chase::Base<T>* ritzv, std::size_t block) override
    {
        SCOPED_NVTX_RANGE();

        runtime_type::device_synchronize();
#if defined(CHASE_PCHASE_GPU_HAS_GPU_COLLECTIVES)
        if constexpr (std::is_same<backend, chase::grid::backend::NCCL>::value)
        {
            CHECK_NCCL_ERROR(chase::nccl::ncclBcastWrapper<T>(
                V1_->l_data() + locked_ * V1_->l_ld(),
                V1_->l_ld() * (V1_->l_cols() - locked_), 0,
                V1_->getMpiGrid()->get_nccl_row_comm()));
        }
        else
#endif
        {
#if defined(HAS_HIP) && !defined(HAS_CUDA)
            chase::linalg::internal::hip_mpi_utils::mpiBcast(
                V1_->l_data() + locked_ * V1_->l_ld(),
                V1_->l_ld() * (V1_->l_cols() - locked_), 0,
                V1_->getMpiGrid()->get_row_comm());
#else
            MPI_Bcast(V1_->l_data() + locked_ * V1_->l_ld(),
                      static_cast<int>(V1_->l_ld() *
                                       (V1_->l_cols() - locked_)),
                      chase::mpi::getMPI_Type<T>(), 0,
                      V1_->getMpiGrid()->get_row_comm());
#endif
        }
        kernel_dispatch::lacpy('A', V2_->l_rows(),
        V2_->l_cols() - locked_, V1_->l_data() + locked_ * V1_->l_ld(), V1_->l_ld(),
        V2_->l_data() + locked_ * V2_->l_ld(), V2_->l_ld());

        if constexpr (std::is_same<typename MatrixType::hermitian_type,
                                   chase::matrix::PseudoHermitian>::value)
        {
#if defined(XGEEV_EXISTS) && !(defined(HAS_HIP) && !defined(HAS_CUDA))
            kernelNamespace::pseudo_hermitian_rayleighRitz(
                blasH_, solverH_, params_, *Hmat_, *V1_, *V2_, *W1_, *W2_,
                *ritzv_, locked_, 2*block, devInfo_, d_work_, lwork_,
                h_work_.get(), lhwork_, A_.get());
#else
            kernelNamespace::pseudo_hermitian_rayleighRitz_v2(
                blasH_, solverH_, params_, *Hmat_, *V1_, *V2_, *W1_, *W2_,
                *ritzv_, locked_, 2*block, devInfo_, d_work_, lwork_, A_.get());
#endif

            kernel_dispatch::lacpy(
                'A', V2_->l_rows(), 2*block, V1_->l_data() + locked_ * V1_->l_ld(),
                V1_->l_ld(), V2_->l_data() + locked_ * V2_->l_ld(), V2_->l_ld());
        }
        else
        {

            kernelNamespace::rayleighRitz(
                blasH_, solverH_, *Hmat_, *V1_, *V2_, *W1_, *W2_, *ritzv_,
                locked_, block, devInfo_, d_work_, lwork_, A_.get());

            kernel_dispatch::lacpy(
                'A', V2_->l_rows(), block, V1_->l_data() + locked_ * V1_->l_ld(),
                V1_->l_ld(), V2_->l_data() + locked_ * V2_->l_ld(), V2_->l_ld());
        }

        runtime_type::device_synchronize();
    }

    void Sort(chase::Base<T>* ritzv, chase::Base<T>* residLast,
              chase::Base<T>* resid) override
    {
    }
    void Resd(chase::Base<T>* ritzv, chase::Base<T>* resd) override
    {
        SCOPED_NVTX_RANGE();

        std::size_t subSize = nevex_ - locked_;
        kernelNamespace::residuals(blasH_, *Hmat_, *V1_, *V2_, *W1_, *W2_,
                                   ritzv_->loc_matrix(), resid_->loc_matrix(),
                                   locked_, subSize);
    }

    void Swap(std::size_t i, std::size_t j) override
    {
        V1_->swap_ij(i, j);
        V2_->swap_ij(i, j);
    }

    void Lock(std::size_t new_converged) override
    {
        SCOPED_NVTX_RANGE();

        locked_ += new_converged;
    }

    void End() override
    {
        SCOPED_NVTX_RANGE();
        V1_->D2H();
    }

private:
    /**
     * @enum NextOp
     * @brief Represents the next operation to be performed in the computation.
     */
    enum NextOp
    {
        cAb, /**< Represents the operation `c = A * b`. */
        bAc  /**< Represents the operation `b = A * c`. */
    };

    NextOp next_; /**< Holds the next operation in the computation sequence. */

    bool is_sym_;        /**< Indicates whether the matrix is symmetric. */
    bool is_pseudoHerm_; /**< Indicates whether the matrix is pseudo-hermitian.
                          */

    std::size_t nev_; /**< Number of eigenvalues to compute. */
    std::size_t
        nex_; /**< Number of additional vectors for iterative refinement. */
    std::size_t nevex_; /**< Total number of vectors (nev + nex) used in the
                           algorithm. */
    std::size_t
        locked_; /**< Count of locked vectors in the eigenvalue problem. */
    std::size_t lanczosIter_; /**< Number of Lanczos Iterations.*/
    std::size_t numLanczos_;  /**< Number of Runs of Lanczos.*/

    std::size_t N_; /**< Dimension of the square matrix. */

    int nprocs_;  /**< Total number of MPI processes. */
    int my_rank_; /**< Rank of the current MPI process. */
    int* coords_; /**< Pointer to coordinates of the current process in the MPI
                     grid. */
    int* dims_;   /**< Pointer to dimensions of the MPI grid. */

    MatrixType* Hmat_; /**< Pointer to the Hamiltonian matrix. */
    InputMultiVectorType*
        V1_; /**< Pointer to the first input multi-vector for eigenvectors. */
    std::unique_ptr<InputMultiVectorType>
        V2_; /**< Unique pointer to the second input multi-vector for
                computations. */
    std::unique_ptr<ResultMultiVectorType>
        W1_; /**< Unique pointer to the first result multi-vector. */
    std::unique_ptr<ResultMultiVectorType>
        W2_; /**< Unique pointer to the second result multi-vector. */

    std::unique_ptr<chase::distMatrix::RedundantMatrix<chase::Base<T>,
                                                       chase::platform::GPU>>
        ritzv_; /**< Matrix holding Ritz values (eigenvalues). */
    std::unique_ptr<chase::distMatrix::RedundantMatrix<chase::Base<T>,
                                                       chase::platform::GPU>>
        resid_; /**< Matrix holding residuals for eigenvalues. */
    std::unique_ptr<chase::distMatrix::RedundantMatrix<T, chase::platform::GPU>>
        A_; /**< Auxiliary matrix for intermediate calculations. */

    execution_context gpu_context_; /**< Owning GPU handles, devInfo, workspace. */
    stream_type stream_ = nullptr; /**< GPU stream for asynchronous operations. */
    blas_handle_type blasH_ = nullptr; /**< Handle to the GPU BLAS library for linear
                                algebra operations. */
    solver_handle_type solverH_ = nullptr; /**< Handle to the GPU solver library. */
    solver_params_type
        params_ = nullptr; /**< Optional solver parameter object for Xgeev. */


    rand_state_type* states_ =
        NULL; /**< Random number generator state for GPU, used for
                 initializations. */

    int* devInfo_ = nullptr; /**< Pointer to device memory for storing operation status
                      (e.g., success or failure) in solver calls. */
    T* d_return_ = nullptr;  /**< Pointer to device memory for storing results of GPU
                      operations. */
    T* d_work_ = nullptr; /**< Pointer to workspace on the device for GPU operations. */
    int lwork_ =
        0; /**< Size of the workspace on the device, used for GPU operations. */
    void* d_work_d = nullptr;

    std::unique_ptr<T[]> h_work_; /**< Pointer to work buffer on host for geev
                                     in the Pseudo Hermitian case. */
    int lhwork_ = 0; /**< Workspace size for host geev operations in the Pseudo
                        Hermitian case. */

    std::size_t* d_diag_xoffs = nullptr; /**< Pointer to device memory holding x offsets
                                  for diagonal elements in computations. */
    std::size_t* d_diag_yoffs = nullptr; /**< Pointer to device memory holding y offsets
                                  for diagonal elements in computations. */
    std::size_t
        diag_cnt; /**< Count of diagonal elements used in the algorithm. */

    chase::ChaseConfig<T>
        config_; /**< Configuration settings for the Chase algorithm, including
                    problem parameters. */

#ifdef CHASE_ENABLE_OZAKI
    size_t ozaki_workspace_size_ = 0; /**< Workspace size for Ozaki-II (GEMMul8). */
    int8_t* ozaki_workspace_ = nullptr; /**< Workspace for Ozaki-II (GEMMul8). */
    chase::Base<T>* d_one_norm_H_ = nullptr; /**< Device scalar for ||H−cI||_1. */
    chase::Base<T>* d_inf_norm_H_ = nullptr; /**< Device scalar for ||H−cI||_∞. */
    size_t ozaki_max_worksize_ = size_t(24) << 30; /**< Max GEMMul8 workspace in bytes. */
    unsigned ozaki_num_moduli_ = 8; /**< Number of moduli for Ozaki-II (GEMMul8). */
    int ozaki_fallback_num_moduli_ =
        -1; /**< Fallback moduli; -1 => FP-GEMM on stagnation. */
    bool ozaki_fallback_disabled_ =
        false; /**< True when CHASE_OZAKI_NUM_MODULI fixes moduli (no stall fallback). */
    bool ozaki_skip_scaling_      = false; /**< Flag to skip scaling Hmat in Ozaki-II (GEMMul8). */
    bool ozaki_save_memory_     = false; /**< Cap Ozaki-II workspace (on by default if GEMMul8 supports it). */
    bool ozaki_fp_hemm_fallback = false; /**< Flag to use FP32/64 fallback in Ozaki-II (GEMMul8). */
    bool ozaki_skip_scaling_activated_ = false; /**< Flag to activate skip scaling Hmat in Ozaki-II (GEMMul8). */
#endif
};

} // namespace Impl
} // namespace chase

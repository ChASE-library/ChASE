// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "linalg/internal/gpu/backend/runtime.hpp"
#include "linalg/internal/gpu/backend/solver.hpp"
#include "linalg/internal/gpu/backend/workspace_plan.hpp"

#include <cstddef>

#if defined(HAS_CUDA)
#include "external/nvidia_gpu/cublaspp/cublaspp.hpp"
#include "external/nvidia_gpu/cusolverpp/cusolverpp.hpp"
#endif

#if defined(HAS_HIP)
#include "external/amd_gpu/rocblaspp/rocblaspp.hpp"
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#endif

/**
 * @page gpu_backend_context_module GpuExecutionContext
 * @brief Per-GPU execution bundle: handles, stream, solver workspace, devInfo.
 *
 * `ChASEGPU`, `pChASEGPU`, and distributed matrix code construct one
 * `GpuExecutionContext<Backend>` per active GPU. The context owns everything
 * that must outlive individual kernel calls but is shared across BLAS, solver,
 * and device kernels on the same stream.
 *
 * @section ctx_owned_resources Owned resources
 *
 * | Resource | Purpose |
 * |----------|---------|
 * | BLAS handle | `blas<Backend>` GEMM, dot, axpy, … |
 * | Solver handle | `solver<Backend>` QR, Cholesky, `heevd`, … |
 * | Stream | Optional; created by default, or injected via `set_stream` |
 * | `dev_info_` | Single device `int` for cuSOLVER / rocSOLVER status |
 * | `workspace_` | Grow-only reusable buffer sized by `ensure_solver_plan` |
 *
 * @section ctx_cuda_vs_hip CUDA vs HIP handles
 *
 * **CUDA** uses distinct library handles:
 * - `cublasCreate` / `cusolverDnCreate` for BLAS and solver.
 * - Each handle is bound to the context stream with `*SetStream`.
 *
 * **HIP** maps both roles to `rocblas_handle`, but ChASE still creates
 * **two** handles (`create_blas` and `create_solver` both call
 * `rocblas_create_handle`). That isolates rocBLAS workspace / stream side
 * effects from rocSOLVER traffic even though the C++ type is the same (see
 * `solver.hpp` workspace-query scratch-handle pattern).
 *
 * @section ctx_workspace Workspace allocation
 *
 * Constructor-time `SolverWorkspacePlan` queries (see
 * `workspace_plan.hpp`) feed `ensure_solver_plan<T>(plan)`, which
 * allocates `plan.device_element_count() * sizeof(T)` bytes. Kernels obtain a
 * typed pointer via `work<T>()` and pass it into `solver<Backend>` routines.
 *
 * `ensure_workspace_bytes` only **grows** the buffer; shrinking is not
 * supported (callers size once at init). Reallocation frees the old block
 * before allocating a larger one.
 *
 * @section ctx_stream Stream ownership
 *
 * By default the constructor creates and owns a stream (`create_stream=true`).
 * `set_stream(external, take_ownership)` replaces it: if the context previously
 * owned its stream, that stream is destroyed; when `take_ownership` is true the
 * context will destroy the new stream in its destructor.
 */

namespace chase
{
namespace linalg
{
namespace internal
{
namespace gpu_backend
{

/**
 * @brief Backend-specific BLAS/solver handle lifecycle for `GpuExecutionContext`.
 *
 * Primary template is declared only; specializations exist per compiled backend
 * (`cuda_backend`, `hip_backend`). Keeps `GpuExecutionContext` free of
 * `#if defined(HAS_CUDA)` branches in its public API.
 */
template <typename Backend>
struct context_handle_traits;

#if defined(HAS_CUDA)
/**
 * @brief CUDA handle traits: separate cuBLAS and cuSOLVER handles.
 */
template <>
struct context_handle_traits<cuda_backend>
{
    using blas_handle_type = typename traits<cuda_backend>::blas_handle;
    using solver_handle_type = typename traits<cuda_backend>::solver_handle;
    using stream_type = typename runtime<cuda_backend>::stream_type;

    static void create_blas(blas_handle_type* handle)
    {
        CHECK_CUBLAS_ERROR(cublasCreate(handle));
    }

    static void create_solver(solver_handle_type* handle)
    {
        CHECK_CUSOLVER_ERROR(cusolverDnCreate(handle));
    }

    static void destroy_blas(blas_handle_type handle)
    {
        CHECK_CUBLAS_ERROR(cublasDestroy(handle));
    }

    static void destroy_solver(solver_handle_type handle)
    {
        CHECK_CUSOLVER_ERROR(cusolverDnDestroy(handle));
    }

    static void set_blas_stream(blas_handle_type handle, stream_type stream)
    {
        CHECK_CUBLAS_ERROR(cublasSetStream(handle, stream));
    }

    static void set_solver_stream(solver_handle_type handle, stream_type stream)
    {
        CHECK_CUSOLVER_ERROR(cusolverDnSetStream(handle, stream));
    }
};
#endif

#if defined(HAS_HIP)
/**
 * @brief HIP handle traits: two rocBLAS handles (BLAS vs solver roles).
 *
 * Both `blas_handle_type` and `solver_handle_type` are `rocblas_handle`.
 * Separate instances avoid workspace and ordering interference between pure
 * rocBLAS calls and rocSOLVER routines that share the rocBLAS handle model.
 */
template <>
struct context_handle_traits<hip_backend>
{
    using blas_handle_type = typename traits<hip_backend>::blas_handle;
    using solver_handle_type = typename traits<hip_backend>::solver_handle;
    using stream_type = typename runtime<hip_backend>::stream_type;

    static void create_blas(blas_handle_type* handle)
    {
        CHECK_ROCBLAS_ERROR(rocblas_create_handle(handle));
    }

    static void create_solver(solver_handle_type* handle)
    {
        CHECK_ROCBLAS_ERROR(rocblas_create_handle(handle));
    }

    static void destroy_blas(blas_handle_type handle)
    {
        CHECK_ROCBLAS_ERROR(rocblas_destroy_handle(handle));
    }

    static void destroy_solver(solver_handle_type handle)
    {
        CHECK_ROCBLAS_ERROR(rocblas_destroy_handle(handle));
    }

    static void set_blas_stream(blas_handle_type handle, stream_type stream)
    {
        CHECK_ROCBLAS_ERROR(rocblas_set_stream(handle, stream));
    }

    static void set_solver_stream(solver_handle_type handle, stream_type stream)
    {
        CHECK_ROCBLAS_ERROR(rocblas_set_stream(handle, stream));
    }
};
#endif

/**
 * @brief RAII bundle of GPU execution state for one device / one stream.
 *
 * Non-copyable. Typical consumers: `ChASEGPU`, `pChASEGPU`, GPU paths in
 * `distMultiVector`. Access handles via `blas()` and `solver()`; pass
 * `dev_info()` into solver calls; reuse `work<T>()` for factorization /
 * eigen workspace.
 *
 * @tparam Backend `cuda_backend` or `hip_backend` (must match build flags).
 */
template <typename Backend>
class GpuExecutionContext
{
public:
    using handle_traits = context_handle_traits<Backend>;
    using blas_handle_type = typename handle_traits::blas_handle_type;
    using solver_handle_type = typename handle_traits::solver_handle_type;
    using runtime_type = runtime<Backend>;
    using stream_type = typename runtime_type::stream_type;

    /**
     * @brief Create BLAS/solver handles, optional stream, and `devInfo` buffer.
     *
     * When @p create_stream is true (default), creates a stream and binds both
     * handles to it. When false, handles are created without a stream until
     * `set_stream` is called.
     *
     * Always allocates a one-element device `int` for solver `devInfo` output.
     */
    explicit GpuExecutionContext(bool create_stream = true)
    {
        handle_traits::create_blas(&blas_handle_);
        handle_traits::create_solver(&solver_handle_);
        owns_stream_ = create_stream;
        if (create_stream)
        {
            runtime_type::stream_create(&stream_);
            handle_traits::set_blas_stream(blas_handle_, stream_);
            handle_traits::set_solver_stream(solver_handle_, stream_);
        }
        runtime_type::device_malloc(reinterpret_cast<void**>(&dev_info_),
                                    sizeof(int));
    }

    ~GpuExecutionContext()
    {
        if (workspace_ != nullptr)
        {
            runtime_type::device_free(workspace_);
        }
        if (dev_info_ != nullptr)
        {
            runtime_type::device_free(dev_info_);
        }
        if (blas_handle_ != nullptr)
        {
            handle_traits::destroy_blas(blas_handle_);
        }
        if (solver_handle_ != nullptr)
        {
            handle_traits::destroy_solver(solver_handle_);
        }
        if (owns_stream_ && stream_ != nullptr)
        {
            runtime_type::stream_destroy(stream_);
        }
    }

    GpuExecutionContext(const GpuExecutionContext&) = delete;
    GpuExecutionContext& operator=(const GpuExecutionContext&) = delete;

    /** @brief cuBLAS / rocBLAS handle for `blas<Backend>`. */
    blas_handle_type blas() const { return blas_handle_; }

    /** @brief cuSOLVER / rocBLAS-solver handle for `solver<Backend>`. */
    solver_handle_type solver() const { return solver_handle_; }

    /** @brief Execution stream (may be null if constructed with `create_stream=false`). */
    stream_type stream() const { return stream_; }

    /** @brief Device buffer for solver `devInfo` (single `int`). */
    int* dev_info() const { return dev_info_; }

    /**
     * @brief Bind BLAS and solver handles to @p stream and update ownership.
     *
     * If the context owned a previous stream, it is destroyed when the pointer
     * changes. When @p take_ownership is true, the destructor will destroy
     * @p stream; otherwise the caller retains stream lifetime responsibility.
     */
    void set_stream(stream_type stream, bool take_ownership = false)
    {
        if (stream_ != stream)
        {
            if (owns_stream_ && stream_ != nullptr)
            {
                runtime_type::stream_destroy(stream_);
            }
            stream_ = stream;
            owns_stream_ = take_ownership;
        }
        handle_traits::set_blas_stream(blas_handle_, stream_);
        handle_traits::set_solver_stream(solver_handle_, stream_);
    }

    /**
     * @brief Grow device workspace to at least @p bytes (no-op if already large enough).
     *
     * Reallocates by freeing the existing block when a larger size is requested.
     */
    void ensure_workspace_bytes(std::size_t bytes)
    {
        if (bytes <= workspace_bytes_)
        {
            return;
        }
        if (workspace_ != nullptr)
        {
            runtime_type::device_free(workspace_);
            workspace_ = nullptr;
            workspace_bytes_ = 0;
        }
        runtime_type::device_malloc(&workspace_, bytes);
        workspace_bytes_ = bytes;
    }

    /**
     * @brief Grow workspace to hold at least @p elements of type @p T.
     */
    template <typename T>
    void ensure_workspace(std::size_t elements)
    {
        ensure_workspace_bytes(elements * sizeof(T));
    }

    /**
     * @brief Allocate solver workspace from a constructor-time `SolverWorkspacePlan`.
     *
     * Equivalent to `ensure_workspace<T>(plan.device_element_count())`. Called
     * from `ChASEGPU` / `pChASEGPU` after `plan_query_*` helpers populate the plan.
     */
    template <typename T>
    void ensure_solver_plan(const SolverWorkspacePlan& plan)
    {
        ensure_workspace<T>(
            static_cast<std::size_t>(plan.device_element_count()));
    }

    /** @brief Raw device workspace pointer (untyped). */
    void* workspace() const { return workspace_; }

    /**
     * @brief Typed view of the shared solver workspace buffer.
     *
     * Returns null until `ensure_workspace` / `ensure_solver_plan` has run.
     */
    template <typename T>
    T* work() const
    {
        return static_cast<T*>(workspace_);
    }

    /** @brief Current workspace allocation size in bytes. */
    std::size_t workspace_bytes() const { return workspace_bytes_; }

private:
    blas_handle_type blas_handle_ = nullptr;
    solver_handle_type solver_handle_ = nullptr;
    stream_type stream_ = nullptr;
    int* dev_info_ = nullptr;
    void* workspace_ = nullptr;
    std::size_t workspace_bytes_ = 0;
    bool owns_stream_ = true;
};

} // namespace gpu_backend
} // namespace internal
} // namespace linalg
} // namespace chase

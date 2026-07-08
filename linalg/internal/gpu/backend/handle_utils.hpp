// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "linalg/internal/gpu/backend/blas.hpp"
#include "linalg/internal/gpu/backend/runtime.hpp"
#include "linalg/internal/gpu/backend/solver.hpp"

#include <type_traits>

/**
 * @page gpu_backend_handle_utils_module handle_utils
 * @brief BLAS/solver handle helpers for shared GPU algorithm code.
 *
 * Runtime calls (alloc, memcpy, streams, events) belong in `runtime.hpp`.
 * Device kernels belong in `gpu_device::kernels`. This header only re-exports
 * BLAS/solver enums, status checks, and stream binding on existing handles.
 * `blas_side_stream_scope` runs BLAS on a temporary side stream with event ordering.
 */

namespace chase
{
namespace linalg
{
namespace internal
{
namespace gpu_backend
{

template <typename Backend>
struct handle_utils
{
    using runtime_type = runtime<Backend>;
    using blas_dispatch = blas<Backend>;
    using solver_dispatch = solver<Backend>;
    using stream_type = typename runtime_type::stream_type;
    using event_type = typename runtime_type::event_type;
    using blas_handle = typename traits<Backend>::blas_handle;
    using solver_handle = typename traits<Backend>::solver_handle;
    using info_type = typename traits<Backend>::info_type;

    using stream_t = stream_type;
    using info_t = info_type;
    using event_t = event_type;
    using blas_handle_t = blas_handle;
    using solver_handle_t = solver_handle;

    using operation = typename blas_dispatch::operation;
    using fill_mode = typename blas_dispatch::fill_mode;
    using side_mode = typename blas_dispatch::side_mode;
    using diag_mode = typename blas_dispatch::diag_mode;
    using eig_mode_t = typename solver_dispatch::eig_mode;

    using device_deleter = runtime_device_deleter<Backend>;

    static constexpr operation op_n() { return blas_dispatch::op_n; }
    static constexpr operation op_t() { return blas_dispatch::op_t; }
    static constexpr operation op_c() { return blas_dispatch::op_c; }
    static constexpr operation op_none() { return op_n(); }
    static constexpr operation op_conj_transpose() { return op_c(); }

    static constexpr fill_mode fill_lower() { return blas_dispatch::fill_lower; }
    static constexpr fill_mode fill_upper() { return blas_dispatch::fill_upper; }
    static constexpr side_mode side_left() { return blas_dispatch::side_left; }
    static constexpr side_mode side_right() { return blas_dispatch::side_right; }
    static constexpr diag_mode diag_non_unit()
    {
        return blas_dispatch::diag_non_unit;
    }

    static eig_mode_t eig_original() { return solver_dispatch::eig_mode_vector; }

    static void check_blas(typename blas_dispatch::status_type status)
    {
        blas_dispatch::check_status(status);
    }

    static void check_solver(typename solver_dispatch::status_type status)
    {
        solver_dispatch::check_status(status);
    }

    static stream_type get_blas_stream(blas_handle handle)
    {
        stream_type stream{};
        check_blas(blas_dispatch::get_stream(handle, &stream));
        return stream;
    }

    static void set_blas_stream(blas_handle handle, stream_type stream)
    {
        check_blas(blas_dispatch::set_stream(handle, stream));
    }

    static void get_blas_stream(blas_handle handle, stream_type* stream)
    {
        check_blas(blas_dispatch::get_stream(handle, stream));
    }

    static void set_stream(blas_handle handle, stream_type stream)
    {
        set_blas_stream(handle, stream);
    }

    static void get_stream(blas_handle handle, stream_type* stream)
    {
        get_blas_stream(handle, stream);
    }

    static void set_solver_stream(solver_handle handle, stream_type stream)
    {
        check_solver(solver_dispatch::set_stream(handle, stream));
    }

    static void get_solver_stream(solver_handle handle, stream_type* stream)
    {
        check_solver(solver_dispatch::get_stream(handle, stream));
    }

    static void synchronize_blas_stream(blas_handle handle)
    {
        stream_synchronize<Backend>(get_blas_stream(handle));
    }

    template <typename T>
    static operation herk_transpose()
    {
        if constexpr (std::is_same_v<T, std::complex<float>> ||
                      std::is_same_v<T, std::complex<double>>)
        {
            return op_c();
        }
        else
        {
            return op_t();
        }
    }
};

#if defined(HAS_CUDA) || defined(HAS_HIP)
/**
 * @brief Temporarily routes a BLAS handle to a private non-blocking stream.
 *
 * Saves the handle's current stream, runs work on a side stream with event
 * ordering against the previous stream, then restores the original stream on
 * finish() or destruction.
 */
template <typename Backend>
class blas_side_stream_scope
{
public:
    using blas_handle = typename traits<Backend>::blas_handle;
    using stream_type = typename runtime<Backend>::stream_type;
    using event_type = typename runtime<Backend>::event_type;

    explicit blas_side_stream_scope(blas_handle handle) : handle_(handle)
    {
        handle_utils<Backend>::get_stream(handle_, &previous_);
        stream_ = create_nonblocking_stream<Backend>();
        handle_utils<Backend>::set_stream(handle_, stream_);
        begin_ = create_event<Backend>();
        record_event<Backend>(begin_, previous_);
        stream_wait_event<Backend>(stream_, begin_);
    }

    ~blas_side_stream_scope()
    {
        try
        {
            finish();
        }
        catch (...)
        {
        }
    }

    blas_side_stream_scope(const blas_side_stream_scope&) = delete;
    blas_side_stream_scope& operator=(const blas_side_stream_scope&) = delete;

    stream_type stream() const { return stream_; }

    void finish()
    {
        if (finished_)
        {
            return;
        }

        event_type end = create_event<Backend>();
        record_event<Backend>(end, stream_);
        stream_wait_event<Backend>(previous_, end);
        destroy_event<Backend>(begin_);
        destroy_event<Backend>(end);
        handle_utils<Backend>::set_stream(handle_, previous_);
        destroy_stream<Backend>(stream_);
        finished_ = true;
    }

private:
    blas_handle handle_{};
    stream_type previous_{};
    stream_type stream_{};
    event_type begin_{};
    bool finished_ = false;
};
#endif

} // namespace gpu_backend
} // namespace internal
} // namespace linalg
} // namespace chase

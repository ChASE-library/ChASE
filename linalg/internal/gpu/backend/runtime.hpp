// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "linalg/internal/gpu/backend/types.hpp"
#include "linalg/internal/gpu/backend/error.hpp"

#include <cstddef>

#if defined(HAS_CUDA)
#include <cuda_runtime.h>
#include <curand_kernel.h>
#endif

#if defined(HAS_HIP)
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>
#endif

/**
 * @page gpu_backend_runtime_module chase::linalg::internal::gpu_backend::runtime
 * @brief Vendor-neutral GPU runtime dispatch (CUDA or HIP).
 *
 * `runtime<Backend>` wraps device allocation, streams, events, memset, and
 * host/device copies. Higher layers (`GpuExecutionContext`, `matrix_gpu_runtime`,
 * `dist_gpu_portability`) call these static methods instead of using
 * `cuda*` / `hip*` APIs directly.
 *
 * The active build backend is exposed as `active_backend` and aliases to
 * `cuda_backend` or `hip_backend` depending on CMake flags.
 */

namespace chase
{
namespace linalg
{
namespace internal
{
namespace gpu_backend
{

#if defined(HAS_CUDA)
/** @brief Backend tag selected when the translation unit is built with CUDA. */
using active_backend = cuda_backend;
#elif defined(HAS_HIP)
/** @brief Backend tag selected when the translation unit is built with HIP. */
using active_backend = hip_backend;
#endif

/**
 * @brief GPU runtime facade specialized per backend.
 *
 * @tparam Backend `cuda_backend` or `hip_backend`.
 */
template <typename Backend>
struct runtime;

template <typename Backend>
struct runtime_device_deleter;

template <typename Backend>
void stream_synchronize(typename runtime<Backend>::stream_type stream);

#if defined(HAS_CUDA)
/**
 * @brief CUDA runtime specialization (`cuda_runtime.h`, `curand_kernel.h`).
 */
template <>
struct runtime<cuda_backend>
{
    using stream_type = cudaStream_t;   ///< CUDA stream handle.
    using event_type = cudaEvent_t;     ///< CUDA event handle.
    using rand_state_type = curandStatePhilox4_32_10_t; ///< cuRAND Philox state.

    /// @name Stream helpers
    /// @{

    /** @brief Legacy default stream (`nullptr`). */
    static stream_type default_stream() { return nullptr; }

    /**
     * @brief Resolve an optional stream pointer.
     * @param stream Null selects `default_stream()`.
     */
    static stream_type resolve_stream(stream_type* stream)
    {
        return stream == nullptr ? default_stream() : *stream;
    }

    /** @brief Block until all device work on the current device completes. */
    static void device_synchronize()
    {
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }

    /** @brief Create a new CUDA stream. */
    static void stream_create(stream_type* stream)
    {
        CHECK_CUDA_ERROR(cudaStreamCreate(stream));
    }

    /** @brief Create a non-blocking CUDA stream. */
    static void stream_create_nonblocking(stream_type* stream)
    {
        CHECK_CUDA_ERROR(
            cudaStreamCreateWithFlags(stream, cudaStreamNonBlocking));
    }

    /** @brief Destroy a CUDA stream. */
    static void stream_destroy(stream_type stream)
    {
        CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
    }

    /** @brief Wait until all work enqueued on @p stream completes. */
    static void stream_synchronize(stream_type stream)
    {
        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    }

    /** @brief Make @p stream wait on @p event. */
    static void stream_wait_event(stream_type stream, event_type event)
    {
        CHECK_CUDA_ERROR(cudaStreamWaitEvent(stream, event, 0));
    }

    /// @}

    /// @name Event helpers
    /// @{

    /** @brief Create a CUDA event. */
    static void event_create(event_type* event)
    {
        CHECK_CUDA_ERROR(cudaEventCreate(event));
    }

    /** @brief Record @p event on @p stream (default stream if null). */
    static void event_record(event_type event, stream_type stream = nullptr)
    {
        CHECK_CUDA_ERROR(cudaEventRecord(event, stream));
    }

    /** @brief Block the host until @p event completes. */
    static void event_synchronize(event_type event)
    {
        CHECK_CUDA_ERROR(cudaEventSynchronize(event));
    }

    /** @brief Write elapsed milliseconds between two recorded events to @p ms. */
    static void event_elapsed_time(float* ms, event_type start,
                                   event_type stop)
    {
        CHECK_CUDA_ERROR(cudaEventElapsedTime(ms, start, stop));
    }

    /** @brief Destroy a CUDA event. */
    static void event_destroy(event_type event)
    {
        CHECK_CUDA_ERROR(cudaEventDestroy(event));
    }

    /// @}

    /// @name Device memory
    /// @{

    /** @brief Allocate @p bytes on the device; address stored in @p *ptr. */
    static void device_malloc(void** ptr, std::size_t bytes)
    {
        CHECK_CUDA_ERROR(cudaMalloc(ptr, bytes));
    }

    /** @brief Asynchronously allocate device memory on @p stream. */
    static void device_malloc_async(void** ptr, std::size_t bytes,
                                    stream_type stream)
    {
        CHECK_CUDA_ERROR(cudaMallocAsync(ptr, bytes, stream));
    }

    /** @brief Free device memory. No-op for null @p ptr. */
    static void device_free(void* ptr)
    {
        CHECK_CUDA_ERROR(cudaFree(ptr));
    }

    /** @brief Set each byte of a device range to @p value. */
    static void device_memset(void* ptr, int value, std::size_t bytes)
    {
        CHECK_CUDA_ERROR(cudaMemset(ptr, value, bytes));
    }

    /** @brief Asynchronous `device_memset` on @p stream. */
    static void device_memset_async(void* ptr, int value, std::size_t bytes,
                                    stream_type stream)
    {
        CHECK_CUDA_ERROR(cudaMemsetAsync(ptr, value, bytes, stream));
    }

    /// @}

    /// @name Pinned host memory
    /// @{

    /** @brief Allocate page-locked host memory. */
    static void host_pinned_malloc(void** ptr, std::size_t bytes)
    {
        CHECK_CUDA_ERROR(cudaMallocHost(ptr, bytes));
    }

    /** @brief Free pinned host memory. */
    static void host_pinned_free(void* ptr)
    {
        CHECK_CUDA_ERROR(cudaFreeHost(ptr));
    }

    /// @}

    /// @name Linear device copies
    /// @{

    /** @brief Copy @p bytes host → device (synchronous). */
    static void memcpy_h2d(void* dst, const void* src, std::size_t bytes)
    {
        CHECK_CUDA_ERROR(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
    }

    /** @brief Copy @p bytes host → device on @p stream. */
    static void memcpy_h2d_async(void* dst, const void* src, std::size_t bytes,
                                 stream_type stream)
    {
        CHECK_CUDA_ERROR(
            cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream));
    }

    /** @brief Copy @p bytes device → host (synchronous). */
    static void memcpy_d2h(void* dst, const void* src, std::size_t bytes)
    {
        CHECK_CUDA_ERROR(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost));
    }

    /** @brief Copy @p bytes device → device (synchronous). */
    static void memcpy_d2d(void* dst, const void* src, std::size_t bytes)
    {
        CHECK_CUDA_ERROR(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice));
    }

    /** @brief Copy @p bytes device → device on @p stream. */
    static void memcpy_d2d_async(void* dst, const void* src, std::size_t bytes,
                                 stream_type stream)
    {
        CHECK_CUDA_ERROR(
            cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, stream));
    }

    /** @brief Copy @p bytes device → host on @p stream. */
    static void memcpy_d2h_async(void* dst, const void* src, std::size_t bytes,
                                 stream_type stream)
    {
        CHECK_CUDA_ERROR(
            cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, stream));
    }

    /// @}

    /// @name Pitched 2-D copies
    /// @{

    /**
     * @brief 2-D host → device copy with separate source/destination pitch.
     * @param width Bytes per row to copy.
     * @param height Number of rows.
     */
    static void memcpy_2d_h2d(void* dst, std::size_t dst_pitch,
                              const void* src, std::size_t src_pitch,
                              std::size_t width, std::size_t height)
    {
        CHECK_CUDA_ERROR(cudaMemcpy2D(dst, dst_pitch, src, src_pitch, width,
                                      height, cudaMemcpyHostToDevice));
    }

    /**
     * @brief 2-D device → host copy with separate source/destination pitch.
     * @param width Bytes per row to copy.
     * @param height Number of rows.
     */
    static void memcpy_2d_d2h(void* dst, std::size_t dst_pitch,
                              const void* src, std::size_t src_pitch,
                              std::size_t width, std::size_t height)
    {
        CHECK_CUDA_ERROR(cudaMemcpy2D(dst, dst_pitch, src, src_pitch, width,
                                      height, cudaMemcpyDeviceToHost));
    }

    /// @}

    /// @name Dense matrix copies (leading dimensions)
    /// @{

    /**
     * @brief Copy a dense submatrix host → device using leading dimensions.
     * @param elem_size Element size in bytes.
     * @param src_ld Host leading dimension (elements).
     * @param dst_ld Device leading dimension (elements).
     */
    static void copy_matrix_h2d(std::size_t rows, std::size_t cols,
                                std::size_t elem_size, const void* src,
                                std::size_t src_ld, void* dst,
                                std::size_t dst_ld)
    {
        memcpy_2d_h2d(dst, dst_ld * elem_size, src, src_ld * elem_size,
                      rows * elem_size, cols);
    }

    /**
     * @brief Copy a dense submatrix device → host using leading dimensions.
     * @param elem_size Element size in bytes.
     * @param src_ld Device leading dimension (elements).
     * @param dst_ld Host leading dimension (elements).
     */
    static void copy_matrix_d2h(std::size_t rows, std::size_t cols,
                                std::size_t elem_size, const void* src,
                                std::size_t src_ld, void* dst,
                                std::size_t dst_ld)
    {
        memcpy_2d_d2h(dst, dst_ld * elem_size, src, src_ld * elem_size,
                      rows * elem_size, cols);
    }

    /// @}
};
#endif

#if defined(HAS_HIP)
/**
 * @brief HIP runtime specialization (`hip_runtime.h`, `hiprand_kernel.h`).
 *
 * Exposes the same static API and semantics as `runtime<cuda_backend>`; only
 * the underlying vendor calls differ.
 */
template <>
struct runtime<hip_backend>
{
    using stream_type = hipStream_t;
    using event_type = hipEvent_t;
    using rand_state_type = hiprandStatePhilox4_32_10_t;

    static stream_type default_stream() { return nullptr; }

    static stream_type resolve_stream(stream_type* stream)
    {
        return stream == nullptr ? default_stream() : *stream;
    }

    static void device_synchronize()
    {
        CHECK_HIP_ERROR(hipDeviceSynchronize());
    }

    static void stream_create(stream_type* stream)
    {
        CHECK_HIP_ERROR(hipStreamCreate(stream));
    }

    static void stream_create_nonblocking(stream_type* stream)
    {
        CHECK_HIP_ERROR(hipStreamCreateWithFlags(stream, hipStreamNonBlocking));
    }

    static void stream_destroy(stream_type stream)
    {
        CHECK_HIP_ERROR(hipStreamDestroy(stream));
    }

    static void stream_synchronize(stream_type stream)
    {
        CHECK_HIP_ERROR(hipStreamSynchronize(stream));
    }

    static void stream_wait_event(stream_type stream, event_type event)
    {
        CHECK_HIP_ERROR(hipStreamWaitEvent(stream, event, 0));
    }

    static void event_create(event_type* event)
    {
        CHECK_HIP_ERROR(hipEventCreate(event));
    }

    static void event_record(event_type event, stream_type stream = nullptr)
    {
        CHECK_HIP_ERROR(hipEventRecord(event, stream));
    }

    static void event_synchronize(event_type event)
    {
        CHECK_HIP_ERROR(hipEventSynchronize(event));
    }

    static void event_elapsed_time(float* ms, event_type start,
                                   event_type stop)
    {
        CHECK_HIP_ERROR(hipEventElapsedTime(ms, start, stop));
    }

    static void event_destroy(event_type event)
    {
        CHECK_HIP_ERROR(hipEventDestroy(event));
    }

    static void device_malloc(void** ptr, std::size_t bytes)
    {
        CHECK_HIP_ERROR(hipMalloc(ptr, bytes));
    }

    static void device_malloc_async(void** ptr, std::size_t bytes,
                                    stream_type stream)
    {
        CHECK_HIP_ERROR(hipMallocAsync(ptr, bytes, stream));
    }

    static void device_free(void* ptr)
    {
        CHECK_HIP_ERROR(hipFree(ptr));
    }

    static void device_memset(void* ptr, int value, std::size_t bytes)
    {
        CHECK_HIP_ERROR(hipMemset(ptr, value, bytes));
    }

    static void device_memset_async(void* ptr, int value, std::size_t bytes,
                                    stream_type stream)
    {
        CHECK_HIP_ERROR(hipMemsetAsync(ptr, value, bytes, stream));
    }

    static void host_pinned_malloc(void** ptr, std::size_t bytes)
    {
        CHECK_HIP_ERROR(hipHostMalloc(ptr, bytes));
    }

    static void host_pinned_free(void* ptr)
    {
        CHECK_HIP_ERROR(hipHostFree(ptr));
    }

    static void memcpy_h2d(void* dst, const void* src, std::size_t bytes)
    {
        CHECK_HIP_ERROR(hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice));
    }

    static void memcpy_h2d_async(void* dst, const void* src, std::size_t bytes,
                                 stream_type stream)
    {
        CHECK_HIP_ERROR(
            hipMemcpyAsync(dst, src, bytes, hipMemcpyHostToDevice, stream));
    }

    static void memcpy_d2h(void* dst, const void* src, std::size_t bytes)
    {
        CHECK_HIP_ERROR(hipMemcpy(dst, src, bytes, hipMemcpyDeviceToHost));
    }

    static void memcpy_d2d(void* dst, const void* src, std::size_t bytes)
    {
        CHECK_HIP_ERROR(hipMemcpy(dst, src, bytes, hipMemcpyDeviceToDevice));
    }

    static void memcpy_d2d_async(void* dst, const void* src, std::size_t bytes,
                                 stream_type stream)
    {
        CHECK_HIP_ERROR(
            hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToDevice, stream));
    }

    static void memcpy_d2h_async(void* dst, const void* src, std::size_t bytes,
                                 stream_type stream)
    {
        CHECK_HIP_ERROR(
            hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToHost, stream));
    }

    static void memcpy_2d_h2d(void* dst, std::size_t dst_pitch,
                              const void* src, std::size_t src_pitch,
                              std::size_t width, std::size_t height)
    {
        CHECK_HIP_ERROR(hipMemcpy2D(dst, dst_pitch, src, src_pitch, width,
                                    height, hipMemcpyHostToDevice));
    }

    static void memcpy_2d_d2h(void* dst, std::size_t dst_pitch,
                              const void* src, std::size_t src_pitch,
                              std::size_t width, std::size_t height)
    {
        CHECK_HIP_ERROR(hipMemcpy2D(dst, dst_pitch, src, src_pitch, width,
                                    height, hipMemcpyDeviceToHost));
    }

    static void copy_matrix_h2d(std::size_t rows, std::size_t cols,
                                std::size_t elem_size, const void* src,
                                std::size_t src_ld, void* dst,
                                std::size_t dst_ld)
    {
        memcpy_2d_h2d(dst, dst_ld * elem_size, src, src_ld * elem_size,
                      rows * elem_size, cols);
    }

    static void copy_matrix_d2h(std::size_t rows, std::size_t cols,
                                std::size_t elem_size, const void* src,
                                std::size_t src_ld, void* dst,
                                std::size_t dst_ld)
    {
        memcpy_2d_d2h(dst, dst_ld * elem_size, src, src_ld * elem_size,
                      rows * elem_size, cols);
    }
};
#endif

#if defined(HAS_CUDA) || defined(HAS_HIP)

/** @brief Deleter for pointers from `device_malloc`. */
template <typename Backend>
struct runtime_device_deleter
{
    template <typename T>
    void operator()(T* ptr) const
    {
        if (ptr != nullptr)
        {
            runtime<Backend>::device_free(ptr);
        }
    }
};

template <typename Backend>
inline typename runtime<Backend>::stream_type create_stream()
{
    typename runtime<Backend>::stream_type stream{};
    runtime<Backend>::stream_create(&stream);
    return stream;
}

template <typename Backend>
inline typename runtime<Backend>::stream_type create_nonblocking_stream()
{
    typename runtime<Backend>::stream_type stream{};
    runtime<Backend>::stream_create_nonblocking(&stream);
    return stream;
}

template <typename Backend>
inline void destroy_stream(typename runtime<Backend>::stream_type stream)
{
    runtime<Backend>::stream_destroy(stream);
}

template <typename Backend>
inline void stream_synchronize(typename runtime<Backend>::stream_type stream)
{
    runtime<Backend>::stream_synchronize(stream);
}

template <typename Backend>
inline void device_synchronize()
{
    runtime<Backend>::device_synchronize();
}

template <typename Backend>
inline typename runtime<Backend>::event_type create_event()
{
    typename runtime<Backend>::event_type event{};
    runtime<Backend>::event_create(&event);
    return event;
}

template <typename Backend>
inline void destroy_event(typename runtime<Backend>::event_type event)
{
    runtime<Backend>::event_destroy(event);
}

template <typename Backend>
inline void record_event(typename runtime<Backend>::event_type event,
                         typename runtime<Backend>::stream_type stream)
{
    runtime<Backend>::event_record(event, stream);
}

template <typename Backend>
inline void stream_wait_event(typename runtime<Backend>::stream_type stream,
                              typename runtime<Backend>::event_type event)
{
    runtime<Backend>::stream_wait_event(stream, event);
}

/**
 * @brief Pointer to @p stream for kernel launchers that accept `stream_type*`.
 *
 * Returns null when @p stream is the legacy default stream (`nullptr`), matching
 * `runtime<Backend>::resolve_stream` in the opposite direction.
 */
template <typename Stream>
inline Stream* optional_stream_ptr(Stream& stream)
{
    return stream == nullptr ? nullptr : &stream;
}

template <typename Backend, typename T>
inline void device_malloc(T** ptr, std::size_t count)
{
    runtime<Backend>::device_malloc(reinterpret_cast<void**>(ptr),
                                    count * sizeof(T));
}

template <typename Backend, typename T>
inline void device_free(T* ptr)
{
    if (ptr != nullptr)
    {
        runtime<Backend>::device_free(ptr);
    }
}

template <typename Backend, typename T>
inline void memset_zero_async(T* ptr, std::size_t count,
                              typename runtime<Backend>::stream_type stream)
{
    runtime<Backend>::device_memset_async(ptr, 0, count * sizeof(T), stream);
}

template <typename Backend, typename T>
inline void memcpy_d2h(T* dst, const T* src, std::size_t count)
{
    runtime<Backend>::memcpy_d2h(dst, src, count * sizeof(T));
}

template <typename Backend, typename T>
inline void memcpy_h2d(T* dst, const T* src, std::size_t count)
{
    runtime<Backend>::memcpy_h2d(dst, src, count * sizeof(T));
}

template <typename Backend, typename T>
inline void memcpy_d2d(T* dst, const T* src, std::size_t count)
{
    runtime<Backend>::memcpy_d2d(dst, src, count * sizeof(T));
}

template <typename Backend, typename T>
inline void memcpy_d2h_async(T* dst, const T* src, std::size_t count,
                             typename runtime<Backend>::stream_type stream)
{
    runtime<Backend>::memcpy_d2h_async(dst, src, count * sizeof(T), stream);
}

template <typename Backend, typename T>
inline void memcpy_h2d_async(T* dst, const T* src, std::size_t count,
                             typename runtime<Backend>::stream_type stream)
{
    runtime<Backend>::memcpy_h2d_async(dst, src, count * sizeof(T), stream);
}

template <typename Backend, typename T>
inline void memcpy_d2d_async(T* dst, const T* src, std::size_t count,
                             typename runtime<Backend>::stream_type stream)
{
    runtime<Backend>::memcpy_d2d_async(dst, src, count * sizeof(T), stream);
}

template <typename Backend, typename T>
inline T* device_malloc_array(std::size_t count)
{
    T* ptr = nullptr;
    device_malloc<Backend, T>(&ptr, count);
    return ptr;
}

template <typename Backend, typename T>
inline void device_malloc_async(T** ptr, std::size_t count,
                                typename runtime<Backend>::stream_type stream)
{
    runtime<Backend>::device_malloc_async(reinterpret_cast<void**>(ptr),
                                          count * sizeof(T), stream);
}

template <typename Backend>
inline void memset_async(void* ptr, int value, std::size_t bytes,
                         typename runtime<Backend>::stream_type stream)
{
    runtime<Backend>::device_memset_async(ptr, value, bytes, stream);
}

template <typename Backend, typename T>
inline void copy_to_host(T* dst, const T* src, std::size_t count)
{
    memcpy_d2h<Backend, T>(dst, src, count);
}

template <typename Backend, typename T>
inline void copy_to_device(T* dst, const T* src, std::size_t count)
{
    memcpy_h2d<Backend, T>(dst, src, count);
}

template <typename Backend, typename T>
inline void memcpy_async_device_to_host(T* host_dst, const T* device_src,
                                        std::size_t count,
                                        typename runtime<Backend>::stream_type stream)
{
    memcpy_d2h_async<Backend, T>(host_dst, device_src, count, stream);
}

template <typename Backend, typename T>
inline void memcpy_async_host_to_device(T* device_dst, const T* host_src,
                                        std::size_t count,
                                        typename runtime<Backend>::stream_type stream)
{
    memcpy_h2d_async<Backend, T>(device_dst, host_src, count, stream);
}

#endif // defined(HAS_CUDA) || defined(HAS_HIP)

} // namespace gpu_backend
} // namespace internal
} // namespace linalg
} // namespace chase

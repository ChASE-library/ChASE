// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#if defined(HAS_CUDA) || defined(HAS_HIP)

#define CHASE_GPU_MATRIX 1

#include "linalg/internal/gpu/kernels/shared/device_kernels.hpp"
#include "linalg/internal/gpu/backend/runtime.hpp"

#include <cstddef>
#include <cstring>
#include <complex>
#include <functional>
#include <memory>
#include <stdexcept>

/**
 * @page matrix_gpu_runtime_module chase::matrix::gpu_runtime
 * @brief GPU runtime facade for serial `Matrix<T, GPU>` and related helpers.
 *
 * This header is the matrix-layer portability boundary for CUDA and HIP builds.
 * It forwards device allocation, host/device copies, precision-conversion
 * kernels, and stream helpers to `gpu_backend::runtime` and
 * `gpu_device::kernels` without exposing vendor-specific APIs to
 * `matrix.hpp` or `matrix_gpu.hpp`.
 *
 * When either `HAS_CUDA` or `HAS_HIP` is defined, this header also defines
 * `CHASE_GPU_MATRIX`, which gates GPU matrix specializations.
 */

namespace chase
{
namespace matrix
{
namespace gpu_runtime
{

using runtime_backend = chase::linalg::internal::gpu_backend::active_backend;
using runtime_type = chase::linalg::internal::gpu_backend::runtime<runtime_backend>;
using kernel_type = chase::linalg::internal::gpu_device::kernels<runtime_backend>;
using chase_gpu_stream_t = typename runtime_type::stream_type;
using chase_gpu_stream_ptr_t = chase_gpu_stream_t*;

/**
 * @brief Resolve an optional GPU stream pointer to a concrete stream handle.
 *
 * A null pointer selects the backend default stream (typically the legacy
 * default stream).
 *
 * @param stream Optional stream pointer.
 * @return Stream handle used for subsequent GPU operations.
 */
inline chase_gpu_stream_t resolve_stream(chase_gpu_stream_ptr_t stream)
{
    return runtime_type::resolve_stream(stream);
}

/**
 * @brief Allocate device memory.
 *
 * @param ptr Output pointer to the allocated device address.
 * @param bytes Number of bytes to allocate.
 */
inline void device_malloc(void** ptr, std::size_t bytes)
{
    runtime_type::device_malloc(ptr, bytes);
}

/**
 * @brief Free device memory previously allocated by `device_malloc`.
 *
 * @param ptr Device pointer to release. No-op if null.
 */
inline void device_free(void* ptr)
{
    runtime_type::device_free(ptr);
}

/**
 * @brief Set device memory to a byte value.
 *
 * @param ptr Device pointer.
 * @param value Byte value written to each byte of the range.
 * @param bytes Number of bytes to initialize.
 */
inline void device_memset(void* ptr, int value, std::size_t bytes)
{
    runtime_type::device_memset(ptr, value, bytes);
}

/**
 * @brief Allocate page-locked (pinned) host memory for faster H2D/D2H copies.
 *
 * @param ptr Output pointer to the allocated host address.
 * @param bytes Number of bytes to allocate.
 */
inline void host_pinned_malloc(void** ptr, std::size_t bytes)
{
    runtime_type::host_pinned_malloc(ptr, bytes);
}

/**
 * @brief Free pinned host memory allocated by `host_pinned_malloc`.
 *
 * @param ptr Host pointer to release. No-op if null.
 */
inline void host_pinned_free(void* ptr)
{
    runtime_type::host_pinned_free(ptr);
}

/**
 * @brief Copy a dense submatrix from host to device.
 *
 * Supports general leading dimensions (`src_ld`, `dst_ld`) so non-contiguous
 * host or device storage can be copied row-wise.
 *
 * @param rows Number of rows to copy.
 * @param cols Number of columns to copy.
 * @param elem_size Size of one matrix element in bytes.
 * @param src Host source buffer.
 * @param src_ld Leading dimension of the host source.
 * @param dst Device destination buffer.
 * @param dst_ld Leading dimension of the device destination.
 */
inline void copy_matrix_h2d(std::size_t rows, std::size_t cols, std::size_t elem_size,
                            const void* src, std::size_t src_ld, void* dst,
                            std::size_t dst_ld)
{
    runtime_type::copy_matrix_h2d(rows, cols, elem_size, src, src_ld, dst,
                                  dst_ld);
}

/**
 * @brief Copy a dense submatrix from device to host.
 *
 * @param rows Number of rows to copy.
 * @param cols Number of columns to copy.
 * @param elem_size Size of one matrix element in bytes.
 * @param src Device source buffer.
 * @param src_ld Leading dimension of the device source.
 * @param dst Host destination buffer.
 * @param dst_ld Leading dimension of the host destination.
 */
inline void copy_matrix_d2h(std::size_t rows, std::size_t cols, std::size_t elem_size,
                            const void* src, std::size_t src_ld, void* dst,
                            std::size_t dst_ld)
{
    runtime_type::copy_matrix_d2h(rows, cols, elem_size, src, src_ld, dst,
                                  dst_ld);
}

/**
 * @brief Convert double-precision values to single precision on the device.
 *
 * @param input Device input buffer (`double`).
 * @param output Device output buffer (`float`).
 * @param size Number of elements to convert.
 * @param stream Optional GPU stream.
 */
inline void convert_dp_to_sp(double* input, float* output, std::size_t size,
                             chase_gpu_stream_ptr_t stream = nullptr)
{
    kernel_type::convert_dp_to_sp(input, output, size, stream);
}

/**
 * @brief Convert double-precision complex values to single precision on the device.
 *
 * @param input Device input buffer (`std::complex<double>`).
 * @param output Device output buffer (`std::complex<float>`).
 * @param size Number of complex elements to convert.
 * @param stream Optional GPU stream.
 */
inline void convert_dp_to_sp(std::complex<double>* input, std::complex<float>* output,
                             std::size_t size,
                             chase_gpu_stream_ptr_t stream = nullptr)
{
    kernel_type::convert_dp_to_sp(input, output, size, stream);
}

/**
 * @brief Convert single-precision values to double precision on the device.
 *
 * @param input Device input buffer (`float`).
 * @param output Device output buffer (`double`).
 * @param size Number of elements to convert.
 * @param stream Optional GPU stream.
 */
inline void convert_sp_to_dp(float* input, double* output, std::size_t size,
                             chase_gpu_stream_ptr_t stream = nullptr)
{
    kernel_type::convert_sp_to_dp(input, output, size, stream);
}

/**
 * @brief Convert single-precision complex values to double precision on the device.
 *
 * @param input Device input buffer (`std::complex<float>`).
 * @param output Device output buffer (`std::complex<double>`).
 * @param size Number of complex elements to convert.
 * @param stream Optional GPU stream.
 */
inline void convert_sp_to_dp(std::complex<float>* input, std::complex<double>* output,
                             std::size_t size,
                             chase_gpu_stream_ptr_t stream = nullptr)
{
    kernel_type::convert_sp_to_dp(input, output, size, stream);
}

/**
 * @ingroup Allocators
 * @brief STL-style allocator for GPU device memory (CUDA or HIP).
 *
 * Used by `Matrix<T, chase::platform::GPU>` and mixed-precision matrix helpers
 * to allocate through the backend-neutral `device_malloc` / `device_free`
 * entry points.
 *
 * @tparam T Element type stored in device memory.
 */
template <typename T>
class GpuAllocator
{
public:
    using value_type = T;

    GpuAllocator() = default;

    /**
     * @brief Rebind-compatible copy constructor.
     * @tparam U Other element type.
     */
    template <class U>
    constexpr GpuAllocator(const GpuAllocator<U>&) noexcept
    {
    }

    /**
     * @brief Allocate `n` elements on the device.
     * @param n Number of elements.
     * @return Device pointer to allocated storage.
     */
    T* allocate(std::size_t n)
    {
        T* ptr = nullptr;
        void* raw = nullptr;
        device_malloc(&raw, n * sizeof(T));
        ptr = static_cast<T*>(raw);
        return ptr;
    }

    /**
     * @brief Deallocate device storage.
     * @param ptr Pointer returned by `allocate`.
     * @param Unused element count (STL allocator interface).
     */
    void deallocate(T* ptr, std::size_t) noexcept
    {
        device_free(ptr);
    }
};

} // namespace gpu_runtime
} // namespace matrix
} // namespace chase

#endif // defined(HAS_CUDA) || defined(HAS_HIP)

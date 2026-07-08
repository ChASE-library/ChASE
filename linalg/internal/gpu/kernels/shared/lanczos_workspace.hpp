// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#if defined(HAS_CUDA) || defined(HAS_HIP)

#include "algorithm/types.hpp"
#include "linalg/internal/gpu/backend/runtime.hpp"

#include <cstddef>

namespace chase::linalg::internal::gpu_device
{

namespace detail
{

template <typename Backend, typename T>
inline void alloc_device_buffer(
    T** ptr, std::size_t count,
    typename gpu_backend::runtime<Backend>::stream_type stream,
    bool stream_ordered)
{
    if (stream_ordered)
    {
        gpu_backend::device_malloc_async<Backend, T>(ptr, count, stream);
    }
    else
    {
        gpu_backend::device_malloc<Backend, T>(ptr, count);
    }
}

} // namespace detail

/** Device scalars and tridiagonal buffers for standard Hermitian Lanczos. */
template <typename Backend, typename T>
struct lanczos_hermitian_workspace
{
    using RealT = chase::Base<T>;
    using stream_t = typename gpu_backend::runtime<Backend>::stream_type;

    T* d_alpha = nullptr;
    RealT* d_real_alpha = nullptr;
    RealT* d_r_beta = nullptr;
    RealT* d_beta_prev = nullptr;
    T* d_beta_neg = nullptr;
    RealT* d_de = nullptr;
    RealT* d_d = nullptr;
    RealT* e_d = nullptr;

    lanczos_hermitian_workspace(std::size_t M, std::size_t numvec,
                               stream_t stream, bool stream_ordered)
    {
        detail::alloc_device_buffer<Backend, T>(&d_alpha, numvec, stream,
                                               stream_ordered);
        detail::alloc_device_buffer<Backend, RealT>(&d_real_alpha, numvec, stream,
                                                   stream_ordered);
        detail::alloc_device_buffer<Backend, RealT>(&d_r_beta, numvec, stream,
                                                   stream_ordered);
        detail::alloc_device_buffer<Backend, RealT>(&d_beta_prev, numvec, stream,
                                                    stream_ordered);
        detail::alloc_device_buffer<Backend, T>(&d_beta_neg, numvec, stream,
                                               stream_ordered);
        detail::alloc_device_buffer<Backend, RealT>(&d_de, 2 * M * numvec, stream,
                                                    stream_ordered);
        d_d = d_de;
        e_d = d_de + M * numvec;
    }

    lanczos_hermitian_workspace(const lanczos_hermitian_workspace&) = delete;
    lanczos_hermitian_workspace& operator=(const lanczos_hermitian_workspace&) =
        delete;

    ~lanczos_hermitian_workspace()
    {
        gpu_backend::device_free<Backend>(d_alpha);
        gpu_backend::device_free<Backend>(d_real_alpha);
        gpu_backend::device_free<Backend>(d_r_beta);
        gpu_backend::device_free<Backend>(d_beta_prev);
        gpu_backend::device_free<Backend>(d_beta_neg);
        gpu_backend::device_free<Backend>(d_de);
    }

    void zero_scalars(std::size_t M, std::size_t numvec, stream_t stream)
    {
        gpu_backend::memset_async<Backend>(d_alpha, 0, numvec * sizeof(T), stream);
        gpu_backend::memset_async<Backend>(d_real_alpha, 0, numvec * sizeof(RealT),
                                           stream);
        gpu_backend::memset_async<Backend>(d_r_beta, 0, numvec * sizeof(RealT),
                                           stream);
        gpu_backend::memset_async<Backend>(d_beta_prev, 0, numvec * sizeof(RealT),
                                           stream);
        gpu_backend::memset_async<Backend>(e_d, 0, M * numvec * sizeof(RealT),
                                           stream);
    }
};

/** Device scalars and tridiagonal buffers for pseudo-Hermitian Lanczos. */
template <typename Backend, typename T>
struct lanczos_pseudo_hermitian_workspace
{
    using RealT = chase::Base<T>;
    using stream_t = typename gpu_backend::runtime<Backend>::stream_type;

    T* d_alpha = nullptr;
    T* d_beta = nullptr;
    RealT* d_real_alpha = nullptr;
    RealT* d_real_beta = nullptr;
    RealT* d_real_beta_prev = nullptr;
    RealT* d_de = nullptr;
    RealT* d_d = nullptr;
    RealT* e_d = nullptr;

    lanczos_pseudo_hermitian_workspace(std::size_t M, std::size_t numvec,
                                       stream_t stream, bool stream_ordered)
    {
        detail::alloc_device_buffer<Backend, T>(&d_alpha, numvec, stream,
                                               stream_ordered);
        detail::alloc_device_buffer<Backend, T>(&d_beta, numvec, stream,
                                               stream_ordered);
        detail::alloc_device_buffer<Backend, RealT>(&d_real_alpha, numvec, stream,
                                                   stream_ordered);
        detail::alloc_device_buffer<Backend, RealT>(&d_real_beta, numvec, stream,
                                                   stream_ordered);
        detail::alloc_device_buffer<Backend, RealT>(&d_real_beta_prev, numvec,
                                                    stream, stream_ordered);
        detail::alloc_device_buffer<Backend, RealT>(&d_de, 2 * M * numvec, stream,
                                                    stream_ordered);
        d_d = d_de;
        e_d = d_de + M * numvec;
    }

    lanczos_pseudo_hermitian_workspace(const lanczos_pseudo_hermitian_workspace&) =
        delete;
    lanczos_pseudo_hermitian_workspace& operator=(
        const lanczos_pseudo_hermitian_workspace&) = delete;

    ~lanczos_pseudo_hermitian_workspace()
    {
        gpu_backend::device_free<Backend>(d_alpha);
        gpu_backend::device_free<Backend>(d_beta);
        gpu_backend::device_free<Backend>(d_real_alpha);
        gpu_backend::device_free<Backend>(d_real_beta);
        gpu_backend::device_free<Backend>(d_real_beta_prev);
        gpu_backend::device_free<Backend>(d_de);
    }

    void zero_scalars(std::size_t M, std::size_t numvec, stream_t stream)
    {
        gpu_backend::memset_async<Backend>(d_alpha, 0, numvec * sizeof(T), stream);
        gpu_backend::memset_async<Backend>(d_beta, 0, numvec * sizeof(T), stream);
        gpu_backend::memset_async<Backend>(d_real_alpha, 0, numvec * sizeof(RealT),
                                           stream);
        gpu_backend::memset_async<Backend>(d_real_beta, 0, numvec * sizeof(RealT),
                                           stream);
        gpu_backend::memset_async<Backend>(d_real_beta_prev, 0,
                                           numvec * sizeof(RealT), stream);
        gpu_backend::memset_async<Backend>(e_d, 0, M * numvec * sizeof(RealT),
                                           stream);
    }
};

} // namespace gpu_device

#endif

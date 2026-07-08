// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#if defined(HAS_CUDA) || defined(HAS_HIP)

#include "linalg/internal/gpu/backend/blas.hpp"
#include "linalg/internal/gpu/backend/context.hpp"
#include "linalg/internal/gpu/kernels/shared/device_kernels.hpp"
#include "linalg/internal/gpu/backend/runtime.hpp"
#include "linalg/internal/gpu/dist_mpi/impl/mpi_collectives.hpp"

#if defined(HAS_NCCL) || defined(HAS_RCCL)
#include "grid/nccl_utils.hpp"
#define CHASE_HAS_DIST_GPU_COLLECTIVE 1
#endif

#include <complex>
#include <limits>
#include <mpi.h>
#include <stdexcept>

namespace chase
{
namespace distMatrix
{
namespace gpu_portability
{

using runtime_backend = chase::linalg::internal::gpu_backend::active_backend;
using runtime_type = chase::linalg::internal::gpu_backend::runtime<runtime_backend>;
using kernel_type = chase::linalg::internal::gpu_device::kernels<runtime_backend>;
using blas_type = chase::linalg::internal::gpu_backend::blas<runtime_backend>;
using swap_context_type =
    chase::linalg::internal::gpu_backend::GpuExecutionContext<runtime_backend>;
using stream_type = typename runtime_type::stream_type;

inline swap_context_type& swap_blas_context()
{
    static thread_local swap_context_type context(false);
    return context;
}

#ifdef CHASE_HAS_DIST_GPU_COLLECTIVE
using collective_stream_type = GpuCollectiveStream_t;

inline collective_stream_type default_collective_stream()
{
    return runtime_type::default_stream();
}

inline collective_stream_type resolve_collective_stream(
    collective_stream_type* stream)
{
    return stream == nullptr ? default_collective_stream() : *stream;
}

template <typename T>
inline void collectiveBcast(T* buffer, std::size_t count, int root,
                            ncclComm_t comm,
                            collective_stream_type* stream = nullptr)
{
    CHECK_NCCL_ERROR(
        chase::nccl::ncclBcastWrapper(buffer, count, root, comm, stream));
}

template <typename T>
inline void collectiveSendrecv(T* sendbuffer, std::size_t sendcount,
                               int dest_rank, T* recvbuffer,
                               std::size_t recvcount, int src_rank,
                               ncclComm_t comm,
                               collective_stream_type* stream = nullptr)
{
    CHECK_NCCL_ERROR(chase::nccl::ncclSendrecvWrapper(
        sendbuffer, sendcount, dest_rank, recvbuffer, recvcount, src_rank, comm,
        stream));
}
#endif

inline void deviceSynchronize()
{
    runtime_type::device_synchronize();
}

template <typename T>
inline void t_lacpy(char uplo, std::size_t m, std::size_t n, T* dA,
                    std::size_t ldda, T* dB, std::size_t lddb)
{
    kernel_type::lacpy(uplo, m, n, dA, ldda, dB, lddb);
}

template <typename T>
inline void t_lacpy(char uplo, std::size_t m, std::size_t n, T* dA,
                    std::size_t ldda, T* dB, std::size_t lddb,
                    stream_type* stream_)
{
    kernel_type::lacpy(uplo, m, n, dA, ldda, dB, lddb, stream_);
}

template <typename T>
inline void mpiBcast(T* device_buf, std::size_t count, int root, MPI_Comm comm)
{
    chase::linalg::internal::gpu_mpi::mpiBcast(device_buf, count, root,
                                                      comm);
}

template <typename T>
inline void mpiSendrecv(const T* send_device, std::size_t sendcount, int dest,
                        int sendtag, T* recv_device, std::size_t recvcount,
                        int source, int recvtag, MPI_Comm comm)
{
    chase::linalg::internal::gpu_mpi::mpiSendrecv(
        send_device, sendcount, dest, sendtag, recv_device, recvcount, source,
        recvtag, comm);
}

inline void convert_DP_TO_SP_GPU(double* input, float* output, std::size_t size)
{
    kernel_type::convert_dp_to_sp(input, output, size);
}

inline void convert_DP_TO_SP_GPU(double* input, float* output, std::size_t size,
                                 stream_type* stream)
{
    kernel_type::convert_dp_to_sp(input, output, size, stream);
}

inline void convert_DP_TO_SP_GPU(std::complex<double>* input,
                                 std::complex<float>* output, std::size_t size)
{
    kernel_type::convert_dp_to_sp(input, output, size);
}

inline void convert_DP_TO_SP_GPU(std::complex<double>* input,
                                 std::complex<float>* output, std::size_t size,
                                 stream_type* stream)
{
    kernel_type::convert_dp_to_sp(input, output, size, stream);
}

inline void convert_SP_TO_DP_GPU(float* input, double* output, std::size_t size)
{
    kernel_type::convert_sp_to_dp(input, output, size);
}

inline void convert_SP_TO_DP_GPU(float* input, double* output, std::size_t size,
                                 stream_type* stream)
{
    kernel_type::convert_sp_to_dp(input, output, size, stream);
}

inline void convert_SP_TO_DP_GPU(std::complex<float>* input,
                                 std::complex<double>* output, std::size_t size)
{
    kernel_type::convert_sp_to_dp(input, output, size);
}

inline void convert_SP_TO_DP_GPU(std::complex<float>* input,
                                 std::complex<double>* output, std::size_t size,
                                 stream_type* stream)
{
    kernel_type::convert_sp_to_dp(input, output, size, stream);
}

template <typename T>
inline void conjugate_inplace(T* input, std::size_t rows, std::size_t cols,
                              std::size_t ld,
                              stream_type stream = runtime_type::default_stream())
{
    kernel_type::conjugate_inplace(input, rows, cols, ld, stream);
}

template <typename T>
inline void t_swap(std::size_t n, T* x, std::size_t incx, T* y,
                   std::size_t incy)
{
    blas_type::check_status(
        blas_type::swap(swap_blas_context().blas(), n, x, incx, y, incy));
}

} // namespace gpu_portability
} // namespace distMatrix
} // namespace chase

#endif // defined(HAS_CUDA) || defined(HAS_HIP)

// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "grid/mpiTypes.hpp"
#include "linalg/internal/gpu/backend/runtime.hpp"
#include "mpi.h"

#include <limits>
#include <stdexcept>

namespace chase
{
namespace linalg
{
namespace internal
{
namespace gpu_mpi
{

using mpi_stream_type =
    typename gpu_backend::runtime<gpu_backend::active_backend>::stream_type;

inline void deviceSynchronizeForMpi()
{
#if defined(HAS_CUDA) || defined(HAS_HIP)
    gpu_backend::device_synchronize<gpu_backend::active_backend>();
#endif
}

inline void syncStreamForMpi(mpi_stream_type stream)
{
#if defined(HAS_CUDA) || defined(HAS_HIP)
    gpu_backend::stream_synchronize<gpu_backend::active_backend>(stream);
#endif
}

/** Device-buffer MPI collectives (CUDA/HIP-aware MPI parity). */
template <typename T>
void mpiAllreduceSum(T* device_buf, std::size_t count, MPI_Comm comm,
                     mpi_stream_type stream)
{
    syncStreamForMpi(stream);
    if (count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::runtime_error("GPU MPI Allreduce count exceeds MPI int range");
    }
    MPI_Allreduce(MPI_IN_PLACE, device_buf, static_cast<int>(count),
                  chase::mpi::getMPI_Type<T>(), MPI_SUM, comm);
}

template <typename T>
void mpiAllreduceSum(T* device_buf, std::size_t count, MPI_Comm comm)
{
    deviceSynchronizeForMpi();
    if (count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::runtime_error("GPU MPI Allreduce count exceeds MPI int range");
    }
    MPI_Allreduce(MPI_IN_PLACE, device_buf, static_cast<int>(count),
                  chase::mpi::getMPI_Type<T>(), MPI_SUM, comm);
    deviceSynchronizeForMpi();
}

template <typename T>
void mpiBcast(T* device_buf, std::size_t count, int root, MPI_Comm comm)
{
    deviceSynchronizeForMpi();
    if (count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::runtime_error("GPU MPI Bcast count exceeds MPI int range");
    }
    MPI_Bcast(device_buf, static_cast<int>(count), chase::mpi::getMPI_Type<T>(),
              root, comm);
    deviceSynchronizeForMpi();
}

template <typename T>
void mpiSendrecv(const T* send_device, std::size_t sendcount, int dest,
                 int sendtag, T* recv_device, std::size_t recvcount, int source,
                 int recvtag, MPI_Comm comm)
{
    deviceSynchronizeForMpi();
    if (sendcount > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        recvcount > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::runtime_error("GPU MPI Sendrecv count exceeds MPI int range");
    }
    MPI_Sendrecv(send_device, static_cast<int>(sendcount),
                 chase::mpi::getMPI_Type<T>(), dest, sendtag, recv_device,
                 static_cast<int>(recvcount), chase::mpi::getMPI_Type<T>(),
                 source, recvtag, comm, MPI_STATUS_IGNORE);
    deviceSynchronizeForMpi();
}

} // namespace gpu_mpi
} // namespace internal
} // namespace linalg
} // namespace chase

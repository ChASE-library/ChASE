// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "linalg/internal/gpu/backend/error.hpp"
#include "external/amd_gpu/rocblaspp/rocblaspp.hpp"
#include "linalg/internal/gpu/dist_mpi/impl/mpi_collectives.hpp"

namespace chase
{
namespace linalg
{
namespace internal
{
namespace hip_mpi_utils
{

struct DeviceDeleter
{
    void operator()(void* ptr) const
    {
        if (ptr)
        {
            CHECK_HIP_ERROR(hipFree(ptr));
        }
    }
};

inline hipStream_t rocblasHipStream(rocblas_handle handle)
{
    hipStream_t stream = nullptr;
    CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
    return stream;
}

inline void syncHipStream(hipStream_t stream)
{
    CHECK_HIP_ERROR(hipStreamSynchronize(stream));
}

inline void syncRocblasHipStream(rocblas_handle handle)
{
    syncHipStream(rocblasHipStream(handle));
}

template <typename T>
void mpiAllreduceSum(T* device_buf, std::size_t count, MPI_Comm comm)
{
    gpu_mpi::mpiAllreduceSum(device_buf, count, comm);
}

template <typename T>
void mpiBcast(T* device_buf, std::size_t count, int root, MPI_Comm comm)
{
    gpu_mpi::mpiBcast(device_buf, count, root, comm);
}

template <typename T>
void mpiSendrecv(const T* send_device, std::size_t sendcount, int dest,
                 int sendtag, T* recv_device, std::size_t recvcount, int source,
                 int recvtag, MPI_Comm comm)
{
    gpu_mpi::mpiSendrecv(send_device, sendcount, dest, sendtag,
                               recv_device, recvcount, source, recvtag, comm);
}

} // namespace hip_mpi_utils
} // namespace internal
} // namespace linalg
} // namespace chase

// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

/**
 * @brief Configure- and run-time probe for GPU-aware MPI with HIP (Frontier Cray MPICH).
 *
 * Exit codes:
 *   0 — GPU-aware MPI works (query API and/or device buffer collective).
 *   1 — HIP runtime error.
 *   2 — MPI reports no GPU-aware support.
 *   3 — device-buffer MPI collective failed plausibility check.
 */

#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <mpi.h>

#if defined(MPICH) && __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#endif

namespace
{

int query_gpu_mpi()
{
#if defined(MPIX_Query_hip_support)
    return MPIX_Query_hip_support() ? 1 : 0;
#elif defined(MPIX_Query_cuda_support)
    // Cray MPICH on AMD Frontier historically exposes this name for GPU buffers.
    return MPIX_Query_cuda_support() ? 1 : 0;
#else
    return -1;
#endif
}

bool device_collective_probe(int rank, int size)
{
    constexpr int count = 4;
    float* device_buffer = nullptr;
    if (hipMalloc(&device_buffer, count * sizeof(float)) != hipSuccess)
    {
        return false;
    }

    if (rank == 0)
    {
        const float host_init[count] = {1.0f, 2.0f, 3.0f, 4.0f};
        if (hipMemcpy(device_buffer, host_init, count * sizeof(float),
                      hipMemcpyHostToDevice) != hipSuccess)
        {
            hipFree(device_buffer);
            return false;
        }
    }

    const int collective_rc =
        MPI_Bcast(device_buffer, count, MPI_FLOAT, 0, MPI_COMM_WORLD);
    if (collective_rc != MPI_SUCCESS)
    {
        hipFree(device_buffer);
        return false;
    }

    float host_result[count] = {};
    if (hipMemcpy(host_result, device_buffer, count * sizeof(float),
                  hipMemcpyDeviceToHost) != hipSuccess)
    {
        hipFree(device_buffer);
        return false;
    }
    hipFree(device_buffer);

    if (size == 1)
    {
        return host_result[0] == 1.0f && host_result[3] == 4.0f;
    }

    return host_result[0] == 1.0f && host_result[1] == 2.0f &&
           host_result[2] == 3.0f && host_result[3] == 4.0f;
}

} // namespace

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int aware = query_gpu_mpi();
    if (aware == 0)
    {
        if (rank == 0)
        {
            std::fprintf(stderr,
                         "HIP-aware MPI probe: MPIX_Query_* reports no "
                         "GPU-aware support (set MPICH_GPU_SUPPORT_ENABLED=1 "
                         "on Frontier).\n");
        }
        MPI_Finalize();
        return 2;
    }

    if (!device_collective_probe(rank, size))
    {
        if (rank == 0)
        {
            std::fprintf(stderr,
                         "HIP-aware MPI probe: device-buffer MPI_Bcast failed "
                         "(GPU-aware MPI not active or no GPU visible).\n");
        }
        MPI_Finalize();
        return 3;
    }

    if (rank == 0 && aware < 0)
    {
        std::fprintf(
            stderr,
            "HIP-aware MPI probe: no MPIX_Query_* API; device MPI_Bcast OK.\n");
    }

    MPI_Finalize();
    return 0;
}

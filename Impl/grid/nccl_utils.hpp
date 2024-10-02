#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include <complex>
#include <nccl.h>

#ifdef HAS_NCCL
#define CHECK_NCCL_ERROR(val) checkNccl((val), #val, __FILE__, __LINE__)

void checkNccl(ncclResult_t err, const char* const func, const char* const file, const int line)
{
    if (err != ncclSuccess)
    {
        std::cerr << "NCCL Runtime Error at: " << file << ":" << line << std::endl;
        std::cerr << ncclGetErrorString(err) << " in function " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}
#endif

namespace chase{
namespace Impl{
namespace nccl{

template <typename T>
struct NcclType;

template <>
struct NcclType<float> {
    static constexpr ncclDataType_t value = ncclFloat;
};

template <>
struct NcclType<double> {
    static constexpr ncclDataType_t value = ncclDouble;
};

// Since NCCL doesn't natively support complex types, we map complex types to their real counterparts
template <>
struct NcclType<std::complex<float>> {
    static constexpr ncclDataType_t value = ncclFloat;  // We treat complex<float> as float
};

template <>
struct NcclType<std::complex<double>> {
    static constexpr ncclDataType_t value = ncclDouble; // We treat complex<double> as double
};

// Helper trait to adjust the data count for complex types
template <typename T>
struct DataCountMultiplier {
    static constexpr std::size_t value = 1;  // No adjustment for real types
};

template <>
struct DataCountMultiplier<std::complex<float>> {
    static constexpr std::size_t value = 2;  // Complex<float> has 2 components (real, imag)
};

template <>
struct DataCountMultiplier<std::complex<double>> {
    static constexpr std::size_t value = 2;  // Complex<double> has 2 components (real, imag)
};


// Example function that uses the traits
//templated NCCL allreduce
template <typename T>
ncclResult_t ncclAllReduceWrapper(T* sendbuf, T* recvbuf, std::size_t count, ncclRedOp_t op, ncclComm_t comm, cudaStream_t *stream = nullptr) {
    ncclDataType_t ncclType = NcclType<T>::value;   // Get the corresponding NCCL type
    std::size_t adjustedCount = count * DataCountMultiplier<T>::value;  // Adjust count if necessary
    cudaStream_t usedStream = (stream == nullptr) ? 0 : *stream;
    // Call the NCCL function
    ncclResult_t status;
    status = ncclAllReduce(sendbuf, recvbuf, adjustedCount, ncclType, op, comm, usedStream);
    return status;
}

template <typename T>
ncclResult_t ncclBcastWrapper(T* buffer, std::size_t count, int root, ncclComm_t comm, cudaStream_t *stream = nullptr) {
    ncclDataType_t ncclType = NcclType<T>::value;   // Get the corresponding NCCL type
    std::size_t adjustedCount = count * DataCountMultiplier<T>::value;  // Adjust count for complex types
    cudaStream_t usedStream = (stream == nullptr) ? 0 : *stream;
    ncclResult_t status;
    // Call the NCCL broadcast function
    status = ncclBroadcast(buffer, buffer, adjustedCount, ncclType, root, comm, usedStream);
    return status;
}

}
}
}
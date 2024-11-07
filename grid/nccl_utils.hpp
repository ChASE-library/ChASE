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

/**
 * @page chase_nccl_namespace chase::nccl Namespace
 * @brief  Namespace for NCCL type mappings, helper functions and templated communication functions.
 *
 * 
 * The `chase::nccl` namespace contains type mappings between C++ data types 
 * and their corresponding NCCL data types. It also provides helper structures 
 * and functions to support operations like all-reduce and broadcast for 
 * both real and complex types.
 */

/**
 * @defgroup nccl_type_mappings NCCL Type Mappings
 * 
 * @brief Functions for mapping C++ types to NCCL types.
 * 
 * This group includes functions that provide the corresponding NCCL data 
 * types for various C++ data types.
 */ 

/**
 * @defgroup nccl_templated_communications NCCL Routines
 * 
 * @brief Functions the templated NCCL communicaiton functions which support both real and complex types in single/double precisions.
 * 
 * This group currently support only allreduce and Broadcast operations.
 */ 

namespace chase{
namespace nccl{

/**
 * @ingroup nccl_type_mappings
 * @struct NcclType
 * 
 * @brief A type trait to map C++ types to their corresponding NCCL data types.
 * 
 * This struct provides a mapping between C++ types and NCCL data types 
 * using specializations. The `value` member is the corresponding NCCL data 
 * type for the given C++ type.
 * 
 * @tparam T The C++ type for which the NCCL data type is required.
 */
template <typename T>
struct NcclType;

/**
 * @ingroup nccl_type_mappings
 * @brief Specialization of `NcclType` for `float`.
 * 
 * This specialization maps `float` to `ncclFloat`.
 */
template <>
struct NcclType<float> {
    static constexpr ncclDataType_t value = ncclFloat;
};

/**
 * @ingroup nccl_type_mappings
 * @brief Specialization of `NcclType` for `double`.
 * 
 * This specialization maps `double` to `ncclDouble`.
 */
template <>
struct NcclType<double> {
    static constexpr ncclDataType_t value = ncclDouble;
};

/**
 * @ingroup nccl_type_mappings
 * @brief Specialization of `NcclType` for `std::complex<float>`.
 * 
 * Since NCCL does not natively support complex types, `std::complex<float>` 
 * is mapped to `ncclFloat`.
 */
template <>
struct NcclType<std::complex<float>> {
    static constexpr ncclDataType_t value = ncclFloat;  // We treat complex<float> as float
};

/**
 * @ingroup nccl_type_mappings
 * @brief Specialization of `NcclType` for `std::complex<double>`.
 * 
 * Since NCCL does not natively support complex types, `std::complex<double>` 
 * is mapped to `ncclDouble`.
 */
template <>
struct NcclType<std::complex<double>> {
    static constexpr ncclDataType_t value = ncclDouble; // We treat complex<double> as double
};

/**
 * @ingroup nccl_type_mappings
 * @struct DataCountMultiplier
 * 
 * @brief A helper trait to adjust the data count for complex types.
 * 
 * This struct helps adjust the data count for complex types, as they have 
 * both real and imaginary components. For real types, the multiplier is 1, 
 * but for complex types, the multiplier is 2.
 * 
 * @tparam T The C++ type for which the data count adjustment is required.
 */
template <typename T>
struct DataCountMultiplier {
    static constexpr std::size_t value = 1;  // No adjustment for real types
};

/**
 * @ingroup nccl_type_mappings
 * @brief Specialization of `DataCountMultiplier` for `std::complex<float>`.
 * 
 * This specialization adjusts the data count for `std::complex<float>`, 
 * multiplying by 2 to account for the real and imaginary components.
 */
template <>
struct DataCountMultiplier<std::complex<float>> {
    static constexpr std::size_t value = 2;  // Complex<float> has 2 components (real, imag)
};

/**
 * @ingroup nccl_type_mappings
 * @brief Specialization of `DataCountMultiplier` for `std::complex<double>`.
 * 
 * This specialization adjusts the data count for `std::complex<double>`, 
 * multiplying by 2 to account for the real and imaginary components.
 */
template <>
struct DataCountMultiplier<std::complex<double>> {
    static constexpr std::size_t value = 2;  // Complex<double> has 2 components (real, imag)
};


//templated NCCL allreduce
/**
 * @ingroup nccl_templated_communications
 * @brief Templated NCCL AllReduce function wrapper.
 * 
 * This function wraps the NCCL all-reduce operation, adjusting the data 
 * count for complex types and handling different data types. It uses the 
 * appropriate NCCL data type and handles complex types by adjusting the 
 * data count accordingly.
 * 
 * @tparam T The type of the elements to be reduced (supports real and complex types).
 * @param sendbuf Pointer to the input buffer to send.
 * @param recvbuf Pointer to the output buffer to receive the result.
 * @param count The number of elements to reduce.
 * @param op The reduction operation (e.g., sum, max).
 * @param comm The NCCL communicator.
 * @param stream The CUDA stream for asynchronous operations (optional).
 * 
 * @return NCCL result status.
 */
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

/**
 * @ingroup nccl_templated_communications
 * @brief Templated NCCL Broadcast function wrapper.
 * 
 * This function wraps the NCCL broadcast operation, adjusting the data 
 * count for complex types and handling different data types. It uses the 
 * appropriate NCCL data type and handles complex types by adjusting the 
 * data count accordingly.
 * 
 * @tparam T The type of the elements to broadcast (supports real and complex types).
 * @param buffer Pointer to the buffer to broadcast.
 * @param count The number of elements to broadcast.
 * @param root The root rank that broadcasts the data.
 * @param comm The NCCL communicator.
 * @param stream The CUDA stream for asynchronous operations (optional).
 * 
 * @return NCCL result status.
 */
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
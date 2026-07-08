// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "linalg/internal/gpu/backend/error.hpp"

#include <complex>
#include <cstddef>

#if defined(CHASE_GPU_DEVICE_BACKEND_CUDA)
#include <cuComplex.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>

#define CHASE_GPU_DEVICE_NAMESPACE cuda
#define CHASE_GPU_PH_DIAG_NAMESPACE cuda_nccl_ph_diag
#define CHASE_GPU_STREAM_T cudaStream_t
#define CHASE_GPU_FLOAT_COMPLEX cuComplex
#define CHASE_GPU_DOUBLE_COMPLEX cuDoubleComplex
#define CHASE_GPU_MAKE_FLOAT_COMPLEX make_cuFloatComplex
#define CHASE_GPU_MAKE_DOUBLE_COMPLEX make_cuDoubleComplex
#define CHASE_GPU_CREAL_FLOAT cuCrealf
#define CHASE_GPU_CIMAG_FLOAT cuCimagf
#define CHASE_GPU_CREAL_DOUBLE cuCreal
#define CHASE_GPU_CIMAG_DOUBLE cuCimag
#define CHASE_GPU_CABS_FLOAT cuCabsf
#define CHASE_GPU_CABS_DOUBLE cuCabs
#define CHASE_GPU_CMUL_FLOAT cuCmulf
#define CHASE_GPU_CMUL_DOUBLE cuCmul
#define CHASE_GPU_CADD_FLOAT cuCaddf
#define CHASE_GPU_CADD_DOUBLE cuCadd
#define CHASE_GPU_CSUB_FLOAT cuCsubf
#define CHASE_GPU_CSUB_DOUBLE cuCsub
#define CHASE_GPU_CONJ_FLOAT cuConjf
#define CHASE_GPU_CONJ_DOUBLE cuConj
#define CHASE_GPU_MEMSET_ASYNC cudaMemsetAsync
#define CHASE_GPU_MALLOC cudaMalloc
#define CHASE_GPU_FREE cudaFree
#define CHASE_GPU_MEMCPY_ASYNC cudaMemcpyAsync
#define CHASE_GPU_MEMCPY_DEVICE_TO_HOST cudaMemcpyDeviceToHost
#define CHASE_GPU_MEMCPY_HOST_TO_DEVICE cudaMemcpyHostToDevice
#define CHASE_GPU_MEMCPY_DEVICE_TO_DEVICE cudaMemcpyDeviceToDevice
#define CHASE_GPU_STREAM_SYNCHRONIZE cudaStreamSynchronize
#define CHASE_GPU_ERROR_T cudaError_t
#define CHASE_GPU_SUCCESS cudaSuccess
#define CHASE_GPU_GET_ERROR_STRING cudaGetErrorString
#define CHASE_GPU_GET_LAST_ERROR cudaGetLastError
#define CHASE_GPU_PEEK_AT_LAST_ERROR cudaPeekAtLastError
#define CHASE_GPU_RAND_STATE curandStatePhilox4_32_10_t
#define CHASE_GPU_RAND_INIT curand_init
#define CHASE_GPU_RAND_NORMAL curand_normal
#define CHASE_GPU_RAND_NORMAL_DOUBLE curand_normal_double
#define CHASE_GPU_DOUBLE2FLOAT_RN(x) __double2float_rn(x)

#elif defined(CHASE_GPU_DEVICE_BACKEND_HIP)
#include <hip/hip_complex.h>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define CHASE_GPU_DEVICE_NAMESPACE hip
#define CHASE_GPU_PH_DIAG_NAMESPACE hip_ph_diag
#define CHASE_GPU_STREAM_T hipStream_t
#define CHASE_GPU_FLOAT_COMPLEX hipFloatComplex
#define CHASE_GPU_DOUBLE_COMPLEX hipDoubleComplex
#define CHASE_GPU_MAKE_FLOAT_COMPLEX make_hipFloatComplex
#define CHASE_GPU_MAKE_DOUBLE_COMPLEX make_hipDoubleComplex
#define CHASE_GPU_CREAL_FLOAT hipCrealf
#define CHASE_GPU_CIMAG_FLOAT hipCimagf
#define CHASE_GPU_CREAL_DOUBLE hipCreal
#define CHASE_GPU_CIMAG_DOUBLE hipCimag
#define CHASE_GPU_CABS_FLOAT hipCabsf
#define CHASE_GPU_CABS_DOUBLE hipCabs
#define CHASE_GPU_CMUL_FLOAT hipCmulf
#define CHASE_GPU_CMUL_DOUBLE hipCmul
#define CHASE_GPU_CADD_FLOAT hipCaddf
#define CHASE_GPU_CADD_DOUBLE hipCadd
#define CHASE_GPU_CSUB_FLOAT hipCsubf
#define CHASE_GPU_CSUB_DOUBLE hipCsub
#define CHASE_GPU_CONJ_FLOAT hipConjf
#define CHASE_GPU_CONJ_DOUBLE hipConj
#define CHASE_GPU_MEMSET_ASYNC hipMemsetAsync
#define CHASE_GPU_MALLOC hipMalloc
#define CHASE_GPU_FREE hipFree
#define CHASE_GPU_MEMCPY_ASYNC hipMemcpyAsync
#define CHASE_GPU_MEMCPY_DEVICE_TO_HOST hipMemcpyDeviceToHost
#define CHASE_GPU_MEMCPY_HOST_TO_DEVICE hipMemcpyHostToDevice
#define CHASE_GPU_MEMCPY_DEVICE_TO_DEVICE hipMemcpyDeviceToDevice
#define CHASE_GPU_STREAM_SYNCHRONIZE hipStreamSynchronize
#define CHASE_GPU_ERROR_T hipError_t
#define CHASE_GPU_SUCCESS hipSuccess
#define CHASE_GPU_GET_ERROR_STRING hipGetErrorString
#define CHASE_GPU_GET_LAST_ERROR hipGetLastError
#define CHASE_GPU_PEEK_AT_LAST_ERROR hipPeekAtLastError
#define CHASE_GPU_RAND_STATE hiprandStatePhilox4_32_10_t
#define CHASE_GPU_RAND_INIT hiprand_init
#define CHASE_GPU_RAND_NORMAL hiprand_normal
#define CHASE_GPU_RAND_NORMAL_DOUBLE hiprand_normal_double
#define CHASE_GPU_DOUBLE2FLOAT_RN(x) __double2float_rn(x)

#else
#error "Define CHASE_GPU_DEVICE_BACKEND_CUDA or CHASE_GPU_DEVICE_BACKEND_HIP before including backend_config.hpp"
#endif

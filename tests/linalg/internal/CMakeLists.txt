file(COPY QR_matrices DESTINATION ${CMAKE_BINARY_DIR}/tests/linalg/internal)
file(COPY BSE_matrices DESTINATION ${CMAKE_BINARY_DIR}/tests/linalg/internal)

# Absolute test-data root for unit tests (QR/BSE matrices). Nested test dirs
# (gpu/dist_mpi/*, gpu/dist_xccl/*) are deeper than the CPU test dirs; relative
# ../ paths no longer reach tests/linalg/internal/QR_matrices.
add_compile_definitions(
  CHASE_LINALG_INTERNAL_TEST_DATA_DIR="${CMAKE_BINARY_DIR}/tests/linalg/internal")

add_subdirectory(cpu)
if(TARGET pchase_cpu)
add_subdirectory(mpi)
endif()

if(TARGET linalg_internal_cuda)
add_subdirectory(gpu/serial/cuda)
endif()

if(TARGET linalg_internal_gpu_device_hip)
add_subdirectory(gpu/kernels/hip)
add_subdirectory(gpu/serial/hip)
elseif(TARGET linalg_internal_hip)
add_subdirectory(gpu/kernels/hip)
add_subdirectory(gpu/serial/hip)
endif()

if(TARGET linalg_internal_cuda_mpi)
add_subdirectory(gpu/dist_mpi/cuda)
endif()

if(TARGET linalg_internal_hip_mpi)
add_subdirectory(gpu/dist_mpi/hip)
endif()

if(TARGET linalg_internal_nccl)
add_subdirectory(gpu/dist_xccl/nccl)
endif()

if(TARGET linalg_internal_rccl)
add_subdirectory(gpu/dist_xccl/rccl)
endif()

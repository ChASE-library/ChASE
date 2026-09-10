add_library(pchase_gpu INTERFACE)

if(TARGET linalg_internal_cuda_mpi)
  target_link_libraries(pchase_gpu
    INTERFACE
      chase_nvidia_gpu_linalg
      linalg_internal_cuda
      linalg_internal_cuda_mpi
      distMatrix_gpu
  )
  target_compile_definitions(pchase_gpu INTERFACE HAS_CUDA=1)
endif()

if(TARGET linalg_internal_hip_mpi)
  target_link_libraries(pchase_gpu
    INTERFACE
      chase_amd_gpu_linalg
      chase_hip_kernels
      linalg_internal_hip
      linalg_internal_hip_mpi
      distMatrix_hip
  )
  target_compile_definitions(pchase_gpu INTERFACE HAS_HIP=1)
  if(NOT TARGET chase_scalapack_linalg)
    message(WARNING
      "pchase_gpu HIP: ScaLAPACK not found. Distributed Householder QR fallback "
      "requires HAS_SCALAPACK on AMD (see docs/amd_port_convergence_plan.md).")
  endif()
endif()

target_link_libraries(pchase_gpu
  INTERFACE
    chase_algorithm
    chase_cpu_linalg
    linalg_internal_cpu
    linalg_internal_mpi
)

if(TARGET mpi_grid_nccl)
  target_link_libraries(pchase_gpu INTERFACE linalg_internal_nccl mpi_grid_nccl)
endif()

if(TARGET mpi_grid_rccl)
  target_link_libraries(pchase_gpu INTERFACE mpi_grid_rccl)
  target_compile_definitions(pchase_gpu INTERFACE HAS_RCCL=1)
  if(CHASE_HIP_DISTRIBUTED_COLLECTIVES STREQUAL "MPI")
    target_compile_definitions(pchase_gpu INTERFACE CHASE_DISTRIBUTED_GPU_BACKEND_MPI=1)
  endif()
endif()

if(TARGET linalg_internal_rccl)
  target_link_libraries(pchase_gpu INTERFACE linalg_internal_rccl)
endif()

target_include_directories(pchase_gpu INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/Impl>
  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/Impl/pchase_gpu>"
)

if(CHASE_ENABLE_MPI_IO)
  target_compile_definitions(pchase_gpu INTERFACE "-DUSE_MPI_IO")
endif()

if(TARGET chase_scalapack_linalg)
  target_link_libraries(pchase_gpu INTERFACE chase_scalapack_linalg)
  target_compile_definitions(pchase_gpu INTERFACE HAS_SCALAPACK=1)
endif()

if(TARGET gemmul8pp)
  target_link_libraries(pchase_gpu INTERFACE gemmul8pp)
endif()

install(TARGETS pchase_gpu
  EXPORT chase_targets
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/Impl
  FILES_MATCHING
    PATTERN "*.hpp"
)

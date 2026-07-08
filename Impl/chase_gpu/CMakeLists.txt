add_library(chase_gpu INTERFACE)

target_link_libraries(chase_gpu
  INTERFACE
    chase_algorithm
    chase_cpu_linalg
    linalg_internal_cpu
)

if(TARGET linalg_internal_cuda)
  target_link_libraries(chase_gpu
    INTERFACE
      chase_nvidia_gpu_linalg
      linalg_internal_cuda
      linalg_internal_gpu_serial_cuda
      matrix_gpu
  )
  target_compile_definitions(chase_gpu INTERFACE HAS_CUDA=1)
endif()

if(TARGET linalg_internal_hip)
  target_link_libraries(chase_gpu
    INTERFACE
      chase_amd_gpu_linalg
      chase_hip_kernels
      linalg_internal_hip
      linalg_internal_gpu_serial_hip
      matrix_hip
  )
  target_compile_definitions(chase_gpu INTERFACE HAS_HIP=1)
endif()

target_include_directories(chase_gpu INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/Impl/chase_gpu>"
)

if(TARGET linalg_internal_hip)
  add_library(chase_hip ALIAS chase_gpu)
endif()

install(TARGETS chase_gpu
  EXPORT chase_targets
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/Impl
  FILES_MATCHING
    PATTERN "*.hpp"
)

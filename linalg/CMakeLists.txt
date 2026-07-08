add_library(linalg_internal_cpu INTERFACE)
target_include_directories(linalg_internal_cpu INTERFACE 
                                               $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/internal/cpu>
                                               "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/linalg/internal/cpu>")

install(TARGETS linalg_internal_cpu
  EXPORT chase_targets
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY internal/cpu DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/linalg/internal
  FILES_MATCHING
    PATTERN "*.hpp"
)

install(DIRECTORY internal/gpu/backend DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/linalg/internal/gpu
  FILES_MATCHING
    PATTERN "*.hpp"
)

add_library(matrix INTERFACE)
target_include_directories(matrix INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/matrix>
                                            "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/linalg/matrix>")

target_link_libraries(matrix INTERFACE chase_cpu_linalg linalg_internal_cpu)

install(TARGETS matrix
  EXPORT chase_targets
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY matrix DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/linalg
  FILES_MATCHING
    PATTERN "*.hpp"
)

add_library(distMatrix INTERFACE)
find_package(MPI)
if(MPI_FOUND)
  add_library(linalg_internal_mpi INTERFACE)
  target_include_directories(linalg_internal_mpi INTERFACE 
                                                $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/internal/mpi>
                                                "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/linalg/internal/mpi>")
  target_link_libraries(linalg_internal_mpi INTERFACE MPI::MPI_CXX)
  target_include_directories(distMatrix INTERFACE 
                                                $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/distMatrix>
                                                "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/linalg/distMatrix>")

  target_link_libraries(distMatrix INTERFACE chase_cpu_linalg linalg_internal_mpi)
endif()

if(TARGET chase_scalapack_linalg)
  target_link_libraries(distMatrix INTERFACE chase_scalapack_linalg)
endif()

if(MPI_FOUND)
  install(TARGETS linalg_internal_mpi
    EXPORT chase_targets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

  install(DIRECTORY internal/mpi DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/linalg/internal
    FILES_MATCHING
      PATTERN "*.hpp"
  )

  install(FILES
      ${CMAKE_CURRENT_SOURCE_DIR}/internal/typeTraits.hpp
      DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/linalg/internal
  )

  install(TARGETS distMatrix
    EXPORT chase_targets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

  install(DIRECTORY distMatrix DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/linalg
    FILES_MATCHING
      PATTERN "*.hpp"
  ) 
endif()

include(CheckLanguage)

if(CHASE_ENABLE_HIP_EFFECTIVE)
  check_language(HIP)
  if(NOT CMAKE_HIP_COMPILER)
    message(FATAL_ERROR "CHASE_ENABLE_HIP=ON but no HIP compiler was found")
  endif()

  if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES "gfx90a")
  endif()
  enable_language(HIP)

  find_package(hip REQUIRED CONFIG)
  message(STATUS "Configuring experimental HIP support for ${CMAKE_HIP_ARCHITECTURES}")

  add_library(chase_hip_kernels
    ${CMAKE_CURRENT_SOURCE_DIR}/internal/gpu/kernels/hip/kernels/shift_diagonal.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/internal/gpu/kernels/hip/kernels/matrix_diag.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/internal/gpu/kernels/hip/kernels/flip_sign.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/internal/gpu/kernels/hip/kernels/lacpy.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/internal/gpu/kernels/hip/kernels/abs_trace.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/internal/gpu/kernels/hip/kernels/conjugate.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/internal/gpu/kernels/hip/kernels/force_hermitian.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/internal/gpu/kernels/hip/kernels/precision_conversion.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/internal/gpu/kernels/hip/kernels/random_normal_distribution.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/internal/gpu/kernels/hip/kernels/residuals.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/internal/gpu/kernels/hip/kernels/lanczos_reduction.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/internal/gpu/kernels/hip/kernels/pseudo_hermitian_lanczos_diag.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/internal/gpu/kernels/hip/kernels/householder_scalar_kernel.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/internal/gpu/kernels/hip/kernels/orthog_kernels.hip
  )
  target_include_directories(chase_hip_kernels
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/internal/gpu/kernels/hip/launchers>
      "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/linalg/internal/gpu/kernels/hip/launchers>"
  )
  if(TARGET hip::host)
    target_link_libraries(chase_hip_kernels PUBLIC hip::host)
  endif()
  find_package(hiprand CONFIG QUIET)
  if(TARGET roc::hiprand)
    target_link_libraries(chase_hip_kernels PUBLIC roc::hiprand)
  elseif(TARGET hip::hiprand)
    target_link_libraries(chase_hip_kernels PUBLIC hip::hiprand)
  else()
    target_link_libraries(chase_hip_kernels PUBLIC hiprand)
  endif()
  target_compile_definitions(chase_hip_kernels PUBLIC HAS_HIP=1)

  add_library(linalg_internal_gpu_device_hip INTERFACE)
  target_include_directories(linalg_internal_gpu_device_hip INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/internal/gpu/kernels/hip/launchers>
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/linalg/internal/gpu/kernels/hip/launchers>"
  )
  target_link_libraries(linalg_internal_gpu_device_hip INTERFACE chase_hip_kernels
                        chase_amd_gpu_linalg)
  target_compile_definitions(linalg_internal_gpu_device_hip INTERFACE HAS_HIP=1)
  add_library(linalg_internal_hip ALIAS linalg_internal_gpu_device_hip)

  add_library(linalg_internal_gpu_serial_impl INTERFACE)
  target_include_directories(linalg_internal_gpu_serial_impl INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/internal/gpu/serial/impl>
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/linalg/internal/gpu/serial/impl>")
  target_compile_definitions(linalg_internal_gpu_serial_impl INTERFACE HAS_HIP=1)

  add_library(linalg_internal_gpu_serial_hip INTERFACE)
  target_include_directories(linalg_internal_gpu_serial_hip INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/internal/gpu/serial/hip>
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/linalg/internal/gpu/serial/hip>")
  target_link_libraries(linalg_internal_gpu_serial_hip INTERFACE
    linalg_internal_gpu_device_hip linalg_internal_gpu_serial_impl)
  target_compile_definitions(linalg_internal_gpu_serial_hip INTERFACE HAS_HIP=1)

  add_library(linalg_internal_gpu_device INTERFACE)
  target_include_directories(linalg_internal_gpu_device INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/internal/gpu/kernels/shared>
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/linalg/internal/gpu/kernels/shared>")
  target_link_libraries(linalg_internal_gpu_device INTERFACE
    linalg_internal_gpu_device_hip)

  add_library(matrix_hip INTERFACE)
  target_include_directories(matrix_hip INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/matrix>
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/linalg/matrix>")
  target_link_libraries(matrix_hip INTERFACE chase_cpu_linalg linalg_internal_cpu)
  target_link_libraries(matrix_hip INTERFACE chase_amd_gpu_linalg linalg_internal_hip)
  target_compile_definitions(matrix_hip INTERFACE HAS_HIP=1)

  install(TARGETS chase_hip_kernels linalg_internal_gpu_device_hip
                  linalg_internal_gpu_serial_hip linalg_internal_gpu_serial_impl
                  linalg_internal_gpu_device matrix_hip
    EXPORT chase_targets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )
  install(DIRECTORY internal/gpu/kernels/hip/launchers
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/linalg/internal/gpu/kernels/hip
    FILES_MATCHING
      PATTERN "*.hpp"
  )
  install(DIRECTORY internal/gpu/serial/hip
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/linalg/internal/gpu/serial
    FILES_MATCHING
      PATTERN "*.hpp"
  )
  install(DIRECTORY internal/gpu/serial/impl
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/linalg/internal/gpu/serial
    FILES_MATCHING
      PATTERN "*.hpp"
  )
  install(DIRECTORY internal/gpu/kernels/shared
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/linalg/internal/gpu/kernels
    FILES_MATCHING
      PATTERN "*.hpp"
  )

  add_library(distMatrix_hip INTERFACE)
  target_include_directories(distMatrix_hip INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/distMatrix>
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/linalg/distMatrix>")
  target_link_libraries(distMatrix_hip INTERFACE chase_cpu_linalg linalg_internal_mpi)
  target_link_libraries(distMatrix_hip INTERFACE chase_amd_gpu_linalg linalg_internal_hip)
  target_compile_definitions(distMatrix_hip INTERFACE HAS_HIP=1)

  install(TARGETS distMatrix_hip
    EXPORT chase_targets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

  if(MPI_FOUND)
    set(HIP_AWARE_MPI_TEST_FILE
        ${CMAKE_SOURCE_DIR}/cmake/tests/hip_mpi_test.cpp)

    set(_chase_hip_mpi_probe_libs MPI::MPI_CXX)
    if(TARGET hip::host)
      list(APPEND _chase_hip_mpi_probe_libs hip::host)
    endif()

    try_compile(HIP_AWARE_MPI_COMPILES
                ${CMAKE_BINARY_DIR}/cmake/hip_mpi_compile_test
                ${HIP_AWARE_MPI_TEST_FILE}
                LINK_LIBRARIES ${_chase_hip_mpi_probe_libs})

    if(HIP_AWARE_MPI_COMPILES)
      message(STATUS "HIP-aware MPI probe: compile/link with MPI + HIP succeeded")
      set(HIP_AWARE_MPI_SUPPORTED TRUE CACHE INTERNAL
          "HIP-aware MPI compile/link probe passed")
    else()
      message(WARNING
        "HIP-aware MPI probe: failed to compile/link with MPI + HIP "
        "(distributed pChASEHIP will need Cray GPU-aware MPICH + ROCm headers)")
      set(HIP_AWARE_MPI_SUPPORTED FALSE CACHE INTERNAL
          "HIP-aware MPI compile/link probe passed")
    endif()

    if(HIP_AWARE_MPI_COMPILES)
      add_executable(hip_mpi_probe ${HIP_AWARE_MPI_TEST_FILE})
      target_link_libraries(hip_mpi_probe PRIVATE
                            ${_chase_hip_mpi_probe_libs})
      target_compile_features(hip_mpi_probe PRIVATE cxx_std_17)
      # Install at build tree root so it is easy to find (not under linalg/).
      set_target_properties(hip_mpi_probe PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
    endif()

    if(CHASE_PROBE_HIP_AWARE_MPI_RUNTIME AND HIP_AWARE_MPI_COMPILES)
      set(ENV{MPICH_GPU_SUPPORT_ENABLED} "1")
      try_run(HIP_AWARE_MPI_RUN_RESULT HIP_AWARE_MPI_RUN_COMPILED
              ${CMAKE_BINARY_DIR}/cmake/hip_mpi_run_test
              ${HIP_AWARE_MPI_TEST_FILE}
              LINK_LIBRARIES ${_chase_hip_mpi_probe_libs}
              RUN_OUTPUT_VARIABLE HIP_AWARE_MPI_RUN_OUTPUT
              COMPILE_OUTPUT_VARIABLE HIP_AWARE_MPI_COMPILE_OUTPUT)
      if(HIP_AWARE_MPI_RUN_COMPILED AND HIP_AWARE_MPI_RUN_RESULT EQUAL 0)
        message(STATUS "HIP-aware MPI probe: runtime check passed")
        set(HIP_AWARE_MPI_RUNTIME_OK TRUE CACHE INTERNAL
            "HIP-aware MPI runtime probe passed")
      else()
        message(WARNING
          "HIP-aware MPI runtime probe failed (result=${HIP_AWARE_MPI_RUN_RESULT}). "
          "Use a GPU allocation, source scripts/frontier_hip_env.sh, and run: "
          "srun -n 2 ${CMAKE_BINARY_DIR}/hip_mpi_probe")
        if(HIP_AWARE_MPI_RUN_OUTPUT)
          message(STATUS "HIP-aware MPI probe output: ${HIP_AWARE_MPI_RUN_OUTPUT}")
        endif()
        set(HIP_AWARE_MPI_RUNTIME_OK FALSE CACHE INTERNAL
            "HIP-aware MPI runtime probe passed")
      endif()
    endif()

    if(HIP_AWARE_MPI_SUPPORTED)
      if(NOT TARGET linalg_internal_gpu_mpi)
        add_library(linalg_internal_gpu_mpi INTERFACE)
        target_include_directories(linalg_internal_gpu_mpi INTERFACE
          $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/internal/gpu/dist_mpi/impl>
          "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/linalg/internal/gpu/dist_mpi/impl>")

        install(TARGETS linalg_internal_gpu_mpi
          EXPORT chase_targets
          INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        )

        install(DIRECTORY internal/gpu/dist_mpi/impl
          DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/linalg/internal/gpu/dist_mpi
          FILES_MATCHING
            PATTERN "*.hpp"
        )
      endif()

      add_library(linalg_internal_hip_mpi INTERFACE)
      target_include_directories(linalg_internal_hip_mpi INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/internal/gpu/dist_mpi/hip>
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/linalg/internal/gpu/dist_mpi/hip>")
      target_link_libraries(linalg_internal_hip_mpi INTERFACE mpi_grid
                            linalg_internal_gpu_device_hip linalg_internal_gpu_mpi
                            linalg_internal_gpu_device)

      install(TARGETS linalg_internal_hip_mpi
        EXPORT chase_targets
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
      )

      install(DIRECTORY internal/gpu/dist_mpi/hip
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/linalg/internal/gpu/dist_mpi
        FILES_MATCHING
          PATTERN "*.hpp"
      )
    endif()

    if(TARGET mpi_grid_rccl AND NOT TARGET linalg_internal_gpu_xccl)
      add_library(linalg_internal_gpu_xccl INTERFACE)
      target_include_directories(linalg_internal_gpu_xccl INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/internal/gpu/dist_xccl/impl>
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/linalg/internal/gpu/dist_xccl/impl>")

      install(TARGETS linalg_internal_gpu_xccl
        EXPORT chase_targets
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
      )

      install(DIRECTORY internal/gpu/dist_xccl/impl
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/linalg/internal/gpu/dist_xccl
        FILES_MATCHING
          PATTERN "*.hpp"
      )
    endif()
    if(TARGET mpi_grid_rccl)
      target_link_libraries(linalg_internal_gpu_xccl INTERFACE
        mpi_grid_rccl linalg_internal_gpu_device_hip linalg_internal_gpu_device)
    endif()

    if(TARGET mpi_grid_rccl AND NOT TARGET linalg_internal_rccl)
      add_library(linalg_internal_rccl INTERFACE)
      target_include_directories(linalg_internal_rccl INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/internal/gpu/dist_xccl/rccl>
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/linalg/internal/gpu/dist_xccl/rccl>")
      target_link_libraries(linalg_internal_rccl INTERFACE
        mpi_grid_rccl linalg_internal_gpu_device_hip linalg_internal_gpu_xccl
        linalg_internal_gpu_device)

      install(TARGETS linalg_internal_rccl
        EXPORT chase_targets
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
      )

      install(DIRECTORY internal/gpu/dist_xccl/rccl
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/linalg/internal/gpu/dist_xccl
        FILES_MATCHING
          PATTERN "*.hpp"
      )
    endif()
  else()
    message(STATUS "HIP-aware MPI probe: skipped (MPI not found)")
    set(HIP_AWARE_MPI_SUPPORTED FALSE CACHE INTERNAL
        "HIP-aware MPI compile/link probe passed")
  endif()
endif()

check_language(CUDA)
if(CMAKE_CUDA_COMPILER)

  if (${CMAKE_VERSION} VERSION_LESS_EQUAL "3.18.0")
    find_package(CUDA REQUIRED)
  elseif(${CMAKE_VERSION} VERSION_GREATER "3.18.0")
    find_package(CUDAToolkit REQUIRED)	  
    set(CUDA_CUBLAS_LIBRARIES CUDA::cublas)
    set(CUDA_cusolver_LIBRARY CUDA::cusolver)
    set(CUDA_curand_LIBRARY CUDA::curand)    
  endif()

  enable_language(CUDA)

  # If variable not defined, set it to default value in order to comply with CMake Policy CMP0104
  if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      set (CMAKE_CUDA_ARCHITECTURES "80")
  endif()

  if (${CMAKE_VERSION} VERSION_LESS "3.18.0")
    message("CMAKE version less than 3.18. Using old CMAKE CUDA policy for defining CUDA code generation flags")
    foreach(CODE ${CMAKE_CUDA_ARCHITECTURES})
      set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${CODE},code=sm_${CODE}")
    endforeach()
  endif()

  message("-- Configuring the build of CUDA with the compute capability of ${CMAKE_CUDA_ARCHITECTURES}")

  # Set the path to the test program
  # !!!!! This is a temporary solution, because the query function is not supported by all MPI providers.
  set(CUDA_AWARE_MPI_TEST_FILE ${CMAKE_SOURCE_DIR}/cmake/tests/cuda_mpi_test.cpp)
  include(CheckIncludeFiles)
  include(CheckFunctionExists)

  # Try to compile and run the test program
  try_compile(CUDA_AWARE_MPI_SUPPORTED
              ${CMAKE_BINARY_DIR}/cuda_mpi_test
              ${CUDA_AWARE_MPI_TEST_FILE}
              LINK_LIBRARIES MPI::MPI_CXX CUDA::cudart)

  if(NOT CUDA_AWARE_MPI_SUPPORTED)
    message(FATAL_ERROR "CUDA-Aware MPI is required, for the distributed multiple-GPU ChASE")
  endif()

  # Specify the CUDA source files
  set(CUDA_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/internal/gpu/kernels/cuda/kernels/precision_conversion.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/internal/gpu/kernels/cuda/kernels/shift_diagonal.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/internal/gpu/kernels/cuda/kernels/flip_sign.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/internal/gpu/kernels/cuda/kernels/conjugate.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/internal/gpu/kernels/cuda/kernels/abs_trace.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/internal/gpu/kernels/cuda/kernels/lacpy.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/internal/gpu/kernels/cuda/kernels/random_normal_distribution.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/internal/gpu/kernels/cuda/kernels/residuals.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/internal/gpu/kernels/cuda/kernels/force_hermitian.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/internal/gpu/kernels/cuda/kernels/orthog_kernels.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/internal/gpu/kernels/cuda/kernels/householder_scalar_kernel.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/internal/gpu/kernels/cuda/kernels/pseudo_hermitian_lanczos_diag.cu
        )
  add_library( chase_cuda_kernels ${CUDA_SOURCES} )
  set_target_properties( chase_cuda_kernels 
	  		                 PROPERTIES
		  		                          CUDA_RESOLVE_DEVICE_SYMBOLS ON
	                                  CUDA_SEPARABLE_COMPILATION ON)

  target_include_directories(chase_cuda_kernels PUBLIC
                                                $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
                                                $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/internal/gpu/kernels/cuda/launchers>
                                                "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
                                                "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/linalg/internal/gpu/kernels/cuda/launchers>")
                                                
  target_include_directories( chase_cuda_kernels INTERFACE
      ${CUDA_INCLUDE_DIRS})

  target_link_libraries(chase_cuda_kernels INTERFACE CUDA::cudart CUDA::curand)

  install(TARGETS chase_cuda_kernels
    EXPORT chase_targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

  install(DIRECTORY internal/gpu/kernels/cuda/launchers
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/linalg/internal/gpu/kernels/cuda
    FILES_MATCHING
      PATTERN "*.cuh"
  )

  add_library(linalg_internal_gpu_device_cuda INTERFACE)
  target_include_directories(linalg_internal_gpu_device_cuda INTERFACE 
                                                $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/internal/gpu/kernels/cuda/launchers>
                                                "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/linalg/internal/gpu/kernels/cuda/launchers>")
  target_link_libraries(linalg_internal_gpu_device_cuda INTERFACE chase_cuda_kernels)
  target_compile_definitions(linalg_internal_gpu_device_cuda INTERFACE HAS_CUDA=1)
  add_library(linalg_internal_cuda ALIAS linalg_internal_gpu_device_cuda)

  add_library(linalg_internal_gpu_serial_cuda INTERFACE)
  target_include_directories(linalg_internal_gpu_serial_cuda INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/internal/gpu/serial/cuda>
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/linalg/internal/gpu/serial/cuda>")
  if(NOT TARGET linalg_internal_gpu_serial_impl)
    add_library(linalg_internal_gpu_serial_impl INTERFACE)
    target_include_directories(linalg_internal_gpu_serial_impl INTERFACE
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/internal/gpu/serial/impl>
      "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/linalg/internal/gpu/serial/impl>")
  endif()
  target_compile_definitions(linalg_internal_gpu_serial_impl INTERFACE HAS_CUDA=1)
  target_link_libraries(linalg_internal_gpu_serial_cuda INTERFACE
    linalg_internal_gpu_device_cuda linalg_internal_gpu_serial_impl)
  target_compile_definitions(linalg_internal_gpu_serial_cuda INTERFACE HAS_CUDA=1)

  if(NOT TARGET linalg_internal_gpu_device)
    add_library(linalg_internal_gpu_device INTERFACE)
    target_include_directories(linalg_internal_gpu_device INTERFACE
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/internal/gpu/kernels/shared>
      "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/linalg/internal/gpu/kernels/shared>")
  endif()
  target_link_libraries(linalg_internal_gpu_device INTERFACE
    linalg_internal_gpu_device_cuda)

  set(_chase_cuda_linalg_export_targets
      linalg_internal_gpu_device_cuda
      linalg_internal_gpu_serial_cuda
      linalg_internal_gpu_serial_impl)
  if(NOT CHASE_ENABLE_HIP_EFFECTIVE)
    list(APPEND _chase_cuda_linalg_export_targets linalg_internal_gpu_device)
  endif()

  install(TARGETS ${_chase_cuda_linalg_export_targets}
    EXPORT chase_targets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

  install(DIRECTORY internal/gpu/serial/cuda
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/linalg/internal/gpu/serial
    FILES_MATCHING
      PATTERN "*.hpp"
  )
  install(DIRECTORY internal/gpu/serial/impl
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/linalg/internal/gpu/serial
    FILES_MATCHING
      PATTERN "*.hpp"
  )
  install(DIRECTORY internal/gpu/kernels/shared
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/linalg/internal/gpu/kernels
    FILES_MATCHING
      PATTERN "*.hpp"
  )

  add_library(matrix_gpu INTERFACE)
  target_include_directories(matrix_gpu INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/matrix>
                                              "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/linalg/matrix>")
  target_link_libraries(matrix_gpu INTERFACE chase_cpu_linalg linalg_internal_cpu)

  target_link_libraries(matrix_gpu INTERFACE chase_nvidia_gpu_linalg linalg_internal_cuda)
  target_compile_definitions(matrix_gpu INTERFACE HAS_CUDA=1)

  install(TARGETS matrix_gpu
    EXPORT chase_targets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

  add_library(distMatrix_gpu INTERFACE)
  target_include_directories(distMatrix_gpu INTERFACE 
                                                $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/distMatrix>
                                                "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/linalg/distMatrix>")
  target_link_libraries(distMatrix_gpu INTERFACE chase_cpu_linalg linalg_internal_mpi)
  target_link_libraries(distMatrix_gpu INTERFACE chase_nvidia_gpu_linalg linalg_internal_cuda)
  target_compile_definitions(distMatrix_gpu INTERFACE HAS_CUDA=1)

  install(TARGETS distMatrix_gpu
    EXPORT chase_targets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

  if(NOT TARGET linalg_internal_gpu_mpi)
    add_library(linalg_internal_gpu_mpi INTERFACE)
    target_include_directories(linalg_internal_gpu_mpi INTERFACE
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/internal/gpu/dist_mpi/impl>
      "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/linalg/internal/gpu/dist_mpi/impl>")

    install(TARGETS linalg_internal_gpu_mpi
      EXPORT chase_targets
      INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )

    install(DIRECTORY internal/gpu/dist_mpi/impl
      DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/linalg/internal/gpu/dist_mpi
      FILES_MATCHING
        PATTERN "*.hpp"
    )
  endif()

  add_library(linalg_internal_cuda_mpi INTERFACE)
  target_include_directories(linalg_internal_cuda_mpi INTERFACE 
                                                $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/internal/gpu/dist_mpi/cuda>
                                                "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/linalg/internal/gpu/dist_mpi/cuda>")
  target_link_libraries(linalg_internal_cuda_mpi INTERFACE mpi_grid linalg_internal_cuda
                        linalg_internal_gpu_mpi linalg_internal_gpu_device)

  install(TARGETS linalg_internal_cuda_mpi
    EXPORT chase_targets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

  install(DIRECTORY internal/gpu/dist_mpi/cuda DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/linalg/internal/gpu/dist_mpi
    FILES_MATCHING
      PATTERN "*.hpp"
  )
  
  if(TARGET mpi_grid_nccl AND NOT TARGET linalg_internal_gpu_xccl)
    add_library(linalg_internal_gpu_xccl INTERFACE)
    target_include_directories(linalg_internal_gpu_xccl INTERFACE
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/internal/gpu/dist_xccl/impl>
      "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/linalg/internal/gpu/dist_xccl/impl>")

    install(TARGETS linalg_internal_gpu_xccl
      EXPORT chase_targets
      INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )

    install(DIRECTORY internal/gpu/dist_xccl/impl
      DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/linalg/internal/gpu/dist_xccl
      FILES_MATCHING
        PATTERN "*.hpp"
    )
  endif()
  if(TARGET mpi_grid_nccl)
    target_link_libraries(linalg_internal_gpu_xccl INTERFACE
      mpi_grid_nccl linalg_internal_cuda linalg_internal_gpu_device)
  endif()

  if(TARGET mpi_grid_nccl)
    add_library(linalg_internal_nccl INTERFACE)
    target_include_directories(linalg_internal_nccl INTERFACE 
                                                  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/internal/gpu/dist_xccl/nccl>
                                                  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/linalg/internal/gpu/dist_xccl/nccl>")
    target_link_libraries(linalg_internal_nccl INTERFACE mpi_grid_nccl
                          linalg_internal_cuda linalg_internal_gpu_xccl
                          linalg_internal_gpu_device)

    install(TARGETS linalg_internal_nccl
      EXPORT chase_targets
      INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )

    install(DIRECTORY internal/gpu/dist_xccl/nccl DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/linalg/internal/gpu/dist_xccl
      FILES_MATCHING
        PATTERN "*.hpp"
    )

  endif()
else()
  message(STATUS "No CUDA support")
endif()

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

add_library(matrix INTERFACE)
target_include_directories(matrix INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/matrix>
                                            "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/linalg/matrix>")

target_link_libraries(matrix INTERFACE blaspp lapackpp linalg_internal_cpu)

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

  target_link_libraries(distMatrix INTERFACE blaspp lapackpp linalg_internal_mpi)
endif()

if(TARGET scalapackpp)
  target_link_libraries(distMatrix INTERFACE scalapackpp)
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
  set(CUDA_AWARE_MPI_TEST_FILE ${CMAKE_SOURCE_DIR}/cmake/tests/cuda_aware_mpi_test.cpp)
  include(CheckIncludeFiles)
  include(CheckFunctionExists)

  # Try to compile and run the test program
  try_compile(CUDA_AWARE_MPI_SUPPORTED
              ${CMAKE_BINARY_DIR}/cuda_aware_mpi_test
              ${CUDA_AWARE_MPI_TEST_FILE}
              LINK_LIBRARIES MPI::MPI_CXX CUDA::cudart)

  if(NOT CUDA_AWARE_MPI_SUPPORTED)
    message(FATAL_ERROR "CUDA-Aware MPI is required, for the distributed multiple-GPU ChASE")
  endif()

  # Specify the CUDA source files
  set(CUDA_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/internal/cuda/precision_conversion.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/internal/cuda/shiftDiagonal.cu
	${CMAKE_CURRENT_SOURCE_DIR}/internal/cuda/flipSign.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/internal/cuda/absTrace.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/internal/cuda/lacpy.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/internal/cuda/random_normal_distribution.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/internal/cuda/residuals.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/internal/cuda/lanczos_kernels.cu
        )
  add_library( chase_cuda_kernels ${CUDA_SOURCES} )
  set_target_properties( chase_cuda_kernels 
	  		                 PROPERTIES
		  		                          CUDA_RESOLVE_DEVICE_SYMBOLS ON
	                                  CUDA_SEPARABLE_COMPILATION ON)

  target_include_directories(chase_cuda_kernels INTERFACE 
                                                $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/internal/cuda>
                                                "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/linalg/internal/cuda>")
                                                
  target_include_directories( chase_cuda_kernels INTERFACE
      ${CUDA_INCLUDE_DIRS})

  target_link_libraries(chase_cuda_kernels INTERFACE CUDA::cudart CUDA::curand)

  install(TARGETS chase_cuda_kernels
    EXPORT chase_targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

  install(DIRECTORY internal/cuda DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/linalg/internal
    FILES_MATCHING
      PATTERN "*.hpp"
      PATTERN "*.cuh"
  )

  add_library(linalg_internal_cuda INTERFACE)
  target_include_directories(linalg_internal_cuda INTERFACE 
                                                $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/internal/cuda>
                                                "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/linalg/internal/cuda>")
  target_link_libraries(linalg_internal_cuda INTERFACE chase_cuda_kernels)

  install(TARGETS linalg_internal_cuda
    EXPORT chase_targets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

  add_library(matrix_gpu INTERFACE)
  target_include_directories(matrix_gpu INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/matrix>
                                              "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/linalg/matrix>")
  target_link_libraries(matrix_gpu INTERFACE blaspp lapackpp linalg_internal_cpu)

  target_link_libraries(matrix_gpu INTERFACE cublaspp cusolverpp linalg_internal_cuda)
  target_compile_definitions(matrix_gpu INTERFACE HAS_CUDA=1)

  install(TARGETS matrix_gpu
    EXPORT chase_targets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

  add_library(distMatrix_gpu INTERFACE)
  target_include_directories(distMatrix_gpu INTERFACE 
                                                $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/distMatrix>
                                                "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/linalg/distMatrix>")
  target_link_libraries(distMatrix_gpu INTERFACE blaspp lapackpp  linalg_internal_mpi)
  target_link_libraries(distMatrix_gpu INTERFACE cublaspp cusolverpp linalg_internal_cuda)
  target_compile_definitions(distMatrix_gpu INTERFACE HAS_CUDA=1)

  install(TARGETS distMatrix_gpu
    EXPORT chase_targets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

  add_library(linalg_internal_cuda_mpi INTERFACE)
  target_include_directories(linalg_internal_cuda_mpi INTERFACE 
                                                $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/internal/cuda_aware_mpi>
                                                "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/linalg/internal/cuda_aware_mpi>")
  target_link_libraries(linalg_internal_cuda_mpi INTERFACE mpi_grid linalg_internal_cuda)

  install(TARGETS linalg_internal_cuda_mpi
    EXPORT chase_targets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

  install(DIRECTORY internal/cuda_aware_mpi DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/linalg/internal
    FILES_MATCHING
      PATTERN "*.hpp"
  )
  
  if(TARGET mpi_grid_nccl)
    add_library(linalg_internal_nccl INTERFACE)
    target_include_directories(linalg_internal_nccl INTERFACE 
                                                  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/internal/nccl>
                                                  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/linalg/internal/nccl>")
    target_link_libraries(linalg_internal_nccl INTERFACE mpi_grid_nccl linalg_internal_cuda)

    install(TARGETS linalg_internal_nccl
      EXPORT chase_targets
      INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )

    install(DIRECTORY internal/nccl DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/linalg/internal
      FILES_MATCHING
        PATTERN "*.hpp"
    )

  endif()
else()
  message(STATUS "No CUDA support")
endif()

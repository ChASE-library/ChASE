find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

#enable_language( Fortran )
include( FortranCInterface )
FortranCInterface_HEADER( ${CMAKE_BINARY_DIR}/fortran_mangle.h
  MACRO_NAMESPACE "FC_"
  SYMBOL_NAMESPACE "FC_"
  )

list(APPEND includePath ${CMAKE_BINARY_DIR})

# Install fortran_mangle.h to ${CMAKE_INSTALL_INCLUDEDIR}
install(FILES ${CMAKE_BINARY_DIR}/fortran_mangle.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

add_library(blaspp INTERFACE)
target_include_directories(blaspp INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cpu/blaspp>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/external/cpu/blaspp>
)

target_include_directories( blaspp INTERFACE
  "$<BUILD_INTERFACE:${includePath}>"
  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>" 
)
target_link_libraries(blaspp INTERFACE BLAS::BLAS)

add_library(lapackpp INTERFACE)
target_include_directories(lapackpp INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cpu/lapackpp>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/external/cpu/lapackpp>
)
target_include_directories( lapackpp INTERFACE
  "$<BUILD_INTERFACE:${includePath}>"
  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>" 
)
target_link_libraries(lapackpp INTERFACE LAPACK::LAPACK)

add_library(chase_cpu_linalg INTERFACE)
target_link_libraries(chase_cpu_linalg INTERFACE blaspp lapackpp)

find_package(SCALAPACK)
if(SCALAPACK_FOUND)
  add_library(scalapackpp INTERFACE)
  target_include_directories(scalapackpp INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/scalapack/scalapackpp>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/external/scalapack/scalapackpp>
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>" 
  )
  target_compile_definitions(scalapackpp INTERFACE HAS_SCALAPACK=1)
  target_link_libraries(scalapackpp INTERFACE SCALAPACK::SCALAPACK)

  add_library(chase_scalapack_linalg INTERFACE)
  target_link_libraries(chase_scalapack_linalg INTERFACE scalapackpp)
  message(STATUS "ScaLAPACK support enabled.")
else()
    message(STATUS "No ScaLAPACK is detected.")
endif()

find_package(CUDAToolkit)
if(CUDAToolkit_FOUND)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES "80;90" CACHE STRING "CUDA architectures to build for")
  endif()
  add_library(cublaspp INTERFACE)
  target_include_directories(cublaspp INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/nvidia_gpu/cublaspp>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/external/nvidia_gpu/cublaspp>
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>" 
  )
  target_link_libraries(cublaspp INTERFACE CUDA::cublas CUDA::cudart)

  add_library(cusolverpp INTERFACE)
  target_include_directories(cusolverpp INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/nvidia_gpu/cusolverpp>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/external/nvidia_gpu/cusolverpp>
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>" 
  )
  target_link_libraries(cusolverpp INTERFACE CUDA::cusolver CUDA::cudart)

  add_library(chase_nvidia_gpu_linalg INTERFACE)
  target_link_libraries(chase_nvidia_gpu_linalg INTERFACE cublaspp cusolverpp)

  if(CHASE_ENABLE_XGEEV_RAYLEIGHRITZ)
    if(${CUDAToolkit_VERSION} VERSION_LESS 12.6.85)
	    message(WARNING "CUDA VERSION = " ${CUDAToolkit_VERSION} " < 12.6.85! Xgeev kernel is not in this CUDA version. ChASE GPU will use the HEEVD version instead.")
    else()
      target_compile_definitions( cusolverpp  INTERFACE "-DXGEEV_EXISTS" )
      message(STATUS "CUDA VERSION = " ${CUDAToolkit_VERSION} " >= 12.6.85. Xgeev kernel exists. ChASE GPU XGEEV will work for Pseudo-Hermitian problems.")
    endif()
  endif()

endif()

if(CHASE_ENABLE_HIP_EFFECTIVE)
  find_package(rocblas REQUIRED CONFIG)
  find_package(rocsolver REQUIRED CONFIG)

  add_library(rocblaspp INTERFACE)
  target_include_directories(rocblaspp INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/amd_gpu/rocblaspp>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/external/amd_gpu/rocblaspp>
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
  )
  target_link_libraries(rocblaspp INTERFACE roc::rocblas)

  add_library(rocsolverpp INTERFACE)
  target_include_directories(rocsolverpp INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/amd_gpu/rocsolverpp>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/external/amd_gpu/rocsolverpp>
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
  )
  target_link_libraries(rocsolverpp INTERFACE roc::rocsolver roc::rocblas)

  add_library(chase_amd_gpu_linalg INTERFACE)
  target_link_libraries(chase_amd_gpu_linalg INTERFACE rocblaspp rocsolverpp)

  if(CHASE_ENABLE_XGEEV_RAYLEIGHRITZ)
    message(STATUS "CHASE_ENABLE_XGEEV_RAYLEIGHRITZ is set for HIP, but rocSOLVER XGEEV is not wired yet.")
  endif()

  message(STATUS "rocBLAS/rocSOLVER support enabled for HIP.")
endif()

install(TARGETS blaspp lapackpp chase_cpu_linalg
  EXPORT chase_targets
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY cpu/blaspp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/external/cpu
  FILES_MATCHING
    PATTERN "*.hpp"
    PATTERN "*.inc"
)

install(DIRECTORY cpu/lapackpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/external/cpu
  FILES_MATCHING
    PATTERN "*.hpp"
    PATTERN "*.inc"
)

if(SCALAPACK_FOUND)
  install(TARGETS scalapackpp chase_scalapack_linalg
    EXPORT chase_targets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

  install(DIRECTORY scalapack/scalapackpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/external/scalapack
    FILES_MATCHING
      PATTERN "*.hpp"
      PATTERN "*.inc"
  )

endif()

if(CUDAToolkit_FOUND)
  install(TARGETS cublaspp cusolverpp chase_nvidia_gpu_linalg
    EXPORT chase_targets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

  install(DIRECTORY nvidia_gpu/cublaspp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/external/nvidia_gpu
    FILES_MATCHING
      PATTERN "*.hpp"
  )

  install(DIRECTORY nvidia_gpu/cusolverpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/external/nvidia_gpu
    FILES_MATCHING
      PATTERN "*.hpp"
  )
endif()

if(CHASE_ENABLE_HIP_EFFECTIVE)
  install(TARGETS rocblaspp rocsolverpp chase_amd_gpu_linalg
    EXPORT chase_targets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

  install(DIRECTORY amd_gpu/rocblaspp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/external/amd_gpu
    FILES_MATCHING
      PATTERN "*.hpp"
  )

  install(DIRECTORY amd_gpu/rocsolverpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/external/amd_gpu
    FILES_MATCHING
      PATTERN "*.hpp"
  )
endif()

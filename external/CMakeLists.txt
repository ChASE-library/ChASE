find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

# Export the concrete CPU libraries to the top-level pkg-config generator.
set(CHASE_PC_BLAS_LIBRARIES "${BLAS_LIBRARIES}" PARENT_SCOPE)
set(CHASE_PC_LAPACK_LIBRARIES "${LAPACK_LIBRARIES}" PARENT_SCOPE)

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
  set(CHASE_PC_SCALAPACK_LIBRARIES "${SCALAPACK_LIBRARIES}" PARENT_SCOPE)
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

# Optional Ozaki Scheme II backend (GEMMul8). NVIDIA CUDA only.
if(CHASE_ENABLE_OZAKI)
  if(NOT CUDAToolkit_FOUND)
    message(FATAL_ERROR "CHASE_ENABLE_OZAKI=ON requires CUDAToolkit (CUDA).")
  endif()
  if(CHASE_ENABLE_HIP_EFFECTIVE AND NOT CUDAToolkit_FOUND)
    message(FATAL_ERROR "CHASE_ENABLE_OZAKI is not supported on HIP-only builds.")
  endif()

  if(NOT DEFINED CUDAToolkit_NVCC_EXECUTABLE OR NOT EXISTS "${CUDAToolkit_NVCC_EXECUTABLE}")
    message(FATAL_ERROR
      "CHASE_ENABLE_OZAKI=ON requires NVCC (via CUDAToolkit). "
      "CUDAToolkit_NVCC_EXECUTABLE='${CUDAToolkit_NVCC_EXECUTABLE}'")
  endif()

  if(NOT GEMMUL8_ROOT)
    message(FATAL_ERROR "CHASE_ENABLE_OZAKI=ON requires -DGEMMUL8_ROOT=/path/to/GEMMul8 (must contain include/ and lib/).")
  endif()

  set(_GEMMUL8_INCLUDE_DIR "${GEMMUL8_ROOT}/include")
  if(EXISTS "${GEMMUL8_ROOT}/GEMMul8/include")
    set(_GEMMUL8_INCLUDE_DIR "${GEMMUL8_ROOT}/GEMMul8/include")
  endif()

  set(_GEMMUL8_LIB "")
  foreach(_cand
          "${GEMMUL8_ROOT}/lib/libgemmul8.so"
          "${GEMMUL8_ROOT}/lib64/libgemmul8.so"
          "${GEMMUL8_ROOT}/lib/libgemmul8.a"
          "${GEMMUL8_ROOT}/lib64/libgemmul8.a"
          "${GEMMUL8_ROOT}/GEMMul8/lib/libgemmul8.so"
          "${GEMMUL8_ROOT}/GEMMul8/lib/libgemmul8.a")
    if(EXISTS "${_cand}")
      set(_GEMMUL8_LIB "${_cand}")
      break()
    endif()
  endforeach()

  if(NOT EXISTS "${_GEMMUL8_INCLUDE_DIR}/gemmul8.hpp")
    message(FATAL_ERROR
      "GEMMul8 header not found. Expected '${_GEMMUL8_INCLUDE_DIR}/gemmul8.hpp'. "
      "Set GEMMUL8_ROOT to your GEMMul8 build/install prefix (or source root containing GEMMul8/include).")
  endif()

  if(NOT _GEMMUL8_LIB)
    message(FATAL_ERROR
      "libgemmul8 not found under GEMMUL8_ROOT='${GEMMUL8_ROOT}'. "
      "Expected lib/libgemmul8.(so|a) (or GEMMul8/lib/...).")
  endif()

  if(NOT TARGET gemmul8::gemmul8)
    add_library(gemmul8::gemmul8 UNKNOWN IMPORTED)
    set_target_properties(gemmul8::gemmul8 PROPERTIES
      IMPORTED_LOCATION "${_GEMMUL8_LIB}"
      INTERFACE_INCLUDE_DIRECTORIES "${_GEMMUL8_INCLUDE_DIR}"
    )
    target_link_libraries(gemmul8::gemmul8 INTERFACE CUDA::cublas CUDA::cudart)
  endif()

  message(STATUS "Ozaki Scheme II backend enabled (GEMMul8): ${_GEMMUL8_LIB}")

  target_compile_definitions(chase_algorithm INTERFACE CHASE_ENABLE_OZAKI=1)

  enable_language(CUDA)

  add_library(gemmul8pp ${CMAKE_CURRENT_SOURCE_DIR}/gemmul8pp/gemmul8pp.cu)
  set_target_properties(gemmul8pp PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON)
  target_compile_options(gemmul8pp PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_90,code=sm_90>)
  target_include_directories(gemmul8pp PRIVATE "${_GEMMUL8_INCLUDE_DIR}")
  target_include_directories(gemmul8pp PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/gemmul8pp>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/external/gemmul8pp>
  )
  target_link_libraries(gemmul8pp PUBLIC gemmul8::gemmul8 CUDA::cublas CUDA::cudart)
  target_compile_definitions(gemmul8pp PUBLIC CHASE_ENABLE_OZAKI=1)
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

# Install Ozaki/GEMMul8 wrapper (library + headers).
if(CHASE_ENABLE_OZAKI)
  install(TARGETS gemmul8pp
    EXPORT chase_targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )
  install(DIRECTORY gemmul8pp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/external
    FILES_MATCHING
      PATTERN "*.hpp"
      PATTERN "*.cu"
  )
endif()

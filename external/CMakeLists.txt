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
  include(CudaArchitectures)
  chase_require_single_cuda_architecture()
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

# Optional Ozaki filter GEMM. GEMMUL8 builds GEMMul8. CUBLAS uses cuBLAS FP64
# fixed-point emulation and does not compile GEMMul8.
if(CHASE_ENABLE_OZAKI)
  string(TOUPPER "${CHASE_OZAKI_BACKEND}" _CHASE_OZAKI_BACKEND)
  if(_CHASE_OZAKI_BACKEND STREQUAL "CUBLAS")
    if(NOT CUDAToolkit_FOUND)
      message(FATAL_ERROR "CHASE_OZAKI_BACKEND=CUBLAS requires the CUDA toolkit.")
    endif()
    if(CUDAToolkit_VERSION VERSION_LESS 13.5)
      message(FATAL_ERROR
        "CHASE_OZAKI_BACKEND=CUBLAS requires CUDA >= 13.5 (found ${CUDAToolkit_VERSION}). "
        "This toolkit has no CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH. Use CHASE_OZAKI_BACKEND=GEMMUL8.")
    endif()
    target_compile_definitions(chase_algorithm INTERFACE CHASE_ENABLE_OZAKI=1 CHASE_OZAKI_CUBLAS=1)
    message(STATUS "Ozaki filter GEMM: cuBLAS FP64 fixed-point emulation (CUDA ${CUDAToolkit_VERSION})")
  elseif(NOT _CHASE_OZAKI_BACKEND STREQUAL "GEMMUL8")
    message(FATAL_ERROR "CHASE_OZAKI_BACKEND must be GEMMUL8 or CUBLAS, got '${CHASE_OZAKI_BACKEND}'.")
  else()
  include(GEMMul8)
  chase_setup_gemmul8()

  target_compile_definitions(chase_algorithm INTERFACE CHASE_ENABLE_OZAKI=1 CHASE_OZAKI_GEMMUL8=1)

  enable_language(CUDA)

  add_library(gemmul8pp
    ${CMAKE_CURRENT_SOURCE_DIR}/gemmul8pp/gemmul8pp.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/gemmul8pp/utilities.cu)
  set_target_properties(gemmul8pp PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
  target_include_directories(gemmul8pp PRIVATE "${_GEMMUL8_INCLUDE_DIR}")
  target_include_directories(gemmul8pp PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/gemmul8pp>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/external/gemmul8pp>
  )
  if(_GEMMUL8_FROM_FETCH)
    # Build against the copy in the build tree. Installed ChASE links the copy
    # installed next to it, and consumers get an rpath to that lib directory.
    target_link_libraries(gemmul8pp PUBLIC
      "$<BUILD_INTERFACE:gemmul8::gemmul8>"
      "$<INSTALL_INTERFACE:$<INSTALL_PREFIX>/${CMAKE_INSTALL_LIBDIR}/libgemmul8.so>"
      CUDA::cublas
      CUDA::cudart)
    target_link_options(gemmul8pp PUBLIC
      "$<INSTALL_INTERFACE:-Wl,-rpath,$<INSTALL_PREFIX>/${CMAKE_INSTALL_LIBDIR}>")
  else()
    target_link_libraries(gemmul8pp PUBLIC gemmul8::gemmul8 CUDA::cublas CUDA::cudart)
  endif()
  target_compile_definitions(gemmul8pp PUBLIC CHASE_ENABLE_OZAKI=1)
  target_compile_definitions(gemmul8pp PRIVATE "CHASE_GEMMUL8_ROOT=\"${_GEMMUL8_ROOT_LOG}\"")
  if(_GEMMUL8_FROM_FETCH)
    add_dependencies(gemmul8pp gemmul8_ep)
  endif()
  message(STATUS "gemmul8pp uses CUDA architecture ${CMAKE_CUDA_ARCHITECTURES}, the same one as the rest of ChASE")
  endif()
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
if(CHASE_ENABLE_OZAKI AND _CHASE_OZAKI_BACKEND STREQUAL "GEMMUL8")
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

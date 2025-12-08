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
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/blaspp>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/external/blaspp>
)

target_include_directories( blaspp INTERFACE
  "$<BUILD_INTERFACE:${includePath}>"
  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>" 
)
target_link_libraries(blaspp INTERFACE BLAS::BLAS)

add_library(lapackpp INTERFACE)
target_include_directories(lapackpp INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/lapackpp>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/external/lapackpp>
)
target_include_directories( lapackpp INTERFACE
  "$<BUILD_INTERFACE:${includePath}>"
  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>" 
)
target_link_libraries(lapackpp INTERFACE LAPACK::LAPACK)

find_package(SCALAPACK)
if(SCALAPACK_FOUND)
  add_library(scalapackpp INTERFACE)
  target_include_directories(scalapackpp INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/scalapackpp>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/external/scalapackpp>
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>" 
  )
  target_compile_definitions(scalapackpp INTERFACE HAS_SCALAPACK=1)
  target_link_libraries(scalapackpp INTERFACE SCALAPACK::SCALAPACK)
  message(STATUS "ScaLAPACK support enabled.")
else()
    message(STATUS "No ScaLAPACK is detected.")
endif()

find_package(CUDAToolkit)
if(CUDAToolkit_FOUND)
  add_library(cublaspp INTERFACE)
  target_include_directories(cublaspp INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cublaspp>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/external/cublaspp>
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>" 
  )
  target_link_libraries(cublaspp INTERFACE CUDA::cublas CUDA::cudart)

  add_library(cusolverpp INTERFACE)
  target_include_directories(cublaspp INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cusolverpp>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/external/cusolverpp>
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>" 
  )
  target_link_libraries(cusolverpp INTERFACE CUDA::cusolver CUDA::cudart)

  if(CHASE_ENABLE_XGEEV_RAYLEIGHRITZ)
    if(${CUDAToolkit_VERSION} VERSION_LESS 12.6.85)
	    message(WARNING "CUDA VERSION = " ${CUDAToolkit_VERSION} " < 12.6.85! Xgeev kernel is not in this CUDA version. ChASE GPU will use the HEEVD version instead.")
    else()
      target_compile_definitions( cusolverpp  INTERFACE "-DXGEEV_EXISTS" )
      message(STATUS "CUDA VERSION = " ${CUDAToolkit_VERSION} " >= 12.6.85. Xgeev kernel exists. ChASE GPU XGEEV will work for Pseudo-Hermitian problems.")
    endif()
  endif()

endif()

install(TARGETS blaspp
  EXPORT chase_targets
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY blaspp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/external
  FILES_MATCHING
    PATTERN "*.hpp"
    PATTERN "*.inc"
)

#####
install(TARGETS lapackpp
  EXPORT chase_targets
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY lapackpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/external
  FILES_MATCHING
    PATTERN "*.hpp"
    PATTERN "*.inc"
)

####
if(SCALAPACK_FOUND)
  install(TARGETS scalapackpp
    EXPORT chase_targets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

  install(DIRECTORY scalapackpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/external
    FILES_MATCHING
      PATTERN "*.hpp"
      PATTERN "*.inc"
  )

endif()

if(CUDAToolkit_FOUND)
  install(TARGETS cublaspp
    EXPORT chase_targets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

  install(DIRECTORY cublaspp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/external
    FILES_MATCHING
      PATTERN "*.hpp"
  )

  install(TARGETS cusolverpp
    EXPORT chase_targets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

  install(DIRECTORY cusolverpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/external
    FILES_MATCHING
      PATTERN "*.hpp"
  )
endif()

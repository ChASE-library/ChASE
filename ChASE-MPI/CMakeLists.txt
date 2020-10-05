# -*- Mode: cmake -*-

# Specify the minimum version for CMake
cmake_minimum_required( VERSION 3.8 )

# project( ChASE-MPI LANGUAGES C CXX CUDA )

add_library( chase_mpi INTERFACE )
target_link_libraries(chase_mpi INTERFACE chase_algorithm)

#underscores for FORTRAN
enable_language( Fortran )
include( FortranCInterface )
FortranCInterface_HEADER( ${CMAKE_BINARY_DIR}/fortran_mangle.h
  MACRO_NAMESPACE "FC_"
  SYMBOL_NAMESPACE "FC_"
  )


list(APPEND includePath ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories( chase_mpi INTERFACE
  "$<BUILD_INTERFACE:${includePath}>"
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ChASE-MPI>  # <prefix>/include/mylib
)

## BLAS / LAPACK ##
find_package( BLAS   REQUIRED )
find_package( LAPACK REQUIRED )
find_package( MPI    REQUIRED )


target_include_directories( chase_mpi INTERFACE
  ${MPI_CXX_INCLUDE_PATH}
  )

target_link_libraries( chase_mpi INTERFACE
  ${BLAS_LIBRARIES}
  ${LAPACK_LIBRARIES}
  ${MPI_CXX_LIBRARIES}
  )

target_compile_options( chase_mpi INTERFACE
  ${CMAKE_C_FLAGS}
  ${MPI_CXX_COMPILE_FLAGS}
  )

install( TARGETS chase_mpi
  EXPORT chase-mpi
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )

install(DIRECTORY . DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ChASE-MPI/
  FILES_MATCHING
    PATTERN "*.hpp"
    PATTERN "*.inc"
)

install( FILES  ${CMAKE_BINARY_DIR}/fortran_mangle.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

install(EXPORT chase-mpi
  NAMESPACE ChASE::
  FILE chase-mpi.cmake
  EXPORT_LINK_INTERFACE_LIBRARIES
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
  )

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  find_package( CUDA REQUIRED )
  enable_language(CUDA)

  add_library( chase_cuda "kernels/shift.cu" )

  set_target_properties( chase_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
 
  target_include_directories(
      chase_cuda
      PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
             $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ChASE-MPI>)

  install( TARGETS chase_cuda
    EXPORT chase-cuda
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )

  install(EXPORT chase-cuda
    NAMESPACE ChASE::
    FILE chase-cuda.cmake
    EXPORT_LINK_INTERFACE_LIBRARIES
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
    )

  # We need cublas for chase_mpi
  target_include_directories( chase_mpi INTERFACE
    ${CUDA_INCLUDE_DIRS}
    )
  target_link_libraries( chase_mpi INTERFACE
    ${CUDA_CUBLAS_LIBRARIES}
    )


else()
  message(STATUS "No CUDA support")
endif()



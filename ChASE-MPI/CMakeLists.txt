# -*- Mode: cmake -*-

# Specify the minimum version for CMake
cmake_minimum_required( VERSION 3.8 )

add_library( chase_mpi INTERFACE )
target_link_libraries(chase_mpi INTERFACE chase_algorithm)

#underscores for FORTRAN
enable_language( Fortran )
include( FortranCInterface )
FortranCInterface_HEADER( ${CMAKE_BINARY_DIR}/include/fortran_mangle.h
  MACRO_NAMESPACE "FC_"
  SYMBOL_NAMESPACE "FC_"
  )
target_include_directories( chase_mpi INTERFACE
  "${CMAKE_BINARY_DIR}/include/"
  "${CMAKE_CURRENT_LIST_DIR}"
  )

## BLAS / LAPACK ##
find_package( BLAS   REQUIRED )
find_package( LAPACK REQUIRED )
find_package( MPI    REQUIRED )

target_include_directories( chase_mpi INTERFACE
  "${CMAKE_BINARY_DIR}/include/"
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


  add_library( chase_mpi_cuda "kernels/shift.cu" )
  set_target_properties( chase_mpi_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# -*- Mode: cmake -*-

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

find_package( CUDA )
if( CUDA_FOUND )
  target_compile_definitions( chase_mpi INTERFACE "-DBUILD_CUDA" )
  target_include_directories( chase_mpi INTERFACE ${CUDA_INCLUDE_DIRS} )
  target_link_libraries( chase_mpi INTERFACE
    ${CUDA_CUBLAS_LIBRARIES}
    ${CUDA_LIBRARIES}
    )
endif()

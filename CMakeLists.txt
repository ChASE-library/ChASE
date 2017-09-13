# -*- Mode: cmake -*-
# Specify the minimum version for CMake

cmake_minimum_required( VERSION 3.7 )

# Project's name

project( ChASE )

set( ChASE_VERSION_MAJOR 0 )
set( ChASE_VERSION_MINOR 9 )


# ## C++11 ##
set( CMAKE_CXX_STANDARD 11 )

add_definitions( "-DOUTPUT" )

# Set the output folder where your program will be created
set( CMAKE_BINARY_DIR       ${CMAKE_SOURCE_DIR}/build/ )
set( EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR} )
set( LIBRARY_OUTPUT_PATH    ${CMAKE_BINARY_DIR} )

include_directories("${PROJECT_SOURCE_DIR}/algorithm")
include_directories("${PROJECT_SOURCE_DIR}/genera/blas")
include_directories("${PROJECT_SOURCE_DIR}/testframework")

# underscores for FORTRAN
enable_language( Fortran )
include( FortranCInterface )
FortranCInterface_HEADER( ${CMAKE_BINARY_DIR}/include/fortran_mangle.h
  MACRO_NAMESPACE "FC_"
  SYMBOL_NAMESPACE "FC_"
  )
include_directories( blas ${CMAKE_BINARY_DIR}/include/ )

# ## BLAS / LAPACK ##
find_package( LAPACK REQUIRED )
find_package( BLAS   REQUIRED )

# ## OPENMP ##
find_package( OpenMP REQUIRED )
if( OPENMP_FOUND )
  set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}" )
  set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}" )
endif()


# ## BOOST ##
# TODO: do not make required, just don't build driver
find_package( Boost COMPONENTS serialization program_options )
find_package( MPI  )
find_package( MPI  )
find_package( CUDA )


## BLAS implementation ##
SET( LIB_NAME "blas_lib" )
SET( DRIVER_NAME "blas_driver" )

add_library( ${LIB_NAME} STATIC driver/blas/lib.cpp )
target_link_libraries( ${LIB_NAME}
  ${BLAS_LIBRARIES}
  ${LAPACK_LIBRARIES}
  )
target_compile_definitions( ${LIB_NAME} PUBLIC
  MATRIX_FREE_IMPLEMENTATION=MATRIX_FREE_IMPLEMENTATION_BLAS_INPLACE
  )

if( Boost_FOUND )
  add_executable( ${DRIVER_NAME} driver/blas/main.cpp )
  target_link_libraries( ${DRIVER_NAME}
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
    ${Boost_LIBRARIES}
    )
  target_compile_definitions( ${DRIVER_NAME} PUBLIC
    MATRIX_FREE_IMPLEMENTATION=MATRIX_FREE_IMPLEMENTATION_BLAS_INPLACE
    )
endif()

## CUDA implementation ##
if( CUDA_FOUND )
  set( LIB_NAME "cuda_lib" )
  set( DRIVER_NAME "cuda_driver" )

  cuda_add_library( ${LIB_NAME} STATIC driver/blas/lib.cpp )
  target_link_libraries( ${LIB_NAME}
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
    )
  target_compile_definitions( ${LIB_NAME} PUBLIC
    MATRIX_FREE_IMPLEMENTATION=MATRIX_FREE_IMPLEMENTATION_CUDA
    )

  if( Boost_FOUND )
    cuda_add_executable( ${DRIVER_NAME} driver/blas/main.cpp )
    target_link_libraries( ${DRIVER_NAME}
      ${BLAS_LIBRARIES}
      ${LAPACK_LIBRARIES}
      ${Boost_LIBRARIES}
      ${CUDA_LIBRARIES}
      )
    target_compile_definitions( ${DRIVER_NAME} PUBLIC
      MATRIX_FREE_IMPLEMENTATION=MATRIX_FREE_IMPLEMENTATION_CUDA
      )
    cuda_add_cublas_to_target( ${DRIVER_NAME} )
  endif()
endif()

## MPI+BLAS implementation ##
if( MPI_FOUND )
  set( LIB_NAME "mpi_lib" )
  set( DRIVER_NAME "mpi_driver" )

  add_library(${LIB_NAME} STATIC driver/blas/lib.cpp )
  target_link_libraries( ${LIB_NAME}
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
    ${MPI_CXX_LIBRARIES}
    )
  target_compile_definitions( ${LIB_NAME} PUBLIC
    MATRIX_FREE_IMPLEMENTATION=MATRIX_FREE_IMPLEMENTATION_MPI_BLAS
    HAS_MPI=1
    )
  target_include_directories( ${LIB_NAME} PRIVATE ${MPI_CXX_INCLUDE_PATH} )

  if( Boost_FOUND )
    add_executable( ${DRIVER_NAME} driver/blas/main.cpp )
    target_link_libraries( ${DRIVER_NAME}
      ${BLAS_LIBRARIES}
      ${LAPACK_LIBRARIES}
      ${Boost_LIBRARIES}
      ${MPI_CXX_LIBRARIES}
      )
    target_compile_definitions( ${DRIVER_NAME} PUBLIC
      MATRIX_FREE_IMPLEMENTATION=MATRIX_FREE_IMPLEMENTATION_MPI_BLAS
      HAS_MPI=1
      )
  endif()
endif()

## TODO: MPI+CUDA
if( CUDA_FOUND AND MPI_FOUND )
  set( LIB_NAME "mpicuda_lib" )
  set( DRIVER_NAME "mpicuda_driver" )

  if( Boost_FOUND )
    cuda_add_executable( ${DRIVER_NAME} driver/blas/main.cpp )
    target_link_libraries( ${DRIVER_NAME}
      ${BLAS_LIBRARIES}
      ${LAPACK_LIBRARIES}
      ${Boost_LIBRARIES}
      ${CUDA_LIBRARIES}
      ${MPI_CXX_LIBRARIES}
      )
    target_compile_definitions( ${DRIVER_NAME} PUBLIC
      MATRIX_FREE_IMPLEMENTATION=MATRIX_FREE_IMPLEMENTATION_MPI_CUDA
      )
    cuda_add_cublas_to_target( ${DRIVER_NAME} )
endif()
endif()

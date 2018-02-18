# -*- Mode: cmake -*-

# Specify the minimum version for CMake
cmake_minimum_required( VERSION 3.7 )

enable_testing()


#set(CMAKE_BUILD_TYPE Release)
set(CMAKE_BUILD_TYPE Debug)
# Project's name

project( ChASE )

# set( ChASE_VERSION_MAJOR 0 )
# set( ChASE_VERSION_MINOR 9 )


# ## C++11 ##
set( CMAKE_CXX_STANDARD 11 )

add_definitions( "-DOUTPUT" )

# Set the output folder where your program will be created
#set( CMAKE_BINARY_DIR       ${CMAKE_SOURCE_DIR} )
set( EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR} )
set( LIBRARY_OUTPUT_PATH    ${CMAKE_BINARY_DIR} )

include_directories(${PROJECT_SOURCE_DIR})

# underscores for FORTRAN
enable_language( Fortran )
include( FortranCInterface )
FortranCInterface_HEADER( ${CMAKE_BINARY_DIR}/include/fortran_mangle.h
  MACRO_NAMESPACE "FC_"
  SYMBOL_NAMESPACE "FC_"
  )
include_directories( blas "${CMAKE_BINARY_DIR}/include/" )

# ## BLAS / LAPACK ##
find_package( BLAS   REQUIRED )
find_package( LAPACK REQUIRED )

# ## OPENMP ##
# find_package( OpenMP REQUIRED )
# if( OPENMP_FOUND )
#   set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}" )
#   set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}" )
# endif()


# ## BOOST ##
# TODO: do not make required, just don't build driver
find_package( Boost COMPONENTS serialization program_options )
find_package( MPI  REQUIRED)
if( MPI_FOUND )
  # set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${MPI_C_COMPILE_FLAGS}" )
  # set( CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} ${MPI_CXX_COMPILE_FLAGS}" )
  add_definitions( "-DHAS_MPI=1" )
  set(CMAKE_C_COMPILER ${MPI_C_COMPILER})
  set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})
endif()

find_package( CUDA )
if( CUDA_FOUND )
  add_definitions( "-DHAS_CUDA=1" )
endif()

add_executable( "driver" driver/blas/noinput.cpp )
target_link_libraries( "driver"
  ${BLAS_LIBRARIES}
  ${LAPACK_LIBRARIES}
  ${Boost_LIBRARIES}
  ${MPI_CXX_LIBRARIES}
  )
target_compile_definitions( "driver" PUBLIC
  MATRIX_FREE_IMPLEMENTATION=MATRIX_FREE_IMPLEMENTATION_BLAS_INPLACE
  )
target_include_directories( "driver" PRIVATE ${MPI_CXX_INCLUDE_PATH} )



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
    ${MPI_CXX_LIBRARIES}
    )
  target_compile_definitions( ${DRIVER_NAME} PUBLIC
    MATRIX_FREE_IMPLEMENTATION=MATRIX_FREE_IMPLEMENTATION_BLAS_INPLACE
    )

  target_include_directories( ${DRIVER_NAME} PRIVATE ${MPI_CXX_INCLUDE_PATH} ${Boost_INCLUDE_DIRS} )

  add_test(
    NAME "${DRIVER_NAME}_test1"
    COMMAND ${EXECUTABLE_OUTPUT_PATH}/blas_driver --n 22360 --mode "R" --opt "S" --name jlesc --path_in /data/winkelmann/JLESC/SingleP/size23k/bin8/ --nev 100 --bgn 1 --end 1 --nex 50 --tol 1e-5 --deg 8 --double false
    )
endif()

## Basic BLAS implementation ##
SET( LIB_NAME "blas_basic_lib" )
SET( DRIVER_NAME "blas_basic_driver" )

add_library( ${LIB_NAME} STATIC driver/blas/lib.cpp )
target_link_libraries( ${LIB_NAME}
  ${BLAS_LIBRARIES}
  ${LAPACK_LIBRARIES}
  )
target_compile_definitions( ${LIB_NAME} PUBLIC
  MATRIX_FREE_IMPLEMENTATION=MATRIX_FREE_IMPLEMENTATION_BLAS
  )

if( Boost_FOUND )
  add_executable( ${DRIVER_NAME} driver/blas/main.cpp )
  target_link_libraries( ${DRIVER_NAME}
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
    ${Boost_LIBRARIES}
    ${MPI_CXX_LIBRARIES}
    )
  target_compile_definitions( ${DRIVER_NAME} PUBLIC
    MATRIX_FREE_IMPLEMENTATION=MATRIX_FREE_IMPLEMENTATION_BLAS
    )

  target_include_directories( ${DRIVER_NAME} PRIVATE ${MPI_CXX_INCLUDE_PATH} ${Boost_INCLUDE_DIRS} )

  add_test(
    NAME "${DRIVER_NAME}_test1"
    COMMAND ${EXECUTABLE_OUTPUT_PATH}/blas_driver --n 22360 --mode "R" --opt "S" --name jlesc --path_in /data/winkelmann/JLESC/SingleP/size23k/bin8/ --nev 100 --bgn 1 --end 1 --nex 50 --tol 1e-5 --deg 8 --double false
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
      ${MPI_CXX_LIBRARIES}
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

  add_library(${LIB_NAME} STATIC driver/blas/lib.cpp driver/blas/hdf5.c )
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

    target_include_directories( ${DRIVER_NAME} PRIVATE ${MPI_CXX_INCLUDE_PATH} ${Boost_INCLUDE_DIRS} )
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

#find_package(Elemental)
set( Elemental_LIBRARIES  "-lelemental -lpmrrr -lpthread -mkl")
set( Elemental_INCLUDE_DIRS "/homeb/slai/slai10/elemental/include")
set( Elemental_DEFINITIONS "")
set( Elemental_LINK_FLAGS "-L/homeb/slai/slai10/elemental/lib")


SET( DRIVER_NAME "elemental_driver")
add_executable(${DRIVER_NAME}  driver/elemental/main.cpp )
target_link_libraries( ${DRIVER_NAME}
  ${Elemental_LIBRARIES}
  ${MPI_CXX_LIBRARIES}
  )
target_compile_definitions(${DRIVER_NAME} PUBLIC ${Elemental_DEFINITIONS} HAS_ELEMENTAL=1 HAS_MPI=1 
#_GLIBCXX_USE_CXX11_ABI=1
)
target_include_directories(${DRIVER_NAME} PUBLIC ${Elemental_INCLUDE_DIRS})
if(Elemental_LINK_FLAGS)
  set_target_properties(${DRIVER_NAME}
    PROPERTIES LINK_FLAGS ${Elemental_LINK_FLAGS})
endif()


SET( DRIVER_NAME "elemental_direct_driver")
add_executable(${DRIVER_NAME}  driver/elemental/direct.cpp )
target_link_libraries( ${DRIVER_NAME}
  ${Elemental_LIBRARIES}
  ${MPI_CXX_LIBRARIES}
  )
target_compile_definitions(${DRIVER_NAME} PUBLIC ${Elemental_DEFINITIONS} HAS_ELEMENTAL=1 HAS_MPI=1 
#_GLIBCXX_USE_CXX11_ABI=1
)
target_include_directories(${DRIVER_NAME} PUBLIC ${Elemental_INCLUDE_DIRS})
if(Elemental_LINK_FLAGS)
  set_target_properties(${DRIVER_NAME}
    PROPERTIES LINK_FLAGS ${Elemental_LINK_FLAGS})
endif()

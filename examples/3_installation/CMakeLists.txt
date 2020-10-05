cmake_minimum_required(VERSION 3.8)

project(
    chase-app
    VERSION 0.0.1
    LANGUAGES CXX)

find_package( chase REQUIRED CONFIG)
find_package( BLAS   REQUIRED )
find_package( LAPACK REQUIRED )
find_package( OpenMP)

find_package(MPI REQUIRED)

set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

############## Compile the CPU version of ChASE ######################

message(STATUS "Compile the CPU version of ChASE as chase-app")

add_executable(chase-app)

target_sources(chase-app PRIVATE chase-app.cpp)

target_compile_features(chase-app INTERFACE cxx_auto_type)

target_include_directories( chase-app INTERFACE
  ${MPI_CXX_INCLUDE_PATH}
  )

target_link_libraries( chase-app INTERFACE
  ${BLAS_LIBRARIES}
  ${LAPACK_LIBRARIES} 
  ${MPI_CXX_LIBRARIES}
  )

target_link_libraries(chase-app PUBLIC ChASE::chase_mpi)

############## Compile the multi-GPU version of ChASE #################

### compile gpu version of application if ChASE implemented with GPUs #
if(EXISTS "${CMAKE_PREFIX_PATH}/lib64/cmake/ChASE/chase-cuda.cmake")

  message(STATUS "ChASE supports GPUs")

  include(CheckLanguage)

  check_language(CUDA)

  if(CMAKE_CUDA_COMPILER)

    find_package( CUDA REQUIRED )

    enable_language(CUDA)

    message(STATUS "Compile the GPU version of ChASE as chase-app-gpu")

    add_executable(chase-app-gpu)

    target_sources(chase-app-gpu PRIVATE chase-app-gpu.cpp)

    target_compile_features(chase-app-gpu INTERFACE cxx_auto_type)

    target_include_directories( chase-app-gpu INTERFACE
      ${MPI_CXX_INCLUDE_PATH}
      )

    target_link_libraries( chase-app-gpu INTERFACE
      ${BLAS_LIBRARIES}
      ${LAPACK_LIBRARIES}
      ${MPI_CXX_LIBRARIES}
      )  

    target_link_libraries(chase-app-gpu PUBLIC ChASE::chase_mpi)

    target_link_libraries(chase-app-gpu PUBLIC ChASE::chase_cuda)

  endif()
endif()


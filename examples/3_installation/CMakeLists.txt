cmake_minimum_required(VERSION 3.8)

project(chase-app VERSION 0.0.1 LANGUAGES CXX)

find_package( chase REQUIRED CONFIG)
############## Compile the CPU version of ChASE ######################

message(STATUS "Compile the CPU version of ChASE as chase-app")
add_executable(chase-app)
target_sources(chase-app PRIVATE chase-app.cpp)
target_link_libraries(chase-app PUBLIC ChASE::chase_mpi)

############## Compile the multi-GPU version of ChASE #################

### compile gpu version of application if ChASE implemented with GPUs #
if(EXISTS "${CMAKE_PREFIX_PATH}/lib64/cmake/ChASE/chase-cuda.cmake")
  message(STATUS "ChASE supports GPUs")
  message(STATUS "Compile the GPU version of ChASE as chase-app-gpu")

  add_executable(chase-app-gpu)
  target_sources(chase-app-gpu PRIVATE chase-app.cpp)
  target_link_libraries(chase-app-gpu PUBLIC ChASE::chase_mpi ChASE::chase_cuda)
  target_compile_definitions(chase-app-gpu PRIVATE DRIVER_BUILD_MGPU=1)

endif()


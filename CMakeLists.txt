# -*- Mode: cmake -*-
cmake_minimum_required( VERSION 3.8 )

project( ChASE LANGUAGES C CXX )

# FIND_PACKAGE(OpenMP REQUIRED)
# if (OPENMP_FOUND)
#     set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#     set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#     set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
# endif()

# ## algorithm ##
add_library(chase_algorithm INTERFACE)

target_include_directories( chase_algorithm INTERFACE
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>"
  $<INSTALL_INTERFACE:include/>  # <prefix>/include/mylib
)
target_compile_features(chase_algorithm INTERFACE cxx_auto_type)

option( CHASE_OUTPUT "ChASE will provide output at each iteration")
if( CHASE_OUTPUT )
  target_compile_definitions( chase_algorithm  INTERFACE "-DCHASE_OUTPUT" )
endif()

install( TARGETS chase_algorithm
  EXPORT chase-headers
  LIBRARY DESTINATION lib
  INCLUDES DESTINATION include
  ARCHIVE DESTINATION lib
  )


install(DIRECTORY algorithm DESTINATION include/algorithm
  FILES_MATCHING
    PATTERN "*.hpp"
    PATTERN "*.inc"
)

install(EXPORT chase-headers
  NAMESPACE ChASE::
  FILE chase-header.cmake
  EXPORT_LINK_INTERFACE_LIBRARIES
  DESTINATION share
  )

# ## ChASE-MPI ##
add_subdirectory( "ChASE-MPI/")
add_subdirectory( "ChASE-Elemental/")

# Driver
add_executable( "chase_driver" driver/noinput.cpp )
target_link_libraries(chase_driver chase_mpi)

# Unit tests
option(UNITTEST "Build unit tests" OFF)
if(UNITTEST)
	message(STATUS "Building unittests")
	add_subdirectory("./unittests")
endif()

# Simple Driver
option(BUILD_SIMPLEDRIVER "Build Simple Driver" OFF)
if(BUILD_SIMPLEDRIVER)
  message(STATUS "Building Simple Driver")
  add_subdirectory("./simple_driver")
endif()

enable_testing()
add_test( BASIC "chase_driver" )

# Install ChASE as a CMake package
include(CMakePackageConfigHelpers)
configure_package_config_file(
    "cmake/Config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/chase-config.cmake"
    INSTALL_DESTINATION share
)

install( FILES
  "${CMAKE_CURRENT_BINARY_DIR}/chase-config.cmake"
  DESTINATION share
  )

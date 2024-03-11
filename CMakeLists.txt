# -*- Mode: cmake -*-
cmake_minimum_required( VERSION 3.8 )

project( ChASE LANGUAGES C CXX VERSION 1.3.0 )
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
# ## algorithm ##

set(CMAKE_CXX_STANDARD 11)

add_library(chase_algorithm INTERFACE)

include(GNUInstallDirs)

target_include_directories( chase_algorithm INTERFACE
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>"
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>  # <prefix>/include/mylib
)

target_compile_features(chase_algorithm INTERFACE cxx_auto_type)

option( CHASE_OUTPUT "ChASE will provide output at each iteration")
if( CHASE_OUTPUT )
  target_compile_definitions( chase_algorithm  INTERFACE "-DCHASE_OUTPUT" )
endif()

option(ENABLE_NSIGHT "Enable profiling with Nvidia Nsight Systems" OFF)

install( TARGETS chase_algorithm
  EXPORT chase-headers
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )


install(DIRECTORY algorithm DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING
    PATTERN "*.hpp"
    PATTERN "*.inc"
)

install(EXPORT chase-headers
  NAMESPACE ChASE::
  FILE chase-header.cmake
  EXPORT_LINK_INTERFACE_LIBRARIES
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
  )

## ChASE-MPI ##
add_subdirectory( "ChASE-MPI/")

## Tests
add_executable( "chase_driver" tests/noinput.cpp )
if(TARGET chase_cuda )
  enable_language(CUDA)
  target_link_libraries(chase_driver chase_mpi chase_cuda)
  target_compile_definitions(chase_driver PRIVATE USE_GPU=1)
else()
  target_link_libraries(chase_driver chase_mpi)
endif()

add_subdirectory("interface")

# Examples
option(BUILD_WITH_EXAMPLES "Build the examples" OFF)
if(BUILD_WITH_EXAMPLES)
        message(STATUS "Building the examples of ChASE")
        add_subdirectory("./examples")
endif()

set(MPI_RUN mpirun CACHE STRING "MPI runner (mpirun/srun)")
option(ENABLE_TESTS "Enable unit tests." OFF)
if(ENABLE_TESTS)
    MESSAGE("Test enabled. Finding GoogleTests.")
    include(${CMAKE_SOURCE_DIR}/cmake/external/Gtest/FetchGtest.cmake)
    include(CTest)
    add_subdirectory(tests)
endif()

# Documentation
option(BUILD_WITH_DOCS "Build the examples" OFF)
if(BUILD_WITH_DOCS)
    message(STATUS "Building Documentation of ChASE")
    add_subdirectory("./docs")
endif()

# Install ChASE as a CMake package
include(CMakePackageConfigHelpers)

if(TARGET chase_cuda )
configure_package_config_file(
    "cmake/Config_CUDA.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/chase-config.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

else()
configure_package_config_file(
    "cmake/Config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/chase-config.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

endif()

install( FILES
  "${CMAKE_CURRENT_BINARY_DIR}/chase-config.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
  )

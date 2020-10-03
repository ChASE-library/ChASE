# -*- Mode: cmake -*-
cmake_minimum_required( VERSION 3.8 )

project( ChASE LANGUAGES C CXX )

# ## algorithm ##
add_library(chase_algorithm INTERFACE)

include(GNUInstallDirs)

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


install(DIRECTORY algorithm DESTINATION include
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

## Tests
add_subdirectory( "tests/")

## ChASE-MPI ##
add_subdirectory( "ChASE-MPI/")
add_subdirectory( "ChASE-Elemental/")

# Examples
option(BUILD_WITH_EXAMPLES "Build the examples" OFF)
if(BUILD_WITH_EXAMPLES)
        message(STATUS "Building the examples of ChASE")
        add_subdirectory("./examples")
endif()

# Unit tests
option(UNITTEST "Build unit tests" OFF)
if(UNITTEST)
	message(STATUS "Building unittests")
	add_subdirectory("./unittests")
endif()

enable_testing()
add_test( BASIC "./tests/chase_driver" )

# Install ChASE as a CMake package
include(CMakePackageConfigHelpers)
configure_package_config_file(
    "cmake/Config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/chase-config.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

install( FILES
  "${CMAKE_CURRENT_BINARY_DIR}/chase-config.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
  )

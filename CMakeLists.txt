# -*- Mode: cmake -*-

cmake_minimum_required( VERSION 3.7 )

project( ChASE )

# ## algorithm ##
add_library(chase_algorithm INTERFACE)
target_include_directories(chase_algorithm INTERFACE "${CMAKE_CURRENT_LIST_DIR}")
target_compile_features(chase_algorithm INTERFACE cxx_std_11)

# ## ChASE-MPI ##
add_subdirectory( "ChASE-MPI/")
add_subdirectory( "ChASE-Elemental/")


option( CHASE_OUTPUT "ChASE will provide output at each iteration")
if( CHASE_OUTPUT )
  target_compile_definitions( chase_algorithm  INTERFACE "-DCHASE_OUTPUT" )
endif()

set( CMAKE_BUILD_TYPE Debug )

# Driver
#add_executable( "driver" EXCLUDE_FROM_ALL driver/noinput.cpp )
add_executable( "driver" driver/noinput.cpp )
target_link_libraries(driver chase_mpi)

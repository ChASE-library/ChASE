# -*- Mode: cmake -*-

cmake_minimum_required( VERSION 3.6 )

project( ChASE )

# ## algorithm ##
add_library(chase_algorithm INTERFACE)
target_include_directories(chase_algorithm INTERFACE "${CMAKE_CURRENT_LIST_DIR}")
target_compile_features(chase_algorithm INTERFACE cxx_auto_type)

# ## ChASE-MPI ##
add_subdirectory( "ChASE-MPI/")
add_subdirectory( "ChASE-Elemental/")


option( CHASE_OUTPUT "ChASE will provide output at each iteration")
if( CHASE_OUTPUT )
  target_compile_definitions( chase_algorithm  INTERFACE "-DCHASE_OUTPUT" )
endif()

# Driver
add_executable( "chase_driver" EXCLUDE_FROM_ALL driver/noinput.cpp )
target_link_libraries(chase_driver chase_mpi)

# -*- Mode: cmake -*-
cmake_minimum_required( VERSION 3.8 )

project( ChASE LANGUAGES C CXX VERSION 1.3.0 )
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
# ## algorithm ##

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(chase_algorithm INTERFACE)

include(GNUInstallDirs)

target_include_directories( chase_algorithm INTERFACE
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>"
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>  # <prefix>/include/mylib
)

target_compile_features(chase_algorithm INTERFACE cxx_auto_type)

option( CHASE_OUTPUT "ChASE will provide output at each iteration")
# Add an option to enable/disable OpenMP support
option(CHASE_ENABLE_OPENMP "Enable OpenMP support" ON)
option(CHASE_ENABLE_MIXED_PRECISION "Enable mixed precision support" OFF)
option(CHASE_ENABLE_MPI_IO "Enable MPI IO to read Hamiltonian matrix from local" OFF)
option(CHASE_USE_NVTX "Enable NVTX for profiling" OFF)
option(CHASE_BUILD_WITH_DOCS "Build the examples" OFF)

if( CHASE_OUTPUT )
  target_compile_definitions( chase_algorithm  INTERFACE "-DCHASE_OUTPUT" )
endif()
# Find OpenMP package if the option is enabled
# Find OpenMP package if the option is enabled
if(ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
    if(OpenMP_CXX_FOUND)
        message(STATUS "OpenMP found, enabling OpenMP support")
        
        # Add OpenMP compiler flags globally
        add_compile_options(${OpenMP_CXX_FLAGS})
        
        # Link OpenMP libraries globally
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    else()
        message(WARNING "OpenMP not found, building without OpenMP support")
    endif()
endif()

# Add the flag to the preprocessor if the option is enabled
if (CHASE_ENABLE_MIXED_PRECISION)
    add_definitions(-DENABLE_MIXED_PRECISION)
    message(STATUS "Mixed precision support enabled.")
else()
    message(STATUS "Mixed precision support disabled.")
endif()

if(CHASE_USE_NVTX)
  target_compile_definitions( chase_algorithm  INTERFACE "-DUSE_NVTX" )
endif()

add_subdirectory(grid)
add_subdirectory(external)
add_subdirectory(linalg)
add_subdirectory(Impl)
add_subdirectory(examples)

add_executable( "chase_driver" tests/noinput.cpp )
target_link_libraries(chase_driver chase_cpu)

if(TARGET chase_gpu)
add_executable( "chase_driver_gpu" tests/noinput.cpp )
target_link_libraries(chase_driver_gpu chase_gpu)
endif()

option(ENABLE_TESTS "Enable unit tests." OFF)
if(ENABLE_TESTS)
    MESSAGE("Test enabled. Finding GoogleTests.")
    include(${CMAKE_SOURCE_DIR}/cmake/external/Gtest/FetchGtest.cmake)
    include(CTest)

    set(MPI_RUN mpirun CACHE STRING "MPI runner (mpirun/srun)")
    set(MPI_RUN_ARGS  CACHE STRING "")
    set(MPI_TEST ON CACHE BOOL "Run test with mpi")
    add_subdirectory(tests)
endif() 

# Documentation
if(CHASE_BUILD_WITH_DOCS)
    message(STATUS "Building Documentation of ChASE")
    add_subdirectory("docs")
endif()

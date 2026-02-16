# -*- Mode: cmake -*-
cmake_minimum_required( VERSION 3.18 )

project( ChASE LANGUAGES C CXX Fortran VERSION 1.7.0 )
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")
# ## algorithm ##

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(chase_algorithm INTERFACE)

include(GNUInstallDirs)

# Detect features based on what CMake finds
# These will be used in chase_config.h.in
set(HAS_MPI OFF)
set(HAS_CUDA OFF)
set(HAS_NCCL OFF)
set(HAS_SCALAPACK OFF)

# Note: These are set later when dependencies are found
# We'll update them before generating chase_config.h

target_include_directories( chase_algorithm INTERFACE
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>"
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>"  # For generated chase_config.h
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>  # <prefix>/include/mylib
)

# Optional: Force-include chase_config.h in all translation units
# This ensures configuration is always consistent, but may increase compile times
# Uncomment the line below to enable automatic inclusion:
# target_compile_options(chase_algorithm INTERFACE 
#   "$<BUILD_INTERFACE:-include;${CMAKE_CURRENT_BINARY_DIR}/chase_config.h>"
#   "$<INSTALL_INTERFACE:-include;chase_config.h>")

target_compile_features(chase_algorithm INTERFACE cxx_auto_type)

option( CHASE_OUTPUT "ChASE will provide output at each iteration")
# Add an option to enable/disable OpenMP support
option(CHASE_ENABLE_OPENMP "Enable OpenMP support" ON)
option(CHASE_ENABLE_MIXED_PRECISION "Enable mixed precision support" OFF)
option(CHASE_ENABLE_MPI_IO "Enable MPI IO to read Hamiltonian matrix from local" OFF)
option(CHASE_USE_NVTX "Enable NVTX for profiling" OFF)
option(CHASE_BUILD_WITH_EXAMPLES "Build the examples" OFF)
option(CHASE_BUILD_WITH_DOCS "Build the docs" OFF)
option(CHASE_SAVE_RESIDUALS "Print the residuals in an external file" OFF)
option(CHASE_ENABLE_XGEEV_RAYLEIGHRITZ "Enable solving pseudo-hermitian problems with XGEEV cusolver" OFF)
option(ChASE_DISPLAY_COND_V_SVD "Compute and display condition number of V from SVD" OFF)
option(CHASE_QR_DOUBLE_PRECISION "Operate QR in Double Precision" ON)
option(CHASE_RR_DOUBLE_PRECISION "Operate HEEVD in RR for pseudo-hermitian matrices in Double Precision" OFF)
option(CHASE_ENABLE_TESTS "Enable unit tests." OFF)
#XZ: Will be removed once performance is confirmed
option(CHASE_ENABLE_GPU_RESIDENT_LANCZOS "Enable GPU-resident Lanczos with fused kernels and NCCL (experimental)" ON)

if(NOT CMAKE_BUILD_TYPE STREQUAL "Release")
    message(WARNING "For prodution, please consider CMAKE_BUILD_TYPE to be Release or RelWithDebInfo")
endif()

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
    target_compile_definitions(chase_algorithm INTERFACE "-DENABLE_MIXED_PRECISION")
    message(STATUS "Mixed precision support enabled.")
else()
    message(STATUS "Mixed precision support disabled.")
endif()

if(CHASE_USE_NVTX)
  target_compile_definitions( chase_algorithm  INTERFACE "-DUSE_NVTX" )
endif()

if(CHASE_QR_DOUBLE_PRECISION)
  target_compile_definitions(chase_algorithm INTERFACE "-DQR_DOUBLE_PRECISION")
endif()

if(CHASE_RR_DOUBLE_PRECISION)
  target_compile_definitions(chase_algorithm INTERFACE "-DRR_DOUBLE_PRECISION")
endif()

if(ChASE_DISPLAY_COND_V_SVD)
    target_compile_definitions(chase_algorithm INTERFACE "-DChASE_DISPLAY_COND_V_SVD")
endif()

if(CHASE_SAVE_RESIDUALS)
    target_compile_definitions(chase_algorithm INTERFACE "-DCHASE_SAVE_RESIDUALS")
endif()
# XZ: Will be removed once performance is confirmed
if(CHASE_ENABLE_GPU_RESIDENT_LANCZOS)
    target_compile_definitions(chase_algorithm INTERFACE "-DCHASE_ENABLE_GPU_RESIDENT_LANCZOS")
    message(STATUS "GPU-resident Lanczos with fused kernels enabled (experimental)")
endif()

add_subdirectory(external)
add_subdirectory(grid)
add_subdirectory(linalg)
add_subdirectory(Impl)
add_subdirectory(interface)

if(CHASE_BUILD_WITH_EXAMPLES)
    add_subdirectory(examples)
endif()

add_executable( "chase_driver" tests/noinput.cpp )
target_link_libraries(chase_driver chase_cpu)

if(TARGET chase_gpu)
add_executable( "chase_driver_gpu" tests/noinput.cpp )
target_link_libraries(chase_driver_gpu chase_gpu)
endif()

if(CHASE_ENABLE_TESTS)
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

##installation
install( TARGETS chase_algorithm
  EXPORT chase_targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )

install(DIRECTORY algorithm DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING
    PATTERN "*.hpp"
    PATTERN "*.inc"
)

# Install the generated configuration header
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/chase_config.h"
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT chase_targets
        NAMESPACE ChASE::
        EXPORT_LINK_INTERFACE_LIBRARIES
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
        FILE chaseTargets.cmake)

include(CMakePackageConfigHelpers)

# Collect compile definitions for pkg-config
set(CHASE_COMPILE_DEFINITIONS "")
if(CHASE_OUTPUT)
    set(CHASE_COMPILE_DEFINITIONS "${CHASE_COMPILE_DEFINITIONS} -DCHASE_OUTPUT")
endif()
if(CHASE_USE_NVTX)
    set(CHASE_COMPILE_DEFINITIONS "${CHASE_COMPILE_DEFINITIONS} -DUSE_NVTX")
endif()
if(CHASE_ENABLE_MIXED_PRECISION)
    set(CHASE_COMPILE_DEFINITIONS "${CHASE_COMPILE_DEFINITIONS} -DENABLE_MIXED_PRECISION")
endif()
if(CHASE_QR_DOUBLE_PRECISION)
    set(CHASE_COMPILE_DEFINITIONS "${CHASE_COMPILE_DEFINITIONS} -DQR_DOUBLE_PRECISION")
endif()
if(CHASE_RR_DOUBLE_PRECISION)
    set(CHASE_COMPILE_DEFINITIONS "${CHASE_COMPILE_DEFINITIONS} -DRR_DOUBLE_PRECISION")
endif()
if(ChASE_DISPLAY_COND_V_SVD)
    set(CHASE_COMPILE_DEFINITIONS "${CHASE_COMPILE_DEFINITIONS} -DChASE_DISPLAY_COND_V_SVD")
endif()
if(CHASE_SAVE_RESIDUALS)
    set(CHASE_COMPILE_DEFINITIONS "${CHASE_COMPILE_DEFINITIONS} -DCHASE_SAVE_RESIDUALS")
endif()
if(CHASE_ENABLE_GPU_RESIDENT_LANCZOS)
    set(CHASE_COMPILE_DEFINITIONS "${CHASE_COMPILE_DEFINITIONS} -DCHASE_ENABLE_GPU_RESIDENT_LANCZOS")
endif()

set(CHASE_LINK_LIBRARIES "")

set(IS_ENABLE_MPI OFF)
set(IS_ENABLE_SCALAPACK OFF)
set(IS_ENABLE_NCCL OFF)

if(TARGET pchase_cpu)
    set(IS_ENABLE_MPI ON)
    set(HAS_MPI ON)  # For chase_config.h
endif()

if(TARGET scalapackpp)
    set(IS_ENABLE_SCALAPACK ON)
    set(HAS_SCALAPACK ON)  # For chase_config.h
endif()

if(TARGET linalg_internal_nccl)
    set(IS_ENABLE_NCCL ON)
    set(HAS_NCCL ON)  # For chase_config.h
endif()

# Check if CUDA was found
if(CMAKE_CUDA_COMPILER OR TARGET CUDA::cudart)
    set(HAS_CUDA ON)
endif()

# Now generate configuration header with correct feature detection
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/chase_config.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/chase_config.h"
  @ONLY
)

# Convert booleans to "ON"/"OFF" strings for the config file
set(IS_ENABLE_MPI_STRING ${IS_ENABLE_MPI})
set(IS_ENABLE_SCALAPACK_STRING ${IS_ENABLE_SCALAPACK})
set(IS_ENABLE_NCCL_STRING ${IS_ENABLE_NCCL})

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/chase-config.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ChASE
)

install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/chase-config.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ChASE
)

# Generate and install pkg-config file for non-CMake users
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/chase.pc.in"
    "${CMAKE_CURRENT_BINARY_DIR}/chase.pc"
    @ONLY
)

install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/chase.pc"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/cmake/modules
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ChASE)

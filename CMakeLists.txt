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
set(HAS_HIP OFF)
set(HAS_HIP_AWARE_MPI OFF)
set(HAS_NCCL OFF)
set(HAS_RCCL OFF)
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
set(CHASE_ENABLE_HIP "AUTO" CACHE STRING
    "Enable experimental AMD HIP support: AUTO, ON, or OFF")
set_property(CACHE CHASE_ENABLE_HIP PROPERTY STRINGS AUTO ON OFF)
string(TOUPPER "${CHASE_ENABLE_HIP}" CHASE_ENABLE_HIP_MODE)
if(NOT CHASE_ENABLE_HIP_MODE MATCHES "^(AUTO|ON|OFF)$")
  message(FATAL_ERROR
    "CHASE_ENABLE_HIP must be AUTO, ON, or OFF (got '${CHASE_ENABLE_HIP}')")
endif()

set(CHASE_ENABLE_HIP_EFFECTIVE OFF)
if(NOT CHASE_ENABLE_HIP_MODE STREQUAL "OFF")
  include(CheckLanguage)
  check_language(HIP)
  if(CMAKE_HIP_COMPILER)
    set(CHASE_ENABLE_HIP_EFFECTIVE ON)
  elseif(CHASE_ENABLE_HIP_MODE STREQUAL "ON")
    message(FATAL_ERROR "CHASE_ENABLE_HIP=ON but no HIP compiler was found")
  endif()
endif()

if(CHASE_ENABLE_HIP_EFFECTIVE)
  set(CHASE_HIP_DISTRIBUTED_COLLECTIVES "NCCL" CACHE STRING
      "HIP distributed pChASEGPU collective backend: NCCL (RCCL, default) or MPI (GPU-aware MPI opt-out)")
  set_property(CACHE CHASE_HIP_DISTRIBUTED_COLLECTIVES PROPERTY STRINGS MPI NCCL)
  message(STATUS "HIP support: ON (${CHASE_ENABLE_HIP_MODE})")
else()
  message(STATUS "HIP support: OFF (${CHASE_ENABLE_HIP_MODE})")
endif()
option(CHASE_PROBE_HIP_AWARE_MPI_RUNTIME
       "Run HIP-aware MPI probe at CMake configure (needs GPU + MPICH_GPU_SUPPORT_ENABLED)"
       OFF)
option(CHASE_USE_NVTX "Enable NVTX for profiling" OFF)
option(CHASE_BUILD_WITH_EXAMPLES "Build the examples" OFF)
option(CHASE_BUILD_WITH_DOCS "Build the docs" OFF)
option(CHASE_SAVE_RESIDUALS "Print the residuals in an external file" OFF)
option(CHASE_DISABLE_EARLY_LOCKING "Disable early-locking heuristic in locking step" OFF)
option(CHASE_ENABLE_XGEEV_RAYLEIGHRITZ "Enable solving pseudo-hermitian problems with XGEEV cusolver" OFF)
option(ChASE_DISPLAY_COND_V_SVD "Compute and display condition number of V from SVD" OFF)
option(CHASE_QR_DOUBLE_PRECISION "Operate QR in Double Precision" ON)
option(CHASE_RR_DOUBLE_PRECISION "Operate HEEVD in RR for pseudo-hermitian matrices in Double Precision" OFF)
option(CHASE_PANEL_HIPREC "Enable Split-Sync-Fix and high-precision panel protection" OFF)
option(CHASE_ENABLE_TESTS "Enable unit tests." OFF)
# Ozaki Scheme II (GEMMul8) integration (GPU-only backend, optional)
option(CHASE_ENABLE_OZAKI "Enable Ozaki Scheme II GEMM backend (requires GEMMul8)" OFF)
set(GEMMUL8_ROOT "" CACHE PATH "Path to GEMMul8 build/install prefix (must contain include/ and lib/)")
# Optional NCCL warm-up in pChASEGPU constructor (cholQR1 + Lanczos); skews startup timing.
option(CHASE_GPU_CONSTRUCTOR_BENCHMARK_WARMUP
  "Enable NCCL warm-up in pChASEGPU constructor (benchmarks only; extra startup cost)" OFF)

set(CHASE_GPU_CONSTRUCTOR_BENCHMARK_WARMUP_EFFECTIVE OFF)
if(CHASE_GPU_CONSTRUCTOR_BENCHMARK_WARMUP)
  if(CMAKE_BUILD_TYPE)
    string(TOUPPER "${CMAKE_BUILD_TYPE}" _CHASE_BT_UPPER)
    if(_CHASE_BT_UPPER STREQUAL "RELEASE" OR _CHASE_BT_UPPER STREQUAL "RELWITHDEBINFO")
      message(WARNING
        "CHASE_GPU_CONSTRUCTOR_BENCHMARK_WARMUP is ignored for CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} "
        "(Release and RelWithDebInfo builds never enable this warm-up).")
    else()
      set(CHASE_GPU_CONSTRUCTOR_BENCHMARK_WARMUP_EFFECTIVE ON)
    endif()
  else()
    # Multi-config generators may leave CMAKE_BUILD_TYPE empty; honor the option if set.
    set(CHASE_GPU_CONSTRUCTOR_BENCHMARK_WARMUP_EFFECTIVE ON)
  endif()
endif()

if(CHASE_GPU_CONSTRUCTOR_BENCHMARK_WARMUP_EFFECTIVE)
  target_compile_definitions(chase_algorithm INTERFACE CHASE_GPU_CONSTRUCTOR_BENCHMARK_WARMUP=1)
  message(STATUS "GPU constructor benchmark warm-up (pChASEGPU): ON")
else()
  message(STATUS "GPU constructor benchmark warm-up (pChASEGPU): OFF")
endif()

# GPU-resident Lanczos (fused kernels + NCCL) is required for GPU Lanczos paths.
set(CHASE_ENABLE_GPU_RESIDENT_LANCZOS ON)

if( CHASE_OUTPUT )
  target_compile_definitions( chase_algorithm  INTERFACE "-DCHASE_OUTPUT" )
endif()
# Find OpenMP package if the option is enabled
if(CHASE_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
    message(STATUS "OpenMP found, enabling OpenMP support")

    # C++ sources and host links only: nvcc forwards unknown options to its
    # host compiler, which may not accept the C++ compiler's OpenMP flag
    # (e.g. NVHPC -mp with gcc as CUDA host compiler).
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:${OpenMP_CXX_FLAGS}>")
    add_link_options("$<HOST_LINK:${OpenMP_CXX_FLAGS}>")
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

if(CHASE_PANEL_HIPREC)
  target_compile_definitions(chase_algorithm INTERFACE "-DCHASE_PANEL_HIPREC=1")
endif()

if(ChASE_DISPLAY_COND_V_SVD)
    target_compile_definitions(chase_algorithm INTERFACE "-DChASE_DISPLAY_COND_V_SVD")
endif()

if(CHASE_SAVE_RESIDUALS)
    target_compile_definitions(chase_algorithm INTERFACE "-DCHASE_SAVE_RESIDUALS")
endif()
if(CHASE_DISABLE_EARLY_LOCKING)
    target_compile_definitions(chase_algorithm INTERFACE "-DCHASE_DISABLE_EARLY_LOCKING")
endif()
target_compile_definitions(chase_algorithm INTERFACE "-DCHASE_ENABLE_GPU_RESIDENT_LANCZOS")
message(STATUS "GPU-resident Lanczos (required for distributed GPU Lanczos)")

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
if(CHASE_PANEL_HIPREC)
    set(CHASE_COMPILE_DEFINITIONS "${CHASE_COMPILE_DEFINITIONS} -DCHASE_PANEL_HIPREC=1")
endif()
if(ChASE_DISPLAY_COND_V_SVD)
    set(CHASE_COMPILE_DEFINITIONS "${CHASE_COMPILE_DEFINITIONS} -DChASE_DISPLAY_COND_V_SVD")
endif()
if(CHASE_SAVE_RESIDUALS)
    set(CHASE_COMPILE_DEFINITIONS "${CHASE_COMPILE_DEFINITIONS} -DCHASE_SAVE_RESIDUALS")
endif()
if(CHASE_DISABLE_EARLY_LOCKING)
    set(CHASE_COMPILE_DEFINITIONS "${CHASE_COMPILE_DEFINITIONS} -DCHASE_DISABLE_EARLY_LOCKING")
endif()
set(CHASE_COMPILE_DEFINITIONS "${CHASE_COMPILE_DEFINITIONS} -DCHASE_ENABLE_GPU_RESIDENT_LANCZOS")
if(CHASE_GPU_CONSTRUCTOR_BENCHMARK_WARMUP_EFFECTIVE)
  set(CHASE_COMPILE_DEFINITIONS "${CHASE_COMPILE_DEFINITIONS} -DCHASE_GPU_CONSTRUCTOR_BENCHMARK_WARMUP=1")
endif()
if(CHASE_ENABLE_OZAKI)
  set(CHASE_COMPILE_DEFINITIONS "${CHASE_COMPILE_DEFINITIONS} -DCHASE_ENABLE_OZAKI=1")
endif()

set(CHASE_LINK_LIBRARIES "")

# Convert concrete CMake library lists to linker flags without traversing the
# target graph. Absolute libraries and existing flags are preserved; plain
# library names receive the conventional -l prefix.
macro(_chase_pc_append_link_items)
    foreach(_chase_pc_item ${ARGN})
        if(NOT _chase_pc_item STREQUAL "")
            if(IS_ABSOLUTE "${_chase_pc_item}" OR
               _chase_pc_item MATCHES "^-" OR
               _chase_pc_item MATCHES " ")
                string(APPEND CHASE_LINK_LIBRARIES " ${_chase_pc_item}")
            else()
                string(APPEND CHASE_LINK_LIBRARIES " -l${_chase_pc_item}")
            endif()
        endif()
    endforeach()
endmacro()

set(IS_ENABLE_MPI OFF)
set(IS_ENABLE_SCALAPACK OFF)
set(IS_ENABLE_NCCL OFF)
set(IS_ENABLE_RCCL OFF)
set(IS_ENABLE_HIP OFF)

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

if(TARGET mpi_grid_rccl)
    set(IS_ENABLE_RCCL ON)
    set(HAS_RCCL ON)  # For chase_config.h
endif()

# Check if CUDA was found
if(CMAKE_CUDA_COMPILER OR TARGET CUDA::cudart)
    set(HAS_CUDA ON)
endif()

if(TARGET linalg_internal_hip)
    set(IS_ENABLE_HIP ON)
    set(HAS_HIP ON)
endif()

if(HIP_AWARE_MPI_SUPPORTED)
    set(HAS_HIP_AWARE_MPI ON)
endif()

# pkg-config does not understand CMake targets, so describe each compiled
# backend explicitly and use the concrete libraries found by CMake.
if(TARGET chase_cuda_kernels)
    if(CUDA_cudart_LIBRARY)
        get_filename_component(_chase_cuda_libdir "${CUDA_cudart_LIBRARY}" DIRECTORY)
        string(APPEND CHASE_LINK_LIBRARIES " -L${_chase_cuda_libdir}")
    endif()
    string(APPEND CHASE_LINK_LIBRARIES
           " -lchase_cuda_kernels -lcudart -lcublas -lcusolver -lcurand")

    if(IS_ENABLE_NCCL)
        if(NCCL_LIBRARIES)
            list(GET NCCL_LIBRARIES 0 _chase_nccl_library)
            get_filename_component(_chase_nccl_libdir "${_chase_nccl_library}" DIRECTORY)
            string(APPEND CHASE_LINK_LIBRARIES " -L${_chase_nccl_libdir}")
        endif()
        string(APPEND CHASE_LINK_LIBRARIES " -lnccl")
    endif()
endif()

if(TARGET chase_hip_kernels)
    if(CMAKE_HIP_COMPILER_ROCM_ROOT)
        set(_chase_rocm_root "${CMAKE_HIP_COMPILER_ROCM_ROOT}")
    elseif(DEFINED ENV{ROCM_PATH})
        set(_chase_rocm_root "$ENV{ROCM_PATH}")
    elseif(CMAKE_HIP_COMPILER)
        get_filename_component(_chase_rocm_bindir "${CMAKE_HIP_COMPILER}" DIRECTORY)
        get_filename_component(_chase_rocm_root "${_chase_rocm_bindir}" DIRECTORY)
    endif()

    if(EXISTS "${_chase_rocm_root}/lib")
        string(APPEND CHASE_LINK_LIBRARIES " -L${_chase_rocm_root}/lib")
    elseif(EXISTS "${_chase_rocm_root}/lib64")
        string(APPEND CHASE_LINK_LIBRARIES " -L${_chase_rocm_root}/lib64")
    endif()
    string(APPEND CHASE_LINK_LIBRARIES
           " -lchase_hip_kernels -lamdhip64 -lhiprand -lrocsolver -lrocblas")

    if(IS_ENABLE_RCCL)
        if(RCCL_LIBRARIES)
            list(GET RCCL_LIBRARIES 0 _chase_rccl_library)
            get_filename_component(_chase_rccl_libdir "${_chase_rccl_library}" DIRECTORY)
            string(APPEND CHASE_LINK_LIBRARIES " -L${_chase_rccl_libdir}")
        endif()
        string(APPEND CHASE_LINK_LIBRARIES " -lrccl")
    endif()
endif()

# CPU and distributed dependencies are required by both CPU-only and GPU
# interfaces because chase_c also contains the parallel entry points.
if(IS_ENABLE_SCALAPACK)
    _chase_pc_append_link_items(${CHASE_PC_SCALAPACK_LIBRARIES})
endif()
_chase_pc_append_link_items(${CHASE_PC_LAPACK_LIBRARIES})
_chase_pc_append_link_items(${CHASE_PC_BLAS_LIBRARIES})

if(IS_ENABLE_MPI)
    _chase_pc_append_link_items("${CHASE_PC_MPI_LINK_FLAGS}")
    _chase_pc_append_link_items(${CHASE_PC_MPI_LIBRARIES})
endif()

if(CHASE_ENABLE_OPENMP)
    # -fopenmp is enough for the compiler driver. OpenMP_CXX_LIBRARIES also
    # pulls in system archives such as libpthread.a, which do not belong in
    # the pkg-config closure.
    _chase_pc_append_link_items("${OpenMP_CXX_FLAGS}")
endif()

if(CHASE_USE_NVTX)
    string(APPEND CHASE_LINK_LIBRARIES " -lnvToolsExt")
endif()

# ChASE C interface is implemented in C++, while common consumers link with
# a Fortran compiler driver. Take only the C++ standard library selected by
# this compiler; the rest of CMAKE_CXX_IMPLICIT_LINK_LIBRARIES (libgcc, libc)
# is already supplied by the compiler driver.
set(_chase_pc_cxx_runtime "")
foreach(_chase_pc_implib IN LISTS CMAKE_CXX_IMPLICIT_LINK_LIBRARIES)
    if(_chase_pc_implib STREQUAL "c++" OR _chase_pc_implib STREQUAL "stdc++")
        set(_chase_pc_cxx_runtime "-l${_chase_pc_implib}")
        break()
    endif()
endforeach()
if(_chase_pc_cxx_runtime STREQUAL "")
    if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
        set(_chase_pc_cxx_runtime "-lc++")
    else()
        set(_chase_pc_cxx_runtime "-lstdc++")
    endif()
endif()
string(APPEND CHASE_LINK_LIBRARIES " ${_chase_pc_cxx_runtime}")

# CMake dependency lists commonly repeat BLAS, LAPACK, and MPI through
# ScaLAPACK. Join -Wl,-rpath with its path before deduplicating so a repeated
# -Wl,-rpath flag cannot be removed from in front of a different directory.
separate_arguments(_chase_pc_link_items UNIX_COMMAND "${CHASE_LINK_LIBRARIES}")
set(_chase_pc_grouped "")
set(_chase_pc_rpath_pending OFF)
foreach(_chase_pc_item IN LISTS _chase_pc_link_items)
    if(_chase_pc_rpath_pending)
        if(_chase_pc_item MATCHES "^-Wl,(.+)")
            list(APPEND _chase_pc_grouped "-Wl,-rpath,${CMAKE_MATCH_1}")
        else()
            list(APPEND _chase_pc_grouped "-Wl,-rpath,${_chase_pc_item}")
        endif()
        set(_chase_pc_rpath_pending OFF)
    elseif(_chase_pc_item STREQUAL "-Wl,-rpath")
        set(_chase_pc_rpath_pending ON)
    else()
        list(APPEND _chase_pc_grouped "${_chase_pc_item}")
    endif()
endforeach()
if(_chase_pc_rpath_pending)
    list(APPEND _chase_pc_grouped "-Wl,-rpath")
endif()
list(REMOVE_DUPLICATES _chase_pc_grouped)
list(JOIN _chase_pc_grouped " " CHASE_LINK_LIBRARIES)

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
set(IS_ENABLE_RCCL_STRING ${IS_ENABLE_RCCL})
set(IS_ENABLE_HIP_STRING ${IS_ENABLE_HIP})

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

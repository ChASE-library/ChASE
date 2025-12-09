# -*- Mode: cmake -*-
cmake_minimum_required( VERSION 3.18 )

project( ChASE LANGUAGES C CXX Fortran VERSION 1.6.1 )
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")
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
option(CHASE_BUILD_WITH_EXAMPLES "Build the examples" OFF)
option(CHASE_BUILD_WITH_DOCS "Build the docs" OFF)
option(CHASE_SAVE_RESIDUALS "Print the residuals in an external file" OFF)
option(CHASE_ENABLE_XGEEV_RAYLEIGHRITZ "Enable solving pseudo-hermitian problems with XGEEV cusolver" OFF)
option(ChASE_DISPLAY_COND_V_SVD "Compute and display condition number of V from SVD" OFF)
option(CHASE_QR_DOUBLE_PRECISION "Operate QR in Double Precision" ON)
option(CHASE_RR_DOUBLE_PRECISION "Operate HEEVD in RR for pseudo-hermitian matrices in Double Precision" OFF)

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

if(CHASE_QR_DOUBLE_PRECISION)
  add_definitions(-DQR_DOUBLE_PRECISION)
endif()

if(CHASE_RR_DOUBLE_PRECISION)
  add_definitions(-DRR_DOUBLE_PRECISION)
endif()

if(ChASE_DISPLAY_COND_V_SVD)
    add_definitions(-DChASE_DISPLAY_COND_V_SVD)
endif()

if(CHASE_SAVE_RESIDUALS)
	add_definitions(-DCHASE_SAVE_RESIDUALS)
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

#add_executable( "herm_chase_run_cpu" tests/dist_quasi_noinput.cpp )
#target_link_libraries("herm_chase_run_cpu" chase_cpu)

#add_executable( "quasi_chase_run_cpu" tests/dist_quasi_noinput.cpp )
#target_link_libraries("quasi_chase_run_cpu" chase_cpu)
#target_compile_definitions( "quasi_chase_run_cpu" PRIVATE USE_QUASI_HERMITIAN=1)

#if(TARGET chase_gpu)
	#add_executable("herm_chase_run_gpu" tests/dist_quasi_noinput.cpp)
#target_link_libraries("herm_chase_run_gpu" chase_gpu)
#target_compile_definitions("herm_chase_run_gpu" INTERFACE "-DHAS_CUDA")

#add_executable("quasi_chase_run_gpu" tests/dist_quasi_noinput.cpp)
#target_link_libraries("quasi_chase_run_gpu" chase_gpu)
#target_compile_definitions("quasi_chase_run_gpu" PRIVATE USE_QUASI_HERMITIAN=1)
#target_compile_definitions("quasi_chase_run_gpu" INTERFACE "-DHAS_CUDA")
#endif()

#if(TARGET pchase_cpu)
	#add_executable("herm_block_dist_chase_run_cpu" tests/dist_quasi_noinput.cpp)
#target_link_libraries("herm_block_dist_chase_run_cpu" pchase_cpu)
#target_compile_definitions("herm_block_dist_chase_run_cpu" PRIVATE USE_MPI=1)

#add_executable("quasi_block_dist_chase_run_cpu" tests/dist_quasi_noinput.cpp)
#target_link_libraries("quasi_block_dist_chase_run_cpu" pchase_cpu)
#target_compile_definitions("quasi_block_dist_chase_run_cpu" PRIVATE USE_QUASI_HERMITIAN=1)
#target_compile_definitions("quasi_block_dist_chase_run_cpu" PRIVATE USE_MPI=1)

#add_executable("herm_cyclic_dist_chase_run_cpu" tests/dist_quasi_noinput.cpp)
#target_link_libraries("herm_cyclic_dist_chase_run_cpu" pchase_cpu)
#target_compile_definitions("herm_cyclic_dist_chase_run_cpu" PRIVATE USE_BLOCKCYCLIC=1)
#target_compile_definitions("herm_cyclic_dist_chase_run_cpu" PRIVATE USE_MPI=1)

#add_executable("quasi_cyclic_dist_chase_run_cpu" tests/dist_quasi_noinput.cpp )
#target_link_libraries("quasi_cyclic_dist_chase_run_cpu" pchase_cpu)
#target_compile_definitions("quasi_cyclic_dist_chase_run_cpu" PRIVATE USE_BLOCKCYCLIC=1)
#target_compile_definitions("quasi_cyclic_dist_chase_run_cpu" PRIVATE USE_QUASI_HERMITIAN=1)
#target_compile_definitions("quasi_cyclic_dist_chase_run_cpu" PRIVATE USE_MPI=1)
#endif()

#if(TARGET pchase_gpu)
	#add_executable("herm_block_dist_chase_run_gpu" tests/dist_quasi_noinput.cpp)
#target_link_libraries("herm_block_dist_chase_run_gpu" pchase_gpu)
#target_compile_definitions("herm_block_dist_chase_run_gpu" INTERFACE "-DHAS_NCCL -DHAS_CUDA")

#add_executable("quasi_block_dist_chase_run_gpu" tests/dist_quasi_noinput.cpp)
#target_link_libraries("quasi_block_dist_chase_run_gpu"  pchase_gpu)
#target_compile_definitions("quasi_block_dist_chase_run_gpu" PRIVATE USE_QUASI_HERMITIAN=1)
#target_compile_definitions("quasi_block_dist_chase_run_gpu" INTERFACE "-DHAS_NCCL -DHAS_CUDA")

#add_executable("herm_cyclic_dist_chase_run_gpu" tests/dist_quasi_noinput.cpp)
#target_link_libraries("herm_cyclic_dist_chase_run_gpu" pchase_gpu)
#target_compile_definitions("herm_cyclic_dist_chase_run_gpu" PRIVATE USE_BLOCKCYCLIC=1)
#target_compile_definitions("herm_cyclic_dist_chase_run_gpu" INTERFACE "-DHAS_NCCL -DHAS_CUDA")

#add_executable("quasi_cyclic_dist_chase_run_gpu" tests/dist_quasi_noinput.cpp)
#target_link_libraries("quasi_cyclic_dist_chase_run_gpu" pchase_gpu)
#target_compile_definitions("quasi_cyclic_dist_chase_run_gpu" PRIVATE USE_BLOCKCYCLIC=1)
#target_compile_definitions("quasi_cyclic_dist_chase_run_gpu" PRIVATE USE_QUASI_HERMITIAN=1)
#target_compile_definitions("quasi_cyclic_dist_chase_run_gpu" INTERFACE "-DHAS_NCCL" "-DHAS_CUDA")
#endif()

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

#get_target_property(interface_defs mpi_grid_nccl INTERFACE_COMPILE_DEFINITIONS)
#message("INTERFACE_COMPILE_DEFINITIONS: ${interface_defs}")

# List all dependencies (linked targets)
#get_target_property(linked_libraries pchase_gpu INTERFACE_LINK_LIBRARIES)
#message("linked_libraries: ${linked_libraries}")
# Prepare a list to store definitions
#set(all_inherited_defs "")

#foreach(dep ${linked_libraries})
    # Check if the dependency is a valid target
#    if(TARGET ${dep})
        # Get INTERFACE_COMPILE_DEFINITIONS from the dependency
#        get_target_property(dep_defs ${dep} INTERFACE_COMPILE_DEFINITIONS)
        
        # Append to the list
#        if(dep_defs)
#            list(APPEND all_inherited_defs ${dep_defs})
#        endif()
#    endif()
#endforeach()

# Remove duplicates (optional)
#list(REMOVE_DUPLICATES all_inherited_defs)

# Print all inherited definitions
#message("Inherited compile definitions: ${all_inherited_defs}")
#get_target_property(linked_libraries blaspp INTERFACE_LINK_LIBRARIES)
#message("Libraries linked by blaspp: ${linked_libraries}")
#get_target_property(blas_library ${linked_libraries} IMPORTED_LOCATION)
#message("BLAS library path: ${blas_library}")


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

install(EXPORT chase_targets
        NAMESPACE ChASE::
        EXPORT_LINK_INTERFACE_LIBRARIES
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
        FILE chaseTargets.cmake)

include(CMakePackageConfigHelpers)

set(IS_ENABLE_MPI OFF)
set(IS_ENABLE_SCALAPACK OFF)
set(IS_ENABLE_NCCL OFF)

if(TARGET pchase_cpu)
    set(IS_ENABLE_MPI ON)
endif()

if(TARGET scalapackpp)
    set(IS_ENABLE_SCALAPACK ON)
endif()

if(TARGET linalg_internal_nccl)
    set(IS_ENABLE_NCCL ON)
endif()

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

install(DIRECTORY ${CMAKE_SOURCE_DIR}/cmake/modules
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ChASE)

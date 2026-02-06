option(CHASE_INTERFACE_WITH_BLOCKCYLIC "Build the C/FORTRAN interface for Block-Cyclic data layout" ON)

add_library( "chase_c" "chase_c_interface.cpp" )

# Link to appropriate backend libraries and set configuration macros
if(TARGET chase_gpu)
    target_link_libraries("chase_c" PUBLIC chase_gpu)
    target_compile_definitions(chase_c PRIVATE HAS_CUDA=1)
else()
    target_link_libraries("chase_c" PUBLIC chase_cpu)    
endif()

if(TARGET pchase_gpu)
    target_link_libraries("chase_c" PUBLIC pchase_gpu)
    # NCCL detection: check if pchase_gpu has NCCL backend
    if(TARGET NCCL::NCCL)
        target_compile_definitions(chase_c PRIVATE HAS_NCCL=1)
    endif()
else()
    target_link_libraries("chase_c" PUBLIC pchase_cpu)
endif()

# MPI is always available for chase_c (required by pchase_* targets)
if(TARGET MPI::MPI_CXX)
    target_compile_definitions(chase_c PRIVATE HAS_MPI=1)
endif()

# ScaLAPACK detection (if available)
if(ScaLAPACK_FOUND OR TARGET scalapack)
    target_compile_definitions(chase_c PRIVATE HAS_SCALAPACK=1)
endif()

# Add dependency macros from chase_algorithm
# These propagate the build options (GPU_RESIDENT_LANCZOS, QR_DOUBLE_PRECISION, etc.)
target_link_libraries("chase_c" PRIVATE chase_algorithm)
# Note: NO_MPI is not defined here since chase_c always links to parallel targets (pchase_*)
# which require MPI. Serial targets (chase_*) are linked for implementation but don't define NO_MPI
# to avoid conflicts with parallel targets.

target_include_directories("chase_c" 
    INTERFACE 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/interface>"
    PRIVATE
        # For building chase_c itself (needs chase_config.h)
        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>  # For generated chase_config.h
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>  # For other includes
)

if(CHASE_INTERFACE_WITH_BLOCKCYLIC)
  target_compile_definitions(chase_c PUBLIC INTERFACE_BLOCK_CYCLIC)
endif()

install(TARGETS chase_c
  EXPORT chase_targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING
    PATTERN "*.h"
)

#set(CMAKE_Fortran_COMPILER nvfortran)
#enable_language( Fortran )

set(
  MODULE_OUTPUT_DIR
  "${CMAKE_CURRENT_BINARY_DIR}/Mod"
)

add_library("chase_f" "chase_fortran_interface.f90")

set_target_properties(
  "chase_f"
  PROPERTIES Fortran_PREPROCESS ON
  POSITION_INDEPENDENT_CODE TRUE
  Fortran_MODULE_DIRECTORY "${MODULE_OUTPUT_DIR}"
)

target_include_directories(
  "chase_f"
  INTERFACE
  $<BUILD_INTERFACE:${MODULE_OUTPUT_DIR}>
  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"  
)

target_link_libraries("chase_f" PUBLIC chase_c)
if(CHASE_INTERFACE_WITH_BLOCKCYLIC)
  target_compile_definitions(chase_f PUBLIC INTERFACE_BLOCK_CYCLIC)
endif()

install( TARGETS chase_f
  EXPORT chase_targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )

install(
  DIRECTORY
  "${MODULE_OUTPUT_DIR}/"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)


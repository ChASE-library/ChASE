add_library( "chase_c_interface" "chase_c_interface.cpp" )
if(TARGET chase_gpu)
    add_compile_definitions(HAS_CUDA=1)
    target_link_libraries("chase_c_interface" PUBLIC chase_gpu)
else()
    target_link_libraries("chase_c_interface" PUBLIC chase_cpu)    
endif()

if(TARGET chase_nccl)
    add_compile_definitions(HAS_NCCL=1)
    target_link_libraries("chase_c_interface" PUBLIC chase_nccl)
else()
    target_link_libraries("chase_c_interface" PUBLIC chase_mpi)
endif()

target_include_directories("chase_c_interface" PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

enable_language( Fortran )

set(
  MODULE_OUTPUT_DIR
  "${CMAKE_CURRENT_BINARY_DIR}/Mod"
)

add_library("chase_fortran_interface" "chase_fortran_interface.f90")

set_target_properties(
  "chase_fortran_interface"
  PROPERTIES Fortran_PREPROCESS ON
  POSITION_INDEPENDENT_CODE TRUE
  Fortran_MODULE_DIRECTORY "${MODULE_OUTPUT_DIR}"
)

target_include_directories(
  "chase_fortran_interface"
  INTERFACE
  $<BUILD_INTERFACE:${MODULE_OUTPUT_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries("chase_fortran_interface" PUBLIC chase_c_interface)


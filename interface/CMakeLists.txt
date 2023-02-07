add_library( "chase_c" "chase_c.cpp" )

if(TARGET chase_cuda)
    add_compile_definitions(HAS_GPU=1)
endif()

if(TARGET chase_cuda)
  enable_language(CUDA)
  target_link_libraries("chase_c" PUBLIC chase_mpi chase_cuda)
else()
  target_link_libraries("chase_c" PUBLIC chase_mpi)
endif()

install( TARGETS chase_c
  EXPORT chase-c
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )

install(EXPORT chase-c
  NAMESPACE ChASE::
  FILE chase-c.cmake
  EXPORT_LINK_INTERFACE_LIBRARIES
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
  )

enable_language( Fortran )
set(
  MODULE_OUTPUT_DIR
  "${CMAKE_CURRENT_BINARY_DIR}/Mod"
)

add_library("chase_f" "chase_f.f90")
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
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries("chase_f" PUBLIC chase_c)

install( TARGETS chase_f
  EXPORT chase-f
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )

install(
  DIRECTORY
  "${MODULE_OUTPUT_DIR}/"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)

install(EXPORT chase-f
  NAMESPACE ChASE::
  FILE chase-f.cmake
  EXPORT_LINK_INTERFACE_LIBRARIES
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
  )


add_library( "chase_c" "chase_c.cpp" )
message("Use distributed multi-GPU version of ChASE")
target_link_libraries("chase_c" PUBLIC chase_mpi)

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



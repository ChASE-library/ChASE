# -*- Mode: cmake -*-

# for the moment ChASE only supports Elemental v.84
# which does not come with CMake support
if(EXISTS "$ENV{ELEMENTAL_ROOT}/conf/ElemVars")

  add_library( chase_elemental INTERFACE )
  target_link_libraries(chase_elemental INTERFACE chase_algorithm)

  target_include_directories( chase_elemental INTERFACE
    "$ENV{ELEMENTAL_ROOT}/include/"
    )

  target_link_libraries( chase_elemental INTERFACE
    "-L/homeb/slai/slai10/elemental/lib -lelemental -lpmrrr -lpthread"
    )

  find_package( BLAS   REQUIRED )
  find_package( LAPACK REQUIRED )
  find_package( MPI    REQUIRED )


  target_include_directories( chase_elemental INTERFACE
    ${MPI_CXX_INCLUDE_PATH}
    )

  target_link_libraries( chase_elemental INTERFACE
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
    ${MPI_CXX_LIBRARIES}
    )

  target_compile_options( chase_elemental INTERFACE
    ${CMAKE_C_FLAGS}
    ${MPI_CXX_COMPILE_FLAGS}
    )



else()
  message("Can not find Elemental")
endif()

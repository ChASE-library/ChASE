include(${CMAKE_SOURCE_DIR}/cmake/external/popl/Fetchpopl.cmake)

if(POPL_FOUND)
  add_executable( "2_input_seq.exe" "2_input.cpp")
  target_link_libraries( "2_input_seq.exe" PRIVATE chase_cpu )
  target_include_directories( "2_input_seq.exe" PRIVATE ${POPL_INCLUDE_DIR})
  target_compile_definitions( "2_input_seq.exe" PRIVATE PRINT_EIGENVALUES=1)

  add_executable( "2_input.exe" "2_input.cpp")
  target_link_libraries( "2_input.exe" PRIVATE chase_mpi )
  target_include_directories( "2_input.exe" PRIVATE ${POPL_INCLUDE_DIR})
  target_compile_definitions( "2_input.exe" PRIVATE PRINT_EIGENVALUES=1 USE_MPI=1)

else()
  message( "popl not found, not building example 2" )
endif()

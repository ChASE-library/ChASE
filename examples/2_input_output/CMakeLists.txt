include(${CMAKE_SOURCE_DIR}/cmake/external/popl/Fetchpopl.cmake)

if(POPL_FOUND)
  add_executable( "2_input_output_seq" "2_input_output.cpp")
  target_link_libraries( "2_input_output_seq" PRIVATE chase_cpu )
  target_include_directories( "2_input_output_seq" PRIVATE ${POPL_INCLUDE_DIR})
  target_compile_definitions( "2_input_output_seq" PRIVATE PRINT_EIGENVALUES=1)

  if(TARGET chase_gpu)
    add_executable( "2_input_output_seq_gpu" "2_input_output.cpp")
    target_link_libraries( "2_input_output_seq_gpu" PRIVATE chase_cpu chase_gpu)
    target_include_directories( "2_input_output_seq_gpu" PRIVATE ${POPL_INCLUDE_DIR})
    target_compile_definitions( "2_input_output_seq_gpu" PRIVATE PRINT_EIGENVALUES=1)
  endif()

  if(TARGET chase_mpi)  
    add_executable( "2_input_output" "2_input_output.cpp")
    target_link_libraries( "2_input_output" PRIVATE chase_mpi )
    target_include_directories( "2_input_output" PRIVATE ${POPL_INCLUDE_DIR})
    target_compile_definitions( "2_input_output" PRIVATE PRINT_EIGENVALUES=1 USE_MPI=1)

    add_executable( "2_input_output_blockcyclic" "2_input_output.cpp")
    target_link_libraries( "2_input_output_blockcyclic" PRIVATE chase_mpi )
    target_include_directories( "2_input_output_blockcyclic" PRIVATE ${POPL_INCLUDE_DIR})
    target_compile_definitions( "2_input_output_blockcyclic" PRIVATE PRINT_EIGENVALUES=1 USE_MPI=1 USE_BLOCKCYCLIC=1)

  endif()

  if(TARGET chase_nccl)  
    add_executable( "2_input_output_mgpu" "2_input_output.cpp")
    target_link_libraries( "2_input_output_mgpu" PRIVATE chase_nccl )
    target_include_directories( "2_input_output_mgpu" PRIVATE ${POPL_INCLUDE_DIR})
    target_compile_definitions( "2_input_output_mgpu" PRIVATE PRINT_EIGENVALUES=1)

    add_executable( "2_input_output_blockcyclic_mgpu" "2_input_output.cpp")
    target_link_libraries( "2_input_output_blockcyclic_mgpu" PRIVATE chase_nccl )
    target_include_directories( "2_input_output_blockcyclic_mgpu" PRIVATE ${POPL_INCLUDE_DIR})
    target_compile_definitions( "2_input_output_blockcyclic_mgpu" PRIVATE PRINT_EIGENVALUES=1 USE_BLOCKCYCLIC=1)

  endif()

else()
  message( "popl not found, not building example 2" )
endif()

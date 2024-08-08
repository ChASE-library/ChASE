include(${CMAKE_SOURCE_DIR}/cmake/external/popl/Fetchpopl.cmake)

if(POPL_FOUND)
  ##############################################################################
  #               2_input_output_seq: no GPU, $A$ not distributed
  ##############################################################################

  add_executable( "2_input_output_seq"
    "2_input_output.cpp"
    )
  target_link_libraries( "2_input_output_seq" PRIVATE chase_seq )
  target_include_directories( "2_input_output_seq" PRIVATE ${POPL_INCLUDE_DIR})
  target_compile_definitions( "2_input_output_seq" PRIVATE PRINT_EIGENVALUES=1)

  ##############################################################################
  #           2_input_output: no GPU, MPI ($A$ distributed among MPI-ranks)
  ##############################################################################

  add_executable( "2_input_output" "2_input_output.cpp" )
  target_link_libraries( "2_input_output" chase_mpi ${CUDA_nvToolsExt_LIBRARY})
  target_include_directories( "2_input_output" PRIVATE ${POPL_INCLUDE_DIR})
  target_compile_definitions( "2_input_output" PRIVATE USE_MPI=1 PRINT_EIGENVALUES=1)

  add_executable( "2_input_output_block_cyclic" "2_input_output.cpp" )
  target_link_libraries( "2_input_output_block_cyclic" chase_mpi )
  target_include_directories( "2_input_output_block_cyclic" PRIVATE ${POPL_INCLUDE_DIR})
  target_compile_definitions( "2_input_output_block_cyclic" PRIVATE USE_MPI=1 USE_BLOCK_CYCLIC=1 PRINT_EIGENVALUES=1)  

  install (TARGETS 2_input_output_seq
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})  
  install (TARGETS 2_input_output
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
  install (TARGETS 2_input_output_block_cyclic
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

  ##############################################################################
  #   2_input_output_mgpu: multi-GPU/node, MPI ($A$ distributed among MPI-ranks)
  ##############################################################################
  
  if(TARGET chase_cuda )
    enable_language(CUDA)
  
    if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      set (CMAKE_CUDA_ARCHITECTURES "80")
    endif()

    if (${CMAKE_VERSION} VERSION_LESS "3.18.0")
      foreach(CODE ${CMAKE_CUDA_ARCHITECTURES})
        set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${CODE},code=sm_${CODE}")
      endforeach()
    endif()

    add_executable( 2_input_output_mgpu "2_input_output.cpp" )
    target_link_libraries( 2_input_output_mgpu chase_mpi chase_cuda ${CUDA_nvToolsExt_LIBRARY})
    target_include_directories( 2_input_output_mgpu PRIVATE ${POPL_INCLUDE_DIR})
    target_compile_definitions( "2_input_output_mgpu" PRIVATE DRIVER_BUILD_MGPU=1 PRINT_EIGENVALUES=1 USE_MPI=1)

    add_executable( 2_input_output_mgpu_block_cyclic "2_input_output.cpp" )
    target_link_libraries( 2_input_output_mgpu_block_cyclic chase_mpi chase_cuda )
    target_include_directories( 2_input_output_mgpu_block_cyclic PRIVATE ${POPL_INCLUDE_DIR})
    target_compile_definitions( "2_input_output_mgpu_block_cyclic" PRIVATE DRIVER_BUILD_MGPU=1 USE_MPI=1 USE_BLOCK_CYCLIC=1 PRINT_EIGENVALUES=1)

    install (TARGETS 2_input_output_mgpu
             RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    install (TARGETS 2_input_output_mgpu_block_cyclic
             RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
  endif()  

else()
  message( "popl not found, not building example 2" )
endif()


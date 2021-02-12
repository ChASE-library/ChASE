find_package( Boost COMPONENTS serialization program_options )

if( ${Boost_FOUND} )

  ##############################################################################
  #                         chase_mpi_driver interface
  ##############################################################################

  add_library( chase_seq_driver INTERFACE )

  target_link_libraries(chase_seq_driver INTERFACE
    chase_seq
    ${Boost_LIBRARIES}
    )
  target_include_directories(chase_seq_driver INTERFACE
    ${Boost_INCLUDE_DIRS}
    )
  target_include_directories( chase_seq_driver INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>"
    )

  add_library( chase_mpi_driver INTERFACE )

  target_link_libraries(chase_mpi_driver INTERFACE
    chase_mpi
    ${Boost_LIBRARIES}
    )
  target_include_directories(chase_mpi_driver INTERFACE
    ${Boost_INCLUDE_DIRS}
    )
  target_include_directories( chase_mpi_driver INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>"
    )

  ##############################################################################
  #               2_input_output_seq: no GPU, $A$ not distributed
  ##############################################################################

  add_executable( "2_input_output_seq"
    "2_input_output.cpp"
    )
  target_link_libraries( "2_input_output_seq" PRIVATE chase_seq_driver )

  ##############################################################################
  #           2_input_output: no GPU, MPI ($A$ distributed among MPI-ranks)
  ##############################################################################

  add_executable( "2_input_output" "2_input_output.cpp" )
  target_link_libraries( "2_input_output" chase_mpi_driver )
  target_compile_definitions( "2_input_output" PRIVATE USE_MPI=1)

  add_executable( "2_input_output_block_cyclic" "2_input_output.cpp" )
  target_link_libraries( "2_input_output_block_cyclic" chase_mpi_driver )
  target_compile_definitions( "2_input_output_block_cyclic" PRIVATE USE_MPI=1 USE_BLOCK_CYCLIC=1)

  ##############################################################################
  #   2_input_output_mgpu: multi-GPU/node, MPI ($A$ distributed among MPI-ranks)
  ##############################################################################

  if(TARGET chase_cuda )
    enable_language(CUDA)

    option(TIMER "Builder timer routines" OFF)

    add_executable( 2_input_output_mgpu "2_input_output.cpp" )
    target_link_libraries( 2_input_output_mgpu chase_mpi_driver chase_cuda)

    add_executable( 2_input_output_mgpu_block_cyclic "2_input_output.cpp" )
    target_link_libraries( 2_input_output_mgpu_block_cyclic chase_mpi_driver chase_cuda)

    if(TIMER) 
      target_compile_definitions( "2_input_output_mgpu" PRIVATE DRIVER_BUILD_MGPU=1 USE_MPI=1 USE_TIMER=1)
      target_compile_definitions( "2_input_output_mgpu_block_cyclic" PRIVATE DRIVER_BUILD_MGPU=1 USE_MPI=1 USE_TIMER=1 USE_BLOCK_CYCLIC=1)
    else()
      target_compile_definitions( "2_input_output_mgpu" PRIVATE DRIVER_BUILD_MGPU=1 USE_MPI=1 USE_TIMER=0)
      target_compile_definitions( "2_input_output_mgpu_block_cyclic" PRIVATE DRIVER_BUILD_MGPU=1 USE_MPI=1 USE_BLOCK_CYCLIC=1 USE_TIMER=0)      
    endif()
  endif()
else()
  message( "Boost not found, not building this example" )
endif()


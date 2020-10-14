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
  target_compile_definitions( "2_input_output" PRIVATE USE_MPI=1 )

  if(TARGET chase_cuda )
    enable_language(CUDA)

    option(TIMER "Builder timer routines" OFF)

  ##############################################################################
  #   2_input_output_mgpu: multi-GPU/node, MPI ($A$ distributed among MPI-ranks)
  ##############################################################################

    add_executable( 2_input_output_mgpu "2_input_output.cpp" )
    target_link_libraries( 2_input_output_mgpu
      chase_mpi_driver
      chase_cuda
#      ${CUDA_CUBLAS_LIBRARIES}
#      ${CUDA_LIBRARIES}
      )
    if(TIMER) 
      target_compile_definitions( "2_input_output_mgpu" PRIVATE DRIVER_BUILD_MGPU=1 USE_MPI=1 USE_TIMER=1)
    else()
      target_compile_definitions( "2_input_output_mgpu" PRIVATE DRIVER_BUILD_MGPU=1 USE_MPI=1 )
    endif()
  endif()
else()
  message( "Boost not found, not building this example" )
endif()


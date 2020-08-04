# -*- Mode: cmake -*-

find_package( Boost COMPONENTS serialization program_options )
if( ${Boost_FOUND} )

  ##############################################################################
  #                         chase_mpi_driver interface
  ##############################################################################
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
  #                simple_seq_driver: no GPU, $A$ not distributed
  ##############################################################################
  add_executable( "simple_seq_driver"
    "simple_driver.cpp"
    )
  target_link_libraries( "simple_seq_driver" PRIVATE chase_mpi_driver )

  ##############################################################################
  #                simple_driver: no GPU, MPI ($A$ distributed among MPI-ranks)
  ##############################################################################
  add_executable( "simple_driver" "simple_driver.cpp" )
  target_link_libraries( "simple_driver" chase_mpi_driver )
  target_compile_definitions( "simple_driver" PRIVATE USE_MPI=1 )

  # add_executable( "gpu_seq_driver" "simple_driver.cpp" )
  # target_link_libraries( "gpu_seq_driver"
  #   chase_mpi_driver
  #   chase_mpi_cuda
  #   ${CUDA_CUBLAS_LIBRARIES}
  #   ${CUDA_LIBRARIES}
  #  )
  # target_compile_definitions( "gpu_seq_driver" PRIVATE DRIVER_BUILD_CUDA=1 )

  if(TARGET chase_mpi_cuda )
    enable_language(CUDA)

    option(TIMER "Builder timer routines" OFF)

    add_executable( gpu_driver "simple_driver.cpp" )
    target_link_libraries( gpu_driver
      chase_mpi_driver
      chase_mpi_cuda
      ${CUDA_CUBLAS_LIBRARIES}
      ${CUDA_LIBRARIES}
      )
    if(TIMER)
      target_compile_definitions( "gpu_driver" PRIVATE DRIVER_BUILD_CUDA=1 USE_MPI=1 USE_TIMER=1)
	else()
      target_compile_definitions( "gpu_driver" PRIVATE DRIVER_BUILD_CUDA=1 USE_MPI=1 )
	endif()

    add_executable( mgpu_driver "simple_driver.cpp" )
    target_link_libraries( mgpu_driver
      chase_mpi_driver
      chase_mpi_cuda
      ${CUDA_CUBLAS_LIBRARIES}
      ${CUDA_LIBRARIES}
      )
    if(TIMER) 
      target_compile_definitions( "mgpu_driver" PRIVATE DRIVER_BUILD_MGPU=1 USE_MPI=1 USE_TIMER=1)
    else()
      target_compile_definitions( "mgpu_driver" PRIVATE DRIVER_BUILD_MGPU=1 USE_MPI=1 )
    endif()
  endif()

else()
  message( "Boost not found, not building ChASE-MPI driver" )
endif()

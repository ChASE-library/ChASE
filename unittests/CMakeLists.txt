# -*- Mode: cmake -*-

find_package( Boost COMPONENTS serialization program_options )
if( ${Boost_FOUND} )

  #add_library( chase_mpi_driver INTERFACE )
  #target_link_libraries(chase_mpi_driver INTERFACE
  #  chase_mpi
  #  ${Boost_LIBRARIES}
  #  )
  #target_include_directories(chase_mpi_driver INTERFACE
  #  ${Boost_INCLUDE_DIRS}
  #  )
  #target_include_directories( chase_mpi_driver INTERFACE
  #  "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>"
  #  )
  ##############################################################################
  #                test mgpu Hemm driver: Test for multi-gpu hemm class
  ##############################################################################
  if(TARGET chase_mpi_cuda )
    enable_language(CUDA)
    add_executable( test_mgpu_Hemm "test_mgpu_Hemm.cpp" )
    target_link_libraries( test_mgpu_Hemm
	  chase_mpi_driver
      chase_mpi_cuda
      ${CUDA_CUBLAS_LIBRARIES}
      ${CUDA_LIBRARIES}
      )
  	#target_include_directories(test_mgpu_Hemm INTERFACE
    #	${Boost_INCLUDE_DIRS}
    #)
  	#target_include_directories(test_mgpu_Hemm INTERFACE
	#	"$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>"
    #)
    target_compile_definitions( "test_mgpu_Hemm" PRIVATE DRIVER_BUILD_CUDA=1 )
  endif()

else()
  message( "Boost not found, not building ChASE-MPI driver" )
endif()

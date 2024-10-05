add_library(chase_mpi INTERFACE)
target_link_libraries(chase_mpi 
                      INTERFACE 
                              chase_algorithm 
                              blaspp 
                              lapackpp 
                              linalg_internal_cpu 
                              linalg_internal_mpi 
                              mpi_grid 
                              distMatrix)
target_include_directories(chase_mpi INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(chase_mpi INTERFACE NO_MPI=0)

if(CHASE_ENABLE_MPI_IO)
    target_compile_definitions( chase_mpi INTERFACE
	    "-DUSE_MPI_IO"
	)
endif()

if(TARGET scalapackpp)
  message("ScaLAPACK is enabled in ChASE-MPI")
  target_link_libraries(chase_mpi INTERFACE scalapackpp)
  target_compile_definitions(chase_mpi INTERFACE HAS_SCALAPACK=1 )
endif()


add_library(chase_cuda_mpi INTERFACE)
target_link_libraries(chase_cuda_mpi 
                      INTERFACE 
                              chase_algorithm 
                              blaspp 
                              lapackpp 
                              cublaspp 
                              cusolverpp 
                              linalg_internal_cpu 
                              linalg_internal_mpi 
                              linalg_internal_cuda
                              mpi_grid
                              distMatrix_gpu)

target_include_directories(chase_cuda_mpi INTERFACE 
                                               $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                                               $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/Impl>
                                               "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/Impl/cuda_aware_mpi>"
                                               "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/config>")

if(CHASE_ENABLE_MPI_IO)
    target_compile_definitions( chase_cuda_mpi INTERFACE
	    "-DUSE_MPI_IO"
	)
endif()

if(TARGET scalapackpp)
  message("ScaLAPACK is enabled in ChASE-CUDA-MPI")
  target_link_libraries(chase_cuda_mpi INTERFACE scalapackpp)
  target_compile_definitions(chase_cuda_mpi INTERFACE HAS_SCALAPACK=1 )
endif()

install(TARGETS chase_cuda_mpi
  EXPORT chase_targets
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/Impl
  FILES_MATCHING
    PATTERN "*.hpp"
)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/Impl/config DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING
    PATTERN "*.hpp"
)
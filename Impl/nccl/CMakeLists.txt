add_library(chase_nccl INTERFACE)
target_link_libraries(chase_nccl 
                      INTERFACE 
                              chase_algorithm 
                              blaspp 
                              lapackpp 
                              cublaspp 
                              cusolverpp 
                              linalg_internal_cpu 
                              linalg_internal_mpi 
                              linalg_internal_cuda
                              linalg_internal_nccl                              
                              mpi_grid_nccl 
                              distMatrix_gpu)

target_include_directories(chase_nccl INTERFACE 
                                               $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                                               $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/Impl>
                                               "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/Impl/nccl>"
                                               "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/config>")

target_compile_definitions(chase_nccl INTERFACE NO_MPI=0)

if(CHASE_ENABLE_MPI_IO)
    target_compile_definitions( chase_nccl INTERFACE
	    "-DUSE_MPI_IO"
	)
endif()

if(TARGET scalapackpp)
  message("ScaLAPACK is enabled in ChASE-NCCL")
  target_link_libraries(chase_nccl INTERFACE scalapackpp)
  target_compile_definitions(chase_nccl INTERFACE HAS_SCALAPACK=1 )
endif()

install(TARGETS chase_nccl
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
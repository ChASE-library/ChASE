add_library(pchase_gpu INTERFACE)
target_link_libraries(pchase_gpu 
                      INTERFACE 
                              chase_algorithm 
                              blaspp 
                              lapackpp 
                              cublaspp 
                              cusolverpp 
                              linalg_internal_cpu 
                              linalg_internal_mpi 
                              linalg_internal_cuda
                              linalg_internal_cuda_mpi
                              linalg_internal_nccl                              
                              mpi_grid_nccl 
                              distMatrix_gpu)

if(TARGET mpi_grid_nccl)
target_link_libraries(pchase_gpu 
                      INTERFACE 
                              linalg_internal_nccl                              
                              mpi_grid_nccl)
endif()

target_include_directories(pchase_gpu INTERFACE 
                                               $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                                               $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/Impl>
                                               "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/Impl/pchase_gpu>"
                                               "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/config>")

# NO_MPI should not be defined for parallel targets - this allows MPI headers to be included
# Serial targets (chase_cpu/chase_gpu) define NO_MPI to exclude MPI headers

if(CHASE_ENABLE_MPI_IO)
    target_compile_definitions( pchase_gpu INTERFACE
	    "-DUSE_MPI_IO"
	)
endif()

if(TARGET scalapackpp)
  target_link_libraries(pchase_gpu INTERFACE scalapackpp)
  target_compile_definitions(pchase_gpu INTERFACE HAS_SCALAPACK=1 )
endif()

install(TARGETS pchase_gpu
  EXPORT chase_targets
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/Impl
  FILES_MATCHING
    PATTERN "*.hpp"
)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/Impl/config DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/Impl
  FILES_MATCHING
    PATTERN "*.hpp"
)
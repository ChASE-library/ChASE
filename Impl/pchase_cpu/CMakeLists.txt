add_library(pchase_cpu INTERFACE)
target_link_libraries(pchase_cpu 
                      INTERFACE 
                              chase_algorithm 
                              blaspp 
                              lapackpp 
                              linalg_internal_cpu 
                              linalg_internal_mpi 
                              mpi_grid 
                              distMatrix)

target_include_directories(pchase_cpu INTERFACE 
                                               $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                                               "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/Impl/pchase_cpu>")

# NO_MPI should not be defined for parallel targets - this allows MPI headers to be included
# Serial targets (chase_cpu/chase_gpu) define NO_MPI to exclude MPI headers

if(CHASE_ENABLE_MPI_IO)
    target_compile_definitions( pchase_cpu INTERFACE
	    "-DUSE_MPI_IO"
	)
endif()

if(TARGET scalapackpp)
  message("ScaLAPACK is enabled in ChASE-MPI")
  target_link_libraries(pchase_cpu INTERFACE scalapackpp)
  target_compile_definitions(pchase_cpu INTERFACE HAS_SCALAPACK=1 )
endif()

if(CHASE_ENABLE_XGEEV_RAYLEIGHRITZ)
  # For CPU, XGEEV uses LAPACK's zgeev/cgeev functions
  # These should be available in standard LAPACK implementations
  target_compile_definitions(pchase_cpu INTERFACE "-DXGEEV_EXISTS")
  message(STATUS "XGEEV support enabled for ChASE-MPI (CPU). Using LAPACK zgeev/cgeev for Pseudo-Hermitian problems.")
endif()

install(TARGETS pchase_cpu
  EXPORT chase_targets
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/Impl
  FILES_MATCHING
    PATTERN "*.hpp"
)

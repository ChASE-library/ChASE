add_library(chase_cpu INTERFACE)

# Link against linalg libraries
target_link_libraries(chase_cpu INTERFACE chase_algorithm blaspp linalg_internal_cpu)

list(APPEND includePath ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories( chase_cpu INTERFACE
  "$<BUILD_INTERFACE:${includePath}>"
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>  # <prefix>/include/ChASE-MPI
)

target_include_directories(chase_cpu INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(chase_cpu INTERFACE NO_MPI=1 )

find_package(MPI REQUIRED)
add_library(chase_mpi INTERFACE)
target_link_libraries(chase_mpi INTERFACE chase_algorithm blaspp linalg_internal_cpu)
target_include_directories( chase_mpi INTERFACE
  "$<BUILD_INTERFACE:${includePath}>"
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>  # <prefix>/include/ChASE-MPI
)

target_include_directories(chase_mpi INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(chase_mpi INTERFACE NO_MPI=0 )

if(CHASE_ENABLE_MIXED_PRECISION)
    target_compile_definitions( chase_mpi INTERFACE
	    "-DUSE_MPI_IO"
	)
endif()

target_link_libraries(chase_mpi INTERFACE MPI::MPI_CXX)
if(TARGET scalapackpp)
  message("ScaLAPACK is enabled in ChASE-MPI")
  target_link_libraries(chase_mpi INTERFACE scalapackpp)
  target_compile_definitions(chase_mpi INTERFACE HAS_SCALAPACK=1 )
endif()

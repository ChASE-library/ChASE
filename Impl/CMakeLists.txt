add_subdirectory(cpu)

if(TARGET linalg_internal_mpi)
add_subdirectory(grid)
add_subdirectory(mpi)
endif()

if(TARGET linalg_internal_cuda)
add_subdirectory(cuda)
endif()
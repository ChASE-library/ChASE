add_subdirectory(cpu)

if(TARGET linalg_internal_mpi)
add_subdirectory(mpi)
endif()

if(TARGET linalg_internal_cuda)
add_subdirectory(cuda)
endif()

if(TARGET linalg_internal_cuda_mpi)
add_subdirectory(cuda_aware_mpi)
endif()

if(TARGET mpi_grid_nccl)
add_subdirectory(nccl)
endif()
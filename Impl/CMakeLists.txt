add_subdirectory(chase_cpu)

if(TARGET linalg_internal_mpi)
add_subdirectory(pchase_cpu)
endif()

if(TARGET linalg_internal_cuda)
add_subdirectory(chase_gpu)
endif()

if(TARGET linalg_internal_cuda_mpi)
add_subdirectory(pchase_gpu)
endif()
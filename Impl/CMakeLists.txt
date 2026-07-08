add_subdirectory(chase_cpu)

if(TARGET linalg_internal_mpi)
add_subdirectory(pchase_cpu)
endif()

if(TARGET linalg_internal_cuda OR TARGET linalg_internal_hip)
add_subdirectory(chase_gpu)
endif()

if(TARGET linalg_internal_cuda_mpi OR TARGET linalg_internal_hip_mpi)
add_subdirectory(pchase_gpu)
endif()

add_subdirectory(cpu)
if(TARGET chase_mpi)
add_subdirectory(mpi)
endif()

if(TARGET chase_cuda)
add_subdirectory(cuda)
endif()

if(TARGET chase_nccl)
add_subdirectory(nccl)
endif()
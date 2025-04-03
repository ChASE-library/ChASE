file(COPY QR_matrices DESTINATION ${CMAKE_BINARY_DIR}/tests/linalg/internal)
file(COPY BSE_matrices DESTINATION ${CMAKE_BINARY_DIR}/tests/linalg/internal)

add_subdirectory(cpu)
if(TARGET pchase_cpu)
add_subdirectory(mpi)
endif()

if(TARGET chase_gpu)
add_subdirectory(cuda)
endif()

if(TARGET linalg_internal_cuda_mpi)
add_subdirectory(cuda_aware_mpi)
endif()	

if(TARGET linalg_internal_nccl)
add_subdirectory(nccl)
endif()

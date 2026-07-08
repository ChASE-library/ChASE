setup_test_serial(MatrixCPUTest matrix.cpp LIBRARIES chase_cpu)
if(TARGET chase_gpu)
setup_test_serial(MatrixGPUTest matrix.cpp LIBRARIES chase_cpu chase_gpu)
endif()

if(TARGET chase_mpi)
setup_test(MatrixCPUDistTest distMatrix.cpp LIBRARIES chase_mpi)
setup_test(MultiVectorCPUDistTest distMultiVector.cpp LIBRARIES chase_mpi)

if(TARGET linalg_internal_cuda_mpi)
setup_test(MatrixGPUDistTest distMatrix.cpp LIBRARIES chase_mpi chase_gpu distMatrix_gpu mpi_grid_nccl)
setup_test(MultiVectorGPUDistTest distMultiVector.cpp LIBRARIES chase_mpi chase_gpu distMatrix_gpu mpi_grid_nccl)
elseif(TARGET linalg_internal_hip_mpi AND TARGET mpi_grid_rccl)
setup_test(MatrixGPUDistTest distMatrix.cpp LIBRARIES chase_mpi chase_gpu distMatrix_hip mpi_grid_rccl)
setup_test(MultiVectorGPUDistTest distMultiVector.cpp LIBRARIES chase_mpi chase_gpu distMatrix_hip mpi_grid_rccl)
endif()

endif()

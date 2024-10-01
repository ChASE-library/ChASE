setup_test_serial(MatrixCPUTest matrix.cpp LIBRARIES chase_cpu)
if(HAS_GPU_SUPPORT)
setup_test_serial(MatrixGPUTest matrix.cpp LIBRARIES chase_cpu chase_gpu)
endif()
setup_test(MatrixCPUDistTest distMatrix.cpp LIBRARIES chase_mpi)
setup_test(MultiVectorCPUDistTest distMultiVector.cpp LIBRARIES chase_mpi)


setup_test_serial(MatrixCPUTest matrix.cpp LIBRARIES chase_cpu)
setup_test(MatrixCPUDistTest distMatrix.cpp LIBRARIES chase_mpi)
setup_test(MultiVectorCPUDistTest distMultiVector.cpp LIBRARIES chase_mpi)


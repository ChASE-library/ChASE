file(COPY QR_matrices DESTINATION ${CMAKE_BINARY_DIR}/tests/QR)

setup_test(QRTest QR_test.cpp LIBRARIES chase_mpi)
setup_test(QRTestIntegration QR_integration.cpp LIBRARIES chase_mpi)
#setup_test(QRTestCuda QR_test.cpp LIBRARIES chase_mpi chase_cuda)
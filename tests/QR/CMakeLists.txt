file(COPY QR_matrices DESTINATION ${CMAKE_BINARY_DIR}/tests/QR)

setup_test(QRTest QR_test.cpp LIBRARIES chase_mpi)
setup_test(QRTestIntegration QR_integration.cpp LIBRARIES chase_mpi GTest::gmock)

setup_test_serial(QRTestSerial QR_test.cpp LIBRARIES chase_seq)

if(TARGET chase_cuda )
    setup_test(QRTestCuda QR_test.cpp LIBRARIES chase_mpi chase_cuda)
    setup_test_serial(QRTestCudaSerial QR_test.cpp LIBRARIES chase_seq chase_cuda)
endif()

file(COPY QR_matrices DESTINATION ${CMAKE_BINARY_DIR}/tests/linalg/internal)
add_subdirectory(cpu)
add_subdirectory(mpi)
add_subdirectory(cuda)

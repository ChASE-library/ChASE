file(COPY QR_matrices DESTINATION ${CMAKE_BINARY_DIR}/tests/linalg/internal)
add_subdirectory(cpu)

if(TARGET chase_mpi)
add_subdirectory(mpi)
endif()

if(TARGET chase_gpu)
add_subdirectory(cuda)
endif()
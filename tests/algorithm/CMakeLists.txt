# Unit tests for algorithm layer (lanczos_for_H2, etc.) using Chase CPU and optionally GPU.
if(TARGET chase_gpu)
  setup_test_serial(AlgorithmLanczosForH2Test lanczos_for_H2_test.cpp LIBRARIES chase_gpu chase_cpu)
else()
  setup_test_serial(AlgorithmLanczosForH2Test lanczos_for_H2_test.cpp LIBRARIES chase_cpu)
endif()
# BSE matrices are copied to build by tests/linalg/internal; pass build dir so test can find them.
target_compile_definitions(AlgorithmLanczosForH2Test PRIVATE
  CHASE_TEST_BSE_DIR="${CMAKE_BINARY_DIR}/tests/linalg/internal/BSE_matrices")

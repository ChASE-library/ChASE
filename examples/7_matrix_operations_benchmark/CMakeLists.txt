if(TARGET pchase_gpu AND (TARGET linalg_internal_cuda_mpi OR TARGET linalg_internal_hip_mpi))
    add_executable(matrix_operations_bench matrix_operations_benchmark.cpp)
    target_link_libraries(matrix_operations_bench pchase_gpu)

    if(TARGET mpi_grid_nccl)
        target_compile_definitions(matrix_operations_bench PRIVATE HAS_NCCL=1)
    endif()
    if(TARGET mpi_grid_rccl)
        target_compile_definitions(matrix_operations_bench PRIVATE HAS_RCCL=1)
    endif()
else()
    message(STATUS
      "Example 7 (matrix_operations_benchmark) requires distributed GPU support; skipped.")
endif()

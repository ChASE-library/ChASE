if(POPL_FOUND)
    if(TARGET pchase_gpu AND TARGET mpi_grid_nccl)
        add_executable("householder_block_cyclic_bench" householder_block_cyclic_benchmark.cpp)
        target_link_libraries("householder_block_cyclic_bench" pchase_gpu mpi_grid_nccl)
        target_compile_definitions("householder_block_cyclic_bench" PRIVATE HAS_NCCL=1)
        target_include_directories("householder_block_cyclic_bench" PRIVATE ${POPL_INCLUDE_DIR})
    elseif(TARGET pchase_gpu)
        message(STATUS
          "Example 6 (householder_block_cyclic_benchmark) is CUDA/NCCL-only; "
          "skipped on HIP builds (see docs/amd_port_convergence_plan.md).")
    endif()
else()
  message("popl not found, not building householder block-cyclic benchmark example")
endif()


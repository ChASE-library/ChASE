add_executable( "1_sequence_eigenproblems" 1_sequence_eigenproblems.cpp )
target_link_libraries(1_sequence_eigenproblems chase_mpi)

add_executable( "1_sequence_eigenproblems_block_cyclic" 1_sequence_eigenproblems.cpp )
target_link_libraries(1_sequence_eigenproblems_block_cyclic chase_mpi)
target_compile_definitions(1_sequence_eigenproblems_block_cyclic PRIVATE USE_BLOCK_CYCLIC=1)

if(TARGET chase_cuda )
    enable_language(CUDA)
    if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      set (CMAKE_CUDA_ARCHITECTURES "80")
    endif()

    if (${CMAKE_VERSION} VERSION_LESS "3.18.0")
      foreach(CODE ${CMAKE_CUDA_ARCHITECTURES})
        set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${CODE},code=sm_${CODE}")
      endforeach()
    endif()
    add_executable( "1_sequence_eigenproblems_mgpu" 1_sequence_eigenproblems.cpp )
    target_link_libraries(1_sequence_eigenproblems_mgpu chase_mpi chase_cuda ${CUDA_nvToolsExt_LIBRARY})
    target_compile_definitions( "1_sequence_eigenproblems_mgpu" PRIVATE DRIVER_BUILD_MGPU=1)

endif()

install (TARGETS 1_sequence_eigenproblems
          RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install (TARGETS 1_sequence_eigenproblems_block_cyclic
          RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})



add_executable( "1_sequence_eigenproblems" 1_sequence_eigenproblems.cpp )
target_link_libraries(1_sequence_eigenproblems chase_mpi)

add_executable( "1_sequence_eigenproblems_block_cyclic" 1_sequence_eigenproblems.cpp )
target_link_libraries(1_sequence_eigenproblems_block_cyclic chase_mpi)
target_compile_definitions(1_sequence_eigenproblems_block_cyclic PRIVATE USE_BLOCK_CYCLIC=1)

install (TARGETS 1_sequence_eigenproblems
          RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install (TARGETS 1_sequence_eigenproblems_block_cyclic
          RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})



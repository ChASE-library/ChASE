add_executable( "0_hello_world" 0_hello_world.cpp )
target_link_libraries(0_hello_world chase_mpi)

add_executable( "0_hello_world_block_cyclic" 0_hello_world.cpp )
target_link_libraries(0_hello_world_block_cyclic chase_mpi)
target_compile_definitions(0_hello_world_block_cyclic PRIVATE USE_BLOCK_CYCLIC=1)

install (TARGETS 0_hello_world 
	  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install (TARGETS 0_hello_world_block_cyclic
          RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})  

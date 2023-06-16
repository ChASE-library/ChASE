add_executable( "0_hello_world" 0_hello_world.cpp )
target_link_libraries(0_hello_world chase_mpi)

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
    add_executable( "0_hello_world_mgpu" 0_hello_world.cpp )
    target_link_libraries(0_hello_world_mgpu chase_mpi chase_cuda ${CUDA_nvToolsExt_LIBRARY})
    target_compile_definitions( "0_hello_world_mgpu" PRIVATE DRIVER_BUILD_MGPU=1)

endif()

add_executable( "0_hello_world_givenDist" 0_hello_world.cpp )
target_link_libraries(0_hello_world_givenDist chase_mpi)
target_compile_definitions(0_hello_world_givenDist PRIVATE USE_GIVEN_DIST=1)

add_executable( "0_hello_world_block_cyclic" 0_hello_world.cpp )
target_link_libraries(0_hello_world_block_cyclic chase_mpi)
target_compile_definitions(0_hello_world_block_cyclic PRIVATE USE_BLOCK_CYCLIC=1)


install (TARGETS 0_hello_world 
	  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install (TARGETS 0_hello_world_givenDist
          RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})  
install (TARGETS 0_hello_world_block_cyclic
          RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})  

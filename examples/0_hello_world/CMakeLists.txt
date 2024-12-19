if(TARGET pchase_cpu)
add_executable( "0_hello_world.exe" "0_hello_world.cpp")
target_link_libraries( "0_hello_world.exe" PRIVATE pchase_cpu )
endif()

if(TARGET pchase_gpu)
add_executable( "0_hello_world_nccl.exe" "0_hello_world.cpp")
target_link_libraries( "0_hello_world_nccl.exe" PRIVATE pchase_gpu )
endif()
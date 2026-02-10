if(TARGET pchase_cpu)
add_executable( "1_hello_world.exe" "1_hello_world.cpp")
target_link_libraries( "1_hello_world.exe" PRIVATE pchase_cpu )
endif()

if(TARGET pchase_gpu)
add_executable( "1_hello_world_gpu.exe" "1_hello_world.cpp")
target_link_libraries( "1_hello_world_gpu.exe" PRIVATE pchase_gpu )
endif()
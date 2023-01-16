add_executable( "chase_driver" noinput.cpp )

if(TARGET chase_cuda )
  enable_language(CUDA)
  target_link_libraries(chase_driver chase_mpi chase_cuda)
  target_compile_definitions(chase_driver PRIVATE USE_GPU=1)
else()
  target_link_libraries(chase_driver chase_mpi)
endif()

add_executable( "random.exe" random.cpp )
target_link_libraries("random.exe" chase_mpi)

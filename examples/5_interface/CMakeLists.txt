enable_language(CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)

add_executable( "5_c_serial_chase.exe" 5_c_serial_chase.c )
target_link_libraries(5_c_serial_chase.exe chase_c chase_cuda)

add_executable( "5_c_dist_chase.exe" 5_c_dist_chase.c )
target_link_libraries(5_c_dist_chase.exe chase_c chase_cuda)

enable_language( Fortran )
set(CMAKE_Fortran_COMPILER ${MPI_Fortran_COMPILER})
add_executable( "5_f_serial_chase.exe" 5_f_serial_chase.f90 )
target_link_libraries(5_f_serial_chase.exe chase_c chase_f chase_cuda )

add_executable( "5_f_dist_chase.exe" 5_f_dist_chase.f90 )
target_link_libraries(5_f_dist_chase.exe chase_c chase_f chase_cuda)



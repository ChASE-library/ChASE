add_executable( "4_c_serial_chase.exe" 4_c_serial_chase.c )
target_link_libraries(4_c_serial_chase.exe chase_c)

add_executable( "4_c_dist_chase.exe" 4_c_dist_chase.c )
target_link_libraries(4_c_dist_chase.exe chase_c)

enable_language( Fortran )
add_executable( "4_f_serial_chase.exe" 4_f_serial_chase.f90 )
target_link_libraries(4_f_serial_chase.exe chase_f )

set(CMAKE_Fortran_COMPILER ${MPI_Fortran_COMPILER})
add_executable( "4_f_dist_chase.exe" 4_f_dist_chase.f90 )
target_link_libraries(4_f_dist_chase.exe chase_f )
set_target_properties(
  "4_f_dist_chase.exe"
  PROPERTIES Fortran_PREPROCESS ON
  POSITION_INDEPENDENT_CODE TRUE
)

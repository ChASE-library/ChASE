add_executable( "5_c_seial_chase.exe" 5_c_seial_chase.c )
target_link_libraries(5_c_seial_chase.exe chase_c)

add_executable( "5_c_dist_chase.exe" 5_c_dist_chase.c )
target_link_libraries(5_c_dist_chase.exe chase_c)

enable_language( Fortran )

add_executable( "5_f_seial_chase.exe" 5_f_seial_chase.f90 )
target_link_libraries(5_f_seial_chase.exe chase_c chase_f)

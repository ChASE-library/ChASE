add_executable( "5_c_interface.exe" 5_c_interface.c )
target_link_libraries(5_c_interface.exe chase_c)

enable_language( Fortran )

add_executable( "5_f_interface.exe" 5_f_interface.f90 )
target_link_libraries(5_f_interface.exe chase_c chase_f)

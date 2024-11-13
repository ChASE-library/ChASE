add_executable( "4_c_serial_chase.exe" 4_c_serial_chase.c )
target_link_libraries(4_c_serial_chase.exe chase_c_interface)

enable_language( Fortran )
add_executable( "4_f_serial_chase.exe" 4_f_serial_chase.f90 )
target_link_libraries(4_f_serial_chase.exe chase_fortran_interface )


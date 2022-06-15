# -*- Mode: cmake -*-

find_package( Boost COMPONENTS serialization program_options )
if( ${Boost_FOUND} )
    add_executable(genRnd.exe genRnd.cpp)
    target_link_libraries(genRnd.exe chase_mpi)

    install (TARGETS genRnd.exe
          RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

else()
  message( "Boost not found, not building ChASE-MPI driver" )
endif()

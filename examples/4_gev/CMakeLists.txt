enable_language( Fortran )
include( FortranCInterface )
FortranCInterface_HEADER( ${CMAKE_BINARY_DIR}/fortran_mangle.h
  MACRO_NAMESPACE "FC_"
  SYMBOL_NAMESPACE "FC_"
)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake-modules")

find_package( SCALAPACK)
find_package( Boost COMPONENTS serialization program_options )

if( ${Boost_FOUND} )

  ##############################################################################
  #                         chase_mpi_driver interface
  ##############################################################################

  if(SCALAPACK_FOUND)
    add_executable( "4_gev" 4_gev.cpp )
    target_link_libraries(4_gev chase_mpi_driver SCALAPACK::SCALAPACK ${Boost_LIBRARIES})
  endif()
else()
  message( "Boost not found, not building this example" )
endif()



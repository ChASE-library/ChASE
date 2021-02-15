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

  if(SCALAPACK_FOUND)

    ##############################################################################
    #           4_gev: no GPU, MPI ($A$ distributed among MPI-ranks)
    ##############################################################################

    add_executable( "4_gev" 4_gev.cpp )
    target_link_libraries("4_gev" chase_mpi SCALAPACK::SCALAPACK ${Boost_LIBRARIES})

    ##############################################################################
    #     4_gev_mgpu: multi-GPU/node, MPI ($A$ distributed among MPI-ranks)
    ##############################################################################
    if(TARGET chase_cuda )
      enable_language(CUDA)
      add_executable( "4_gev_mgpu" 4_gev.cpp )
      target_link_libraries("4_gev_mgpu" chase_mpi chase_cuda SCALAPACK::SCALAPACK ${Boost_LIBRARIES})
      target_compile_definitions( "4_gev_mgpu" PRIVATE DRIVER_BUILD_MGPU=1)
    endif()

  endif()
else()
  message( "Boost not found, not building example 4" )
endif()



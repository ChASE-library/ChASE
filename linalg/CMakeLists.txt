find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

enable_language( Fortran )
include( FortranCInterface )
FortranCInterface_HEADER( ${CMAKE_BINARY_DIR}/fortran_mangle.h
  MACRO_NAMESPACE "FC_"
  SYMBOL_NAMESPACE "FC_"
  )

add_library(blaspp INTERFACE)
target_include_directories(blaspp INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/blaspp)
target_link_libraries(blaspp INTERFACE BLAS::BLAS)

add_library(lapackpp INTERFACE)
target_include_directories(lapackpp INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/lapackpp)
target_link_libraries(lapackpp INTERFACE LAPACK::LAPACK)

add_library(linalg_internal_cpu INTERFACE)
target_include_directories(linalg_internal_cpu INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/internal/cpu)

find_package(SCALAPACK)
if(SCALAPACK_FOUND)
  add_library(scalapackpp INTERFACE)
  target_include_directories(scalapackpp INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/scalapackpp)
  add_definitions(-DHAS_SCALAPACK)
  target_link_libraries(scalapackpp INTERFACE ${SCALAPACK_LIBRARIES})
  message(STATUS "ScaLAPACK support enabled.")
else()
    message(STATUS "No ScaLAPACK is detected.")
endif()


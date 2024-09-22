#add_subdirectory(blaspp)
#add_subdirectory(lapackpp)
#add_subdirectory(internal/cpu)

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

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

enable_language( Fortran )
include( FortranCInterface )
FortranCInterface_HEADER( ${CMAKE_BINARY_DIR}/fortran_mangle.h
  MACRO_NAMESPACE "FC_"
  SYMBOL_NAMESPACE "FC_"
  )

list(APPEND includePath ${CMAKE_BINARY_DIR})

add_library(blaspp INTERFACE)
target_include_directories(blaspp INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/blaspp)
target_include_directories( blaspp INTERFACE
  "$<BUILD_INTERFACE:${includePath}>"
)
target_link_libraries(blaspp INTERFACE BLAS::BLAS)

add_library(lapackpp INTERFACE)
target_include_directories(lapackpp INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/lapackpp)
target_include_directories( lapackpp INTERFACE
  "$<BUILD_INTERFACE:${includePath}>"
)
target_link_libraries(lapackpp INTERFACE LAPACK::LAPACK)


find_package(SCALAPACK)
if(SCALAPACK_FOUND)
  add_library(scalapackpp INTERFACE)
  target_include_directories(scalapackpp INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/scalapackpp)
  target_compile_definitions(scalapackpp INTERFACE HAS_SCALAPACK=1)
  target_link_libraries(scalapackpp INTERFACE ${SCALAPACK_LIBRARIES})
  message(STATUS "ScaLAPACK support enabled.")
else()
    message(STATUS "No ScaLAPACK is detected.")
endif()

find_package(CUDAToolkit)	  

if(CUDAToolkit_FOUND)
  add_library(cublaspp INTERFACE)
  target_include_directories(cublaspp INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/cublaspp)
  target_link_libraries(cublaspp INTERFACE CUDA::cublas CUDA::cudart)

  add_library(cusolverpp INTERFACE)
  target_include_directories(cusolverpp INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/cusolverpp)
  target_link_libraries(cusolverpp INTERFACE CUDA::cusolver CUDA::cudart)
endif()
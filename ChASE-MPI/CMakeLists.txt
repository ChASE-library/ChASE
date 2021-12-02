# -*- Mode: cmake -*-

# Specify the minimum version for CMake
cmake_minimum_required( VERSION 3.8 )

# project( ChASE-MPI LANGUAGES C CXX CUDA )

add_library( chase_seq INTERFACE )
target_link_libraries(chase_seq INTERFACE chase_algorithm)
target_compile_definitions(chase_seq INTERFACE NO_MPI=1 )

add_library( chase_mpi INTERFACE )
target_link_libraries(chase_mpi INTERFACE chase_algorithm)

#underscores for FORTRAN
enable_language( Fortran )
include( FortranCInterface )
FortranCInterface_HEADER( ${CMAKE_BINARY_DIR}/fortran_mangle.h
  MACRO_NAMESPACE "FC_"
  SYMBOL_NAMESPACE "FC_"
  )

list(APPEND includePath ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

target_include_directories( chase_seq INTERFACE
  "$<BUILD_INTERFACE:${includePath}>"
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>  # <prefix>/include/ChASE-MPI
)

target_include_directories( chase_mpi INTERFACE
  "$<BUILD_INTERFACE:${includePath}>"
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>  # <prefix>/include/ChASE-MPI
)

## BLAS / LAPACK ##
find_package( BLAS   REQUIRED )
find_package( LAPACK REQUIRED )
find_package( MPI    REQUIRED )

target_include_directories( chase_seq INTERFACE
  ${MPI_CXX_INCLUDE_PATH}
  )

target_link_libraries( chase_seq INTERFACE
  ${BLAS_LIBRARIES}
  ${LAPACK_LIBRARIES}
  ${MPI_CXX_LIBRARIES}  
  )

target_compile_options( chase_seq INTERFACE
  ${CMAKE_C_FLAGS}
  ${MPI_CXX_COMPILE_FLAGS}  
  )

install( TARGETS chase_seq
  EXPORT chase-seq
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )

install(EXPORT chase-seq
  NAMESPACE ChASE::
  FILE chase-seq.cmake
  EXPORT_LINK_INTERFACE_LIBRARIES
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
  )

target_include_directories( chase_mpi INTERFACE
  ${MPI_CXX_INCLUDE_PATH}
  )

target_link_libraries( chase_mpi INTERFACE
  ${BLAS_LIBRARIES}
  ${LAPACK_LIBRARIES}
  ${MPI_CXX_LIBRARIES}
  )

target_compile_options( chase_mpi INTERFACE
  ${CMAKE_C_FLAGS}
  ${MPI_CXX_COMPILE_FLAGS}
  )

install( TARGETS chase_mpi
  EXPORT chase-mpi
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )

install(DIRECTORY . DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ChASE-MPI/
  FILES_MATCHING
    PATTERN "*.hpp"
    PATTERN "*.inc"
)

install( FILES  ${CMAKE_BINARY_DIR}/fortran_mangle.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

install(EXPORT chase-mpi
  NAMESPACE ChASE::
  FILE chase-mpi.cmake
  EXPORT_LINK_INTERFACE_LIBRARIES
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
  )

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)

  if (${CMAKE_VERSION} VERSION_LESS "3.18.0")
    find_package(CUDA REQUIRED)
  elseif(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.17.0")
    find_package(CUDAToolkit REQUIRED)	  
    set(CUDA_CUBLAS_LIBRARIES CUDA::cublas)
    set(CUDA_cusolver_LIBRARY CUDA::cusolver)    
  endif()

  enable_language(CUDA)

  # If variable not defined, set it to default value in order to comply with CMake Policy CMP0104
  if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      set (CMAKE_CUDA_ARCHITECTURES "80")
  endif()

  # If cuda compute compatibility of your GPU is 8.6 (e.g. RTX 3090) you should build with
  # -DCMAKE_CUDA_ARCHITECTURES=86. In the case you want to build code for more than one 
  # CUDA compute capability (e.g. 70, 75, 80 and 86) then build with
  # -DCMAKE_CUDA_ARCHITECTURES="70;75;80;86"

  # If CMAKE_VERSION < 3.18 then CMake is not compliant with CMAKE policy CMP0104 (introduced 
  # in CMake 3.18) which defines that the variable CMAKE_CUDA_ARCHITECTURES has to be initialized.
  # In that case, the code generation flag has to be set manually.
  # For simplicity and compatibility with newer (3.18+) CMake version, the CMAKE_CUDA_ARCHITECTURES 
  # variable has to be always set, not matter the cmake version.
  if (${CMAKE_VERSION} VERSION_LESS "3.18.0")
    message("CMAKE version less than 3.18. Using old CMAKE CUDA policy for defining CUDA code generation flags")
    foreach(CODE ${CMAKE_CUDA_ARCHITECTURES})
      set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${CODE},code=sm_${CODE}")
    endforeach()
  endif()

  message("-- Configuring the build of CUDA with the compute capability of ${CMAKE_CUDA_ARCHITECTURES}")

  add_library( chase_cuda "kernels/shift.cu" )

  set_target_properties( chase_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
 
  target_include_directories(
      chase_cuda
      PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
             $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ChASE-MPI>)

  target_include_directories( chase_cuda INTERFACE
    ${CUDA_INCLUDE_DIRS}
    )
  target_link_libraries( chase_cuda INTERFACE
    ${CUDA_CUBLAS_LIBRARIES}
    ${CUDA_cusolver_LIBRARY}
    )

  install( TARGETS chase_cuda
    EXPORT chase-cuda
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )

  install(EXPORT chase-cuda
    NAMESPACE ChASE::
    FILE chase-cuda.cmake
    EXPORT_LINK_INTERFACE_LIBRARIES
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
    )

else()
  message(STATUS "No CUDA support")
endif()



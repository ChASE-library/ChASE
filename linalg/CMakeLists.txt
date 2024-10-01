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

set(HAS_GPU_SUPPORT OFF BOOL "HAS_CUDA" PARENT_SCOPE)
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)

  if (${CMAKE_VERSION} VERSION_LESS_EQUAL "3.18.0")
    find_package(CUDA REQUIRED)
  elseif(${CMAKE_VERSION} VERSION_GREATER "3.18.0")
    find_package(CUDAToolkit REQUIRED)	  
    set(CUDA_CUBLAS_LIBRARIES CUDA::cublas)
    set(CUDA_cusolver_LIBRARY CUDA::cusolver)
    set(CUDA_curand_LIBRARY CUDA::curand)    
  endif()

  enable_language(CUDA)
  add_definitions(-DHAS_CUDA)
  set(HAS_GPU_SUPPORT TRUE PARENT_SCOPE)

  # If variable not defined, set it to default value in order to comply with CMake Policy CMP0104
  if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      set (CMAKE_CUDA_ARCHITECTURES "80")
  endif()

  if (${CMAKE_VERSION} VERSION_LESS "3.18.0")
    message("CMAKE version less than 3.18. Using old CMAKE CUDA policy for defining CUDA code generation flags")
    foreach(CODE ${CMAKE_CUDA_ARCHITECTURES})
      set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${CODE},code=sm_${CODE}")
    endforeach()
  endif()

  message("-- Configuring the build of CUDA with the compute capability of ${CMAKE_CUDA_ARCHITECTURES}")

  add_library(cublaspp INTERFACE)
  target_include_directories(cublaspp INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/cublaspp)
  target_link_libraries(cublaspp INTERFACE ${CUDA_CUBLAS_LIBRARIES} CUDA::cudart)
  target_include_directories( cublaspp INTERFACE
    ${CUDA_INCLUDE_DIRS}
    )

  add_library(cusolverpp INTERFACE)
  target_include_directories(cusolverpp INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/cusolverpp)
  target_link_libraries(cusolverpp INTERFACE ${CUDA_cusolver_LIBRARY} CUDA::cudart)
  target_include_directories( cusolverpp INTERFACE
    ${CUDA_INCLUDE_DIRS}
    )
    
  # Specify the CUDA source files
  set(CUDA_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/internal/cuda/precision_conversion.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/internal/cuda/shiftDiagonal.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/internal/cuda/absTrace.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/internal/cuda/lacpy.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/internal/cuda/random_normal_distribution.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/internal/cuda/residuals.cu
        )
  add_library( chase_cuda_kernels ${CUDA_SOURCES} )
  set_target_properties( chase_cuda_kernels 
	  		                 PROPERTIES
		  		                          CUDA_RESOLVE_DEVICE_SYMBOLS ON
	                                  CUDA_SEPARABLE_COMPILATION ON)
  target_include_directories(
      chase_cuda_kernels
      INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/internal/cuda>)

  target_include_directories( chase_cuda_kernels INTERFACE
      ${CUDA_INCLUDE_DIRS})

  target_link_libraries(chase_cuda_kernels INTERFACE CUDA::cudart CUDA::curand)

  else()
  message(STATUS "No CUDA support")
endif()
 
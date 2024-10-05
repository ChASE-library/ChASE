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

add_library(linalg_internal_cpu INTERFACE)
target_include_directories(linalg_internal_cpu INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/internal/cpu)

add_library(matrix INTERFACE)
target_include_directories(matrix INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/matrix)
target_link_libraries(matrix INTERFACE blaspp lapackpp linalg_internal_cpu)

add_library(distMatrix INTERFACE)
find_package(MPI)
if(MPI_FOUND)
add_library(linalg_internal_mpi INTERFACE)
target_include_directories(linalg_internal_mpi INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/internal/mpi)
target_link_libraries(linalg_internal_mpi INTERFACE MPI::MPI_CXX)
target_include_directories(distMatrix INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/distMatrix)
target_link_libraries(distMatrix INTERFACE blaspp lapackpp linalg_internal_mpi)
endif()

find_package(SCALAPACK)
if(SCALAPACK_FOUND)
  add_library(scalapackpp INTERFACE)
  target_include_directories(scalapackpp INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/scalapackpp)
  target_compile_definitions(scalapackpp INTERFACE HAS_SCALAPACK=1)
  target_link_libraries(scalapackpp INTERFACE ${SCALAPACK_LIBRARIES})
  message(STATUS "ScaLAPACK support enabled.")
  target_link_libraries(distMatrix INTERFACE scalapackpp)
else()
    message(STATUS "No ScaLAPACK is detected.")
endif()

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

  add_library(linalg_internal_cuda INTERFACE)
  target_include_directories(linalg_internal_cuda INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/internal/cuda)
  target_link_libraries(linalg_internal_cuda INTERFACE chase_cuda_kernels)

  add_library(matrix_gpu INTERFACE)
  target_include_directories(matrix_gpu INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/matrix)
  target_link_libraries(matrix_gpu INTERFACE blaspp lapackpp linalg_internal_cpu)

  target_link_libraries(matrix_gpu INTERFACE cublaspp cusolverpp linalg_internal_cuda)
  target_compile_definitions(matrix_gpu INTERFACE HAS_CUDA=1)

  add_library(distMatrix_gpu INTERFACE)
  target_include_directories(distMatrix_gpu INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/distMatrix)
  target_link_libraries(distMatrix_gpu INTERFACE blaspp lapackpp  linalg_internal_mpi)
  target_link_libraries(distMatrix_gpu INTERFACE cublaspp cusolverpp linalg_internal_cuda)
  target_compile_definitions(distMatrix_gpu INTERFACE HAS_CUDA=1)

  find_package(NCCL)
  if(NCCL_FOUND)
    add_library(linalg_internal_nccl INTERFACE)
    target_include_directories(linalg_internal_nccl INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/internal/nccl)
    target_link_libraries(linalg_internal_nccl INTERFACE mpi_grid_nccl linalg_internal_cuda)
  endif()
  else()
  message(STATUS "No CUDA support")
endif()

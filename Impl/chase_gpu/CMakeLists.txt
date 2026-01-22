add_library(chase_gpu INTERFACE)
# Link against linalg libraries
target_link_libraries(chase_gpu 
                        INTERFACE 
                        chase_algorithm 
                        blaspp 
                        lapackpp 
                        linalg_internal_cpu 
                        cublaspp 
                        cusolverpp 
                        linalg_internal_cuda
                        matrix_gpu)

target_include_directories(chase_gpu INTERFACE 
                                               $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                                               "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/Impl/chase_gpu>")

# NO_MPI is not defined here to avoid conflicts when chase_c links to both serial and parallel targets
# The interface layer (chase_c) will handle NO_MPI definition based on actual usage
target_compile_definitions(chase_gpu INTERFACE HAS_CUDA=1)

install(TARGETS chase_gpu
  EXPORT chase_targets
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/Impl
  FILES_MATCHING
    PATTERN "*.hpp"
)

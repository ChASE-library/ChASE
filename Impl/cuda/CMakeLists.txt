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

target_include_directories(chase_gpu INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(chase_gpu INTERFACE NO_MPI=1 HAS_CUDA=1)


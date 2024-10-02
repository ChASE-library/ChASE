add_library(chase_cpu INTERFACE)

# Link against linalg libraries
target_link_libraries(chase_cpu INTERFACE chase_algorithm blaspp lapackpp linalg_internal_cpu matrix)
target_include_directories(chase_cpu INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(chase_cpu INTERFACE NO_MPI=1)
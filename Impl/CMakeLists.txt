add_library(chase_cpu INTERFACE)

# Link against linalg libraries
target_link_libraries(chase_cpu INTERFACE chase_algorithm blaspp linalg_internal_cpu)

list(APPEND includePath ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories( chase_cpu INTERFACE
  "$<BUILD_INTERFACE:${includePath}>"
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>  # <prefix>/include/ChASE-MPI
)

target_include_directories(chase_cpu INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(chase_cpu INTERFACE NO_MPI=1 )

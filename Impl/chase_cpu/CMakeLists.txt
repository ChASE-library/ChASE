add_library(chase_cpu INTERFACE)

# Link against linalg libraries
target_link_libraries(chase_cpu INTERFACE chase_algorithm blaspp lapackpp linalg_internal_cpu matrix)

target_include_directories(chase_cpu INTERFACE 
                                               $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                                               "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/Impl/chase_cpu>")

target_compile_definitions(chase_cpu INTERFACE NO_MPI=1)

install(TARGETS chase_cpu
  EXPORT chase_targets
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/Impl
  FILES_MATCHING
    PATTERN "*.hpp"
)
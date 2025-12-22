add_library(chase_cpu INTERFACE)

# Link against linalg libraries
target_link_libraries(chase_cpu INTERFACE chase_algorithm blaspp lapackpp linalg_internal_cpu matrix)

target_include_directories(chase_cpu INTERFACE 
                                               $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                                               "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/Impl/chase_cpu>")

# NO_MPI is not defined here to avoid conflicts when chase_c links to both serial and parallel targets
# The interface layer (chase_c) will handle NO_MPI definition based on actual usage

install(TARGETS chase_cpu
  EXPORT chase_targets
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/Impl
  FILES_MATCHING
    PATTERN "*.hpp"
)
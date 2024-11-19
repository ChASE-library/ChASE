find_package(MPI)

if(MPI_FOUND)
    add_library(mpi_grid INTERFACE)
    target_link_libraries(mpi_grid INTERFACE MPI::MPI_CXX)
    target_include_directories(mpi_grid INTERFACE 
                                                $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                                                "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/grid>")


    install(TARGETS mpi_grid
        EXPORT chase_targets
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )

    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING
        PATTERN "*.hpp"
    )

endif()

find_package(NCCL)
if(NCCL_FOUND)
    add_library(mpi_grid_nccl INTERFACE)
    target_link_libraries(mpi_grid_nccl INTERFACE MPI::MPI_CXX)
    target_include_directories(mpi_grid_nccl INTERFACE 
                                                $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                                                "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/grid>")
                                                
    target_link_libraries(mpi_grid_nccl INTERFACE ${NCCL_LIBRARIES})
    #target_include_directories( mpi_grid_nccl INTERFACE ${NCCL_INCLUDE_DIRS})
    target_compile_definitions( mpi_grid_nccl INTERFACE "-DHAS_NCCL")

    install(TARGETS mpi_grid_nccl
        EXPORT chase_targets
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )

endif()



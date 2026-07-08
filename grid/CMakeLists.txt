find_package(MPI)

if(MPI_FOUND)
    add_library(mpi_grid INTERFACE)
    target_link_libraries(mpi_grid INTERFACE MPI::MPI_CXX)
    
    if(TARGET chase_scalapack_linalg)
        target_link_libraries(mpi_grid INTERFACE chase_scalapack_linalg)
    endif()

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
    
    if(TARGET chase_scalapack_linalg)
        target_link_libraries(mpi_grid_nccl INTERFACE chase_scalapack_linalg)
    endif()

    target_include_directories(mpi_grid_nccl INTERFACE 
                                                $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                                                "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/grid>")
                                                
    target_link_libraries(mpi_grid_nccl INTERFACE NCCL::NCCL)
    target_compile_definitions(mpi_grid_nccl INTERFACE HAS_NCCL)

    install(TARGETS mpi_grid_nccl
        EXPORT chase_targets
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
endif()

# AMD RCCL (NCCL-compatible API on ROCm). Separate from mpi_grid_nccl / NVIDIA NCCL.
if(CHASE_ENABLE_HIP_EFFECTIVE AND MPI_FOUND)
    find_package(RCCL)
    if(RCCL_FOUND)
        add_library(mpi_grid_rccl INTERFACE)
        target_link_libraries(mpi_grid_rccl INTERFACE MPI::MPI_CXX)

        if(TARGET chase_scalapack_linalg)
            target_link_libraries(mpi_grid_rccl INTERFACE chase_scalapack_linalg)
        endif()

        target_include_directories(mpi_grid_rccl INTERFACE
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
            "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/grid>")

        target_link_libraries(mpi_grid_rccl INTERFACE RCCL::RCCL)
        target_compile_definitions(mpi_grid_rccl INTERFACE HAS_RCCL)

        install(TARGETS mpi_grid_rccl
            EXPORT chase_targets
            INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        )
    else()
        message(STATUS "RCCL not found; mpi_grid_rccl and RCCL grid tests are disabled")
    endif()
endif()

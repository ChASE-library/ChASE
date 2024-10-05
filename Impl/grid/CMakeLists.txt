add_library(mpi_grid INTERFACE)
target_link_libraries(mpi_grid INTERFACE linalg_internal_mpi)
target_include_directories(mpi_grid INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

find_package(NCCL)
if(NCCL_FOUND)
    add_library(mpi_grid_nccl INTERFACE)
    target_link_libraries(mpi_grid_nccl INTERFACE linalg_internal_mpi)
    target_include_directories(mpi_grid_nccl INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

    target_link_libraries(mpi_grid_nccl INTERFACE ${NCCL_LIBRARIES})
    target_include_directories( mpi_grid_nccl INTERFACE ${NCCL_INCLUDE_DIRS})
    target_compile_definitions( mpi_grid_nccl INTERFACE "-DHAS_NCCL")
endif()
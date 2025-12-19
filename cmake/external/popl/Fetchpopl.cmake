if(NOT EXISTS "${CMAKE_BINARY_DIR}/_deps/poplproject-src")
    include(FetchContent)

    FetchContent_Declare(
        poplProject
        GIT_REPOSITORY https://github.com/badaix/popl.git
        GIT_TAG master
    )

    # Allow old Populate API (required for patching)
    if (POLICY CMP0169)
        cmake_policy(SET CMP0169 OLD)
    endif()

    FetchContent_GetProperties(poplProject)
    if (NOT poplproject_POPULATED)
        # Explicit populate step
        FetchContent_Populate(poplProject)

        # Apply your patch BEFORE popl's CMakeLists.txt is processed
        execute_process(
            COMMAND git apply ${CMAKE_SOURCE_DIR}/cmake/external/popl/popl.patch
            WORKING_DIRECTORY ${poplproject_SOURCE_DIR}
        )

        # Now configure patched popl
        add_subdirectory(${poplproject_SOURCE_DIR} ${poplproject_BINARY_DIR})
    endif()

    message(STATUS "popl fetched + patched successfully.")
    set(POPL_FOUND TRUE CACHE BOOL "external popl project found variable")
    set(POPL_INCLUDE_DIR ${poplproject_SOURCE_DIR}/include CACHE STRING 
        "external popl project include directory")
endif()
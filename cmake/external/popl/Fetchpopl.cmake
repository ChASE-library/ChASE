if(NOT EXISTS "${CMAKE_BINARY_DIR}/_deps/poplproject-src")
    include(FetchContent)

    FetchContent_Declare(
        poplProject
        GIT_REPOSITORY https://github.com/badaix/popl.git
        GIT_TAG master )

        set(FETCHCONTENT_QUIET ON)
        FetchContent_GetProperties(poplProject)

        if(NOT poplproject_POPULATED)
            FetchContent_Populate(poplProject)
            execute_process(
                COMMAND git apply ${CMAKE_SOURCE_DIR}/cmake/external/popl/popl.patch
                WORKING_DIRECTORY ${poplproject_SOURCE_DIR} )
            add_subdirectory(${poplproject_SOURCE_DIR} ${poplproject_BINARY_DIR} )
        endif()
    
    message(STATUS "Fetching Program Options Parser Library (popl)  was successful.")
    set(POPL_FOUND TRUE CACHE BOOL "external popl project found variable")
    set(POPL_INCLUDE_DIR ${poplproject_SOURCE_DIR}/include CACHE STRING 
        "external popl project include directory")
endif()
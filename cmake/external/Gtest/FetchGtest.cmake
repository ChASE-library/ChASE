include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG main 
    )

FetchContent_GetProperties(googletest)

#FetchContent_MakeAvailable(googletest)

if(NOT googletest_POPULATED)
    FetchContent_Populate(googletest)

    add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR} )
endif()

message(STATUS "Fetching Googletest was successful.")
set(GTEST_FOUND TRUE CACHE BOOL "external GTest project found variable")

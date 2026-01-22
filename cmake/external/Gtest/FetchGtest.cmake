include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG main 
    )

FetchContent_MakeAvailable(googletest)

message(STATUS "Fetching Googletest was successful.")
set(GTEST_FOUND TRUE CACHE BOOL "external GTest project found variable")

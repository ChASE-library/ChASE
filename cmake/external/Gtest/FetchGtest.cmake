include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG main 
    FIND_PACKAGE_ARGS NAMES GTest )

FetchContent_MakeAvailable(googletest)
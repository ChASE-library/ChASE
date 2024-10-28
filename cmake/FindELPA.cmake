# FindELPA.cmake - Locate ELPA with support for multiple variants via pkg-config
# Sets the imported target `ELPA::ELPA` if found

# Ensure the PkgConfig module is available
find_package(PkgConfig REQUIRED)

# Allow the user to specify the preferred ELPA variant and installation path
set(ELPA_PACKAGE_NAME "elpa" CACHE STRING "Name of the ELPA package to find (e.g., elpa, elpa_openmp)")
set(ELPA_ROOT_DIR "" CACHE PATH "Root directory of the ELPA installation (if not in default locations)")

# Check for lib and lib64 directories
if(EXISTS "${ELPA_ROOT_DIR}/lib64")
    set(ELPA_LIB_DIR "${ELPA_ROOT_DIR}/lib64")
elseif(EXISTS "${ELPA_ROOT_DIR}/lib")
    set(ELPA_LIB_DIR "${ELPA_ROOT_DIR}/lib")
else()
    message(FATAL_ERROR "Neither lib64 nor lib directory found in ${ELPA_ROOT_DIR}.")
endif()


# Set PKG_CONFIG_PATH to include lib and lib64 paths
set(ENV{PKG_CONFIG_PATH} "${ELPA_LIB_DIR}:$ENV{PKG_CONFIG_PATH}")
set(ENV{PKG_CONFIG_PATH} "$ENV{PKG_CONFIG_PATH}:${ELPA_LIB_DIR}/pkgconfig")

# Check for the specified ELPA package using pkg-config
pkg_check_modules(ELPA REQUIRED ${ELPA_PACKAGE_NAME})

# Make sure ELPA's include directories and libraries are visible in the cache
# Store the include and library paths
include_directories(${ELPA_INCLUDE_DIR})
set(ELPA_LIBRARY ${ELPA_LIBRARIES})
set(ELPA_INCLUDE_DIRS ${ELPA_INCLUDE_DIR})

# Add any additional library paths if required
if(ELPA_LIB_DIR)
    set(ELPA_LIBRARY "${ELPA_LIBRARY};-L${ELPA_LIB_DIR}")
endif()

message("--   FOUND ELPA_INCLUDE_DIR = ${ELPA_INCLUDE_DIR}")
message("--   FOUND ELPA_LIBRARY = ${ELPA_LIBRARY}")

# Create an imported target for ELPA if pkg-config succeeded
if(ELPA_FOUND)
    add_library(ELPA::ELPA INTERFACE IMPORTED)

    # Link the include directories and library flags from pkg-config
    set_target_properties(ELPA::ELPA PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${ELPA_INCLUDE_DIRS}"
        INTERFACE_LINK_LIBRARIES "${ELPA_LIBRARY}"
    )

    message(STATUS "Found ELPA (${ELPA_PACKAGE_NAME}) via pkg-config")
else()
    message(FATAL_ERROR "ELPA package '${ELPA_PACKAGE_NAME}' not found.")
endif()
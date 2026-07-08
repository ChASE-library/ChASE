# Find the RCCL libraries (ROCm collective communication, NCCL-compatible API).
#
# Frontier layout (rocm module):
#   $ROCM_PATH/include/rccl/rccl.h
#   $ROCM_PATH/lib/librccl.so
#
# Optional cache variables:
#   RCCL_ROOT          Base directory (default: $ENV{ROCM_PATH})
#   RCCL_INCLUDE_DIR   Directory containing rccl/rccl.h (usually $ROCM_PATH/include)
#   RCCL_LIB_DIR       Directory containing librccl (usually $ROCM_PATH/lib)
#
# On success sets:
#   RCCL_FOUND, RCCL_INCLUDE_DIRS, RCCL_LIBRARIES, RCCL::RCCL

if(DEFINED ENV{RCCL_INCLUDE_DIR} AND NOT RCCL_INCLUDE_DIR)
  set(RCCL_INCLUDE_DIR $ENV{RCCL_INCLUDE_DIR} CACHE PATH
      "Folder containing rccl/rccl.h (parent of rccl/)")
endif()
if(DEFINED ENV{RCCL_LIB_DIR} AND NOT RCCL_LIB_DIR)
  set(RCCL_LIB_DIR $ENV{RCCL_LIB_DIR} CACHE PATH "Folder containing librccl")
endif()

if(NOT RCCL_ROOT)
  if(DEFINED ENV{RCCL_ROOT})
    set(RCCL_ROOT $ENV{RCCL_ROOT})
  elseif(DEFINED ENV{ROCM_PATH})
    set(RCCL_ROOT $ENV{ROCM_PATH})
  elseif(DEFINED ENV{ROCM_ROOT})
    set(RCCL_ROOT $ENV{ROCM_ROOT})
  endif()
endif()
set(RCCL_ROOT "${RCCL_ROOT}" CACHE PATH "ROCm/RCCL install root")

list(APPEND CMAKE_PREFIX_PATH ${RCCL_ROOT})

set(_rccl_include_hints ${RCCL_INCLUDE_DIR})
if(RCCL_ROOT)
  list(APPEND _rccl_include_hints
       "${RCCL_ROOT}/include"
       "${RCCL_ROOT}/include/rccl")
endif()

find_path(RCCL_INCLUDE_DIRS
  NAMES rccl/rccl.h
  HINTS ${_rccl_include_hints}
  DOC "RCCL include directory (parent of rccl/)")

set(_rccl_lib_hints ${RCCL_LIB_DIR})
if(RCCL_ROOT)
  list(APPEND _rccl_lib_hints
       "${RCCL_ROOT}/lib"
       "${RCCL_ROOT}/lib64")
endif()

find_library(RCCL_LIBRARIES
  NAMES rccl
  HINTS ${_rccl_lib_hints}
  DOC "RCCL shared library")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(RCCL DEFAULT_MSG RCCL_INCLUDE_DIRS RCCL_LIBRARIES)

if(RCCL_FOUND)
  if(NOT TARGET RCCL::RCCL)
    add_library(RCCL::RCCL INTERFACE IMPORTED)
    set_target_properties(RCCL::RCCL PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${RCCL_INCLUDE_DIRS}"
      INTERFACE_LINK_LIBRARIES "${RCCL_LIBRARIES}")
  endif()
  mark_as_advanced(RCCL_INCLUDE_DIRS RCCL_LIBRARIES)
  message(STATUS "Found RCCL (include: ${RCCL_INCLUDE_DIRS}, library: ${RCCL_LIBRARIES})")
endif()

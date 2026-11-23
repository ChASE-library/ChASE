# Optional Ozaki Scheme II dependency (GEMMul8). NVIDIA CUDA only.
# GEMMUL8_ROOT if set, otherwise clone and build GEMM-only INT8 when
# CHASE_FETCH_GEMMUL8 is ON. Sets these variables in the caller:
#   _GEMMUL8_INCLUDE_DIR, _GEMMUL8_FROM_FETCH, _GEMMUL8_ROOT_LOG
# Creates gemmul8::gemmul8, and gemmul8_ep when ChASE builds the library.

function(chase_resolve_gemmul8_gpu_arch out_var)
  # GEMMul8's Makefile accepts one GPU_ARCH. It is the same value as
  # CMAKE_CUDA_ARCHITECTURES, which chase_require_single_cuda_architecture() keeps singular.
  if(CHASE_GEMMUL8_GPU_ARCH AND NOT CHASE_GEMMUL8_GPU_ARCH STREQUAL "${CMAKE_CUDA_ARCHITECTURES}")
    message(FATAL_ERROR
      "CHASE_GEMMUL8_GPU_ARCH=${CHASE_GEMMUL8_GPU_ARCH} does not match "
      "CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}. "
      "ChASE and GEMMul8 use that one architecture.")
  endif()
  if(NOT CMAKE_CUDA_ARCHITECTURES MATCHES "^[0-9]+$")
    message(FATAL_ERROR
      "GEMMul8 GPU_ARCH needs one numeric architecture, got '${CMAKE_CUDA_ARCHITECTURES}'.")
  endif()
  set(${out_var} "${CMAKE_CUDA_ARCHITECTURES}" PARENT_SCOPE)
endfunction()

function(chase_setup_gemmul8)
  if(NOT CUDAToolkit_FOUND)
    message(FATAL_ERROR "CHASE_ENABLE_OZAKI=ON requires CUDAToolkit (CUDA).")
  endif()
  if(NOT DEFINED CUDAToolkit_NVCC_EXECUTABLE OR NOT EXISTS "${CUDAToolkit_NVCC_EXECUTABLE}")
    message(FATAL_ERROR
      "CHASE_ENABLE_OZAKI=ON requires NVCC (via CUDAToolkit). "
      "CUDAToolkit_NVCC_EXECUTABLE='${CUDAToolkit_NVCC_EXECUTABLE}'")
  endif()

  if(NOT GEMMUL8_ROOT AND DEFINED ENV{GEMMUL8_ROOT} AND NOT "$ENV{GEMMUL8_ROOT}" STREQUAL "")
    set(GEMMUL8_ROOT "$ENV{GEMMUL8_ROOT}")
  endif()

  set(_GEMMUL8_INCLUDE_DIR "")
  set(_GEMMUL8_LIB "")
  set(_GEMMUL8_ROOT_LOG "")
  set(_GEMMUL8_FROM_FETCH OFF)

  if(GEMMUL8_ROOT)
    set(_GEMMUL8_INCLUDE_DIR "${GEMMUL8_ROOT}/include")
    if(EXISTS "${GEMMUL8_ROOT}/GEMMul8/include")
      set(_GEMMUL8_INCLUDE_DIR "${GEMMUL8_ROOT}/GEMMul8/include")
    endif()

    foreach(_cand
            "${GEMMUL8_ROOT}/lib/libgemmul8.so"
            "${GEMMUL8_ROOT}/lib64/libgemmul8.so"
            "${GEMMUL8_ROOT}/lib/libgemmul8.a"
            "${GEMMUL8_ROOT}/lib64/libgemmul8.a"
            "${GEMMUL8_ROOT}/GEMMul8/lib/libgemmul8.so"
            "${GEMMUL8_ROOT}/GEMMul8/lib/libgemmul8.a")
      if(EXISTS "${_cand}")
        set(_GEMMUL8_LIB "${_cand}")
        break()
      endif()
    endforeach()

    if(NOT EXISTS "${_GEMMUL8_INCLUDE_DIR}/gemmul8.hpp")
      message(FATAL_ERROR
        "GEMMul8 header not found. Expected '${_GEMMUL8_INCLUDE_DIR}/gemmul8.hpp'. "
        "Set GEMMUL8_ROOT to your GEMMul8 build/install prefix (or source root containing GEMMul8/include).")
    endif()

    if(NOT _GEMMUL8_LIB)
      message(FATAL_ERROR
        "libgemmul8 not found under GEMMUL8_ROOT='${GEMMUL8_ROOT}'. "
        "Expected lib/libgemmul8.(so|a) (or GEMMul8/lib/...).")
    endif()
    set(_GEMMUL8_ROOT_LOG "${GEMMUL8_ROOT}")
    message(STATUS "Ozaki Scheme II backend enabled (GEMMul8 prefix): ${_GEMMUL8_LIB}")
  elseif(CHASE_FETCH_GEMMUL8)
    set(_GEMMUL8_FROM_FETCH ON)
  else()
    message(FATAL_ERROR
      "CHASE_ENABLE_OZAKI=ON needs -DGEMMUL8_ROOT=/path/to/GEMMul8 (include/ and lib/), "
      "or -DCHASE_FETCH_GEMMUL8=ON to download and build GEMM-only INT8.")
  endif()

  if(_GEMMUL8_FROM_FETCH)
    chase_resolve_gemmul8_gpu_arch(_GEMMUL8_GPU_ARCH)
    get_filename_component(_CHASE_CUDA_BIN "${CUDAToolkit_NVCC_EXECUTABLE}" DIRECTORY)
    get_filename_component(_CHASE_CUDA_PATH "${_CHASE_CUDA_BIN}" DIRECTORY)
    find_program(CHASE_GEMMUL8_MAKE NAMES gmake make REQUIRED)
    if(CMAKE_BUILD_PARALLEL_LEVEL)
      set(_GEMMUL8_JOBS "${CMAKE_BUILD_PARALLEL_LEVEL}")
    elseif(DEFINED ENV{CMAKE_BUILD_PARALLEL_LEVEL} AND NOT "$ENV{CMAKE_BUILD_PARALLEL_LEVEL}" STREQUAL "")
      set(_GEMMUL8_JOBS "$ENV{CMAKE_BUILD_PARALLEL_LEVEL}")
    else()
      set(_GEMMUL8_JOBS 16)
    endif()

    set(_GEMMUL8_SRC "${CMAKE_BINARY_DIR}/_deps/gemmul8-src")
    set(_GEMMUL8_INCLUDE_DIR "${_GEMMUL8_SRC}/include")
    set(_GEMMUL8_LIB "${_GEMMUL8_SRC}/lib/libgemmul8.so")
    set(_GEMMUL8_ROOT_LOG "${_GEMMUL8_SRC}")
    set(_GEMMUL8_TAG_FILE "${_GEMMUL8_SRC}/.chase-gemmul8-tag")

    set(_GEMMUL8_NEED_CLONE ON)
    if(EXISTS "${_GEMMUL8_INCLUDE_DIR}/gemmul8.hpp" AND EXISTS "${_GEMMUL8_TAG_FILE}")
      file(READ "${_GEMMUL8_TAG_FILE}" _GEMMUL8_HAVE_TAG)
      string(STRIP "${_GEMMUL8_HAVE_TAG}" _GEMMUL8_HAVE_TAG)
      if(_GEMMUL8_HAVE_TAG STREQUAL "${CHASE_GEMMUL8_GIT_TAG}")
        set(_GEMMUL8_NEED_CLONE OFF)
      endif()
    endif()
    if(_GEMMUL8_NEED_CLONE)
      message(STATUS "Cloning GEMMul8 ${CHASE_GEMMUL8_GIT_TAG}")
      file(REMOVE_RECURSE "${_GEMMUL8_SRC}")
      file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/_deps")
      execute_process(
        COMMAND git clone --depth 1 --branch "${CHASE_GEMMUL8_GIT_TAG}"
                "${CHASE_GEMMUL8_GIT_REPOSITORY}" "${_GEMMUL8_SRC}"
        RESULT_VARIABLE _GEMMUL8_CLONE_RC
        OUTPUT_VARIABLE _GEMMUL8_CLONE_OUT
        ERROR_VARIABLE _GEMMUL8_CLONE_ERR
      )
      if(NOT _GEMMUL8_CLONE_RC EQUAL 0)
        message(FATAL_ERROR
          "Failed to clone GEMMul8 ${CHASE_GEMMUL8_GIT_TAG} from ${CHASE_GEMMUL8_GIT_REPOSITORY} "
          "(exit ${_GEMMUL8_CLONE_RC}).\n${_GEMMUL8_CLONE_ERR}${_GEMMUL8_CLONE_OUT}")
      endif()
      file(WRITE "${_GEMMUL8_TAG_FILE}" "${CHASE_GEMMUL8_GIT_TAG}\n")
    endif()

    # GEMMul8's shared library has no DT_SONAME. CMake links build-tree libraries by a
    # relative path, and ld then stores that path in NEEDED, which the loader does
    # not search through RPATH. Pass an explicit soname so NEEDED is "libgemmul8.so".
    # Command-line LINK_FLAGS replaces the Makefile variable, so keep its aarch64 flag.
    set(_GEMMUL8_LINK_FLAGS "-Xlinker -soname -Xlinker libgemmul8.so")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
      set(_GEMMUL8_LINK_FLAGS "-Xlinker --stub-group-size=33554432 ${_GEMMUL8_LINK_FLAGS}")
    endif()

    set(_GEMMUL8_SPEC "${CMAKE_BINARY_DIR}/_deps/gemmul8-buildspec.txt")
    set(_GEMMUL8_SPEC_CONTENT
      "repo=${CHASE_GEMMUL8_GIT_REPOSITORY}\ntag=${CHASE_GEMMUL8_GIT_TAG}\narch=${_GEMMUL8_GPU_ARCH}\nops=gemm\noz2=INT8\ncuda=${_CHASE_CUDA_PATH}\nlink=${_GEMMUL8_LINK_FLAGS}\n")
    set(_GEMMUL8_SPEC_OLD "")
    if(EXISTS "${_GEMMUL8_SPEC}")
      file(READ "${_GEMMUL8_SPEC}" _GEMMUL8_SPEC_OLD)
    endif()
    if(NOT _GEMMUL8_SPEC_OLD STREQUAL _GEMMUL8_SPEC_CONTENT)
      file(WRITE "${_GEMMUL8_SPEC}" "${_GEMMUL8_SPEC_CONTENT}")
    endif()

    add_custom_command(
      OUTPUT "${_GEMMUL8_LIB}"
      COMMAND ${CMAKE_COMMAND} -E rm -f "${_GEMMUL8_LIB}"
      COMMAND ${CMAKE_COMMAND} -E env
        "PATH=${_CHASE_CUDA_BIN}:$ENV{PATH}"
        "CUDA_PATH=${_CHASE_CUDA_PATH}"
        ${CHASE_GEMMUL8_MAKE} -C "${_GEMMUL8_SRC}" -j${_GEMMUL8_JOBS}
        BACKEND=cuda
        GPU_ARCH=${_GEMMUL8_GPU_ARCH}
        OPS=gemm
        OZ2_BACKENDS=INT8
        CUDA_PATH=${_CHASE_CUDA_PATH}
        "LINK_FLAGS=${_GEMMUL8_LINK_FLAGS}"
      DEPENDS "${_GEMMUL8_SPEC}"
      COMMENT "Building GEMMul8 (OPS=gemm OZ2_BACKENDS=INT8 GPU_ARCH=${_GEMMUL8_GPU_ARCH})"
      VERBATIM
      USES_TERMINAL
    )
    add_custom_target(gemmul8_ep DEPENDS "${_GEMMUL8_LIB}")
    install(FILES "${_GEMMUL8_LIB}" DESTINATION ${CMAKE_INSTALL_LIBDIR})
    message(STATUS
      "Ozaki Scheme II backend: GEMMul8 ${CHASE_GEMMUL8_GIT_TAG} will be built "
      "(OPS=gemm OZ2_BACKENDS=INT8 GPU_ARCH=${_GEMMUL8_GPU_ARCH}) and installed "
      "to ${CMAKE_INSTALL_LIBDIR}/libgemmul8.so")
  endif()

  if(NOT TARGET gemmul8::gemmul8)
    add_library(gemmul8::gemmul8 UNKNOWN IMPORTED GLOBAL)
    set_target_properties(gemmul8::gemmul8 PROPERTIES
      IMPORTED_LOCATION "${_GEMMUL8_LIB}"
      INTERFACE_INCLUDE_DIRECTORIES "${_GEMMUL8_INCLUDE_DIR}"
    )
    target_link_libraries(gemmul8::gemmul8 INTERFACE CUDA::cublas CUDA::cudart)
  endif()

  if(EXISTS "${_GEMMUL8_INCLUDE_DIR}/version.hpp")
    message(STATUS "GEMMul8 memory-saving API detected (version.hpp); save-memory is on unless CHASE_OZAKI_WORKSPACE_GB is set")
  elseif(_GEMMUL8_FROM_FETCH)
    message(STATUS "GEMMul8 memory-saving API follows version.hpp in the fetched tree")
  else()
    message(STATUS "GEMMul8 memory-saving API not present; save-memory stays off (old GEMMul8)")
  endif()

  set(_GEMMUL8_INCLUDE_DIR "${_GEMMUL8_INCLUDE_DIR}" PARENT_SCOPE)
  set(_GEMMUL8_FROM_FETCH "${_GEMMUL8_FROM_FETCH}" PARENT_SCOPE)
  set(_GEMMUL8_ROOT_LOG "${_GEMMUL8_ROOT_LOG}" PARENT_SCOPE)
endfunction()

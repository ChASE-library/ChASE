# One CUDA architecture for ChASE and for a ChASE-built GEMMul8.
# Unset or "native": first GPU from nvidia-smi. No numeric fallback.
# A value the user set must already be a single architecture.

function(chase_cuda_arch_from_nvidia_smi out_var)
  set(_detected "")
  execute_process(
    COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
    OUTPUT_VARIABLE _caps
    ERROR_VARIABLE _caps_err
    RESULT_VARIABLE _rc
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(_rc EQUAL 0 AND NOT "${_caps}" STREQUAL "")
    string(REPLACE "\r" "" _caps "${_caps}")
    string(REPLACE "\n" ";" _lines "${_caps}")
    foreach(_line IN LISTS _lines)
      string(STRIP "${_line}" _line)
      string(REPLACE "." "" _arch "${_line}")
      if(_arch MATCHES "^[0-9]+$")
        list(APPEND _detected "${_arch}")
      endif()
    endforeach()
  endif()

  if(NOT _detected)
    message(FATAL_ERROR
      "CMAKE_CUDA_ARCHITECTURES is not set and nvidia-smi --query-gpu=compute_cap "
      "did not return a compute capability. "
      "A login node often has the CUDA toolkit but no GPU. "
      "Configure on a compute node, or pass one architecture, "
      "for example -DCMAKE_CUDA_ARCHITECTURES=90.\n${_caps_err}")
  endif()

  list(GET _detected 0 _first)
  list(REMOVE_DUPLICATES _detected)
  list(LENGTH _detected _n)
  if(_n GREATER 1)
    message(WARNING
      "nvidia-smi reported compute capabilities ${_detected}. "
      "ChASE and GEMMul8 use the first GPU only: ${_first}.")
  endif()
  set(${out_var} "${_first}" PARENT_SCOPE)
endfunction()

function(chase_log_cuda_architecture _one _automatic)
  if(_automatic)
    message(STATUS
      "CUDA architecture ${_one} selected automatically from the first GPU "
      "(nvidia-smi --query-gpu=compute_cap). "
      "To target another architecture, pass -DCMAKE_CUDA_ARCHITECTURES=<arch>.")
  else()
    message(STATUS
      "CUDA architecture ${_one} from CMAKE_CUDA_ARCHITECTURES. "
      "ChASE and GEMMul8 use this architecture.")
  endif()
endfunction()

function(chase_require_single_cuda_architecture)
  get_property(_decided GLOBAL PROPERTY CHASE_CUDA_ARCH_DECIDED)
  if(_decided)
    return()
  endif()
  set_property(GLOBAL PROPERTY CHASE_CUDA_ARCH_DECIDED ON)

  set(_entries "${CMAKE_CUDA_ARCHITECTURES}")
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR _entries STREQUAL "" OR _entries STREQUAL "native")
    chase_cuda_arch_from_nvidia_smi(_one)
    set(CMAKE_CUDA_ARCHITECTURES "${_one}" CACHE STRING
      "Single CUDA architecture. Unset: first GPU from nvidia-smi." FORCE)
    set(CHASE_CUDA_ARCH_FROM_SMI ON CACHE INTERNAL
      "ON when CMAKE_CUDA_ARCHITECTURES was chosen by nvidia-smi")
    set(CHASE_CUDA_ARCH_SMI_VALUE "${_one}" CACHE INTERNAL
      "Architecture last chosen by nvidia-smi")
    chase_log_cuda_architecture("${_one}" ON)
    return()
  endif()

  if(CHASE_CUDA_ARCH_FROM_SMI AND _entries STREQUAL "${CHASE_CUDA_ARCH_SMI_VALUE}")
    chase_log_cuda_architecture("${_entries}" ON)
    return()
  endif()

  set(_nums "")
  foreach(_raw IN LISTS _entries)
    string(REGEX REPLACE "-real$|-virtual$" "" _arch "${_raw}")
    if(NOT _arch MATCHES "^[0-9]+$")
      message(FATAL_ERROR
        "CMAKE_CUDA_ARCHITECTURES='${CMAKE_CUDA_ARCHITECTURES}' is not one numeric architecture. "
        "Pass a single value, for example -DCMAKE_CUDA_ARCHITECTURES=90, "
        "or leave it unset to take the first GPU from nvidia-smi.")
    endif()
    list(APPEND _nums "${_arch}")
  endforeach()
  list(REMOVE_DUPLICATES _nums)
  list(LENGTH _nums _n)
  if(_n GREATER 1)
    message(FATAL_ERROR
      "ChASE and GEMMul8 use one CUDA architecture. "
      "CMAKE_CUDA_ARCHITECTURES='${CMAKE_CUDA_ARCHITECTURES}' has more than one. "
      "Pass a single value, for example -DCMAKE_CUDA_ARCHITECTURES=90, "
      "or leave it unset to take the first GPU from nvidia-smi.")
  endif()
  list(GET _nums 0 _one)
  if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "${_one}")
    set(CMAKE_CUDA_ARCHITECTURES "${_one}" CACHE STRING
      "Single CUDA architecture. Unset: first GPU from nvidia-smi." FORCE)
  endif()
  set(CHASE_CUDA_ARCH_FROM_SMI OFF CACHE INTERNAL
    "ON when CMAKE_CUDA_ARCHITECTURES was chosen by nvidia-smi")
  chase_log_cuda_architecture("${_one}" OFF)
endfunction()

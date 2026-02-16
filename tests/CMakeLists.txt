include(GoogleTest)
find_package( MPI    REQUIRED )

function(setup_test name)
  set(sources)
  set(libraries)
  set(is_library FALSE)
  foreach(arg IN LISTS ARGN)
    if(arg STREQUAL "LIBRARIES")
      set(is_library TRUE)
    elseif(is_library)
      list(APPEND libraries ${arg})
    else()
      list(APPEND sources ${arg})
    endif()
  endforeach()

  add_executable(${name} ${sources} ${CMAKE_SOURCE_DIR}/tests/main.cpp)
  target_include_directories(${name} PRIVATE ${MPI_CXX_INCLUDE_PATH})
  target_link_libraries(${name} ${libraries} ${MPI_CXX_LIBRARIES} GTest::gtest_main)
  
  if(MPI_TEST)
    target_compile_definitions(${name} PRIVATE MPI_NUM_PROCS=4)
    if(${MPI_RUN} STREQUAL "srun")
      add_test(NAME ${name} COMMAND ${MPI_RUN} ${MPI_RUN_ARGS} -n 4 $<TARGET_FILE:${name}>)
    else()#mpirun
      add_test(NAME ${name} COMMAND ${MPI_RUN} ${MPI_RUN_ARGS} -n 1 $<TARGET_FILE:${name}> --gtest_output=xml:unittest_reports/${name}_report.xml : -n 3 $<TARGET_FILE:${name}>)
    endif()
  else()
    target_compile_definitions(${name} PRIVATE MPI_NUM_PROCS=1)
    add_test(NAME ${name} COMMAND ${MPI_RUN} ${MPI_RUN_ARGS} -n 1 $<TARGET_FILE:${name}> --gtest_output=xml:unittest_reports/${name}_report.xml)
  endif()
endfunction()

function(setup_test_serial name)
  set(sources)
  set(libraries)
  set(is_library FALSE)
  foreach(arg IN LISTS ARGN)
    if(arg STREQUAL "LIBRARIES")
      set(is_library TRUE)
    elseif(is_library)
      list(APPEND libraries ${arg})
    else()
      list(APPEND sources ${arg})
    endif()
  endforeach()

  add_executable(${name} ${sources} ${CMAKE_SOURCE_DIR}/tests/main.cpp)
  target_include_directories(${name} PRIVATE ${MPI_CXX_INCLUDE_PATH})
  target_link_libraries(${name} ${libraries} ${MPI_CXX_LIBRARIES} GTest::gtest_main)

  target_compile_definitions(${name} PRIVATE MPI_NUM_PROCS=1 SERIAL_TEST=1)
  add_test(NAME ${name} COMMAND ${MPI_RUN} ${MPI_RUN_ARGS} -n 1 $<TARGET_FILE:${name}> --gtest_output=xml:unittest_reports/${name}_report.xml)
endfunction()

# -----------------------------------------------------------------------------
# Full solve tests (based on examples: noinput.cpp and 1_hello_world.cpp)
# -----------------------------------------------------------------------------
if(TARGET chase_gpu)
  setup_test_serial(ChaseSerialSolveTest chase_serial_solve.cpp LIBRARIES chase_gpu chase_cpu)
else()
  setup_test_serial(ChaseSerialSolveTest chase_serial_solve.cpp LIBRARIES chase_cpu)
endif()

if(TARGET pchase_gpu)
  setup_test(ChaseDistributedSolveTest chase_distributed_solve.cpp LIBRARIES pchase_gpu pchase_cpu)
elseif(TARGET pchase_cpu)
  setup_test(ChaseDistributedSolveTest chase_distributed_solve.cpp LIBRARIES pchase_cpu)
endif()

# -----------------------------------------------------------------------------
# Fortran interface tests (analogous to chase_serial_solve and chase_distributed_solve)
# -----------------------------------------------------------------------------
if(TARGET chase_f)
  # Serial: single rank, Clement matrix, s/d/c/z chase (like ChaseSerialSolveTest)
  add_executable(ChaseFortranSerialSolveTest chase_fortran_serial_solve.f90)
  target_link_libraries(ChaseFortranSerialSolveTest PRIVATE chase_f MPI::MPI_Fortran)
  set_target_properties(ChaseFortranSerialSolveTest PROPERTIES
    LINKER_LANGUAGE Fortran
    Fortran_PREPROCESS ON
  )
  add_test(NAME ChaseFortranSerialSolveTest
    COMMAND ${MPI_RUN} ${MPI_RUN_ARGS} -n 1 $<TARGET_FILE:ChaseFortranSerialSolveTest>)

  # Distributed: MPI block-block p*chase with Clement (like ChaseDistributedSolveTest)
  add_executable(ChaseFortranDistributedSolveTest chase_fortran_distributed_solve.f90)
  target_link_libraries(ChaseFortranDistributedSolveTest PRIVATE chase_f MPI::MPI_Fortran)
  set_target_properties(ChaseFortranDistributedSolveTest PROPERTIES
    LINKER_LANGUAGE Fortran
    Fortran_PREPROCESS ON
  )
  add_test(NAME ChaseFortranDistributedSolveTest
    COMMAND ${MPI_RUN} ${MPI_RUN_ARGS} -n 4 $<TARGET_FILE:ChaseFortranDistributedSolveTest>)
endif()

# -----------------------------------------------------------------------------
add_subdirectory(linalg)
add_subdirectory(matrix)
if(TARGET mpi_grid)
add_subdirectory(grid)
endif()
include(GoogleTest)
find_package( MPI    REQUIRED )

add_executable(FirstTest first_test.cpp main.cpp)
target_include_directories( FirstTest PRIVATE
  ${MPI_CXX_INCLUDE_PATH}
  )

target_link_libraries(FirstTest 
  ${MPI_CXX_LIBRARIES}
  GTest::gtest_main
  )

add_test( BASIC ${CMAKE_BINARY_DIR}/chase_driver )
add_test(NAME FirstTest COMMAND ${MPI_RUN} ${MPI_RUN_ARGS} -n 1 $<TARGET_FILE:FirstTest> --gtest_output=xml:unittest_reports/FirstTest_report.xml : -n 3 $<TARGET_FILE:FirstTest>)
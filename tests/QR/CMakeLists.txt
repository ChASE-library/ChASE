file(COPY QR_matrices DESTINATION ${CMAKE_BINARY_DIR}/tests/QR)

add_executable(QRTest QR_test.cpp ../main.cpp)
target_include_directories( QRTest PRIVATE
  ${MPI_CXX_INCLUDE_PATH}
  )

target_link_libraries(QRTest 
  ${MPI_CXX_LIBRARIES}
  GTest::gtest_main
  chase_mpi
  )

add_executable(QRTestIntegration QR_integration.cpp ../main.cpp)
target_include_directories( QRTestIntegration PRIVATE
  ${MPI_CXX_INCLUDE_PATH}
  )

target_link_libraries(QRTestIntegration 
  ${MPI_CXX_LIBRARIES}
  GTest::gtest_main
  chase_mpi
  )

add_test(NAME QRTest COMMAND ${MPI_RUN} ${MPI_RUN_ARGS} -n 1 $<TARGET_FILE:QRTest> --gtest_output=xml:unittest_reports/FirstTest_report.xml : -n 3 $<TARGET_FILE:QRTest>)
add_test(NAME QRTestIntegration COMMAND ${MPI_RUN} ${MPI_RUN_ARGS} -n 1 $<TARGET_FILE:QRTestIntegration> --gtest_output=xml:unittest_reports/FirstTest_report.xml : -n 3 $<TARGET_FILE:QRTestIntegration>)
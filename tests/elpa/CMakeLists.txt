find_package(ELPA)

if(ELPA_FOUND)
message("-- Enable test for elpa")
add_executable( "elpa_test.exe" elpa.cpp )
target_link_libraries( "elpa_test.exe" PRIVATE pchase_gpu ELPA::ELPA )

add_executable( "cusolver_test.exe" cusolver.cpp )
target_link_libraries( "cusolver_test.exe" PRIVATE cusolverpp pchase_gpu)

endif()
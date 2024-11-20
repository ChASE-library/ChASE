cmake_minimum_required(VERSION 3.8)

project(chase-app VERSION 0.0.1 LANGUAGES C CXX Fortran)

#find_package(BLAS REQUIRED)
find_package( chase REQUIRED CONFIG)

add_executable( "chase_app_cpu.exe" "chase-app.cpp")
target_link_libraries( "chase_app_cpu.exe" PRIVATE ChASE::chase_mpi )

if(TARGET ChASE::chase_nccl)
    add_executable( "chase_app_gpu.exe" "chase-app.cpp")
    target_link_libraries( "chase_app_gpu.exe" PRIVATE ChASE::chase_nccl )
    target_compile_definitions( "chase_app_gpu.exe" PRIVATE USE_GPU=1)
endif()

if(TARGET ChASE::chase_c)
    message("-- ChASE C interface is detected")
    add_executable( "chase_app_c.exe" chase-c.c )
    target_link_libraries(chase_app_c.exe ChASE::chase_c)
endif()

if(TARGET ChASE::chase_f)
    message("-- ChASE fortran interface is detected")
    set(CMAKE_Fortran_COMPILER ${MPI_Fortran_COMPILER})
    add_executable( "chase_app_fortran.exe" chase-f.f90 )
    target_link_libraries(chase_app_fortran.exe  ChASE::chase_f )
    set_target_properties(
        "chase_app_fortran.exe"
        PROPERTIES Fortran_PREPROCESS ON
        POSITION_INDEPENDENT_CODE TRUE
    )
endif()


cmake_minimum_required(VERSION 3.8)

project(chase-app VERSION 0.0.1 LANGUAGES CXX Fortran)

find_package( chase REQUIRED CONFIG)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(MPI REQUIRED)
add_executable( "chase_app_cpu.exe" "chase-app.cpp")
target_link_libraries( "chase_app_cpu.exe" PRIVATE ChASE::chase_mpi )

if(TARGET ChASE::chase_nccl)
    find_package(CUDAToolkit REQUIRED)	
    add_executable( "chase_app_gpu.exe" "chase-app.cpp")
    target_link_libraries( "chase_app_gpu.exe" PRIVATE ChASE::chase_nccl )
    target_compile_definitions( "chase_app_gpu.exe" PRIVATE USE_GPU=1)
endif()

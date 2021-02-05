# ChASE Version 0.9rc
=============================

The Chebyshev Accelerated Subspace iteration Eigensolver.

We provide two versions of ChASE:
1. **ChASE-MPI**

   Using an custom MPI-based routine for the `HEMM`.

2. **ChASE-Elemental**

   Using the Elemental distributed-memory linear algebra framework.
   http://libelemental.org/

## Building ChASE
--------------

### ChASE build with the CMake system.

The following should generate a driver that demonstrates how to use ChASE:

    cd ChASE/
    mkdir build
    cd build/
    cmake ..
    make

### SIMPLE DRIVER

For a quick test and usage of the library, we provide a ready-to-use simple driver. In order to build the simple driver together with ChASE the sequence of building commands is slightly modified as below:

    cd ChASE/
    mkdir build
    cd build/    
    cmake .. -DBUILD_SIMPLEDRIVER=ON
    make

### Build with support to multithreaded BLIS library

If you want to build the ChASE against the multithreaded BLIS library one have to provide the full path to multithreaded BLIS library, such as:

    cmake .. -DBLAS_LIBRARIES="<path-to-instal-dir>/lib/libblas-mt.so"
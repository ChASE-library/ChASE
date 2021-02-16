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

### Build with Examples 

For a quick test and usage of the library, we provide several ready-to-use examples. In order to build these examples with ChASE the sequence of building commands is slightly modified as below:

    cd ChASE/
    mkdir build
    cd build/    
    cmake .. -DBUILD_WITH_EXAMPLES=ON
    make

In order to quick test of ChASE using the previous simple driver, please use follow example instead:

```bash
./examples/2_input_output/2_input_output --path_in=${MATRIX_BINARY}
```

For the test of multi-GPU support ChASE, please use:

```bash
./examples/2_input_output/2_input_output_mgpu --path_in=${MATRIX_BINARY}
```

### Build with support to multithreaded BLIS library

If you want to build the ChASE against the multithreaded BLIS library one have to provide the full path to multithreaded BLIS library, such as:

    cmake .. -DBLAS_LIBRARIES="<path-to-instal-dir>/lib/libblas-mt.so"

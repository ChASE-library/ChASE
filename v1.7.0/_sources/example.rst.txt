.. _examples-chase:

*********************************
Examples
*********************************

ChASE provides multiple examples to help users get familiar with the library. 
This section introduces these examples and explains what each one demonstrates.

.. note::

   All examples are compiled when ChASE is built with the option 
   ``-DCHASE_BUILD_WITH_EXAMPLES=ON``.

   The examples use ``popl`` (Program Options Parser Library) for command-line 
   argument parsing, which is automatically downloaded by CMake via FetchContent.
   No additional dependencies are required.

.. _hello-world-chase:

Hello World
================

The first example is 
`1_hello_world <https://github.com/ChASE-library/ChASE/tree/master/examples/1_hello_world>`_. 
This example demonstrates:

  * Basic setup of ChASE for parallel computation (CPU or GPU)
  * Construction of a Clement matrix
  * Solving a sequence of eigenproblems with matrix perturbation
  * Using the performance decorator to measure kernel performance
  * Configuration of ChASE parameters

The example constructs a simple Clement matrix and finds a given number of its 
eigenpairs. It then perturbs the matrix and solves the sequence of related 
eigenproblems, demonstrating how to reuse approximate solutions from previous 
problems for faster convergence.

Parallel I/O and Configuration
===================================

The second example is 
`2_input_output <https://github.com/ChASE-library/ChASE/tree/master/examples/2_input_output>`_. 
This example demonstrates:

  * Command-line configuration of ChASE parameters using ``popl``
  * Parallel I/O for loading distributed matrices from disk
  * Handling sequences of eigenproblems with file-based input
  * Support for both sequential and parallel implementations
  * CPU and GPU implementations

The example shows how to configure ChASE parameters from the command line and 
load matrices in parallel across multiple MPI processes. It supports both 
sequential (single-node) and parallel (multi-node) execution modes.

.. toctree::
   :maxdepth: 0

   example/io

Installation and Linking
============================

The third example is 
`3_installation <https://github.com/ChASE-library/ChASE/tree/master/examples/3_installation>`_. 
This example demonstrates how to link ChASE to external applications through CMake. 
It includes:

  * C++ application linking example (``chase-app.cpp``)
  * C interface linking example (``chase-c.c``)
  * Fortran interface linking example (``chase-f.f90``)
  * CMake configuration files for linking ChASE

This example does not build automatically with the compilation/installation of ChASE. 
It must be compiled independently by the user. For more information about linking 
ChASE to your application, please visit :ref:`link_by_cmake`.

C/Fortran Interface
============================

The fourth example is 
`4_interface <https://github.com/ChASE-library/ChASE/tree/master/examples/4_interface>`_. 
This example demonstrates the use of ChASE's C and Fortran interfaces:

  * **C Interface Examples**:
    - ``4_c_serial_chase.c``: Sequential ChASE using C interface
    - ``4_c_dist_chase.c``: Parallel ChASE using C interface

  * **Fortran Interface Examples**:
    - ``4_f_serial_chase.f90``: Sequential ChASE using Fortran interface
    - ``4_f_dist_chase.f90``: Parallel ChASE using Fortran interface

These examples show how to use ChASE from C and Fortran applications without 
directly using the C++ API. For more information about the C/Fortran interface, 
please visit :ref:`c_fortran_interface`.

BSE Benchmark
==================

The fifth example is 
`5_bse_benchmark <https://github.com/ChASE-library/ChASE/tree/master/examples/5_bse_benchmark>`_. 
This example demonstrates:

  * Solving pseudo-Hermitian eigenvalue problems (Bethe-Salpeter Equation)
  * Using ``chase::matrix::PseudoHermitianMatrix`` for BSE problems
  * Command-line configuration with extensive parameter options
  * Support for both single and double precision
  * Sequential and parallel implementations (CPU and GPU)
  * Matrix input from files

This example is specifically designed for BSE benchmark problems and shows how 
to use ChASE for pseudo-Hermitian eigenvalue problems, which require special 
handling compared to standard Hermitian problems.
